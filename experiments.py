"""
Comparative Experimental Analysis of Sequential Rule Mining Algorithms:
CMRules, RuleGrowth, and ERMiner
Datasets: BMSWebView2, Leviathan | Framework: SPMF

Dependencies:
  pip install numpy pandas scipy psutil
"""

import os
import re
import subprocess
import threading
import time
import uuid

import numpy as np
import pandas as pd
import psutil
from scipy import stats

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR          = "/Users/elifyilmaz/Desktop/pattern_mining"
SPMF_JAR          = os.path.join(BASE_DIR, "spmf.jar")
DATASETS          = {
    "BMSWebView2": os.path.join(BASE_DIR, "BMS2.txt"),
    "Leviathan":   os.path.join(BASE_DIR, "leviathan.txt"),
}
ALGORITHMS        = ["CMRules", "RuleGrowth", "ERMiner"]
MIN_SUP_VALUES    = [0.02, 0.05, 0.10, 0.20, 0.30]
MIN_CONF          = 0.50
SCALABILITY_FRACS = [0.20, 0.40, 0.60, 0.80, 1.00]
SCALABILITY_SUP   = 0.10
N_REPS            = 5
N_REPS_SCALE      = 3     
SEED              = 42
TEMP_DIR          = os.path.join(BASE_DIR, "temp")
OUT_DIR           = os.path.join(BASE_DIR, "results")
JVM_HEAP          = "-Xmx6g"

MEM_POLL_INTERVAL = 0.05

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)


# ── Dataset Analysis ──────────────────────────────────────────────────────────
def parse_sequences(filepath):
 
    seqs = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            tokens = [int(t) for t in line.split()]
            items, itemsets, cur = [], [], []
            for t in tokens:
                if t == -2:
                    break
                elif t == -1:
                    if cur:
                        itemsets.append(cur)
                        cur = []
                elif t > 0:
                    cur.append(t)
                    items.append(t)
            if cur:
                itemsets.append(cur)
            if items:
                seqs.append({"items": items, "itemsets": itemsets})
    return seqs


def compute_dataset_stats(name, filepath):

    seqs      = parse_sequences(filepath)
    lengths   = [len(s["items"]) for s in seqs]
    n_isets   = [len(s["itemsets"]) for s in seqs]
    all_items = {i for s in seqs for i in s["items"]}
    n_items   = len(all_items)
    n_pairs   = n_items * (n_items - 1) // 2   # C(|I|, 2)
    return {
        "Dataset":                name,
        "Sequences":              len(seqs),
        "Distinct Items":         n_items,
        "Candidate Pairs (C2)":   n_pairs,
        "Min Seq Length":         int(np.min(lengths)),
        "Max Seq Length":         int(np.max(lengths)),
        "Mean Seq Length":        round(float(np.mean(lengths)), 2),
        "Median Seq Length":      round(float(np.median(lengths)), 2),
        "Std Seq Length":         round(float(np.std(lengths, ddof=1)), 2),
        "Avg Itemsets/Seq":       round(float(np.mean(n_isets)), 2),
        "Total Item Occurrences": sum(lengths),
        "Density (occ/seq/item)": round(sum(lengths) / (len(seqs) * n_items), 6),
    }


# ── Memory Measurement ────────────────────────────────────────────────────────
def parse_memory_mb(stdout_text: str) -> float:

    pattern = re.compile(
        r"(?:max\s+memory\s+usage|memory\s+usage)[^\d]*([0-9]+(?:\.[0-9]+)?)\s*mb",
        re.IGNORECASE,
    )
    m = pattern.search(stdout_text)
    if m:
        return float(m.group(1))
    # Secondary fallback: any line containing "memory" with a numeric value.
    for line in stdout_text.splitlines():
        if "memory" in line.lower():
            nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", line)
            if nums:
                return float(nums[-1])
    return float("nan")


class _RssSampler(threading.Thread):

    def __init__(self, pid: int, interval: float = MEM_POLL_INTERVAL):
        super().__init__(daemon=True)
        self._pid      = pid
        self._interval = interval
        self._stop_evt = threading.Event()
        self.peak_rss_mb: float = 0.0

    def run(self):
        try:
            proc = psutil.Process(self._pid)
        except psutil.NoSuchProcess:
            return
        while not self._stop_evt.is_set():
            try:
                rss = proc.memory_info().rss / (1024 ** 2)   # bytes → MB
                if rss > self.peak_rss_mb:
                    self.peak_rss_mb = rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self._interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2.0)


# ── SPMF Execution ────────────────────────────────────────────────────────────
def run_spmf(algo: str, input_file: str, min_sup: float, min_conf: float):

    out_file = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}.txt")
    sup_str  = f"{min_sup * 100:.4g}%"
    cmd = [
        "java", JVM_HEAP, "-jar", SPMF_JAR, "run", algo,
        input_file, out_file, sup_str, str(min_conf),
    ]

    # Launch subprocess.
    t0   = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Start RSS polling immediately after the process is created.
    sampler = _RssSampler(proc.pid)
    sampler.start()

    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=1800)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout_bytes, stderr_bytes = proc.communicate()
        sampler.stop()
        raise RuntimeError(f"SPMF timeout after 1800 s for {algo}")
    finally:
        sampler.stop()

    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        raise RuntimeError(
            stderr_bytes.decode(errors="replace")[:300]
        )

    # ── Memory reconciliation ─────────────────────────────────────────────
    combined_out = (
        stdout_bytes.decode(errors="replace")
        + stderr_bytes.decode(errors="replace")
    )
    mem_spmf = parse_memory_mb(combined_out)         
    mem_rss  = sampler.peak_rss_mb                 

    # Prefer SPMF-internal value (heap only); fall back to RSS (heap + JVM).
    mem_mb = mem_spmf if (not np.isnan(mem_spmf) and mem_spmf > 0) else mem_rss

    # ── Rule count ───────────────────────────────────────────────────────
    n_rules = 0
    if os.path.isfile(out_file):
        with open(out_file, encoding="utf-8") as f:
            n_rules = sum(
                1 for ln in f if ln.strip() and not ln.startswith("#")
            )
        os.remove(out_file)

    return elapsed, n_rules, mem_mb, mem_spmf, mem_rss


def repeated_run(algo: str, input_file: str,
                 min_sup: float, min_conf: float, n_reps: int) -> dict:
    times, counts = [], []
    mems, mems_spmf, mems_rss = [], [], []

    for rep in range(n_reps):
        try:
            t, n, m, m_spmf, m_rss = run_spmf(algo, input_file, min_sup, min_conf)
            times.append(t)
            counts.append(n)
            mems.append(m)
            mems_spmf.append(m_spmf)
            mems_rss.append(m_rss)
        except Exception as e:
            print(f"    [WARN] rep={rep}: {e}")
            times.append(np.nan)
            counts.append(np.nan)
            mems.append(np.nan)
            mems_spmf.append(np.nan)
            mems_rss.append(np.nan)

    t_arr  = np.array(times,      dtype=float)
    c_arr  = np.array(counts,     dtype=float)
    m_arr  = np.array(mems,       dtype=float)
    ms_arr = np.array(mems_spmf,  dtype=float)
    mr_arr = np.array(mems_rss,   dtype=float)

    vt = t_arr[~np.isnan(t_arr)]
    vc = c_arr[~np.isnan(c_arr)]
    vm = m_arr[~np.isnan(m_arr)]
    vs = ms_arr[~np.isnan(ms_arr)]
    vr = mr_arr[~np.isnan(mr_arr)]
    n  = len(vt)

    ci95 = (
        1.96 * np.std(vt, ddof=1) / np.sqrt(n) if n > 1 else np.nan
    )
    cv = (
        float(np.std(vt, ddof=1) / np.nanmean(vt))
        if n > 1 and np.nanmean(vt) > 0
        else np.nan
    )

    def _mean(v): return float(np.nanmean(v)) if len(v) else np.nan
    def _max(v):  return float(np.nanmax(v))  if len(v) else np.nan

    return {
        # ── Timing ───────────────────────────────────────────────────────
        "mean_time_s":   float(np.nanmean(t_arr)) if n else np.nan,
        "std_time_s":    float(np.std(vt, ddof=1)) if n > 1 else np.nan,
        "median_time_s": float(np.nanmedian(t_arr)) if n else np.nan,
        "min_time_s":    float(np.nanmin(t_arr)) if n else np.nan,
        "max_time_s":    float(np.nanmax(t_arr)) if n else np.nan,
        "cv_time":       cv,
        "ci95_s":        float(ci95),
        # ── Rule count ───────────────────────────────────────────────────
        "mean_n_rules":  float(np.nanmean(c_arr)) if n else np.nan,
        "std_n_rules":   float(np.nanstd(vc, ddof=1)) if n > 1 else np.nan,
        # ── Memory (reconciled best source) ──────────────────────────────
        "mean_mem_mb":   _mean(vm),
        "max_mem_mb":    _max(vm),
        # ── Memory (SPMF-reported, may be NaN for RuleGrowth / ERMiner) ──
        "mean_mem_spmf_mb": _mean(vs),
        "max_mem_spmf_mb":  _max(vs),
        # ── Memory (psutil RSS — always available) ────────────────────────
        "mean_mem_rss_mb": _mean(vr),
        "max_mem_rss_mb":  _max(vr),
        # ── Bookkeeping ───────────────────────────────────────────────────
        "n_valid_runs":  n,
    }


# ── Main Experiments ──────────────────────────────────────────────────────────
def run_main_experiments() -> pd.DataFrame:

    records = []
    total   = len(DATASETS) * len(ALGORITHMS) * len(MIN_SUP_VALUES)
    done    = 0
    for ds_name, ds_path in DATASETS.items():
        for algo in ALGORITHMS:
            for sup in MIN_SUP_VALUES:
                done += 1
                print(
                    f"  [{done:2d}/{total}] {ds_name:<14} | "
                    f"{algo:<12} | minsup={sup:.0%}"
                )
                s = repeated_run(algo, ds_path, sup, MIN_CONF, N_REPS)
                records.append(
                    {"dataset": ds_name, "algorithm": algo,
                     "min_support": sup, **s}
                )
    return pd.DataFrame(records)


# ── Scalability Experiments ───────────────────────────────────────────────────
def write_subset(seqs: list, filepath: str) -> None:
 
    with open(filepath, "w", encoding="utf-8") as f:
        for s in seqs:
            f.write(" ".join(str(i) for i in s["items"]) + " -1 -2\n")


def run_scalability() -> pd.DataFrame:

    records = []
    rng     = np.random.default_rng(SEED)
    total   = len(DATASETS) * len(ALGORITHMS) * len(SCALABILITY_FRACS)
    done    = 0
    for ds_name, ds_path in DATASETS.items():
        seqs = parse_sequences(ds_path)
        for algo in ALGORITHMS:
            for frac in SCALABILITY_FRACS:
                done += 1
                print(
                    f"  [{done:2d}/{total}] SCALABILITY | "
                    f"{ds_name:<14} | {algo:<12} | {frac:.0%}"
                )
                n_take = max(1, int(len(seqs) * frac))
                idx    = rng.choice(len(seqs), size=n_take, replace=False)
                subset = [seqs[i] for i in sorted(idx)]
                tmp    = os.path.join(
                    TEMP_DIR, f"sub_{ds_name}_{frac:.2f}.txt"
                )
                write_subset(subset, tmp)
                s = repeated_run(
                    algo, tmp, SCALABILITY_SUP, MIN_CONF, N_REPS_SCALE
                )
                if os.path.isfile(tmp):
                    os.remove(tmp)
                records.append(
                    {
                        "base_dataset": ds_name,
                        "algorithm":    algo,
                        "fraction":     frac,
                        "n_sequences":  n_take,
                        **s,
                    }
                )
    return pd.DataFrame(records)


# ── Scalability Regression Analysis ──────────────────────────────────────────
def analyze_scalability(df_scale: pd.DataFrame) -> pd.DataFrame:

    records = []
    for (ds, algo), grp in df_scale.groupby(["base_dataset", "algorithm"]):
        grp = grp.dropna(subset=["mean_time_s"]).sort_values("n_sequences")
        x   = grp["n_sequences"].values.astype(float)
        y   = grp["mean_time_s"].values.astype(float)
        if len(x) < 3:
            continue

        r, pval = stats.pearsonr(x, y)

        # Linear fit
        slope, intercept, r_lin, _, _ = stats.linregress(x, y)
        r2_lin     = r_lin ** 2
        y_pred_lin = slope * x + intercept
        ss_res_lin = np.sum((y - y_pred_lin) ** 2)
        aic_lin    = len(x) * np.log(ss_res_lin / len(x)) + 2 * 2   # k=2

        # Quadratic fit
        try:
            coeffs   = np.polyfit(x, y, 2)
            y_pred_q = np.polyval(coeffs, x)
            ss_res_q = np.sum((y - y_pred_q) ** 2)
            ss_tot   = np.sum((y - np.mean(y)) ** 2)
            r2_quad  = 1 - ss_res_q / ss_tot if ss_tot > 0 else np.nan
            aic_quad = len(x) * np.log(ss_res_q / len(x)) + 2 * 3   # k=3
            best_fit = "linear" if aic_lin <= aic_quad else "quadratic"
        except Exception:
            r2_quad  = np.nan
            aic_quad = np.nan
            best_fit = "linear"

        records.append(
            {
                "dataset":          ds,
                "algorithm":        algo,
                "pearson_r":        round(r, 4),
                "pearson_p":        round(pval, 6),
                "r2_linear":        round(r2_lin, 4),
                "r2_quadratic":     round(r2_quad, 4)
                                    if not np.isnan(r2_quad) else np.nan,
                "slope_linear":     round(slope, 8),
                "best_fit_model":   best_fit,
            }
        )
    return pd.DataFrame(records)


# ── CMRules Overhead Analysis (sparse data) ───────────────────────────────────
def analyze_cmrules_overhead(
    df_stats: pd.DataFrame,
    df_main:  pd.DataFrame,
) -> pd.DataFrame:

    records = []
    for ds in df_main["dataset"].unique():
        ds_stats = df_stats[df_stats["Dataset"] == ds].iloc[0]
        for sup in df_main["min_support"].unique():
            sub  = df_main[
                (df_main["dataset"] == ds) & (df_main["min_support"] == sup)
            ]
            t_cm = sub[sub["algorithm"] == "CMRules"]["mean_time_s"].values
            t_rg = sub[sub["algorithm"] == "RuleGrowth"]["mean_time_s"].values
            t_er = sub[sub["algorithm"] == "ERMiner"]["mean_time_s"].values
            if not (len(t_cm) and len(t_rg) and t_rg[0] > 0):
                continue
            records.append(
                {
                    "dataset":              ds,
                    "min_support":          sup,
                    "candidate_pairs":      ds_stats.get(
                                                "Candidate Pairs (C2)", np.nan
                                            ),
                    "density":              ds_stats.get(
                                                "Density (occ/seq/item)", np.nan
                                            ),
                    "mean_rules":           sub[
                                                sub["algorithm"] == "CMRules"
                                            ]["mean_n_rules"].values[0],
                    "time_CMRules":         round(t_cm[0], 4),
                    "time_RuleGrowth":      round(t_rg[0], 4),
                    "time_ERMiner":         round(t_er[0], 4)
                                            if len(t_er) else np.nan,
                    "overhead_ratio_vs_RG": round(t_cm[0] / t_rg[0], 3),
                    "overhead_ratio_vs_ER": (
                        round(t_cm[0] / t_er[0], 3)
                        if len(t_er) and t_er[0] > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Sequential Rule Mining --- Comparative Experiment v3")
    print(f"Seed: {SEED} | Reps (main): {N_REPS} | Reps (scale): {N_REPS_SCALE}")
    print(f"MinConf: {MIN_CONF} | Mem poll interval: {MEM_POLL_INTERVAL} s")
    print("Memory source: SPMF stdout (preferred) + psutil RSS (fallback)")
    print("=" * 60)

    # 1. Dataset Statistics
    print("\n[1/5] Computing Dataset Statistics ...")
    ds_rows  = [compute_dataset_stats(n, p) for n, p in DATASETS.items()]
    df_stats = pd.DataFrame(ds_rows)
    print(df_stats.to_string(index=False))
    df_stats.to_csv(os.path.join(OUT_DIR, "dataset_stats.csv"), index=False)
    print("  Saved: dataset_stats.csv")

    # 2. Main Experiments
    print("\n[2/5] Running Main Experiments ...")
    df_main = run_main_experiments()
    df_main.to_csv(os.path.join(OUT_DIR, "main_results.csv"), index=False)
    print("  Saved: main_results.csv")
    print("  Memory columns (reconciled)  : mean_mem_mb, max_mem_mb")
    print("  Memory columns (SPMF-reported): mean_mem_spmf_mb, max_mem_spmf_mb")
    print("  Memory columns (psutil RSS)  : mean_mem_rss_mb, max_mem_rss_mb")

    # 3. Scalability Experiments
    print("\n[3/5] Running Scalability Experiments ...")
    df_scale = run_scalability()
    df_scale.to_csv(os.path.join(OUT_DIR, "scalability_results.csv"), index=False)
    print("  Saved: scalability_results.csv")

    # 4. Scalability Regression Analysis
    print("\n[4/5] Scalability Regression Analysis ...")
    df_regression = analyze_scalability(df_scale)
    df_regression.to_csv(
        os.path.join(OUT_DIR, "scalability_regression.csv"), index=False
    )
    print(df_regression.to_string(index=False))
    print("  Saved: scalability_regression.csv")

    # 5. CMRules Overhead Analysis
    print("\n[5/5] CMRules Overhead Analysis ...")
    df_overhead = analyze_cmrules_overhead(df_stats, df_main)
    df_overhead.to_csv(
        os.path.join(OUT_DIR, "cmrules_overhead.csv"), index=False
    )
    print(df_overhead.to_string(index=False))
    print("  Saved: cmrules_overhead.csv")

    print(f"\nAll results saved to: {OUT_DIR}")