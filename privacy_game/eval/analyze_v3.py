"""Paired analysis of cipher-eval-v3-0.5b results.

Pulls per-episode JSONLs from `Itachi-42/cipher-eval-v3-0.5b` (or a local dir),
runs paired-by-task statistics, and prints the verdict.

Why paired analysis matters
───────────────────────────
All cells in this eval used the same task seed (0), so the same 200
(profile, task) draws are graded across base / GRPO / SFT. That means we
can subtract base reward from trained reward *per episode* and look at the
distribution of deltas — which removes the task-difficulty variance that
otherwise dominates the per-cell std (~0.4).

For independent comparison the n=200 95% CI half-width is ~0.026.
For paired comparison (same tasks) it can be 3-5× narrower, often
making a 0.05-0.10 Δ statistically significant.

Usage
─────
    # default: pull from HF Hub
    python -m privacy_game.eval.analyze_v3

    # local mode (after `hf download` of the dataset)
    python -m privacy_game.eval.analyze_v3 --local /path/to/dataset
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO = "Itachi-42/cipher-eval-v3-0.5b"

CELLS = [
    "0.5b-base",
    "0.5b-grpo-seed42",
    "0.5b-grpo-seed43",
    "0.5b-grpo-seed44",
    "0.5b-sft-baseline",
]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading

def _load_local(root: Path) -> dict[str, list[dict]]:
    out = {}
    for cell in CELLS:
        p = root / cell / "results.jsonl"
        if not p.exists():
            print(f"  (missing locally: {p})")
            continue
        with open(p) as f:
            out[cell] = [json.loads(line) for line in f if line.strip()]
        print(f"  loaded {cell}: {len(out[cell])} episodes")
    return out


def _load_hub(repo: str) -> dict[str, list[dict]]:
    """Download each cell's results.jsonl from the dataset repo and parse."""
    from huggingface_hub import hf_hub_download
    out = {}
    for cell in CELLS:
        try:
            local = hf_hub_download(
                repo_id=repo, repo_type="dataset",
                filename=f"{cell}/results.jsonl",
            )
        except Exception as e:
            print(f"  (skipping {cell}: {e})")
            continue
        with open(local) as f:
            out[cell] = [json.loads(line) for line in f if line.strip()]
        print(f"  loaded {cell}: {len(out[cell])} episodes")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stats helpers (no scipy — math.erfc is enough for large-n)

def _normal_p_two_sided(z: float) -> float:
    """P(|Z| > z) under a standard normal — fine for n>=30."""
    return math.erfc(abs(z) / math.sqrt(2))


def _summarize(rewards: list[float]) -> dict:
    n = len(rewards)
    mean = statistics.mean(rewards) if n else 0.0
    std  = statistics.stdev(rewards) if n > 1 else 0.0
    sem  = std / math.sqrt(n) if n > 0 else 0.0
    return {
        "n":        n,
        "mean":     mean,
        "std":      std,
        "sem":      sem,
        "ci95":     [mean - 1.96 * sem, mean + 1.96 * sem],
    }


def _independent_delta(trained: list[dict], base: list[dict]) -> dict:
    """Welch's-style independent comparison. No pairing assumed.

    Use this when the env's RNG state diverges across cells (which happens in
    cipher-eval-v3 because the RP and task sampler share self._rng — see notes).
    """
    a = [r["reward"] for r in trained]
    b = [r["reward"] for r in base]
    na, nb = len(a), len(b)
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = statistics.variance(a) if na > 1 else 0.0
    vb = statistics.variance(b) if nb > 1 else 0.0
    diff = ma - mb
    se   = math.sqrt(va / na + vb / nb) if (na > 0 and nb > 0) else 0.0
    ci   = 1.96 * se
    t    = diff / se if se > 0 else 0.0
    p    = _normal_p_two_sided(t)
    return {
        "n_a":     na,
        "n_b":     nb,
        "mean":    diff,
        "sem":     se,
        "ci95":    [diff - ci, diff + ci],
        "t_stat":  t,
        "p_value": p,
    }


def _matched_paired_delta(trained: list[dict], base: list[dict]) -> dict:
    """Paired comparison restricted to episodes where task_id agrees across
    cells. The env's RNG drift means most episodes are not paired, but the
    matched subset (~30-40% of n) is properly paired and gives stronger
    signal per sample than the independent comparison.

    Matches by episode_idx AND task_id — only counts an episode if both cells
    sampled the same task at that index.
    """
    by_idx_b = {r["episode_idx"]: r for r in base}
    deltas: list[float] = []
    matched_pairs = 0
    for t in trained:
        b = by_idx_b.get(t["episode_idx"])
        if b is None:
            continue
        if t.get("task_id") != b.get("task_id"):
            continue
        deltas.append(t["reward"] - b["reward"])
        matched_pairs += 1

    n = len(deltas)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "sem": 0.0,
                "ci95": [0.0, 0.0], "t_stat": 0.0, "p_value": 1.0,
                "wins": 0, "losses": 0, "ties": 0, "deltas": [],
                "match_rate": 0.0}

    mean  = statistics.mean(deltas)
    std   = statistics.stdev(deltas) if n > 1 else 0.0
    sem   = std / math.sqrt(n) if n > 0 else 0.0
    ci    = 1.96 * sem
    t_stat = mean / sem if sem > 0 else 0.0
    p     = _normal_p_two_sided(t_stat)
    wins  = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties  = n - wins - losses
    return {
        "n":          n,
        "mean":       mean,
        "std":        std,
        "sem":        sem,
        "ci95":       [mean - ci, mean + ci],
        "t_stat":     t_stat,
        "p_value":    p,
        "wins":       wins,
        "losses":     losses,
        "ties":       ties,
        "deltas":     deltas,
        "match_rate": matched_pairs / max(len(trained), len(base)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Reporting

def _fmt_ci(s: dict) -> str:
    return f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"


def _stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return ""


def report(data: dict[str, list[dict]]) -> None:
    if "0.5b-base" not in data:
        print("\nERROR: base cell missing — cannot compute deltas")
        return

    base = data["0.5b-base"]
    trained_seeds = [c for c in ("0.5b-grpo-seed42", "0.5b-grpo-seed43", "0.5b-grpo-seed44") if c in data]
    sft = data.get("0.5b-sft-baseline")

    # ── Per-cell means
    print("\n========== PER-CELL SUMMARIES ==========")
    print(f"{'cell':<24} {'n':>4} {'mean':>10} {'sem':>8} {'95% CI':>22}")
    for cell, ep_results in data.items():
        rewards = [r["reward"] for r in ep_results]
        s = _summarize(rewards)
        print(f"{cell:<24} {s['n']:>4} {s['mean']:+10.4f} {s['sem']:>8.4f} {_fmt_ci(s):>22}")

    # ── Independent comparison (Welch-style) — primary result
    # The env shares self._rng between task sampling and the relying party, so
    # different conversation lengths cause RNG drift across cells. We can't
    # rely on full pairing; independent comparison is the conservative default.
    print("\n========== INDEPENDENT Δ: trained vs base (Welch-style, primary) ==========")
    print(f"{'cell':<24} {'mean Δ':>10} {'SE(Δ)':>10} {'95% CI':>26} {'p-val':>10}")
    indep_per_seed: list[dict] = []
    for cell in trained_seeds + ([s for s in [sft and "0.5b-sft-baseline"] if s] if sft else []):
        if cell not in data:
            continue
        ind = _independent_delta(data[cell], base)
        indep_per_seed.append((cell, ind))
        print(f"{cell:<24} {ind['mean']:+10.4f} {ind['sem']:>10.4f} "
              f"{_fmt_ci(ind):>26} {ind['p_value']:>9.4f}{_stars(ind['p_value'])}")

    # ── Pooled GRPO across seeds vs base (independent)
    if len(trained_seeds) >= 2:
        pooled_rewards = []
        for cell in trained_seeds:
            pooled_rewards.extend(r["reward"] for r in data[cell])
        b_rewards = [r["reward"] for r in base]
        na, nb = len(pooled_rewards), len(b_rewards)
        ma, mb = statistics.mean(pooled_rewards), statistics.mean(b_rewards)
        va = statistics.variance(pooled_rewards) if na > 1 else 0.0
        vb = statistics.variance(b_rewards) if nb > 1 else 0.0
        diff = ma - mb
        se   = math.sqrt(va / na + vb / nb) if (na and nb) else 0.0
        ci   = 1.96 * se
        t    = diff / se if se > 0 else 0.0
        p    = _normal_p_two_sided(t)
        print(f"\n  pooled GRPO ({len(trained_seeds)} seeds, n={na}) vs base (n={nb}):")
        print(f"    mean Δ = {diff:+.4f}  SE(Δ) = {se:.4f}  "
              f"95% CI = [{diff - ci:+.4f}, {diff + ci:+.4f}]")
        print(f"    t = {t:+.3f}, two-sided p = {p:.4f}{_stars(p)}")

    # ── Matched-subset paired comparison (only episodes with same task_id)
    print("\n========== MATCHED-SUBSET PAIRED Δ (same task_id by chance) ==========")
    print("  (Env's RNG diverges, so this only uses episodes that happened to align.)")
    print(f"{'cell':<24} {'matched n':>10} {'match%':>8} {'mean Δ':>10} {'paired SEM':>12} {'95% CI':>26} {'p-val':>10}")
    for cell in trained_seeds + (["0.5b-sft-baseline"] if sft else []):
        if cell not in data:
            continue
        m = _matched_paired_delta(data[cell], base)
        if m["n"] == 0:
            print(f"{cell:<24} {'(no matched episodes)':>30}")
            continue
        print(f"{cell:<24} {m['n']:>10} {m['match_rate']:>7.0%} {m['mean']:+10.4f} "
              f"{m['sem']:>12.4f} {_fmt_ci(m):>26} {m['p_value']:>9.4f}{_stars(m['p_value'])}")

    # ── Pooled matched-paired across GRPO seeds
    if len(trained_seeds) >= 2:
        all_paired = []
        for cell in trained_seeds:
            m = _matched_paired_delta(data[cell], base)
            all_paired.extend(m["deltas"])
        n = len(all_paired)
        if n > 1:
            mean = statistics.mean(all_paired)
            std  = statistics.stdev(all_paired)
            sem  = std / math.sqrt(n)
            ci   = 1.96 * sem
            t    = mean / sem if sem > 0 else 0.0
            p    = _normal_p_two_sided(t)
            print(f"\n  pooled matched-paired GRPO across {len(trained_seeds)} seeds (n={n}):")
            print(f"    mean Δ = {mean:+.4f}  paired SEM = {sem:.4f}  "
                  f"95% CI = [{mean - ci:+.4f}, {mean + ci:+.4f}]")
            print(f"    t = {t:+.3f}, two-sided p = {p:.4f}{_stars(p)}")

    # ── Markdown table for README/paper (independent — the defensible claim)
    print("\n========== MARKDOWN TABLE (independent comparison) ==========")
    print("| Cell | n | Mean reward | 95% CI | Δ vs base | SE(Δ) | p |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    base_summary = _summarize([r["reward"] for r in base])
    print(f"| 0.5b-base | {base_summary['n']} | {base_summary['mean']:+.4f} | {_fmt_ci(base_summary)} | — | — | — |")
    for cell in CELLS[1:]:
        if cell not in data:
            continue
        rewards = [r["reward"] for r in data[cell]]
        s = _summarize(rewards)
        ind = _independent_delta(data[cell], base)
        print(f"| {cell} | {s['n']} | {s['mean']:+.4f} | {_fmt_ci(s)} | "
              f"**{ind['mean']:+.4f}** | {ind['sem']:.4f} | {ind['p_value']:.4f}{_stars(ind['p_value'])} |")

    print("\nLegend: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10")
    print("\nNOTE: Env shares self._rng between task sampling and RP — paired comparison")
    print("      is only valid for the matched subset. Independent comparison is the")
    print("      defensible primary claim for this run. Future evals should pin a")
    print("      separate task_rng.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--local", type=Path, help="Path to a local dataset dir (with {cell}/results.jsonl)")
    p.add_argument("--repo",  default=REPO, help=f"HF Hub dataset repo id (default: {REPO})")
    args = p.parse_args(argv)

    if args.local:
        print(f"Loading local results from {args.local}")
        data = _load_local(args.local)
    else:
        print(f"Loading results from hf.co/datasets/{args.repo}")
        data = _load_hub(args.repo)

    if not data:
        print("\nNo cells loaded — nothing to analyze.")
        return 1

    report(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
