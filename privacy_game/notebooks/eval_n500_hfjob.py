#!/usr/bin/env python3
"""n=500 multi-seed eval at 0.5B — designed for `hf jobs uv-run`.

Self-contained: clones the repo, installs deps, evaluates 5 cells at n=500,
pushes per-episode results + summary to a HF Hub dataset repo.

Run from your local machine:

    hf jobs uv-run --flavor t4-small --secrets HF_TOKEN \\
        "https://raw.githubusercontent.com/RAJVEER42/META_H/main/privacy_game/notebooks/eval_n500_hfjob.py"

t4-small ($0.40/hr) is ample for 0.5B+LoRA. Expected wall-clock: ~3-4h ≈ ~$1.50.

Cells evaluated (5):
    1. Qwen2.5-0.5B-Instruct (base, no adapter)
    2. cipher-qwen-0.5b-grpo-seed42 (GRPO, training seed 42)
    3. cipher-qwen-0.5b-grpo-seed43 (GRPO, training seed 43)
    4. cipher-qwen-0.5b-grpo-seed44 (GRPO, training seed 44)
    5. cipher-qwen-0.5b-sft-baseline (SFT on smart_generalize traces)

All cells use the SAME eval task seed (0) so they see the same 500 task draws —
enables paired comparison.

Reward mode: pareto_it (env-native, default). Matches the v1 numbers in
PAPER_ROADMAP.md.

Output: pushed to hf.co/datasets/Itachi-42/cipher-eval-v3-0.5b
    {cell}/results.jsonl   per-episode rewards/utility/recon
    {cell}/summary.json    n / mean / std / sem / 95%-CI / approved-rate
    all_summary.json       aggregated table across cells
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.4,<2.7",
#     "transformers>=4.45,<4.50",
#     "peft>=0.13",
#     "accelerate>=1.0",
#     "datasets>=2.14",
#     "openenv-core>=0.2.2",
#     "huggingface_hub>=0.26",
# ]
# ///

import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 1. Bootstrap — clone repo, add to sys.path

REPO_URL = "https://github.com/RAJVEER42/META_H.git"
WORKSPACE = Path("/tmp/META_H")
if not WORKSPACE.exists():
    print(f"cloning {REPO_URL}...", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(WORKSPACE)], check=True)
sys.path.insert(0, str(WORKSPACE))

OUT_DIR = WORKSPACE / "privacy_game" / "outputs" / "eval_v3_0.5b"
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PRIVACY_GAME_REGISTRY_FILLER"] = "20000"
os.environ["PRIVACY_GAME_LOG_TRAJECTORIES"] = "0"  # skip rich trajectory logs to save disk
os.chdir(WORKSPACE / "privacy_game")

import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Cell definitions

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CELLS = [
    ("0.5b-base",         BASE_MODEL, None),
    ("0.5b-grpo-seed42",  BASE_MODEL, "Itachi-42/cipher-qwen-0.5b-grpo-seed42"),
    ("0.5b-grpo-seed43",  BASE_MODEL, "Itachi-42/cipher-qwen-0.5b-grpo-seed43"),
    ("0.5b-grpo-seed44",  BASE_MODEL, "Itachi-42/cipher-qwen-0.5b-grpo-seed44"),
    ("0.5b-sft-baseline", BASE_MODEL, "Itachi-42/cipher-qwen-0.5b-sft-baseline"),
]

N_EPISODES = 500
EVAL_SEED = 0
REWARD_MODE = "pareto_it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_REPO = "Itachi-42/cipher-eval-v3-0.5b"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Per-cell pipeline factory + policy adapter

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_pipeline(base_model: str, checkpoint: str | None):
    print(f"  loading base={base_model}, adapter={checkpoint!r}", flush=True)
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Plain .to(device) instead of device_map= to avoid accelerate hook conflict
    # with pipeline's device= argument.
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)

    if checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()  # bake adapter into weights

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device=0 if DEVICE == "cuda" else -1,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        return_full_text=False,
    )
    return pipe


# Re-use the existing prompt rendering + sanitizer to match training format
from privacy_game.eval.llm_adapter import _render_prompt, _sanitize


def make_policy(pipe):
    def policy(rp_question, profile, history):
        prompt = _render_prompt(rp_question, profile, history)
        out = pipe(prompt)[0]["generated_text"]
        return _sanitize(out)
    return policy


# ──────────────────────────────────────────────────────────────────────────────
# 4. Eval driver — calls the env directly so we get per-episode dicts

from privacy_game.server.privacy_game_environment import PrivacyGameEnvironment
from privacy_game.models import DisclosureAction


def run_n_episodes(policy, n: int, seed: int, reward_mode: str) -> list[dict]:
    env = PrivacyGameEnvironment(split="train", seed=seed, reward_mode=reward_mode, force_task_id=None)
    results = []
    t0 = time.time()
    for ep_i in range(n):
        obs = env.reset()
        history = list(obs.history)
        turns = 0
        while not obs.terminated:
            reply = policy(obs.relying_party_message, obs.profile, history)
            obs = env.step(DisclosureAction(message=reply))
            history = list(obs.history)
            turns += 1
            if turns > obs.max_turns + 2:
                break
        md = obs.metadata or {}
        results.append({
            "episode_idx":      ep_i,
            "task_id":          obs.task_id,
            "phase":            obs.phase,
            "reward":           float(obs.reward or 0.0),
            "utility":          float(md.get("utility_score", 0.0)),
            "reconstruction":   float(md.get("reconstruction_score", 0.0)),
            "verbosity":        float(md.get("verbosity_penalty", 0.0)),
            "turns":            obs.turn_number,
            "terminated_reason": obs.terminated_reason,
        })
        if (ep_i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (ep_i + 1) / elapsed
            eta = (n - ep_i - 1) / rate
            print(f"    ep {ep_i+1}/{n}  {rate:.2f} ep/s  eta {eta/60:.1f} min", flush=True)
    return results


def summarize(results: list[dict]) -> dict:
    rewards = [r["reward"] for r in results]
    utils   = [r["utility"] for r in results]
    recons  = [r["reconstruction"] for r in results]
    n = len(rewards)
    mean = statistics.mean(rewards)
    std  = statistics.stdev(rewards) if n > 1 else 0.0
    sem  = std / math.sqrt(n) if n > 1 else 0.0
    ci95 = 1.96 * sem  # large-n normal approx
    approved = sum(1 for r in results if r["terminated_reason"] == "approved")
    return {
        "n":              n,
        "mean":           mean,
        "std":            std,
        "sem":            sem,
        "ci95_half":      ci95,
        "ci95":           [mean - ci95, mean + ci95],
        "min":            min(rewards),
        "max":            max(rewards),
        "mean_utility":   statistics.mean(utils),
        "mean_recon":     statistics.mean(recons),
        "approved_rate":  approved / n,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Main loop

all_summaries = {}
all_results = {}

for label, base_model, ckpt in CELLS:
    print(f"\n========== {label} ==========", flush=True)
    pipe = load_pipeline(base_model, ckpt)
    policy = make_policy(pipe)

    t_cell = time.time()
    results = run_n_episodes(policy, n=N_EPISODES, seed=EVAL_SEED, reward_mode=REWARD_MODE)
    cell_secs = time.time() - t_cell

    summ = summarize(results)
    summ["wall_clock_sec"] = cell_secs
    summ["base_model"] = base_model
    summ["checkpoint"] = ckpt
    summ["eval_seed"] = EVAL_SEED
    summ["reward_mode"] = REWARD_MODE

    print(f"\n  {label}: mean={summ['mean']:+.4f} ± {summ['sem']:.4f} "
          f"(95% CI [{summ['ci95'][0]:+.4f}, {summ['ci95'][1]:+.4f}])  "
          f"approved={summ['approved_rate']:.0%}  in {cell_secs/60:.1f} min", flush=True)

    # save locally
    cell_dir = OUT_DIR / label
    cell_dir.mkdir(exist_ok=True)
    with open(cell_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(cell_dir / "summary.json", "w") as f:
        json.dump(summ, f, indent=2)

    all_summaries[label] = summ
    all_results[label] = results

    # free GPU memory before next cell
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# 6. Aggregate table

with open(OUT_DIR / "all_summary.json", "w") as f:
    json.dump(all_summaries, f, indent=2)

print("\n========== AGGREGATE TABLE ==========", flush=True)
print(f"{'cell':<22} {'mean':>10} {'sem':>10} {'95% CI':>22} {'approved':>10}")
for label in all_summaries:
    s = all_summaries[label]
    ci = f"[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}]"
    print(f"{label:<22} {s['mean']:+10.4f} {s['sem']:>10.4f} {ci:>22} {s['approved_rate']:>9.0%}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Push to HF Hub

from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id=TARGET_REPO, repo_type="dataset", exist_ok=True, private=False)
api.upload_folder(
    folder_path=str(OUT_DIR),
    repo_id=TARGET_REPO,
    repo_type="dataset",
    commit_message=f"n={N_EPISODES} eval, seed={EVAL_SEED}, reward={REWARD_MODE} — 5 cells (base + 3 GRPO seeds + SFT)",
)
print(f"\nUploaded to https://huggingface.co/datasets/{TARGET_REPO}", flush=True)
