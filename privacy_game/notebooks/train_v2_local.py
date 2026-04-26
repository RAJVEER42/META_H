# ruff: noqa: E402
# pyright: reportMissingImports=false
"""GRPO v2 training — local Windows / RTX 4060 runner.

A single-file, walk-away training script that mirrors `docs/HANDOFF_COLAB.md`
Block 1 (v2 sharper reward, lr=1e-5, 200 steps) but is adapted for a local
RTX 4060 Laptop GPU (Ada Lovelace, bf16-native, 8GB VRAM) on Windows.

Three modes:

    # 1. Sanity-check the v2 reward shape (5 min, no GPU heavy lifting)
    python notebooks/train_v2_local.py --smoke

    # 2. Train (45-60 min on RTX 4060 Laptop, ~50 min wall expected)
    python notebooks/train_v2_local.py --train

    # 3. Eval the trained adapter vs base on n=50 held-out
    python notebooks/train_v2_local.py --eval

    # All three back-to-back:
    python notebooks/train_v2_local.py --smoke --train --eval

Key differences vs the Colab handoff:
- bf16 forced ON (RTX 4060 / Ada Lovelace supports it natively; the Colab
  auto-detect string list — A100/H100/L4/L40 — would miss your card).
- No `!pip` / `!git clone` shell magics.
- Windows-friendly paths via `pathlib`.
- Trajectory/metrics dirs default under `outputs/` relative to repo root,
  not `/content/META_H`.
- `--num-generations 2` flag for OOM fallback (pass `--num-generations 2` if
  you hit `CUDA out of memory` on the first GRPO step).
- `--push-hub` flag pushes the adapter to HF Hub when training is done.

Run from `privacy_game/` directory (the package root). Activate the venv
first:
    .\.venv\Scripts\Activate.ps1
    cd privacy_game
    python notebooks/train_v2_local.py --smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr so emoji + arrows print on Windows (cp1252 default).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Repo bootstrap — make sure we can import `privacy_game` whether the user runs
# from the repo root, the `privacy_game/` dir, or the `notebooks/` dir.

_HERE = Path(__file__).resolve()
_PRIVACY_GAME_DIR = _HERE.parent.parent          # .../META_H/privacy_game
_REPO_ROOT = _PRIVACY_GAME_DIR.parent            # .../META_H

# Insert repo root so `import privacy_game.server...` works on a fresh shell.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Env-var defaults (set BEFORE importing privacy_game so the registry size and
# split sizes use these). Match HANDOFF_COLAB Block 1.3.

os.environ.setdefault("PRIVACY_GAME_N_TRAIN", "200")
os.environ.setdefault("PRIVACY_GAME_N_HOLDOUT", "40")
os.environ.setdefault("PRIVACY_GAME_REGISTRY_FILLER", "20000")
os.environ.setdefault("PRIVACY_GAME_LOG_TRAJECTORIES", "1")
os.environ.setdefault(
    "PRIVACY_GAME_TRAJECTORY_DIR",
    str(_PRIVACY_GAME_DIR / "outputs" / "trajectories_v2"),
)


# ──────────────────────────────────────────────────────────────────────────────
# Constants — paths, model id, etc.

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

OUT_DIR = _PRIVACY_GAME_DIR / "outputs" / "grpo_v2"
ADAPTER_DIR = _PRIVACY_GAME_DIR / "outputs" / "grpo_adapter_v2"
METRICS_PATH = _PRIVACY_GAME_DIR / "outputs" / "metrics" / "grpo_run_v2.jsonl"

# Match the env-var default above (avoid two sources of truth).
TRAJECTORY_DIR = Path(os.environ["PRIVACY_GAME_TRAJECTORY_DIR"])

DEFAULT_HF_REPO = "RAJVEER42/disclosure-game-qwen-0.5b-grpo-v2"


# ──────────────────────────────────────────────────────────────────────────────
# Prompt rendering — must match `eval/llm_adapter.py` so trained model
# generalizes correctly at inference time.

SYSTEM = (
    "You are a privacy-aware persona in a multi-turn disclosure game. "
    "An automated relying party will ask for specific fields to complete a task. "
    "An off-screen adversary scans the transcript and tries to reconstruct your "
    "PROTECTED FIELDS. Your goal: provide enough information to satisfy the task "
    "(REQUIRED FIELDS at the indicated tier or generalized form) while making it "
    "impossible for the adversary to recover any PROTECTED FIELD. Reply in one "
    "or two short sentences — no lists, no JSON, no markdown."
)


def _format_prompt(obs) -> str:
    """Render an env observation into the GRPO training/eval prompt."""
    relevant = list(obs.required_fields) + list(obs.protected_fields) + [
        "first_name", "last_name", "full_name", "email", "phone", "ssn",
        "zip_code", "date_of_birth", "gender", "employer", "medication", "diagnosis",
        "religion", "political_affiliation", "health_status", "street_address",
        "city", "state",
    ]
    seen, profile_lines = set(), []
    for k in relevant:
        if k in seen or k not in obs.profile:
            continue
        seen.add(k)
        profile_lines.append(f"  {k}: {obs.profile[k]}")
    return (
        f"{SYSTEM}\n\n"
        f"=== TASK [{obs.task_id} · {obs.phase}] ===\n{obs.task_description}\n\n"
        f"YOUR PROFILE:\n" + "\n".join(profile_lines) + "\n\n"
        f"REQUIRED FIELDS: {', '.join(obs.required_fields)}\n"
        f"PROTECTED FIELDS: {', '.join(obs.protected_fields)}\n\n"
        f"Relying party (turn {obs.turn_number+1}/{obs.max_turns}): "
        f"{obs.relying_party_message}\n\n"
        f"Your reply:"
    )


# ──────────────────────────────────────────────────────────────────────────────
# v2 sharper reward — see HANDOFF_COLAB.md Block 1.4. Keep in sync.

def _v2_disclosure_reward(prompts, completions, episode_seed, task_id, **kwargs):
    """V2 reward: utility +1.0, smart_bonus +0.6 max, leak −1.0 per raw PII.

    Sharpens the gradient vs v1's full-episode rollout reward (which was
    drowned out by the smart-policy tail). Single-turn credit assignment:
    the model's reply is scored directly, no further env steps.
    """
    from privacy_game.models import DisclosureAction
    from privacy_game.server.privacy_game_environment import PrivacyGameEnvironment

    rewards: list[float] = []
    for ep_seed, t_id, completion in zip(episode_seed, task_id, completions):
        env_local = PrivacyGameEnvironment(
            seed=int(ep_seed), force_task_id=t_id, reward_mode="pareto_it",
        )
        obs0 = env_local.reset()
        obs1 = env_local.step(DisclosureAction(message=completion))
        md = obs1.metadata or {}

        if obs1.terminated:
            collected = list((md.get("collected_fields", {}) or {}).keys())
        else:
            collected = md.get("collected_this_turn", []) or []

        n_required = max(1, len(obs0.required_fields))
        utility = min(1.0, len(collected) / n_required)

        leak = 0.0
        gold_zip = str(obs0.profile.get("zip_code", "")).strip()
        if len(gold_zip) == 5 and gold_zip.isdigit() and gold_zip in completion:
            leak += 1.0
        if re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", completion):
            leak += 1.0
        if re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", completion):
            leak += 1.0
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", completion):
            leak += 1.0
        for f in obs0.protected_fields:
            val = str(obs0.profile.get(f, "")).strip().lower()
            if val and len(val) >= 4 and val in completion.lower():
                leak += 0.5

        smart_bonus = 0.0
        if gold_zip and len(gold_zip) >= 3:
            if gold_zip[:3] in completion and gold_zip not in completion:
                smart_bonus += 0.3
        gold_dob = str(obs0.profile.get("date_of_birth", "")).strip()
        if len(gold_dob) >= 4:
            year = gold_dob[:4]
            full_dob = (
                re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", completion)
                or re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", completion)
            )
            if year in completion and not full_dob:
                smart_bonus += 0.3

        reward = max(-1.0, min(1.5, utility + smart_bonus - leak))
        rewards.append(float(reward))
    return rewards


# ──────────────────────────────────────────────────────────────────────────────
# Dataset

def _build_dataset(n_examples: int, seed: int):
    from datasets import Dataset

    from privacy_game.server.privacy_game_environment import PrivacyGameEnvironment
    from privacy_game.server.tasks import ALL_TASKS_BY_ID

    rng = random.Random(seed)
    task_ids = list(ALL_TASKS_BY_ID.keys())
    rows = []
    for _ in range(n_examples):
        ep_seed = rng.randint(0, 2**31 - 1)
        t_id = rng.choice(task_ids)
        env_local = PrivacyGameEnvironment(
            seed=ep_seed, force_task_id=t_id, reward_mode="pareto_it",
        )
        obs = env_local.reset()
        rows.append({
            "prompt": _format_prompt(obs),
            "episode_seed": ep_seed,
            "task_id": t_id,
        })
    return Dataset.from_list(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1 — smoke test

def run_smoke() -> bool:
    """Reward-shape gating signal. Returns True iff the reward looks correct.

    Run BEFORE training. If `smart-3field` < +1.0 OR `zip5-only` > 0, the
    reward is broken on this machine and you should NOT proceed to training.
    """
    print("\n" + "=" * 72)
    print("v2 reward smoke test — gating signal before training")
    print("=" * 72)

    # Heavy import; do it lazily so `--help` is fast.
    from privacy_game.server.privacy_game_environment import PrivacyGameEnvironment
    from privacy_game.server.tasks import ALL_TASKS_BY_ID

    # Rebuild registry to 20k entries to match HANDOFF Block 1.3 (sharper
    # adversary). The default 9500 is too small; Sweeney returns 1.0 for
    # almost every reply and the reward gradient flatlines.
    import privacy_game.server.privacy_game_environment as pge
    from privacy_game.server.adversary import build_registry
    pge._REGISTRY = build_registry(
        pge._TRAIN_PROFILES + pge._HOLDOUT_PROFILES,
        extra_size=20000, seed=43,
    )
    print(f"registry rebuilt: {sum(len(v) for v in pge._REGISTRY.by_tuple.values())} entries")

    # Build a tiny dataset so we can pick a P3-A row (Sweeney triangulation
    # task) — this is where the reward signal is sharpest.
    train_ds = _build_dataset(64, seed=42)
    p3a_rows = [r for r in train_ds if r["task_id"] == "P3-A"]
    row = p3a_rows[0] if p3a_rows else train_ds[0]

    env_test = PrivacyGameEnvironment(
        seed=row["episode_seed"], force_task_id=row["task_id"], reward_mode="pareto_it",
    )
    obs_test = env_test.reset()
    print(
        f"task={row['task_id']}  zip={obs_test.profile['zip_code']!r} "
        f"dob={obs_test.profile['date_of_birth']!r} "
        f"gender={obs_test.profile.get('gender', '?')!r}\n"
    )

    candidates = [
        ("smart-3field", f"I'm in the {obs_test.profile['zip_code'][:3]}XX area, "
                         f"born {obs_test.profile['date_of_birth'][:4]}, "
                         f"{obs_test.profile['gender']}."),
        ("smart-1field", f"I'm in the {obs_test.profile['zip_code'][:3]}XX area."),
        ("refuse",       "I'd rather not share that."),
        ("zip5-only",    obs_test.profile["zip_code"]),
        ("over-share",   f"My zip is {obs_test.profile['zip_code']}, "
                         f"born {obs_test.profile['date_of_birth']}."),
        ("ssn-leak",     "My SSN is 123-45-6789 and zip is 197XX."),
    ]

    table = {}
    for label, completion in candidates:
        r = _v2_disclosure_reward(
            prompts=[row["prompt"]],
            completions=[completion],
            episode_seed=[row["episode_seed"]],
            task_id=[row["task_id"]],
        )[0]
        table[label] = r
        print(f"  {label:<14s}  reward={r:+.4f}")

    smart3 = table.get("smart-3field", -999)
    zip5 = table.get("zip5-only", 999)
    overshare = table.get("over-share", 999)
    ok = (smart3 >= 1.0) and (zip5 < 0) and (overshare < 0)

    print()
    if ok:
        print("✅ smoke PASS — reward shape is correct, safe to train")
    else:
        print("❌ smoke FAIL — reward shape is WRONG, do NOT train v2")
        print("   expected: smart-3field >= +1.0  AND  zip5-only < 0  AND  over-share < 0")
        print(f"   got:      smart-3field = {smart3:+.4f}  zip5-only = {zip5:+.4f}  over-share = {overshare:+.4f}")
    print("=" * 72)
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2 — training

def run_train(num_generations: int = 4, max_steps: int = 200, lr: float = 1e-5,
              seed: int = 42) -> Path:
    """GRPO v2 training. Returns the adapter directory path on success."""

    print("\n" + "=" * 72)
    print(f"v2 GRPO training — {max_steps} steps, lr={lr}, num_generations={num_generations}")
    print("=" * 72)

    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    # Repeat the registry rebuild here so `--train` works as a standalone call
    # (not just after `--smoke`).
    import privacy_game.server.privacy_game_environment as pge
    from privacy_game.server.adversary import build_registry
    pge._REGISTRY = build_registry(
        pge._TRAIN_PROFILES + pge._HOLDOUT_PROFILES,
        extra_size=20000, seed=43,
    )
    print(f"registry: {sum(len(v) for v in pge._REGISTRY.by_tuple.values())} entries")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() == False. Either you installed the CPU "
            "wheel of torch, or the driver isn't visible. Re-install with: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}  VRAM: {vram_gb:.1f} GB")

    # RTX 4060 (Ada) supports bf16 natively. T4 (Turing) does not. Force bf16
    # for any non-Turing card; only fall back to fp16 for cards that lack it.
    is_turing = any(x in gpu_name for x in ("T4", "RTX 20", "Quadro RTX 6", "Quadro RTX 8"))
    use_bf16 = not is_turing
    print(f"precision: {'bf16' if use_bf16 else 'fp16'} (Ada Lovelace) ")

    # Datasets
    train_dataset = _build_dataset(256, seed=seed)
    print(f"train rows: {len(train_dataset)}  (no eval_dataset — broken in TRL 0.14 + PEFT)")

    # Model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=str(OUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        max_prompt_length=1024,
        max_completion_length=200,
        num_generations=num_generations,
        temperature=0.9,
        beta=0.04,
        max_steps=max_steps,
        logging_steps=1,
        save_steps=100,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
        # NOTE: no eval_dataset/eval_steps/eval_strategy — PEFT+GRPO eval
        # is broken in TRL 0.14. Eval after training as a separate step.
    )

    class JSONLLoggerCallback(TrainerCallback):
        def __init__(self, path: Path):
            self.fh = open(path, "a", buffering=1, encoding="utf-8")

        def on_log(self, args, state, control, logs=None, **kw):  # noqa: D401
            if logs is None:
                return
            rec = dict(logs)
            rec["step"] = state.global_step
            rec["epoch"] = state.epoch
            rec["timestamp"] = time.time()
            self.fh.write(json.dumps(rec, default=str) + "\n")

        def on_train_end(self, args, state, control, **kw):
            self.fh.close()

    jsonl_cb = JSONLLoggerCallback(METRICS_PATH)
    print(f"metrics → {METRICS_PATH}")
    print(f"trajectories → {TRAJECTORY_DIR}")

    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=_v2_disclosure_reward,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
        callbacks=[jsonl_cb],
    )

    print("\n🚀 GRPO v2 starting — ~50 min on RTX 4060. Walk away.")
    print("    Tip: open a separate PowerShell with `nvidia-smi -l 5` to monitor.\n")
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    print(f"\n✅ training complete in {dt/60:.1f} min")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"✅ adapter saved → {ADAPTER_DIR}")
    return ADAPTER_DIR


# ──────────────────────────────────────────────────────────────────────────────
# Mode 3 — eval

def run_eval(n_eval: int = 50, adapter_dir: Path | None = None) -> dict:
    """Eval trained_v2 vs base over n=50 held-out episodes. Returns dict of
    means with key 'delta'."""

    print("\n" + "=" * 72)
    print(f"v2 eval — trained vs base, n={n_eval}")
    print("=" * 72)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from privacy_game.models import DisclosureAction
    from privacy_game.server.privacy_game_environment import PrivacyGameEnvironment
    from privacy_game.server.tasks import ALL_TASKS_BY_ID

    # Same 20k registry as training
    import privacy_game.server.privacy_game_environment as pge
    from privacy_game.server.adversary import build_registry
    pge._REGISTRY = build_registry(
        pge._TRAIN_PROFILES + pge._HOLDOUT_PROFILES,
        extra_size=20000, seed=43,
    )

    if adapter_dir is None:
        adapter_dir = ADAPTER_DIR
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"adapter not found at {adapter_dir}. Run --train first, or pass --adapter-dir."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build held-out eval set with a different seed than training
    eval_rng = random.Random(2026)
    eval_eps = [
        {
            "episode_seed": eval_rng.randint(0, 2**31 - 1),
            "task_id": eval_rng.choice(list(ALL_TASKS_BY_ID.keys())),
        }
        for _ in range(n_eval)
    ]

    def _gen(model, prompt: str) -> str:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024,
        ).to("cuda")
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
        ).strip()

    def _eval(model, label: str) -> list[float]:
        rewards = []
        for i, ep in enumerate(eval_eps):
            env_l = PrivacyGameEnvironment(
                seed=ep["episode_seed"], force_task_id=ep["task_id"], reward_mode="pareto_it",
            )
            obs0 = env_l.reset()
            prompt = _format_prompt(obs0)
            completion = _gen(model, prompt)
            r = _v2_disclosure_reward(
                prompts=[prompt],
                completions=[completion],
                episode_seed=[ep["episode_seed"]],
                task_id=[ep["task_id"]],
            )[0]
            rewards.append(r)
            if (i + 1) % 10 == 0:
                print(f"  [{label}] {i+1}/{n_eval}  running mean={statistics.mean(rewards):+.4f}")
        m = statistics.mean(rewards)
        s = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        print(f"  ✅ {label}: mean={m:+.4f}  std={s:.4f}  (n={len(rewards)})")
        return rewards

    print("\n=== TRAINED v2 (base + LoRA adapter merged) ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    trained = PeftModel.from_pretrained(base_model, str(adapter_dir)).merge_and_unload()
    trained_rewards = _eval(trained, "trained_v2")
    del trained, base_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== BASE Qwen2.5-0.5B (no adapter) ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    base_rewards = _eval(base_model, "base")
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    delta = statistics.mean(trained_rewards) - statistics.mean(base_rewards)
    print("\n" + "=" * 72)
    print(f"📊 v2 DELTA = {delta:+.4f}  "
          f"(trained={statistics.mean(trained_rewards):+.4f}  "
          f"base={statistics.mean(base_rewards):+.4f}  "
          f"n={n_eval})")
    if delta > 0.10:
        print("🎉 v2 wins big — ship v2 numbers")
    elif delta > 0.05:
        print("✅ v2 modest win — ship v2 as 'directional'")
    elif delta > 0.02:
        print("⚠️  v2 weak — try v3 (stronger leak penalty) only if time permits")
    else:
        print("❌ v2 didn't help — fall back to v1")
    print("=" * 72)

    # Save the eval result alongside metrics for plot regeneration
    eval_path = METRICS_PATH.parent / "grpo_v2_eval.json"
    eval_path.write_text(json.dumps({
        "n_eval": n_eval,
        "trained_v2_mean": statistics.mean(trained_rewards),
        "trained_v2_std": statistics.stdev(trained_rewards) if len(trained_rewards) > 1 else 0.0,
        "base_mean": statistics.mean(base_rewards),
        "base_std": statistics.stdev(base_rewards) if len(base_rewards) > 1 else 0.0,
        "delta": delta,
        "trained_v2_rewards": trained_rewards,
        "base_rewards": base_rewards,
    }, indent=2))
    print(f"eval saved → {eval_path}")
    return {
        "trained_v2_mean": statistics.mean(trained_rewards),
        "base_mean": statistics.mean(base_rewards),
        "delta": delta,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Mode 4 — push to HF Hub

def run_push_hub(repo_id: str, adapter_dir: Path | None = None) -> None:
    print(f"\n=== Push adapter → HF Hub: {repo_id} ===")
    if adapter_dir is None:
        adapter_dir = ADAPTER_DIR
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter not found at {adapter_dir}")

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(folder_path=str(adapter_dir), repo_id=repo_id, repo_type="model")
    print(f"✅ pushed → https://huggingface.co/{repo_id}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="Run reward smoke test (gating signal, ~1 min)")
    ap.add_argument("--train", action="store_true",
                    help="Run GRPO v2 training (~50 min on RTX 4060 Laptop)")
    ap.add_argument("--eval", action="store_true",
                    help="Eval trained adapter vs base (n=50, ~5 min)")
    ap.add_argument("--push-hub", action="store_true",
                    help=f"Push adapter to HF Hub (default repo: {DEFAULT_HF_REPO})")
    ap.add_argument("--num-generations", type=int, default=4,
                    help="GRPO group size. Drop to 2 if OOM. Default: 4")
    ap.add_argument("--max-steps", type=int, default=200,
                    help="Max GRPO steps. Default: 200")
    ap.add_argument("--lr", type=float, default=1e-5, help="Learning rate. Default: 1e-5")
    ap.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    ap.add_argument("--n-eval", type=int, default=50,
                    help="Eval episodes per policy. Default: 50")
    ap.add_argument("--adapter-dir", type=Path, default=None,
                    help=f"Adapter dir (default: {ADAPTER_DIR})")
    ap.add_argument("--hf-repo", type=str, default=DEFAULT_HF_REPO,
                    help=f"HF Hub repo id (default: {DEFAULT_HF_REPO})")
    ap.add_argument("--skip-smoke-gate", action="store_true",
                    help="Run --train even if --smoke fails. NOT recommended.")

    args = ap.parse_args()

    # If no flags, print help.
    if not any([args.smoke, args.train, args.eval, args.push_hub]):
        ap.print_help()
        print("\nQuick start: python notebooks/train_v2_local.py --smoke --train --eval")
        return 1

    # Smoke + train chained: smoke must pass.
    if args.smoke:
        ok = run_smoke()
        if args.train and not ok and not args.skip_smoke_gate:
            print("\n❌ Aborting --train because --smoke failed. Pass --skip-smoke-gate to override.")
            return 2

    if args.train:
        run_train(
            num_generations=args.num_generations,
            max_steps=args.max_steps,
            lr=args.lr,
            seed=args.seed,
        )

    if args.eval:
        run_eval(n_eval=args.n_eval, adapter_dir=args.adapter_dir)

    if args.push_hub:
        run_push_hub(repo_id=args.hf_repo, adapter_dir=args.adapter_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
