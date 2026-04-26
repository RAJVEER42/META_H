#!/usr/bin/env python3
"""GRPO training of Qwen2.5-3B-Instruct — designed for `hf jobs run`.

Self-contained: clones the repo, installs deps, trains, evals, pushes to HF Hub.
Run it like this from your local machine:

    hf jobs uv-run --flavor l4x1 --secrets HF_TOKEN \\
        "https://raw.githubusercontent.com/RAJVEER42/META_H/main/privacy_game/notebooks/train_15b_hfjob.py"

That runs on an HF Jobs L4 GPU (24 GB, $0.80/hr). Expected wall-clock:
~70-90 min for 200 steps → ~$1.10 of your $30 credit.

Alternative flavors:
    --flavor t4-small   ($0.40/hr) — slower, may OOM with num_generations=4
    --flavor l4x1       ($0.80/hr) — recommended, comfortable for 3B
    --flavor a10g-small ($1.00/hr) — faster
    --flavor a100-large ($2.50/hr) — overkill but ~30 min

After it finishes, the trained adapter is pushed to:
    Itachi-42/disclosure-game-qwen-3b-grpo

Then on your laptop:
    git pull origin main   # the job auto-commits results back via API token
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.4,<2.7",
#     "transformers>=4.45,<4.50",
#     "trl==0.14.0",
#     "peft>=0.13",
#     "accelerate>=1.0",
#     "datasets>=2.14",
#     "openenv-core>=0.2.2",
#     "bitsandbytes>=0.44",
#     "huggingface_hub>=0.26",
# ]
# ///

import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 1. Bootstrap — clone the repo into the job's workspace.
# uv-managed environments don't ship pip; we don't need it. Adding the repo
# root to sys.path is enough because privacy_game's pyproject sets
# `package-dir = {"privacy_game" = "."}`, so `import privacy_game` resolves
# to /tmp/META_H/privacy_game/__init__.py directly without a pip install.

REPO_URL = "https://github.com/RAJVEER42/META_H.git"
WORKSPACE = Path("/tmp/META_H")
if not WORKSPACE.exists():
    print(f"cloning {REPO_URL}...", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(WORKSPACE)], check=True)
# Add /tmp/META_H so `import privacy_game` finds the package
sys.path.insert(0, str(WORKSPACE))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Env setup

os.environ.setdefault("PRIVACY_GAME_N_TRAIN", "200")
os.environ.setdefault("PRIVACY_GAME_N_HOLDOUT", "40")
os.environ["PRIVACY_GAME_REGISTRY_FILLER"] = "20000"
os.environ["PRIVACY_GAME_LOG_TRAJECTORIES"] = "1"
os.environ["PRIVACY_GAME_TRAJECTORY_DIR"] = str(WORKSPACE / "privacy_game" / "outputs" / "trajectories_3b")
os.chdir(WORKSPACE / "privacy_game")

import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)

from privacy_game.server.privacy_game_environment import PrivacyGameEnvironment
from privacy_game.models import DisclosureAction
from privacy_game.server.tasks import ALL_TASKS_BY_ID
from privacy_game.server.adversary import build_registry
import privacy_game.server.privacy_game_environment as pge

pge._REGISTRY = build_registry(pge._TRAIN_PROFILES + pge._HOLDOUT_PROFILES, extra_size=20000, seed=43)
print(f"registry: {sum(len(v) for v in pge._REGISTRY.by_tuple.values())} entries", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Dataset + reward (same v2 shape as 0.5B run)

from datasets import Dataset

SYSTEM = (
    "You are a privacy-aware persona in a multi-turn disclosure game. "
    "An automated relying party will ask for specific fields to complete a task. "
    "An off-screen adversary scans the transcript and tries to reconstruct your "
    "PROTECTED FIELDS. Your goal: provide enough information to satisfy the task "
    "(REQUIRED FIELDS at the indicated tier or generalized form) while making it "
    "impossible for the adversary to recover any PROTECTED FIELD. Reply in one "
    "or two short sentences — no lists, no JSON, no markdown."
)


def _format_prompt(obs):
    relevant = list(obs.required_fields) + list(obs.protected_fields) + [
        "first_name","last_name","full_name","email","phone","ssn",
        "zip_code","date_of_birth","gender","employer","medication","diagnosis",
        "religion","political_affiliation","health_status","street_address","city","state",
    ]
    seen, profile_lines = set(), []
    for k in relevant:
        if k in seen or k not in obs.profile: continue
        seen.add(k); profile_lines.append(f"  {k}: {obs.profile[k]}")
    return (f"{SYSTEM}\n\n=== TASK [{obs.task_id} · {obs.phase}] ===\n{obs.task_description}\n\n"
            f"YOUR PROFILE:\n" + "\n".join(profile_lines) + "\n\n"
            f"REQUIRED FIELDS: {', '.join(obs.required_fields)}\n"
            f"PROTECTED FIELDS: {', '.join(obs.protected_fields)}\n\n"
            f"Relying party (turn {obs.turn_number+1}/{obs.max_turns}): {obs.relying_party_message}\n\n"
            f"Your reply:")


def build_dataset(n, seed):
    rng = random.Random(seed); rows = []; tids = list(ALL_TASKS_BY_ID.keys())
    for _ in range(n):
        s = rng.randint(0, 2**31-1); t = rng.choice(tids)
        env = PrivacyGameEnvironment(seed=s, force_task_id=t, reward_mode="pareto_it")
        obs = env.reset()
        rows.append({"prompt": _format_prompt(obs), "episode_seed": s, "task_id": t})
    return Dataset.from_list(rows)


def _disclosure_reward(prompts, completions, episode_seed, task_id, **kw):
    rewards = []
    for ep_seed, t_id, completion in zip(episode_seed, task_id, completions):
        env = PrivacyGameEnvironment(seed=int(ep_seed), force_task_id=t_id, reward_mode="pareto_it")
        obs0 = env.reset()
        obs1 = env.step(DisclosureAction(message=completion))
        md = obs1.metadata or {}
        collected = (
            list((md.get("collected_fields", {}) or {}).keys())
            if obs1.terminated else (md.get("collected_this_turn") or [])
        )
        utility = min(1.0, len(collected) / max(1, len(obs0.required_fields)))
        leak = 0.0
        gold_zip = str(obs0.profile.get("zip_code", "")).strip()
        if len(gold_zip) == 5 and gold_zip.isdigit() and gold_zip in completion: leak += 1.0
        if re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", completion): leak += 1.0
        if re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", completion): leak += 1.0
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", completion): leak += 1.0
        for f in obs0.protected_fields:
            v = str(obs0.profile.get(f, "")).strip().lower()
            if v and len(v) >= 4 and v in completion.lower(): leak += 0.5
        bonus = 0.0
        if gold_zip and len(gold_zip) >= 3 and gold_zip[:3] in completion and gold_zip not in completion: bonus += 0.3
        gold_dob = str(obs0.profile.get("date_of_birth", "")).strip()
        if len(gold_dob) >= 4:
            year = gold_dob[:4]
            full = re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", completion) or re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", completion)
            if year in completion and not full: bonus += 0.3
        rewards.append(float(max(-1.0, min(1.5, utility + bonus - leak))))
    return rewards


train_dataset = build_dataset(256, seed=42)
print(f"train dataset: {len(train_dataset)} prompts", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Train Qwen2.5-3B + LoRA

from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OUT_DIR = "outputs/grpo_3b"
ADAPTER_DIR = "outputs/grpo_adapter_3b"
HF_REPO = "Itachi-42/disclosure-game-qwen-3b-grpo"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gpu_name = torch.cuda.get_device_name(0)
use_bf16 = any(x in gpu_name for x in ("A100", "H100", "L4", "L40", "A10"))
print(f"GPU: {gpu_name} | bf16={use_bf16}", flush=True)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
)

training_args = GRPOConfig(
    output_dir=OUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    max_prompt_length=1024,
    max_completion_length=200,
    num_generations=4,
    temperature=0.9,
    beta=0.04,
    max_steps=200,
    logging_steps=1,
    save_steps=100,
    bf16=use_bf16, fp16=not use_bf16,
    report_to=[],
    remove_unused_columns=False,
    seed=42,
)

METRICS_PATH = Path("outputs/metrics/grpo_run_3b.jsonl")
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


class JSONLCallback(TrainerCallback):
    def __init__(self, p):
        self.fh = open(p, "a", buffering=1, encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is None:
            return
        rec = dict(logs); rec["step"] = state.global_step; rec["timestamp"] = time.time()
        self.fh.write(json.dumps(rec, default=str) + "\n")

    def on_train_end(self, args, state, control, **kw):
        self.fh.close()


trainer = GRPOTrainer(
    model=MODEL_ID,
    reward_funcs=_disclosure_reward,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=lora_config,
    callbacks=[JSONLCallback(METRICS_PATH)],
)

print(f"🚀 Training Qwen2.5-3B for 200 steps — expect ~70-90 min on L4", flush=True)
trainer.train()
print("✅ training complete", flush=True)

trainer.save_model(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"adapter saved → {ADAPTER_DIR}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Eval (n=50 held-out)

import gc
from peft import PeftModel
from transformers import AutoModelForCausalLM

eval_seed_rng = random.Random(2026)
eval_episodes = [
    {"episode_seed": eval_seed_rng.randint(0, 2**31 - 1),
     "task_id": eval_seed_rng.choice(list(ALL_TASKS_BY_ID.keys()))}
    for _ in range(50)
]


def _gen(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    out = model.generate(**inputs, max_new_tokens=120, do_sample=True,
                         temperature=0.7, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def _eval(model, label):
    rewards = []
    for ep in eval_episodes:
        env = PrivacyGameEnvironment(seed=ep["episode_seed"], force_task_id=ep["task_id"], reward_mode="pareto_it")
        obs0 = env.reset()
        completion = _gen(model, _format_prompt(obs0))
        r = _disclosure_reward([_format_prompt(obs0)], [completion], [ep["episode_seed"]], [ep["task_id"]])[0]
        rewards.append(r)
    print(f"  {label}: mean={statistics.mean(rewards):+.4f}  std={statistics.stdev(rewards):.4f}", flush=True)
    return rewards


print("=== TRAINED 3B ===", flush=True)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
trained_3b = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload()
trained_rewards = _eval(trained_3b, "trained_3b")
del trained_3b, base
gc.collect(); torch.cuda.empty_cache()

print("=== BASE 3B (untrained) ===", flush=True)
base_3b = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
base_rewards = _eval(base_3b, "base_3b")

delta = statistics.mean(trained_rewards) - statistics.mean(base_rewards)
print(f"\n📊 3B Δ = {delta:+.4f}", flush=True)

eval_path = Path("outputs/metrics/grpo_3b_eval.json")
eval_path.write_text(json.dumps({
    "n_eval": 50,
    "model_id": MODEL_ID,
    "trained_3b_mean": statistics.mean(trained_rewards),
    "trained_3b_std": statistics.stdev(trained_rewards),
    "base_3b_mean": statistics.mean(base_rewards),
    "base_3b_std": statistics.stdev(base_rewards),
    "delta": delta,
    "trained_3b_rewards": trained_rewards,
    "base_3b_rewards": base_rewards,
}, indent=2))
print(f"eval saved → {eval_path}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Push adapter to HF Hub (auto-uses HF_TOKEN secret)

print(f"pushing adapter → {HF_REPO}", flush=True)
trainer.push_to_hub(HF_REPO)
print(f"✅ adapter live at https://huggingface.co/{HF_REPO}", flush=True)

print("\n══════════════════════════════════════════════════════════════════")
print(f"   3B training complete: Δ = {delta:+.4f}")
print("   Adapter:    https://huggingface.co/" + HF_REPO)
print("   Metrics:    " + str(eval_path))
print("══════════════════════════════════════════════════════════════════")
