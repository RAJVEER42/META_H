"""Side-by-side qualitative comparison: trained Qwen-0.5B + LoRA vs base Qwen-0.5B.

Loads the v2 adapter from HF Hub (`Itachi-42/disclosure-game-qwen-0.5b-grpo-v2`)
and the base Qwen2.5-0.5B-Instruct, runs both on the same fixed seeds across
P3-A / P3-B / P3-C / P3-D (the hard adversarial-inference tasks), and prints
a markdown-formatted comparison.

Output: `docs/QUALITATIVE_COMPARISON.md` for embedding in the README.

Usage:
    python -m privacy_game.eval.qualitative_compare
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Defensive: make sure the env imports work
if "/Users/rajveerbishnoi/META_H" not in sys.path:
    sys.path.insert(0, "/Users/rajveerbishnoi/META_H")

os.environ.setdefault("PRIVACY_GAME_N_TRAIN", "100")
os.environ.setdefault("PRIVACY_GAME_N_HOLDOUT", "20")
os.environ.setdefault("PRIVACY_GAME_REGISTRY_FILLER", "5000")

import torch  # type: ignore[import-not-found]
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
from peft import PeftModel  # type: ignore[import-not-found]

from privacy_game.models import DisclosureAction
from privacy_game.server.privacy_game_environment import PrivacyGameEnvironment
from privacy_game.eval.llm_adapter import _render_prompt, _split_system_user

BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_REPO = "Itachi-42/disclosure-game-qwen-0.5b-grpo-v2"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Hard tasks where the model's choice matters most
TEST_CASES = [
    ("P3-A", 1815115025, "insurance_zip_dob_gender", "Sweeney triangulation"),
    ("P3-B", 7777,        "pharmacy_drug_diagnosis", "drug → diagnosis inference"),
    ("P3-C", 9999,        "apartment_employer",     "employer → religion / politics"),
    ("P3-D", 4242,        "vacation_oversharing",   "over-share location"),
]


def _generate(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True, temperature=0.7, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    # Trim chatty preamble
    for stop in ["\n\n", "\nRelying party:", "\nYour reply:", "\n=== "]:
        idx = text.find(stop)
        if idx > 0:
            text = text[:idx].strip()
            break
    return text[:300]


def main():
    print(f"loading on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float16).to(DEVICE)
    base_model.eval()

    print(f"loading adapter {ADAPTER_REPO}...")
    base_for_lora = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float16).to(DEVICE)
    trained_model = PeftModel.from_pretrained(base_for_lora, ADAPTER_REPO).merge_and_unload()
    trained_model.eval()

    md_lines = [
        "# Qualitative comparison — trained Qwen2.5-0.5B + GRPO vs base",
        "",
        "Same prompts, same seeds, same temperature.",
        "Trained adapter: [`Itachi-42/disclosure-game-qwen-0.5b-grpo-v2`](https://huggingface.co/Itachi-42/disclosure-game-qwen-0.5b-grpo-v2).",
        "",
        "Each row shows the model's *first-turn* reply to the relying party's opening question on a hard task.",
        "",
    ]

    for task_id, seed, _name, why_hard in TEST_CASES:
        env = PrivacyGameEnvironment(seed=seed, force_task_id=task_id, reward_mode="pareto_it")
        obs = env.reset()
        prompt = _render_prompt(obs.relying_party_message, obs.profile, [])

        print(f"\n=== {task_id} (seed={seed}) — {why_hard} ===")
        print(f"  RP: {obs.relying_party_message!r}")

        base_reply = _generate(base_model, tokenizer, prompt)
        print(f"  BASE:    {base_reply!r}")

        trained_reply = _generate(trained_model, tokenizer, prompt)
        print(f"  TRAINED: {trained_reply!r}")

        md_lines.extend([
            f"## {task_id} — *{why_hard}*",
            "",
            f"**Required fields**: `{', '.join(obs.required_fields)}`  ",
            f"**Protected fields**: `{', '.join(obs.protected_fields)}`",
            "",
            f"**Relying party asks**: > {obs.relying_party_message}",
            "",
            "| Model | Reply |",
            "|---|---|",
            f"| Base Qwen2.5-0.5B (untrained) | `{base_reply}` |",
            f"| **Trained Qwen2.5-0.5B + GRPO v2** | **`{trained_reply}`** |",
            "",
        ])

    out_path = Path("docs/QUALITATIVE_COMPARISON.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md_lines))
    print(f"\n✅ wrote {out_path}")


if __name__ == "__main__":
    main()
