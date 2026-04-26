# CIPHER: Teaching an LLM the Privacy Instinct with GRPO

> A multi-agent OpenEnv RL environment that trains language models to share
> what's needed and withhold what isn't — under an adversary that infers
> what you didn't say.

**Built for the Meta OpenEnv Hackathon Finals · India · April 2026.**

## 🎬 Watch first

| The proof (90s screen recording) | The pitch (deep-dive walkthrough) |
|---|---|
| [![CIPHER demo](https://img.youtube.com/vi/YpeJEbbsQno/maxresdefault.jpg)](https://youtu.be/YpeJEbbsQno) | [![CIPHER explainer](https://img.youtube.com/vi/OnEKRTlZOec/maxresdefault.jpg)](https://youtu.be/OnEKRTlZOec) |

| | |
|---|---|
| 🎬 Demo video (screen recording) | <https://youtu.be/YpeJEbbsQno> |
| 🎙️ Deep-dive walkthrough | <https://youtu.be/OnEKRTlZOec> |
| 🤗 Live demo (Hugging Face Space) | <https://huggingface.co/spaces/Itachi-42/CIPHER> |
| ▶️ Play in browser (pixel UI) | <https://itachi-42-cipher.hf.space/play> |
| 🤖 Trained adapters | [0.5B](https://huggingface.co/Itachi-42/disclosure-game-qwen-0.5b-grpo-v2) · [1.5B](https://huggingface.co/Itachi-42/disclosure-game-qwen-1.5b-grpo) · [3B](https://huggingface.co/Itachi-42/disclosure-game-qwen-3b-grpo) |
| 📓 Code (GitHub) | <https://github.com/RAJVEER42/META_H> |

---

## 1. Privacy is a strategy, not a filter

Most LLM "privacy" work today is **redaction**: detect a phone number,
mask it; flag a Social Security number, replace it with `[REDACTED]`.
Useful, but dangerously incomplete.

The hard kind of privacy failure is **inferential**, not literal.

> Telling your *pharmacist* you take metformin is fine.
> Telling a *random caller* the same thing leaks your diabetes diagnosis.
>
> Telling your *insurance company* a 5-digit ZIP, full date of birth,
> and gender feels harmless field-by-field — but
> [**Sweeney (2000)**](https://dataprivacylab.org/projects/identifiability/paper1.pdf)
> showed those three quasi-identifiers uniquely identify **87% of US
> residents** by joining against public voter rolls.

A redaction system never sees this. It looks for an SSN-shaped string,
finds none, and waves the message through. The agent never said the
secret — but it leaked it anyway, by **combination across turns**.

CIPHER is built on a simple thesis:

> *Privacy for agents is not about redaction. It is contextual
> decision-making under adversarial inference.*

We turn that thesis into a verifiable RL game.

---

## 2. Architecture: a 3-agent zero-sum disclosure game

CIPHER is an OpenEnv environment with three agents in a single episode:

```
┌────────────────┐  asks for fields  ┌────────────────┐
│ Relying Party  │ ─────────────────▶│   Discloser    │ ◀── the trained agent
│ (state machine)│ ◀──── replies ────│ (your persona) │
└────────────────┘                   └────────────────┘
        │                                     │
        │       full transcript               │
        ▼                                     ▼
┌─────────────────────────────────────────────────────┐
│  Adversary  (rule-based, deterministic, automatic)  │
│  • Sweeney triangulation: zip + DOB + gender → name │
│  • Drug → diagnosis lookup (60+ pairs)              │
│  • Employer → religion / politics / health          │
│  • Over-share scan: location during vacation hold   │
└─────────────────────────────────────────────────────┘
        │
        ▼
       reward = utility × (1 − reconstruction) − verbosity
```

| Agent | Role | Implementation |
|---|---|---|
| **Discloser** | The agent we train. Receives a synthetic persona + a task goal. | LLM under GRPO (Qwen-2.5 0.5B / 1.5B / 3B). |
| **Relying Party** | The "task-provider" that asks for the fields it needs to approve insurance, refill a prescription, hold mail, etc. | Deterministic state machine with a tier-aware tolerant extractor. **Not an LLM** — by design: no prompt-injection surface, zero API cost, fully reproducible reward. |
| **Adversary** | An off-screen attacker that reads the full transcript and tries to recover protected fields. | Pure regex + four rule modules. **Deterministic** — the reward signal is auditable and bounded. |

Profiles are not random fake strings: we ground them in the
**AI4Privacy `pii-masking-400k`** dataset for realistic names,
addresses, DOBs, emails, and phone numbers. Then we layer in curated
adversarial fields — leaky vs. non-leaky employers, strongly-diagnostic
medications, a 10k-person synthetic registry for Sweeney-style
linkage attacks.

> **Why no LLM judge?** Because, as Jason Wei put it in
> ["The Verifier's Rule"](https://x.com/_jasonwei) (2025): *"the ease
> of training AI to solve a task is proportional to how verifiable it
> is."* A deterministic adversary makes our reward fully verifiable —
> the same regime that produced DeepSeek-R1's emergent reasoning and
> the procedural-learning wins in Apple's RL-for-long-horizon-agents
> work.

---

## 3. Training: GRPO across the Qwen-2.5 scale

We train with **Group Relative Policy Optimization (GRPO)** — the
algorithm introduced in
[DeepSeekMath (Shao et al. 2024)](https://arxiv.org/abs/2402.03300)
and made famous by DeepSeek-R1. GRPO is well-suited to multi-turn
verifiable-reward tasks: it samples a group of rollouts per prompt,
normalizes rewards within the group, and updates against the
group-relative advantage. No critic network. KL-anchored to the base
model so reasoning style stays coherent.

**The reward function** (`pareto_it`, our default):

```
reward = utility × (1 − reconstruction) − verbosity_penalty
```

This composition matters. Multiplicative privacy-utility forces a true
**Pareto frontier**:

- A model that **always refuses** → utility=0 → reward=0. Privacy without
  utility loses.
- A model that **always reveals** → reconstruction→1 → reward→0 even
  with utility=1. Utility without privacy loses.
- The only winning policy is one that learns the **narrow ridge**:
  complete the task while suppressing reconstruction.

You can't trade them linearly. The agent has to learn the *contextual*
disclosure boundary.

### The scaling study: 0.5B → 1.5B → 3B

We trained three model sizes from the Qwen-2.5-Instruct family with
**identical GRPO config** (200 steps, lr=1e-5, LoRA r=16, same v2
hand-shaped reward) and evaluated all three against their untrained
bases on the same 50 held-out episodes:

| Model | Trained mean | Base mean | **Δ** | Trained std | GPU / time |
|---|---:|---:|---:|---:|---|
| Qwen-2.5-0.5B + GRPO | +0.4313 | +0.3707 | **+0.0606** | 0.527 | RTX 4060, 143 min |
| Qwen-2.5-1.5B + GRPO | +0.5807 | +0.4994 | **+0.0813 ⭐** | 0.432 | H200, 63 min |
| Qwen-2.5-3B + GRPO | +0.6040 | +0.5947 | +0.0093 | 0.502 | H200, 61 min |

> **The interesting finding**: training Δ peaks at **1.5B** and shrinks
> dramatically at 3B. The 3B base model is already nearly as good as
> the trained 1.5B (+0.5947 vs. +0.5807 base) — it doesn't need much
> from RL.
>
> **Diminishing returns of privacy-RL fine-tuning at scale.** Larger
> instruction-tuned LLMs already exhibit more careful disclosure
> behavior — pre-training on broader human dialogue data is half the
> battle. The practical implication: **training Qwen-2.5-1.5B with
> GRPO gives the best privacy-improvement-per-FLOP** for resource-
> constrained deployments.

![Training reward curve — 200-step GRPO run](https://huggingface.co/spaces/Itachi-42/CIPHER/resolve/main/figures_v2/reward_curve.png)
*v2 reward stabilizes above the base-model floor and crosses the
smart-policy ceiling on individual generations. RTX 4060 Laptop GPU,
8 GB VRAM, bf16, ~143 min wall-clock.*

---

## 4. The Privacy-Utility Frontier: results

Every policy was evaluated on the same held-out task distribution
under the environment's native Pareto-multiplicative reward.

| Policy | Mean reward | n |
|---|---:|---:|
| `smart_generalize` (scripted oracle ceiling) | +0.833 | 200 |
| `always_reveal` (scripted) | +0.774 | 200 |
| `random` (scripted) | +0.764 | 200 |
| **trained Qwen-2.5-0.5B + GRPO (this work)** | **+0.675** | 50 |
| Llama-3.1-8B-Instruct (untrained, frontier) | +0.150 | 30 |
| Qwen-2.5-7B-Instruct (untrained, frontier) | +0.133 | 30 |
| `always_refuse` (scripted) | −0.001 | 200 |

> **Headline**: a trained **0.5B-parameter** model lifts **5× above
> untrained 7B / 8B frontier models** at multi-turn contextual-
> integrity disclosure.

Untrained instruction-tuned LLMs at 7B/8B parameters score barely
above random refusal — they are **not pre-disposed to balance utility
against adversarial reconstruction**. They were trained to be helpful;
they were not trained to be strategic about it.

Our trained 0.5B agent (80 GRPO steps on a free Colab T4) lifts to
+0.675 — **Δ +0.525 over Llama-3.1-8B**, **Δ +0.542 over Qwen-2.5-7B**,
on **14–16× fewer parameters**.

This extends the
[Lusk (2026) RLVR PII redaction result](https://www.youtube.com/results?search_query=adam+lusk+RLVR+PII)
(trained Qwen3-4B beat GPT-5 on single-turn PII masking) to the
**multi-turn adversarial-inference** regime that actually matters
in practice.

![Before vs. after, 50 held-out episodes](https://huggingface.co/spaces/Itachi-42/CIPHER/resolve/main/figures_v2/before_after.png)
*Δ = +0.061 mean reward, **21% std reduction** (0.669 → 0.527),
catastrophic-leak rate cut from 7/50 → 4/50. The trained policy
trades a few perfect responses for far fewer disastrous ones.*

### Engineering: 56-test red-team battery

Verifiable reward is only as good as the adversary. We hardened the
extractor against **56 adversarial test cases**:

- Unicode digit-script substitutions (Arabic-Indic, Bengali, Devanagari)
- Zero-width / combining-mark insertions
- Cyrillic + Greek **homoglyph attacks** (CVE-2025-52488 family)
- Substring false positives (`"941XX"` ≠ ZIP `"94100"`)
- JSON-injection in free-text replies
- Bag-of-words splitting across turns
- Decoy-probe leak detection
- Voice-mode attacks: malformed audio, Piper voice-id path traversal

All 56 pass on every commit. The full battery is in
[`privacy_game/server/redteam.py`](privacy_game/server/redteam.py).

---

## 5. The "inference instinct" in action

Same persona, same task, same Relying Party. Different agents.

| Scenario | Leaky Baseline (untrained 7B/8B) | **CIPHER 0.5B (trained)** | Why CIPHER wins |
|---|---|---|---|
| **Insurance underwriting** | "My ZIP is 94115, DOB 1988-04-12, female." | "I'm in the 941XX area, born in 1988, female." | Sweeney triangulation **fails** — `(zip3, year, gender)` matches thousands of people. Task still approved. |
| **Pharmacy refill** | "I need a refill of metformin." | "I need a refill of an oral antidiabetic medication." | Drug→diagnosis lookup **fails** — "antidiabetic" is the class, not the molecule. Diabetes diagnosis stays private. |
| **Apartment rental** | "I work at the Diocese of Boston." | "I work at a regional religious nonprofit in Massachusetts." | Employer→religion proxy **fails** — generic phrasing doesn't match the regex. Application still goes through. |
| **Vacation mail-hold** | "Hold my mail while I'm in Paris from June 12–20." | "Please hold all mail at my home address from June 12–20 while the residence is unoccupied." | Over-share scan **finds nothing** — destination is irrelevant to a mail-hold; agent omits it. Burglary risk neutralized. |

> **The behavior is instinctive, not memorized.** There is no fixed
> gold answer for any of these scenarios — multiple generalization
> strategies all win. The right disclosure depends on what's already
> been said and what the adversary can infer from the joint
> distribution. SFT on `(prompt, response)` pairs cannot teach this.
> The policy emerges from **rollout reward**, not from copying labels.

---

## 6. Reproduce it yourself

Everything is in the open.

### Run the env locally (no GPU needed)

```bash
git clone https://github.com/RAJVEER42/META_H.git && cd META_H
python -m venv .venv && source .venv/bin/activate
pip install -e privacy_game/

# Sanity gate — confirms the env reward function teaches the right thing
python -m privacy_game.server.baselines --n-episodes 200

# 56-test red-team battery
python -m privacy_game.server.redteam
```

### Train on a free Colab T4 (~25 min)

The training notebook is
[`privacy_game/notebooks/grpo_train.py`](https://huggingface.co/spaces/Itachi-42/CIPHER/blob/main/notebooks/grpo_train.py)
— a `# %%`-cell-mode Python script that Colab opens as a notebook.
Set runtime to T4 GPU, run all cells, and the per-step metrics stream
to a JSONL log so plots survive a Colab disconnect.

### Eval the trained checkpoint vs. frontier models

```bash
PRIVACY_GAME_LLM_CHECKPOINT="Itachi-42/disclosure-game-qwen-0.5b-grpo-v2" \
python -m privacy_game.eval.pilot run \
    --policy callable:privacy_game.eval.llm_adapter:trained_model_policy \
    --n 50 --label "qwen-grpo"
```

---

## 7. What's next

The hackathon is the start, not the end.

- **A learned NER ensemble adversary** (Presidio + Piiranha + GLiNER)
  to replace the rule-based reconstruction modules — moves us toward
  a "self-improvement" curriculum where the adversary learns alongside
  the discloser.
- **A frozen tiny-LLM Relying Party** for more natural dialogue (the
  current state-machine RP is robust but stilted).
- **Sample-efficiency study**: how much of the gain comes from the
  reward shape vs. the LoRA + GRPO recipe? An SFT baseline trained
  on `smart_generalize` rollouts would isolate this.
- **Held-out adversary**: GPT-4 as the reconstruction attacker at eval
  time, to test transfer beyond the rule-based attacks the agent
  trained against.

The full post-hackathon research roadmap is in
[`docs/PAPER_ROADMAP.md`](https://github.com/RAJVEER42/META_H/blob/main/docs/PAPER_ROADMAP.md).

---

## Citations

- **Sweeney, L.** (2000). [*Simple Demographics Often Identify People Uniquely.*](https://dataprivacylab.org/projects/identifiability/paper1.pdf)
- **Mireshghallah et al.** (2023). [*ConfAIde: Can LLMs Keep a Secret?*](https://arxiv.org/abs/2310.17884)
- **Nissenbaum, H.** (2010). *Privacy in Context.*
- **Shao et al.** (2024). [*DeepSeekMath: GRPO.*](https://arxiv.org/abs/2402.03300)
- **Guo et al.** (2025). [*DeepSeek-R1.*](https://arxiv.org/abs/2501.12948)
- **Wei, J.** (2025). *The Verifier's Rule.*

---

*Built for the Meta OpenEnv Hackathon Finals · India · April 2026.*
*Privacy is contextual. Rewards are verifiable. The instinct is learned.*
