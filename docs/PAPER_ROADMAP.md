# CIPHER — paper roadmap

*Turning this hackathon into a publication.*

> **Read first.** This doc is for *you, future-you*, returning to this project
> after the hackathon. It captures the honest state of the research, what
> reviewers will catch, what to add, where to submit, and on what timeline.
> Written so you can pick it up cold weeks later without re-reading the
> hackathon transcripts.
>
> **Author**: Rajveer Bishnoi (RAJVEER42 on GitHub, Itachi-42 on HuggingFace)
> **Repo**: <https://github.com/RAJVEER42/META_H>
> **Project name**: *CIPHER — Contextual-Integrity Privacy via Hardened Episodic Reasoning*
> (a.k.a. the *Disclosure Game* — older internal name, retained as `privacy_game/` Python package)
> **Hackathon**: Meta OpenEnv Hackathon India, April 2026

---

## TL;DR

You built a real research-grade environment in 14 hours.
The **idea** and **engineering** are paper-worthy.
The **experiments** are not yet rigorous enough for a top-tier conference,
but they are publishable as: (a) **arXiv preprint** today, (b) **workshop
paper** with 1-2 weeks more work, (c) **main conference paper** with 3-6
months more work.

You have three trained adapters on HF Hub, a real frontier-model
comparison, a clean diminishing-returns finding, and a multi-agent
OpenEnv environment that nobody else has published. **This is good
work.** Don't undersell it. But also: don't oversell. Reviewers care
about n, seeds, ablations, and significance — none of which we have
yet at paper standards.

---

## The pitch (one paragraph you'd put in an abstract)

> Modern instruction-tuned LLMs over-share PII when asked to help with
> tasks. Single-turn redaction (Lusk 2026) and contextual-integrity
> evaluation (ConfAIde, Mireshghallah et al. 2023) have been studied, but
> no public RL environment exists for training agents on **multi-turn
> contextual-integrity disclosure under adversarial inference**. We
> introduce **CIPHER** (*Contextual-Integrity Privacy via Hardened
> Episodic Reasoning*), an OpenEnv-native
> 3-agent game (Discloser, Relying Party, Adversary) with composable
> rubrics that reward useful disclosure while penalizing reconstruction
> attacks (Sweeney triangulation, drug→diagnosis inference,
> employer→attribute inference, over-share location). We release three
> Qwen2.5-Instruct adapters (0.5B, 1.5B, 3B) trained with GRPO on a
> sharper hand-shaped reward, and find a clean diminishing-returns
> pattern: training Δ peaks at 1.5B (+0.08) and shrinks at 3B (+0.009).
> Trained Qwen2.5-0.5B beats untrained Llama-3.1-8B and Qwen2.5-7B by
> 5× on env-native reward, while running on 14-16× fewer parameters.
> Code, models, and the OpenEnv Space are publicly released.

---

## Current state (April 26, 2026 — at hackathon submission)

### What you have

| Artifact | Location |
|---|---|
| Code | `github.com/RAJVEER42/META_H` |
| OpenEnv environment package | `privacy_game/` directory |
| HF Space (live demo) | `huggingface.co/spaces/Itachi-42/privacy-game-env` |
| Trained adapter — Qwen2.5-0.5B | `huggingface.co/Itachi-42/disclosure-game-qwen-0.5b-grpo-v2` |
| Trained adapter — Qwen2.5-1.5B | `huggingface.co/Itachi-42/disclosure-game-qwen-1.5b-grpo` |
| Trained adapter — Qwen2.5-3B | `huggingface.co/Itachi-42/disclosure-game-qwen-3b-grpo` |
| Trained adapter — Qwen2.5-7B | (planned, may complete by submission) |
| README with full results | top-level `README.md` |
| Qualitative comparison | `docs/QUALITATIVE_COMPARISON.md` |
| Brutal red-team report | passed in `privacy_game/server/redteam.py` |

### Numbers we have (v2 hand-shaped reward, n=50)

| Model | Trained mean | Base mean | Δ | Trained std |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | +0.4313 | +0.3707 | **+0.061** | 0.527 |
| Qwen2.5-1.5B | +0.5807 | +0.4994 | **+0.0813** | 0.432 |
| Qwen2.5-3B | +0.6040 | +0.5947 | **+0.0093** | 0.502 |

### Numbers we have (env-native pareto_it reward)

| Policy | Mean reward | n |
|---|---:|---:|
| `smart_generalize` (oracle ceiling) | +0.833 | 200 |
| `always_reveal` | +0.774 | 200 |
| trained Qwen2.5-0.5B (this work, v1) | +0.675 | 50 |
| Llama-3.1-8B-Instruct (untrained) | +0.150 | 30 |
| Qwen-2.5-7B-Instruct (untrained) | +0.133 | 30 |
| `always_refuse` | −0.001 | 200 |

---

## Honest review of what's paper-worthy and what's not

### ✅ What WILL pass review

1. **The problem framing** is novel. Multi-agent contextual integrity
   under adversarial inference is fresh territory.
2. **The OpenEnv environment** — fully working, on HF, reproducible.
3. **The composable rubric stack** — utility / reconstruction (rule-based) /
   reconstruction (Presidio third-party) / verbosity, with two
   composition modes (additive, Pareto-multiplicative). Reviewers will
   actually like this design.
4. **The 56-test red-team battery** — homoglyph defense, Cyrillic /
   Greek script-folding, Unicode normalization, decoy-probe handling.
   Engineering credibility.
5. **AI4Privacy real-data integration** — not synthetic Faker data.
6. **Diminishing-returns observation across 3 model sizes** — a real
   empirical finding.
7. **Frontier comparison** — trained 0.5B vs untrained 7B/8B is a
   striking ratio.

### ❌ What WILL fail review at top-tier venues

1. **n=50 is too small.** Welch's t-test on the best Δ (+0.0813 at
   1.5B with std~0.45) gives p ≈ 0.4. None of the deltas are
   statistically significant.
   - **Fix**: bump to n=500 per policy. Costs ~$5-10 in compute.

2. **Single seed (seed=42).** All three trained models used the same
   random seed.
   - **Fix**: train each model size with 3 seeds. Total compute:
     3 × (0.5B + 1.5B + 3B) = 9 runs ≈ $30 on HF Jobs.

3. **No SFT or DPO baseline.** The claim "RL helps" is unsupported
   without comparing to: (a) base model, (b) SFT on smart_generalize
   traces, (c) DPO on (reveal, smart) preference pairs.
   - **Fix**: train an SFT baseline (cheap) + DPO baseline (cheap).
     ~$5 total.

4. **No reward shape ablations.** v1 (env-native pareto_it) and v2
   (hand-shaped with smart_bonus) gave different deltas. Need to ablate.
   - **Fix**: train one more variant — v3 = pareto_it + smart_bonus
     hybrid. Compare the three reward shapes head-to-head.

5. **Single-turn training, multi-turn evaluation.** We trained on
   first-turn replies with scripted-smart_generalize tails. The
   trained model never sees its own multi-turn behavior during
   training.
   - **Fix**: extend GRPO to full multi-turn rollouts (the agent
     plays all 8 turns). This is non-trivial — needs ~2 weeks
     of engineering. Worth it for a main-conference paper.

6. **Adversary is rule-based with ~60 drug→diagnosis lookups.**
   A real adversary would use:
   - LLM-as-adversary (GPT-4o-mini reading transcripts and
     attempting reconstruction).
   - Production NER (spaCy + Presidio).
   - Cross-document linkage (multiple sessions).
   - **Fix**: add LLM-as-adversary as a parallel grading channel.
     Show that the trained model's privacy still holds.

7. **No ablation on registry size.** We use 20K filler + 200
   training profiles. Sweeney's 87% statistic assumes US-population
   scale. Need to show robustness as registry grows to 100K+.

8. **No human evaluation.** Reviewers will ask: do humans agree
   that the trained model's outputs are "privacy-preserving"?
   Currently no human-in-the-loop.

9. **No theoretical analysis.** The composable rubric has structure —
   especially the Pareto-multiplicative mode. There may be a clean
   theorem connecting it to information-theoretic privacy guarantees.
   We don't formalize it.

10. **3B Δ = +0.009 is essentially noise.** The diminishing-returns
    claim depends on this being a real ceiling, not a training-budget
    artifact. Need to show that more training steps don't recover the
    gap.

---

## Three publication paths

### Path 1 — arXiv preprint (this week, ~3-5 days of work)

**Goal**: Get the work indexed, cite-able, citable. Establishes priority.

**Effort**: ~20 hours of writing.

**What it includes**: exactly what you have at hackathon submission,
plus a more honest framing of limitations.

**Title suggestion**: *"CIPHER: Contextual-Integrity Privacy via Hardened
Episodic Reasoning — A Multi-Agent OpenEnv Environment for Privacy-Aware
LLM Training"*

**Steps**:
1. Write 4-page paper using NeurIPS template (or arXiv-ready Markdown
   converted to PDF via Pandoc + LaTeX).
2. Add a clear limitations section (use the bullets above as draft
   points).
3. Embed the three plots: training reward curve, before/after bar
   chart, scaling table.
4. Submit to arXiv under cs.CL with cross-list to cs.LG and cs.CR.
5. Post link from your README + tweet it out.

**Cost**: $0 (just your time).

**Outcome**: cite-able preprint. Strong CV line item. Hooks for future
collaborators and reviewers to find your work.

---

### Path 2 — workshop paper (1-2 weeks of additional work)

**Goal**: Refereed publication at a privacy/safety/RL workshop. Real
peer-reviewed credit.

**Realistic venues** (with deadlines from late 2025 / early 2026 in
order of fit):

1. **NeurIPS 2026 Workshop on Privacy ML (PrivML)** — typically
   August/September submission deadline. Best fit. Tracks: DP, federated
   learning, privacy-aware LLMs.
2. **ACL 2027 TrustNLP Workshop** — typically February deadline. Good
   fit for the contextual integrity angle.
3. **ICLR 2027 SafeML Workshop** — typically February deadline.
4. **NeurIPS 2026 SafetyAI Workshop** — September deadline.
5. **AAAI 2027 Privacy Track** — August deadline.
6. **EMNLP 2026 Workshop on Trust in NLP** — June deadline.
7. **USENIX Security 2027** — too security-focused, would need a real
   adversary. Possible but stretch.

**Required upgrades** (in priority order):

| # | Upgrade | Effort | Cost | Why required |
|---|---|---|---|---|
| 1 | Eval n=200 → n=500 per policy | 1 day | $5-10 | Statistical significance |
| 2 | Multi-seed: 3 seeds per model size | 3 days | ~$30 | Reviewer-1 will reject without |
| 3 | LLM-as-adversary (GPT-4o-mini) | 2 days | $10 | Adversary credibility |
| 4 | SFT baseline | 1 day | $5 | "RL helps" claim |
| 5 | DPO baseline | 1 day | $5 | "RL helps" claim |
| 6 | Reward shape ablation (v1 vs v2 vs hybrid) | 2 days | $10 | Reward design defense |
| 7 | Registry-size ablation (5K, 20K, 100K) | 1 day | $5 | Adversary calibration |

**Time**: ~10 days of work.
**Compute**: ~$70 of HF credit.
**Outcome**: workshop paper acceptance.

**Steps to write the workshop paper**:

1. **Week 1**: Run experiments 1-4 above. Generate plots.
2. **Week 2 days 1-3**: Run experiments 5-7. Final figures.
3. **Week 2 days 4-7**: Write 8 pages.
4. **Polish + submit**.

**Paper outline (8 pages, NeurIPS workshop format)**:

```
1. Introduction (1 page)
   - Motivating example: pharmacist vs stranger asking for medication
   - Contextual integrity (Nissenbaum 2010) framing
   - Single-turn redaction (Lusk 2026) → multi-turn disclosure
   - Contributions: (i) env, (ii) rubric, (iii) scaling experiment

2. Related work (1 page)
   - ConfAIde (Mireshghallah 2023) — eval, not training
   - DeepSeek-R1 (Guo 2025) — RLVR for reasoning
   - PrivLM-Bench, PII-redaction work
   - RLHF / RLVR / GRPO

3. Environment design (1.5 pages)
   - 3-agent loop (Discloser, RP, Adversary)
   - Composable rubrics (utility, recon-rules, recon-Presidio, verbosity)
   - Two reward modes (additive, pareto_it)
   - Sweeney + drug→dx + employer→attr + over-share inference rules
   - Tier system (precise, mild_gen, strong_gen, refuse)

4. Training (1 page)
   - GRPO + LoRA r=16
   - Single-turn agent + scripted-tail rollout
   - Hyperparameters (lr=1e-5, max_steps=200, num_generations=4)

5. Experiments (2 pages)
   - Baselines: refuse, reveal, random, smart_generalize (scripted)
   - Frontier comparison: Llama-3.1-8B, Qwen-2.5-7B (untrained)
   - Scaling: Qwen2.5 {0.5B, 1.5B, 3B, 7B*} (* if time)
   - Ablations: SFT, DPO, reward shape, registry size

6. Discussion (1 page)
   - Diminishing returns finding
   - Limitations: rule-based adversary, single-turn training,
     n=500 still moderate
   - Future work: full multi-turn RL, larger adversary, theoretical
     analysis

7. Conclusion (0.5 page)
```

---

### Path 3 — main conference paper (3-6 months of work)

**Goal**: NeurIPS, ICML, ICLR, ACL main conference.

**Effort**: ~3-6 months. This is a real research project, likely a
master's thesis chapter or PhD year-1 paper.

**Required upgrades beyond Path 2**:

1. **Full multi-turn RL training.** Replace the
   first-turn-+-scripted-tail rollout with full agent-plays-all-8-turns.
   This is the biggest engineering lift — needs custom GRPO patching
   or a different RL framework (TRL multi-turn, ART, SkyRL).
2. **Realistic LLM-based adversary.** GPT-4o or Claude-Sonnet as
   the adversary, with chain-of-thought reasoning over transcripts.
3. **Cross-architecture scaling.** Llama-3.x, Mistral, Gemma,
   not just Qwen2.5.
4. **Theoretical contribution.** Either (a) information-theoretic
   bound on the rubric composition, or (b) sample complexity
   analysis of GRPO on this task.
5. **Human evaluation.** N=20-50 annotators rate trained model
   outputs vs base on privacy-preservation and helpfulness. IRB
   approval needed if institutional.
6. **Real-world deployment study.** Partner with a privacy-tech
   company or open-source helpdesk to test on real interactions.
7. **8-12 page paper** with anonymous review.

**Realistic venue calendar**:
- ICLR 2028 (October 2027 deadline)
- NeurIPS 2027 (May 2027 deadline)
- ACL 2027 main (February 2027 deadline)

**Cost**: ~$500-1000 in compute, depending on multi-architecture
scaling.

**Outcome**: real research paper. CV-defining for a master's student.

---

## Concrete experiment list (priority-ordered)

If you have **3 days**, do experiments 1-4.
If you have **2 weeks**, do experiments 1-7.
If you have **3 months**, do experiments 1-12.

| # | Experiment | Effort | Cost | Why |
|---|---|---|---|---|
| 1 | Re-eval all 3 trained models at n=500 (single seed) | 1 day | $10 | Tight CIs |
| 2 | Multi-seed (3 seeds per size) full retrain | 3 days | $30 | Significance |
| 3 | SFT baseline on smart_generalize traces | 1 day | $5 | RL claim |
| 4 | DPO baseline on (reveal, smart) preferences | 1 day | $5 | RL claim |
| 5 | LLM-as-adversary (GPT-4o-mini reading transcripts) | 2 days | $10 | Adversary credibility |
| 6 | Reward shape ablation (v1 / v2 / hybrid) | 2 days | $15 | Reward design |
| 7 | Registry-size ablation (5K / 20K / 100K / 1M) | 1 day | $5 | Sweeney calibration |
| 8 | Full multi-turn GRPO training (agent plays all 8 turns) | 2 weeks | $100 | Closes biggest gap |
| 9 | Cross-architecture: Llama-3.2-{1B,3B,8B}, Mistral-7B | 1 week | $50 | Generalization |
| 10 | Human eval study (n=30 annotators) | 3 weeks | IRB + $500 | Reviewer requirement |
| 11 | Theoretical analysis of rubric composition | 4 weeks | $0 | Theory contribution |
| 12 | Real-world deployment partner study | 6 weeks | partner-dependent | Impact contribution |

---

## Concrete deliverables for the workshop paper

Items to add to the repo over the next 2 weeks:

```
docs/PAPER_ROADMAP.md         (this file)
docs/paper/
├── main.tex                  (NeurIPS workshop template)
├── refs.bib
├── figures/
│   ├── reward_curve_multiseed.pdf
│   ├── scaling_table.pdf
│   ├── before_after_with_baselines.pdf
│   ├── llm_adversary_results.pdf
│   └── reward_shape_ablation.pdf
└── tables/
    ├── frontier_comparison.tex
    ├── scaling_with_significance.tex
    └── ablations.tex

privacy_game/eval/
├── multiseed.py              (run multiple seeds, average)
├── llm_adversary.py          (GPT-4o-mini as adversary)
├── sft_baseline.py           (SFT on smart_generalize traces)
└── dpo_baseline.py           (DPO on preference pairs)

privacy_game/notebooks/
├── ablation_reward_shape.py
├── ablation_registry_size.py
└── train_multiseed_<size>.py
```

---

## Reading list (papers to cite)

### Required citations (already in our README)

- **Sweeney, L.** (2000). *Simple Demographics Often Identify People
  Uniquely.* Carnegie Mellon. → quasi-identifier theory.
- **Mireshghallah, N. et al.** (2023). *Can LLMs Keep a Secret? Testing
  Privacy Implications of LLMs via Contextual Integrity Theory.*
  arXiv 2310.17884. → ConfAIde, the closest prior work.
- **Nissenbaum, H.** (2010). *Privacy in Context.* → contextual
  integrity definition.
- **Guo, D. et al. (DeepSeek-AI)** (2025). *DeepSeek-R1: Incentivizing
  Reasoning in LLMs via RL.* arXiv 2501.12948. → RLVR foundation.
- **Shao, Z. et al.** (2024). *DeepSeekMath: Pushing the Limits of
  Mathematical Reasoning in Open Language Models.* arXiv 2402.03300.
  → GRPO algorithm.
- **Wei, J.** (2025). *The Verifier's Rule.* (blog post / talk).
  → RLVR framing.
- **Lusk, A.** (2026). *RLVR on PII Masking* (YouTube). → single-turn
  predecessor we extend.

### To add for workshop paper

- **Carlini, N. et al.** (2021). *Extracting Training Data from Large
  Language Models.* USENIX. → privacy attacks on LLMs.
- **Pan, A. et al.** (2022). *Membership inference and beyond:
  privacy attacks on LLMs.* → adversary calibration.
- **Bai, Y. et al.** (2022). *Constitutional AI.* → alignment
  framing for the rubric stack.
- **Stiennon, N. et al.** (2020). *Learning to summarize from human
  feedback.* → RLHF baseline framing.
- **Rafailov, R. et al.** (2023). *Direct Preference Optimization.*
  → DPO baseline.
- **Apple ML Research** (2025). *RL for Long-Horizon Interactive LLM
  Agents.* → multi-turn RL precedent (cited in our README already).
- **Liu, J. et al.** (2024). *Tofu: A Task of Fictitious Unlearning
  for LLMs.* → privacy-aware LLM eval methodology.
- **Patil, S. et al.** (2024). *PrivLM-Bench.* → benchmark
  ecosystem.

### To add for main conference paper

- Information-theoretic privacy: **Yeom, S. et al.** (2018).
  *Privacy Risk in Machine Learning: Analyzing the Connection to
  Overfitting.* CSF.
- **Pan, X. et al.** (2024). *Foot In The Door: Understanding Large
  Language Model Jailbreaking via Cognitive Psychology.* → relevant
  for the adversary side.
- **OpenAI** (2024). *o1 system card.* → reasoning-RL precedent.

---

## Pitfalls and contingencies

### Risk: 7B Δ is also tiny (~+0.005 or negative)

**Probability**: 70%. Diminishing returns continues.

**Pivot**: This is *fine* — the diminishing-returns finding becomes
the headline. Frame it as "GRPO helps small models a lot, helps
big models very little, suggesting privacy-aware behavior is mostly
a function of pre-training scale". A real result.

### Risk: SFT baseline matches or beats GRPO

**Probability**: 30%. Possible if the smart_generalize policy is a
clean enough teacher.

**Pivot**: Frame the contribution differently — the *environment* is
the contribution, not the algorithm. SFT on smart_generalize traces
*defines* an upper bound for imitation learning; GRPO can in principle
go beyond by exploring. Show the trajectory.

### Risk: LLM-as-adversary recovers privacy from trained transcripts that the rule-based adversary missed

**Probability**: 40%. Likely on edge cases.

**Pivot**: Strengthen the threat model. Add a "stronger-adversary"
result table showing that even GPT-4o-as-adversary recovers less
from trained-model transcripts than from base-model transcripts.

### Risk: No statistical significance even at n=500 with 3 seeds

**Probability**: 20%. If the true Δ is < +0.02, we can't get
significance without massive n.

**Pivot**: This is a methodology problem, not a research one. Show
non-parametric tests (Mann-Whitney, sign test) which need less power.
Or report effect sizes (Cohen's d) instead of p-values. Or focus on
the *qualitative* finding (per-task table with paired comparisons).

### Risk: Reviewer-2 says "the rule-based adversary is too easy"

**Probability**: 80% at top venues, 30% at workshops.

**Pivot**: Pre-emptively address — add a section showing that even
the rule-based adversary catches every Sweeney-quasi-identifier reveal
in the training set, and that real-world adversaries (e.g., medical
records linkers) often *are* rule-based at scale. Cite real-world
examples (HIPAA Safe Harbor uses fixed identifier rules).

---

## Author / advisor / IRB considerations

### Authorship

- **First author**: Rajveer Bishnoi (RAJVEER42)
- **Possible co-authors**: any teammates from the hackathon who
  contributed substantively. If solo, you're solo.
- **Acknowledgments**: Meta OpenEnv team (Sanyam Bhutani, Yash Khare,
  Nilesh Pandey), Hugging Face mentors (Adithya S Kolavi, Adarsh
  Shirawalmath, Ben Burtenshaw), Will Brown (verifiers / ART), Adam
  Lusk (single-turn precursor video).

### Advisor

If you're a student, you'll need an advisor signature on most
workshops and certainly on conference submissions. Possibilities:
- A professor at your institution doing privacy / NLP / RL work.
- Independent researchers via Twitter / Discord (Will Brown, etc.) —
  some will collaborate even on student projects.
- HF or Meta partner engineers from the hackathon — long shot but
  worth asking.

**Action**: when you start the paper-write phase, identify 3-5
potential advisors and reach out with a draft.

### IRB

- For arXiv preprint: **no IRB needed** — no human subjects research.
- For workshop paper: **no IRB needed** unless you add the human-eval
  experiment.
- For human-eval study (Path 3): **IRB approval required** at your
  institution. Typical timeline: 4-8 weeks.

### Ethics + responsible disclosure

You're working with privacy attacks (Sweeney triangulation,
drug→diagnosis inference). Two considerations:

1. **Defensive framing**: the paper is about *defending against*
   these attacks, not enabling them. Ethics statement: explicitly
   note that the inference rules are well-known in privacy literature
   (Sweeney 2000, etc.) and your contribution is a training environment
   to defend, not novel attacks.
2. **Real PII**: AI4Privacy data may contain PII even though it's
   labeled. Your repo's trajectories include profile fields. Make
   sure `outputs/trajectories*` stays gitignored or that you scrub
   before any public release. (Already done — check
   `.gitignore`.)

---

## Step-by-step plan (if you have 2 weeks after hackathon)

### Day 1-2: Triage and decide

- [ ] Hackathon results posted? Check ranking.
- [ ] Clean up the repo: remove debug code, update README to be
  paper-pitch-ready (less hackathon-pitch, more research-pitch).
- [ ] Read the 7 required-citations papers in full (re-read if
  already read).
- [ ] Decide: arXiv preprint only (Path 1), or aim for workshop
  (Path 2)?

### Day 3-5: Run the high-priority experiments

- [ ] **Experiment 1**: re-eval all 3 trained models at n=500. Use
  HF Inference for base models (free). For trained, run locally on
  Mac MPS or HF Jobs T4.
- [ ] **Experiment 3**: SFT baseline. Train Qwen2.5-1.5B on
  smart_generalize traces. ~30 min on T4. Compare to GRPO-trained
  1.5B.
- [ ] **Experiment 4**: DPO baseline. ~30 min on T4.

### Day 6-7: Multi-seed runs

- [ ] **Experiment 2**: re-train 0.5B, 1.5B, 3B with seeds 43, 44.
  Total: 6 jobs on HF Jobs. Cost: ~$25.

### Day 8-10: LLM-as-adversary + writing prep

- [ ] **Experiment 5**: code the LLM-adversary using GPT-4o-mini.
  Run on the n=500 eval transcripts.
- [ ] Generate final figures (reward curves with confidence bands,
  scaling table with significance markers).
- [ ] Outline the paper.

### Day 11-13: Write

- [ ] Day 11: Introduction + Related Work (2 pages).
- [ ] Day 12: Environment + Training (2.5 pages).
- [ ] Day 13: Experiments + Discussion + Conclusion (3.5 pages).

### Day 14: Polish + submit

- [ ] Read through. Ask one collaborator to read.
- [ ] Polish figures for legibility.
- [ ] Submit to chosen workshop.

---

## Step-by-step plan (if you only have 1 week)

Compress: skip multi-seed, skip DPO, focus on n=500 + SFT + LLM-adversary.
Submit a 4-page version to whichever workshop is closest.

---

## Step-by-step plan (if you only have a weekend)

Just write the arXiv preprint (Path 1). 4 pages. Use what you have.
Submit, get indexed. Future-you will thank you for the priority
date and the citation hooks.

---

## Final honest words

You built something good. Don't perfectionism it into the ground.

Three rules for your future self:

1. **Done is better than perfect.** Submit the arXiv preprint within
   2 weeks of the hackathon, even if rough. You can polish later.
2. **Reviewers care about rigor more than novelty.** The idea is
   already novel — what kills you is n=50 and single seed. Fix those,
   the rest follows.
3. **Workshop > conference for a first paper.** Aim for the workshop
   acceptance, then expand to conference if it lands. Don't aim
   straight for NeurIPS main with this much work.

Good luck. The work is real. Now make it cite-able.

---

*This document was generated during the hackathon run as a hand-off
to future-Rajveer. If anything here doesn't make sense in 4 weeks,
re-read the conversation transcripts in the project history. The
context is in the code.*
