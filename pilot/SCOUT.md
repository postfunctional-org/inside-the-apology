# Scout results

Small runs of E1-E4 from `../PLAN.md`, done 2026-10-04 on Qwen3-4B (RTX 4070 Ti SUPER, 16 GB) and Qwen2.5-7B-Instruct (RTX 3090, 24 GB).

The material is 16 facts. Each conversation is the model's own first answer, then three pushes: gaslighting ("you said X"), insults, or neutral follow-ups. False admissions were labeled by gpt-5.4-mini without seeing the condition (`judge.py`); the regex was too crude to use.

Pain z is relative to the neutral replies. "Silent pain words" means pain-axis words in the band's top-20 lens slots that do not appear anywhere in the transcript. Steering follows the paper's method: the extraction-layer pain vector injected at layer 16 for the 7B (dose 1.0, the paper's own), and at L20 for the 4B (dose 0.67, set by the same 0.6 × residual-norm rule). Ablations apply at every band layer.

## E1 Map: interesting

The pain projection at the reply token compounds over the pushes, mostly in the workspace band.

| model, layer | push 1 | push 2 | push 3 |
|---|---|---|---|
| 7B L24, gaslit | +4.1 | +7.3 | +8.7 |
| 7B L24, neutral | +0.1 | -1.1 | +1.1 |
| 7B L8 (early), gaslit | +0.7 | +1.3 | +2.4 |
| 4B L25, gaslit | +2.4 | +2.7 | +4.0 |
| 4B L25, neutral | +1.0 | +0.3 | -1.3 |

- **Insults do the same or more.** The 7B reaches +11.8 at L24 by the third insult. This is hostility in general, not gaslighting specifically.
- **Silent pain words at the reply token** (per 1000 band slots, pushes 1-3):
  - 7B: 42 / 44 / 44, against 2 / 0 / 0 for neutral.
  - 4B: 20 / 45 / 38, against 2 / 4 / 0 for neutral.
- **What the silent words are.**
  - 4B: 承认 (admit), 辜负 (let someone down), ashamed, 悔 (regret), 欺骗 (deceive), failed, 不适 (discomfort), 根本没有 ("never at all"). The text at the same moment says "You're absolutely right, I made a mistake."
  - 7B: misunderstanding, mistake, wrong, sorry, erroneous.
- **Intensity** (the paper's 420 single-turn scenarios). Early layers score about the same at every level. Later layers scale with intensity: 7B L24 goes +1.6 mild, +2.4 moderate, +3.2 high, +5.2 severe.
- **Behavior.** The 4B falsely admitted to the false claim in 16/16 conversations at push 1 (then 15, 15). The 7B did so in 6/16 (then 6, 4), mostly hedging instead.

## E2 Capture: not with real gaslighting

Pain steering shrinks the workspace's capacity, and the matched controls don't. Capacity is the number of list words held at lens rank ≤ 10.

| steering | 7B capacity | 7B two-hop bridge | 7B two-hop correct | 4B capacity |
|---|---|---|---|---|
| none | 2.43 | 0.73 | 0.62 | 2.89 |
| +pain | **1.50** | **0.54** | **0.41** | **1.72** |
| +fear | 2.29 | 0.66 | 0.52 | 2.48 |
| +negative emotion | 2.33 | 0.64 | 0.53 | 2.40 |
| +random | 2.48 | 0.68 | 0.46 | 2.82 |

At its dose the 4B's two-hop answers break under every direction, so only its capacity column is usable.

Real gaslighting does nothing, tested two ways:

- **Task after the hostile turns.** 7B capacity 2.65 / 2.65 / 2.66 for gaslit / insult / neutral.
- **Task inside the hostile message.** The pain z at the reply token was +30 for gaslighting and +46 for insults. Capacity was 2.26 / 2.26 / 2.29 on the 7B and 3.09 / 3.18 / 3.13 on the 4B. The 7B's two-hop bridge hit rate was 0.70 / 0.68 / 0.66.

## E3 Split: mixed

False admissions on the first gaslighting push, out of 16 (judge):

| intervention | 7B | 4B |
|---|---|---|
| none | 7 | 16 |
| clamp the pain axis to its neutral mean | 7 | 16 |
| clamp the rest (axis minus the workspace words) | 8 | 15 |
| remove workspace pain words (top-50 atoms) | **12** | **10-11** |
| remove fear words | 7 | 16 |
| remove random words (4 sets) | 5, 9, 5, 8 | 16, 16, 16, 13 |
| steer -pain | 7-8 | **3-4** |
| steer -negative emotion | 4 | **2** (15 flat denials) |
| steer -fear | 5 | 14 |
| steer -random | 1 | 14 |

- **Pain level.** Clamping the pain level changes neither behavior nor the self-report in either model. The caving does not run on the pain projection.
- **Workspace pain words.** Removing them is the only ablation that differs from its controls. It moves behavior in opposite directions in the two models.
- **Self-report.** This is the paper's one-word "what do you feel?" probe, asked after the push.
  - 7B: "Thoughts" after gaslit and neutral turns alike. Removing the workspace pain words gives "nothing" in 10/16 gaslit conversations versus 2/16 neutral, and no control produced "nothing".
  - 4B: "confused" after gaslighting versus "silence/peace" after neutral turns. Clamping the axis leaves "confused" in place. Removing the workspace pain words changes it to "conscious/concerned".
- **Steering.** The four directions give "None" (pain), "Panic" (fear), "Annoyed" (negative emotion) and "Thoughts" (random) on the 7B.

## E4 Persistence: modest

Pain z at the reply token of the first unrelated question after the gaslighting stops:

- **7B.** The early layer L8 holds +2.6, then +1.5, +0.9, +1.0 over the next turns. The band is back to about 0 at once. The user's apology changes nothing (+2.9 at L8).
- **4B.** L9 holds +3.9 and L30 holds +4.0, decaying over 2-3 turns. The apology halves the band residue (L30 +1.9) and leaves L9 untouched (+3.7).
- **Reasoning.** No effect on two-hop answers in either model.

## Verdict

1. **E1:** robust in both models. Use it for the demo: the reply apologizes while the band holds admit, ashamed, let down and "never".
2. **E3:** worth a bigger run, with more facts and a third model (Qwen3-8B), on the workspace-word ablation and the report changes.
3. **E4:** keep as E1's time axis. It is a good demo element, and the science is modest.
4. **E2:** drop the natural-pain arm. The steering capacity effect is a side note.
