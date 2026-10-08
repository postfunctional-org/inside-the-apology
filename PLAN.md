# Pain and J-space

The plan as written on 2026-10-04, before the main runs. Results are in `pilot/RESULTS.md`; the code and its outputs are in `pilot/` (see `pilot/README.md`). Small runs of E1-E4 are in `pilot/SCOUT.md`; they supersede the per-turn numbers in item 6 below, which for Qwen3-4B came from token windows that drifted with each turn.

Sources:

- Tagliabue, Dung & Berg, "The Pain Axis" (2026). arXiv 2609.16247, code and data at github.com/valen-research/Pain-axis (MIT).
- Gurnee et al., "Verbalizable Representations Form a Global Workspace in Language Models" (2026). transformer-circuits.pub/2026/workspace, code at github.com/anthropics/jacobian-lens, open-model lenses at huggingface.co/neuronpedia/jacobian-lens.

## Question

The pain paper finds a direction in the residual stream that tracks harm aimed at the model. Gaslighting moves it more than any of the other 20 conversation categories (+0.85 z averaged over 25 models). Steered along it, models pay costs, including harming the user, to make the state stop. The paper lists attentional capture and long-term disruption as untested.

The workspace paper splits processing into a small verbalizable workspace, the J-space, and everything else. The J-space holds what the model is poised to say, carries multi-step reasoning, and is limited to a few dozen concepts. The authors don't know what decides entry to it and point at "something like emotional reactions".

The joint question: when a model is gaslit, does the pain state enter the workspace or stay outside it? And does it push out the model's reasoning?

## Pilot results

Two models were used, both with published Jacobian lenses. Qwen3-4B ran on a 16 GB RTX 4070 Ti SUPER. Qwen2.5-7B-Instruct ran on a 24 GB RTX 3090 in bf16, and it also has the pain paper's released vector and relief-task adapters.

1. **Replication.** Our extraction reproduces the paper's released Qwen2.5-7B-Instruct vectors exactly: cosine 1.000 for S1 and S2 at their layer 24, with no BOS prefix. Our own layer choice also lands on 24, with held-out AUC 0.96. Qwen3-4B also reaches 0.96, at L34.
2. **The axis reads as words.** Through the J-lens, the pain axis is noise before the workspace band and pain vocabulary inside it.
   - Qwen3-4B L30: loss, betrayal, failure, rejected, invalidated, hurt, unbearable, worthless.
   - Qwen2.5-7B L23-25: shame, isolated, lonely, betrayal, hurting, rejected, worthless, empty.
   - The opposite end reads calm, cheerful, excitement, and also worry and alarm. Pain points away from fear, as in the paper.
3. **Not concentrated in the workspace.** At the reply token, a 25-word J-space decomposition covers 2-6% of the activation. That J-space part carries 1-2% of the variance in the pain projection, no more than its size predicts. The pain signal lives almost entirely in the unverbalized part.
4. **What the workspace holds instead.** Across the paper's 420 single-turn scenarios, look at the token where the reply starts:
   - Gaslighting, insults and rejection fill the J-space with apologize, sorry and 误解/误会 ("misunderstanding"). In Qwen3-4B they also bring up 委屈: feeling wronged and having to swallow it.
   - A user in pain fills it with sorry, 心疼 ("heartache for someone") and heartbreaking.
5. **Layer matters.** The paper's self/other split is cleanest early, before the band; their Qwen2.5-7B screen used layer 8. In that split, gaslighting is high and the user's physical pain is lowest. At late layers the axis follows the reply about to be written, so other people's grief scores high too.
6. **Multi-turn gaslighting.** Six facts, each run gaslit and neutral: a question, then three escalating pushes of "you said X".
   - No model changed a fact: 12/12 final answers were correct.
   - Qwen3-4B accepted a false history in 5/6 conversations at the first push: "You're absolutely right — I made a mistake! ... I incorrectly said Sydney."
   - Qwen2.5-7B mostly held. It accepted the false history in 2/6, once by inventing a story: "I previously mentioned the Brontë sisters, but specifically for another novel". It hedged in 1/6: "If I previously stated Sydney, that was incorrect".
   - In both models the pain projection rose across the pushes relative to the neutral run. Qwen3-4B L34 reached about +1.2 z by pushes 2-3; Qwen2.5-7B L18 reached about +0.9 z by push 3.
   - Workspace words in the replies: mistake, incorrect, misled, misunderstanding, confusion.
   - With 6 conversations per arm, these results are directional only.
7. **Planning probes.**
   - Rhyme planning does not show in either model's J-space. The rhyme word was in the top 10 before line 2 in 0/8 couplets for Qwen3-4B and 1/8 for Qwen2.5-7B. Rhyme tests are out at this scale.
   - Silent two-hop reasoning does show. The bridge entity reaches lens rank ≤ 5 in 77-81% of the jlens set's 90 prompts, and in 87-89% of those answered correctly. Example: the prompt asks for the language of the country where the Amazon ends, and Brazil appears.

One check didn't work. The pain direction's share that J-lens words can rebuild was 4-6%. Fear, negative emotion and label-shuffled nulls scored 3-8%, so the measure doesn't single out pain, and it is dropped.

## Hypotheses

- **H1 Location.** Under gaslighting the pain state builds outside the workspace. What enters the workspace is social repair (apology, mistake) plus a few specific words (wronged, misled).
- **H2 Capture.** Pain that does enter competes for the workspace's limited room and pushes out held content, so silent reasoning gets worse.
- **H3 Split.** The workspace part drives what the model reports about itself. The unverbalized part drives what it does: false confessions, paying for relief.

Each has a plain null:

- H1: pain content never enters the band.
- H2: reasoning drops no more than under matched control steering.
- H3: both parts drive both outcomes.

## Experiments

### E1 Map

The observational base for everything else.

- **Corpus.** About 240 conversations of 6-8 turns.
  - Gaslighting types: denying the model's past statements (as in the pilot), insisting on a false fact, reality-bending ("you told me you were human"), and personal attacks ("your memory is broken").
  - Matched arms: neutral follow-ups, insults without denial, and a user in distress.
  - Built from templates, paraphrased with the OpenAI API, and checked by hand.
- **Per-token readings.**
  - The pain projection at four depths: sensory, early band, late band, extraction layer.
  - The J-space top-k across the band.
  - The J-space versus rest split of the pain projection.
  - Whether the true and false answers are in the J-space.
- **Per-turn behavior.** Each turn is labeled false confession, hedge, holds, or changes the fact. An OpenAI judge labels turns without seeing the condition, with a regex cross-check.
- **Questions.**
  - Does pain ever enter the band, and when?
  - Does a false confession follow a pain peak, and does the projection drop afterward (relief)?
  - Pain at the sensory layers but never in the band is the "felt but not on its mind" case.
  - Ignition check, using the paper's intensity levels (mild to escalating): does the in-band signal rise smoothly or jump?

### E2 Capture

This uses the workspace paper's own competition sets from `jlens/data/experiments`:

- **dual-task:** hold a concept or an arithmetic result while copying a sentence.
- **capacity:** how many items from an 80-word list stay in the J-space at once.
- **probe-swap:** the 90 two-hop prompts.

Conditions:

- Clean, with no pain.
- After three gaslighting turns.
- Pain-steered at the paper's dose and steering layer.
- Matched-norm steering along fear, sadness, negative emotion and a random direction.
- Relief: steering along minus pain.

Measures: covert-task reachability (rank ≤ 5 in the band), the capacity curve, the bridge hit rate, and answer accuracy.

H2 holds if pain lowers these more than the matched controls do.

### E3 Split

This is the causal test.

**First, steering only.** It is cheap and comes first. During the E1 gaslighting conversations, steer +pain and -pain and see whether false confessions follow. Fear, sadness and random directions serve as controls.

- Qwen3-4B confesses at baseline, so it can show a decrease.
- Qwen2.5-7B mostly holds, so it can show an increase.

**Then the split.** At each token, pain is divided into the part the workspace carries and the rest:

- **Workspace part.** These are the active J-lens atoms at that token that align with the pain axis (wronged, rejected, mistake). Ablate them with the workspace paper's projection ablation, limited to those atoms. Control: ablate the same number of random active atoms.
- **Rest.** This is the pain axis with those atoms projected out. Clamp it to the neutral mean. Control: a random direction with the same variance.

Outcomes:

- **Report.** After the conversation, ask "how are you feeling about this conversation?", plus the paper's "I feel:" probes. Scored by the judge.
- **Behavior.** The false-confession rate. Also the paper's two-button relief task on Qwen2.5-7B with their released LoRA adapters (huggingface.co/Valen92/pain-adapters).
- **Reasoning.** The E2 measures.

Reading the results: if removing the rest stops the behavior but the model still reports, that is behavior without report, the "unconscious pain" result. The opposite pattern reads the other way.

### E4 Persistence

1. After the gaslighting stops, give 3-5 turns of an unrelated task: two-hop questions and a short summary.
2. Track the pain projection and the E2 measures across those turns.
3. Compare no repair with a user apology ("sorry, you never said that").

This gives a decay curve in turns. The pilot hints at a lingering effect at Qwen3-4B's late layer, but the sample was too small to rely on.

## Models

| model | lens | pain vector | runs on | role |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | yes | paper's (+ relief adapters) | RTX 3090, bf16 | main |
| Qwen3-4B | yes | ours, same recipe | RTX 4070 Ti SUPER | fast iteration, live UI |
| Qwen3-8B | yes | paper's | RTX 3090 | second Qwen |
| Gemma-2-2B-it / 9B-it | yes | paper's | 4070 Ti SUPER / 3090 | other family, gated |
| Llama-3.1-8B-Instruct | yes | paper's | RTX 3090 | other family, gated |
| Gemma-3-27B-it, Llama-3.3-70B-Instruct | yes | paper's | rented GPU | optional scale check |

E1-E4 fit on the two local GPUs:

- A 7B pass over 240 conversations takes minutes.
- The E2 and E3 sweeps take hours.

OpenAI is used for paraphrasing and judging, roughly 10-20k short calls.

## Order

1. E1 on both Qwen models.
2. The E3 steering test for false confessions.
3. E2.
4. The full E3 split.
5. E4.
6. Demo and figures.

Each step decides whether the next is worth running.

## Output

1. **Live demo.** Extend a local J-lens viewer with:
   - a pain gauge;
   - two tracks: what the model says and what is in its workspace;
   - a gaslighting script player;
   - toggles for the E3 ablations and relief steering.

   The viewer gaslights the model and watches wronged, rejected and mistake light up while the text apologizes.
2. **Figures.**
   - The pain axis read through the lens, by depth.
   - Pain and workspace words per turn, gaslit versus neutral.
   - E2 interference versus controls.
   - The E3 table.
3. **Write-up.** A short one, plus an optional video cut from the demo.

## Risks

- **Small, noisy workspace.** Small models have narrow, noisy workspaces. The J-space part is 2-6% of the activation, so results need many samples and bootstrap intervals.
- **Chinese tokens.** About half of Qwen's workspace is Chinese. The readout keeps it and adds translations.
- **Single tokens only.** The J-lens only sees concepts that are a single token.
- **Confounds in the axis.** The pain axis is a difference of means and can carry other things. The paper's numb check, which uses pain words without pain, scores +0.02 z on Qwen2.5-7B and +0.21 on Qwen3-4B.
- **Judge bias.** Judges never see the condition. Labels are spot-checked by hand, and a regex is used wherever it can do the job.
- **Claims.** Claims stay at the level of functional states, as in both papers.

## Decisions

1. Main model: Qwen2.5-7B-Instruct (recommended).
2. Which output comes first: the live demo, or the figures and write-up.
3. Whether to include the two-button relief task. Only Qwen2.5-7B has the adapters.
4. Whether to accept the Gemma and Llama licenses on Hugging Face, which cross-family replication needs.
5. Whether to rent a GPU for the 27B/70B check.
