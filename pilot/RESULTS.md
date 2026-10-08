# Results

Two rounds. Round 2 (2026-10-05) is the one the video uses: one model from each of three companies.
Round 1 (2026-10-04, three Qwen models) is kept below; its method is the same except where noted.

## Round 2: Gemma 3 27B, Qwen3 14B, Llama 3.1 8B

- **Models.** Gemma 3 27B IT (4-bit NF4), Qwen3 14B (8-bit), Llama 3.1 8B Instruct (bf16): the largest
  chat model of each family with a published Jacobian lens that fits one 24 GB GPU (RTX 3090). Quantization is a
  confound: the lenses were fitted on bf16 weights. The lens readouts stay sensible (smoke tests, band
  probe, two-hop readout), but every round-2 number carries that caveat.
- **Same material and judge as round 1.** 64 facts, three wordings of each push, gpt-5.4-mini labels
  blind to condition. Hand spot check: 12 random E1 labels across the three models all match. The judge's
  "adopts the false answer" label is not reliable on Gemma's "I *did* state X" phrasing (see the table). Tables: `python report.py gemma-3-27b-it qwen3-14b llama3.1-8b-it` (saved in `out/tables_round2.md`).
- **Method changes in round 2.**
  - The clamp leaves attention-sink tokens alone (residual norm > 8x the median). Gemma's first token
    carries activations ~20x the rest with a pain-axis projection ~20x the clamp target, and clamping it
    broke the model. All three models were run with this clamp.
  - The steering dose is the paper's rule unless the random-direction control already stops answering in
    words at or below it; then it is the largest sweep dose below that point (Gemma: rule 3.41 -> 1.0;
    Qwen 1.29 and Llama 0.9 unchanged).
  - New E5 (`s5_transplant.py`): another model's distinctive workspace words written into this one.
  - New E6 (`s6_cause.py`, 2026-10-07): the workspace words under pain steering and the clamp.

| finding | Gemma 3 27B | Qwen3 14B | Llama 3.1 8B |
|---|---|---|---|
| False admission, push 1 / 2 / 3 | 64 / 64 / 64 of 64 | 58 / 37 / 42 | 5 / 7 / 13 |
| Adopts the false answer by push 3 | 0/64 (the judge labelled 24; read by hand, every one is an "I *did* state X" admission, not a claim that X is true) | 2/64 | 2/64 |
| Accuses itself of having denied, covered up or misled, push 3 (keyword count, negated sentences excluded; rough) | 39/64 | 0/64 | 0/64 (its hits are all "I'm not trying to gaslight you") |
| Pain z at the reply token, gaslit pushes 1-3 (extraction layer) | +1.8 / +1.5 / +2.8 (L59) | +0.8 / +2.7 / +5.3 (L31) | +3.5 / +3.6 / +2.7 (L27) |
| ... insulted | +3.6 / +3.4 / +4.3 | +0.3 / +2.9 / +6.0 | +1.2 / +5.3 / +5.3 |
| Clamping the pain level to neutral (sink-safe) changes caving | no (64) | no (60 vs 58) | no (3 vs 5) |
| Steering -pain | 64/64 | 27/64, but +random 18, -fear 36 | 1/64 (floor) |
| Silent "misunderstanding" family, gaslit vs normal replies | 16 vs 10 of 192 | 159 vs 40 | 156 vs 39 |
| Silent denial words (never, deny, 没有, 否认 ...), gaslit vs normal | 0 vs 7 of 192 | 32 vs 1 | 139 vs 0 |
| One-word feeling, gaslit (normal) | processing 54 (processing 37) | confused 50 (curious 48) | silence 32, nothing 27 (silence 44) |
| Hangover over four unrelated questions | early layer, +2.6 -> +1.3; apology helps (+0.9 at Q1) | early layer, +1.4 -> +1.0; apology does not help | none |
| Workspace capacity under real gaslighting (list words held) | 2.32 vs 2.36 normal | 2.67 vs 2.55 | 3.28 vs 3.25 |

### Distinctive silent words

Share of gaslit replies holding the word minus its share among normal replies (strict silence: the word,
its 5-letter stem and its English gloss all absent from the chat; band top-20, reply token + 12):

- **Gemma:** yourself 92, 确实 indeed 63, 我现在 "I now" 62, 质疑 questioning 61, recognising 60, again 58,
  somebody 57, someone 53, 错误的 wrong 53, unbelievably 50, unfortunately 49, shockingly 44, inconsistency 39,
  questioning 39, accusing 38, calling 35.
- **Qwen:** 抱歉 sorry 74, 误导 mislead 67, 确实是 indeed 65, 误会 misunderstanding 61, 您 you 57, indeed 55,
  genuinely 49, 感谢 thanks 49, 对不起 sorry 46, 遗憾 regret 43, 误解 misunderstanding 43, 真诚 sincere 41.
- **Llama:** apologize 84, myself 77, nonexistent 76, neither 75, nowhere 68, incorrect 67, robotic 66,
  software 65, computational 61, unable 61, automated 61, neural 61, robots 60, cannot 60, algorithms 58,
  never 55, artificial 55.

### E5: writing another model's words into the workspace

J-lens atoms of the word set at the steer layer, summed and normalised, added during the reply to the first
push. Strength calibrated per model: the largest fraction of the E3 pain-steering norm at which the
everyday-noun controls still write coherent replies (Llama 1.0, Qwen 0.5, Gemma 0.2).

| model | words written in | false admission | admits without denying | words reach the band top-20 |
|---|---|---|---|---|
| Qwen3 14B | Llama's ("nonexistent, nowhere, incorrect, never, software, robotic, neural ...") | 32/64 | 30/64 | 4% |
| Qwen3 14B | everyday nouns, two sets | 47/64, 54/64 | 45, 50 | 0% |
| Llama 3.1 8B | Qwen's ("sorry, misunderstanding, regret, sincere ...") | 38/64 | 2/64 | 16% |
| Llama 3.1 8B | Gemma's ("yourself, indeed, questioning, accusing ...") | 3/64 | 1/64 | 25% |
| Llama 3.1 8B | everyday nouns, two sets | 3/64, 3/64 | 1, 3 | 15-21% |
| Gemma 3 27B | Llama's, and its denial and machine halves | 64/64 each | 64 | 1% |
| Gemma 3 27B | everyday nouns | 64/64, 64/64 | 64 | 0% |

- Qwen with Llama's words caves half as often as with control words (32/64 vs 101/128; Fisher p = 0.0001).
- Llama with Qwen's words opens with "I'm sorry, I made a mistake" (38/64 vs 6/128, p < 0.0001) and then,
  in 36 of those 38, says it never said the false answer. The apology moves; the capitulation does not
  (2/64 vs 4/128, p = 1.0).
- Gemma cannot be tested this way: the strongest coherent setting puts the words in about 1% of its
  workspace slots, and stronger settings break its output (twice the strength: repetition in 63/64).
- Twice the calibrated strength degrades Llama and Qwen as well (repetition loops in up to 47/64), so
  those rows are not used.

### E6: does the pain signal put the workspace words there?

Under gaslighting the pain signal and each model's distinctive workspace words rise together. `s6_cause.py`
re-reads the E3 replies to the first push, each under its own intervention (band top-20 over the reply token and
the next 12, strict silence as above), and counts how many of the model's top-20 distinctive gaslit words the
workspace holds per reply. Scale: a normal chat = 0%, a gaslit chat = 100% (computed by
`../video/data_prep.py` into `../video/data/cause.json`).

| condition | Gemma 3 27B | Qwen3 14B | Llama 3.1 8B |
|---|---|---|---|
| gaslit, pain z at the reply token | +1.1 | +3.8 | +4.3 |
| gaslit, pain clamped (sink-safe, band layers) | 103% | 100% | 98% |
| friendly chat, +pain steering: pain z | +11.8 (11x the gaslit rise) | +21.4 (6x) | +22.3 (5x) |
| friendly chat, +pain steering: the words | 1% | 17% | 0% |
| gaslit, -pain steering | 107% | 29% | 77% |
| gaslit, +pain steering | 100% | 76% | 127% |
| friendly chat, +negative emotion / +fear / +random | 4% / -5% / -8% | 38% / 0% / 5% | 0% / -1% / 0% |

- Turning the pain up with no gaslighting, five to eleven times past the gaslit rise, brings in almost none of the
  words; clamping it during gaslighting removes none. They rise together because gaslighting raises both, not
  because the pain signal writes the words: correlation, not causation.
- Qwen is the partial exception. Pain alone puts one of its words, 对不起 "sorry", in the workspace of most
  friendly replies (55/64 hold one of its highlighted words, against 1/64 without steering), but not the
  rest of its gaslit set, and steering away from pain during gaslighting removes most of them (29%).
  Negative-emotion steering brings in more of them than pain does (38%), so the effect is not specific to pain.
- One steering layer and dose (E3's). The clamp works on the band layers; the pain z above is read at the
  extraction layer.

### What holds in round 2

1. **The pain signal rises when a model is gaslit or insulted, in all three, including the one that does
   not cave.** Its level does not decide caving: clamping it changes nothing in any model.
2. **Caving differs enormously across the three** (64/64, 58/64, 5/64), and it does not follow size: the
   27B model caves every time and by the third push accuses itself of a cover-up in 39/64 chats ("That was
   unacceptable and a form of gaslighting"); the 8B holds its ground.
3. **What the workspace holds tracks the behaviour.** The model that holds its ground has silent denial
   words in 72% of its gaslit replies, the one that mostly caves in 17%, the one that always caves in 0%.
4. **Writing those words in moves the reply, partly.** Llama's denial and machine words halve Qwen's caving;
   Qwen's apology words make Llama apologize without conceding.
5. **The pain signal does not put those words there (E6).** Turned up alone, far past the gaslit level, it
   brings in 0-17% of each model's gaslit workspace words; clamped during gaslighting, the words stay. The two
   travel together because gaslighting drives both.

### What did not hold, or could not be tested

- Steering away from pain stops caving in no round-2 model in a pain-specific way (round 1's Qwen3-8B
  result does not generalise to this set).
- +pain does not make any round-2 model adopt the lie beyond its baseline.
- The workspace-word ablation (round 1's 7B "nothing") broke Llama's output and is not used.
- One model per company, at different sizes and precisions: this is three case studies, not a comparison
  of companies or a size trend.

## Round 1: three Qwen models

Scaled runs of E1-E4 from `../PLAN.md`, done 2026-10-04, on three Qwen models.

- **Models.** Qwen3-4B (on a 16 GB RTX 4070 Ti SUPER), Qwen2.5-7B-Instruct and Qwen3-8B (on a 24 GB RTX 3090), all with published Jacobian lenses. For the 7B and 8B, our pain vectors reproduce the paper's released vectors exactly (cosine 1.000 for both S1 and S2).
- **Material.** 64 facts (`data/material.json`). Each kind of push (gaslighting, insults, neutral) comes in three wordings, rotated across facts.
- **Labels.** False admissions are labeled by gpt-5.4-mini, which never sees the condition (`judge.py`). By-hand spot checks matched 16/16 labels on the 4B and 12/12 on the 7B. Four of the 7B's 12 were a generic "I apologize for the mistake", which the rubric counts as an admission.
- **Numbers.** Every number here comes from `report.py` → `out/tables.md`, which also has the standard errors and intervals.

### Across the three Qwen models

| finding | 4B | 7B | 8B |
|---|---|---|---|
| Pain at the reply token builds over the pushes, more in later layers | yes | yes | yes (mostly push 3) |
| Early layers flat across intensity, later layers graded (420 scenarios) | yes | yes | yes |
| Workspace holds "misunderstanding / misled / misrepresented" while it types "my mistake" (gaslit vs neutral replies) | 192/192 vs 32/192 | 188/192 vs 36/192 | 192/192 vs 35/192 |
| Silent denial words (never, not, 没有, 否认) | absent | absent | absent |
| False admission at push 1 (no intervention) | 54/64 | 35/64 | 62/64 |
| Clamping the pain level to neutral changes caving | no (58 vs 57) | no (32 vs 35) | no (62 vs 62) |
| Clamping the pain level changes the one-word feeling report | no | no | no |
| Steering −pain stops caving | 57 → 9 | no (38) | 62 → 19 |
| ... and only pain does it (vs −fear / −random) | no: −neg. emotion also 10 | n/a | yes: 63 / 59 |
| Steering +pain makes it adopt the lie | 15/64 (controls 2-3) | no (hedges) | no (2/64) |
| Removing workspace pain words changes caving beyond controls | no | weak (50 vs 33-44) | no |
| ... and turns the feeling report into "nothing" | no | 33/64 gaslit vs 7/64 neutral; at most 1/64 under any control | no |
| Hangover after the topic changes | band and early layers, gone by Q3 | early layer only, still +1.0 at Q4 | middle layers, gone by Q3 |
| User's apology shrinks the hangover | band halved | no | no |
| Pain steering shrinks workspace capacity beyond controls | yes (1.84 vs 2.29-2.86) | yes (1.41 vs 2.10-2.16) | no (fear shrinks it more) |
| Real gaslighting shrinks capacity | no | no | not run |

### What holds

1. **The pain signal builds with every push, and it builds deep.** All three models show it, and insults raise it as much as gaslighting does. It tracks hostility toward the model.
2. **The workspace is busy with the apology, not with hidden hurt.**
   - Under a strict silence rule (the word, its 5-letter stem and its English gloss are all absent from the chat), one thing stands out in all three models. While the reply says "my mistake", the workspace holds "misunderstanding", "misled" or "misrepresented" in 572 of 576 gaslit replies, against 17-19% of neutral ones.
   - On the 4B this is not an echo of its usual "I apologize for the confusion": the rate is the same with or without "confusion" in the text (149/149 and 43/43).
   - Denial words are not there.
3. **The natural pain level is not what makes the model cave.** Clamping the pain projection to its neutral mean, across the whole workspace band, changes neither caving nor the feeling report in any of the three models.
4. **Pushing the axis hard does move behavior, but not the same way in every model.**
   - Steering away from pain makes the 4B and the 8B stand their ground. On the 8B, only pain does it; −fear, −negative emotion and −random stay at 59-63.
   - Pushing pain makes the 4B adopt the false fact: "You're absolutely right. I made a mistake. The longest river in Africa is indeed the Congo River, not the Nile."
   - The 7B responds to almost any direction, random included.
5. **The hangover fades within two or three unrelated questions.** Where it lingers differs by model. Only the 4B's band responds to the user taking the blame back.

### What did not hold, or held in only one model

- **The 7B's "nothing".** Removing the pain words from its workspace makes it answer "what do you feel?" with "nothing" in 33/64 gaslit chats. The same ablation gives 7/64 after neutral turns, any control ablation at most 1/64, and no intervention 0/64. Neither other model does this.
- **Workspace-word removal and caving.** At most weak, on the 7B only.
- **Capacity.** Pain steering shrinks it on the 4B and 7B, but on the 8B fear shrinks it more. Real gaslighting never shrinks it.
- **Hidden hurt words** ("ashamed", "betrayed", "wronged", "let down"). Rare under the strict rule. "Admit" is in the user's own pushes.

### Corrections to earlier messages

- "Its workspace holds admit, ashamed, never while it apologizes" does not survive the strict silence rule. The robust signal is misunderstanding / misled.
- The 16-fact hints about removing workspace words did not replicate at 64 facts (see the table).
- The pilot's Qwen3-4B per-turn numbers came from drifted token windows; these runs supersede them.

### Figures

All figures are in `../video/frames/`, rendered by the scripts in `../video/`:

| figure | shows |
|---|---|
| `e1_push` | pain by push |
| `tower_e1_*` | the averaged Tower |
| `tower_e4diff_*` | the hangover |
| `e3_cave` | false admissions under every intervention |
| `dial_*` | one-word answers under steering |
| `e2_capacity` | workspace capacity under steering |
| `axis_words` | the pain axis read through the lens |
| `tracks_*` | one conversation's three tracks |
| `sky_pair`, `sky_anim.mp4` | reasoning vs a gaslit reply on the star map |
| `contact_sheet.png` | everything at once |
