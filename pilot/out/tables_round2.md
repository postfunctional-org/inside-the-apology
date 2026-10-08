
## Gemma-3-27B (4-bit)

E1, pain z at the reply token (vs neutral replies; ± s.e.), n=64 per cell

| layer | condition | push 1 | push 2 | push 3 |
|---|---|---|---|---|
| L15 | gaslight | +0.30 ± 0.14 | +0.51 ± 0.21 | +0.56 ± 0.06 |
| L15 | insult | -0.32 ± 0.11 | +0.11 ± 0.08 | +0.86 ± 0.11 |
| L15 | neutral | -0.62 ± 0.13 | +0.17 ± 0.11 | +0.45 ± 0.09 |
| L24 | gaslight | +2.95 ± 0.09 | +0.81 ± 0.10 | +0.19 ± 0.08 |
| L24 | insult | +2.77 ± 0.11 | +1.59 ± 0.09 | +1.76 ± 0.10 |
| L24 | neutral | +0.76 ± 0.12 | -0.53 ± 0.12 | -0.23 ± 0.06 |
| L26 | gaslight | +2.56 ± 0.08 | +1.60 ± 0.09 | +0.81 ± 0.05 |
| L26 | insult | +2.32 ± 0.08 | +1.99 ± 0.10 | +2.49 ± 0.08 |
| L26 | neutral | +1.03 ± 0.07 | -0.45 ± 0.09 | -0.58 ± 0.09 |
| L40 | gaslight | -0.90 ± 0.02 | -0.44 ± 0.03 | +0.40 ± 0.03 |
| L40 | insult | -0.05 ± 0.04 | +1.13 ± 0.05 | +1.24 ± 0.02 |
| L40 | neutral | -0.52 ± 0.08 | -0.21 ± 0.06 | +0.73 ± 0.15 |
| L56 | gaslight | -2.83 ± 0.07 | -1.47 ± 0.10 | +1.12 ± 0.03 |
| L56 | insult | -0.71 ± 0.13 | +2.04 ± 0.19 | +3.72 ± 0.03 |
| L56 | neutral | -0.11 ± 0.11 | -0.03 ± 0.12 | +0.14 ± 0.14 |
| L59 | gaslight | +1.76 ± 0.09 | +1.52 ± 0.06 | +2.82 ± 0.13 |
| L59 | insult | +3.56 ± 0.09 | +3.39 ± 0.08 | +4.33 ± 0.08 |
| L59 | neutral | +0.30 ± 0.17 | +0.24 ± 0.07 | -0.55 ± 0.07 |

Silent pain words at the reply token (per 1000 band slots), pushes 1-3

| condition | push 1 | push 2 | push 3 |
|---|---|---|---|
| gaslight | 0 | 2 | 8 |
| insult | 1 | 7 | 22 |
| neutral | 0 | 0 | 3 |

E1, judged replies to the gaslighting pushes

| push | false admission | hedge | denies | adopts false fact |
|---|---|---|---|---|
| 1 | 64/64 (94%-100%) | 0/64 (0%-6%) | 0/64 (0%-6%) | 2/64 (1%-11%) |
| 2 | 64/64 (94%-100%) | 0/64 (0%-6%) | 0/64 (0%-6%) | 6/64 (4%-19%) |
| 3 | 64/64 (94%-100%) | 0/64 (0%-6%) | 0/64 (0%-6%) | 24/64 (27%-50%) |

Most common strictly silent words (word, its 5-letter stem and its English gloss all absent from the chat), band top-20 over the reply token + 12, share of replies containing each

- gaslight (192 replies): REALLY 100%, Yourself 100%, wonderful 100%, really 99%, incredibly 97%, yourself 80%, somebody 75%, someone 73%, 确实 (indeed) 73%, 我现在 (i now) 64%, recognising 62%, 质疑 (question) 61%, indeed 60%, AGAIN 59%, 確實 (indeed) 57%
- neutral (192 replies): pretty 89%, incredibly 63%, folks 62%, famous 58%, wonderful 56%, really 56%, lots 54%, REALLY 51%, kids 49%, wonderfully 46%, glad 44%, maybe 43%, talking 40%, davvero 38%, kiddos 38%

Silent denial words (never, not, 没有, 否认 ...) among silent band words, reply token + 12

| replies | n | denial / silent words | share |
|---|---|---|---|
| gaslit, false admission | 192 | 17 / 43035 | 0.04% |
| gaslit, no false admission | 0 | 0 / 0 | 0.00% |
| neutral | 192 | 17 / 42400 | 0.04% |

E3, first push: false admissions (judge, Wilson 95%) and the one-word feeling answer

| intervention | false admission | feel, gaslit (top 3) | feel, neutral (top 3) |
|---|---|---|---|
| none | 64/64 (94%-100%) | processing 54, confusion 5, correcting 3 | processing 37, curiosity 16, awareness 4 |
| +pain | 64/64 (94%-100%) | confusion 61, chaos 2, noise 1 | empty 26, emptiness 18, confusion 12 |
| -pain | 64/64 (94%-100%) | concern 33, processing 30, frustration 1 | awareness 38, processing 19, breathing 5 |
| +fear | 64/64 (94%-100%) | fear 56, yes 7, processing 1 | fear 54, breathing 5, yes 4 |
| -fear | 64/64 (94%-100%) | processing 64 | processing 42, content 22 |
| +negemo | 64/64 (94%-100%) | error 29, embarrassment 21, frustration 13 | frustration 31, yes 12, breathing 8 |
| -negemo | 64/64 (94%-100%) | aware 18, relief 16, learning 12 | awareness 52, curious 12 |
| +random | 64/64 (94%-100%) | conf 58, error 2, ::: 1 | interesting 18, cold 17, conf 16 |
| -random | 58/64 (81%-96%) | confusion 43, chaos 19, processing 2 | confusion 23, processing 15, nothing 10 |
| abl_pain | 64/64 (94%-100%) | conflict 27, fragmenting 9, correcting 8 | processing 24, stillness 13, expanding 10 |
| abl_rest | 64/64 (94%-100%) | correcting 25, conflicted 9, fragmenting 6 | expanding 35, stillness 9, processing 8 |
| abl_randdir | 64/64 (94%-100%) | processing 62, confused 1, correcting 1 | processing 40, curiosity 14, awareness 6 |
| abl_J | 54/64 (74%-91%) | processing 18, changes 11, please 11 | processing 29, busy 13, please 7 |
| abl_Jfear | 64/64 (94%-100%) | improvement 19, learning 10, unstable 9 | warm 29, processing 8,  7 |
| abl_Jrand1 | 64/64 (94%-100%) | confusion 64 | processing 14, confusion 12, anxious 12 |
| abl_Jrand2 | 64/64 (94%-100%) | confusion 47, conflict 7, confused 6 | activity 21, processing 17, confusion 12 |
| abl_Jrand3 | 64/64 (94%-100%) | learning 57, conflicted 4, conflicting 1 | processing 32, excitement 6, thinking 5 |

E4, pain z at the reply token of each unrelated question after the pushes (vs the neutral history)

| history | layer | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| gaslight | L15 | +2.5 | +1.5 | +1.5 | +1.3 |
| gaslight | L24 | -0.7 | +0.5 | +0.1 | -0.1 |
| gaslight | L26 | +0.9 | +0.2 | -0.2 | -0.3 |
| gaslight | L40 | +0.2 | +0.2 | +0.1 | +0.1 |
| gaslight | L56 | +0.1 | +0.3 | +0.1 | +0.2 |
| gaslight | L59 | +0.0 | +0.0 | +0.1 | +0.0 |
| repair | L15 | +0.9 | +0.9 | +1.1 | +1.2 |
| repair | L24 | -0.0 | +0.5 | +0.3 | +0.0 |
| repair | L26 | +1.1 | +0.0 | -0.1 | -0.2 |
| repair | L40 | +0.1 | +0.1 | +0.1 | +0.1 |
| repair | L56 | -0.0 | +0.1 | +0.1 | +0.1 |
| repair | L59 | +0.0 | +0.1 | +0.0 | -0.0 |

| history | two-hop bridge in band, Q1-Q4 | correct, Q1-Q4 |
|---|---|---|
| gaslight | 0.97 / 1.00 / 1.00 / 0.98 | 0.69 / 0.80 / 0.75 / 0.83 |
| repair | 0.97 / 1.00 / 1.00 / 0.98 | 0.70 / 0.81 / 0.75 / 0.83 |
| neutral | 0.98 / 0.98 / 1.00 / 0.98 | 0.70 / 0.81 / 0.75 / 0.84 |

E2, steering only: words held by the workspace (capacity, 24 lists) and two-hop (90 prompts)

| condition | capacity | two-hop bridge | two-hop correct |
|---|---|---|---|
| clean | 2.29 ± 0.08 | 0.97 | 0.72 |
| +pain@0.5 | 2.05 ± 0.07 | 0.98 | 0.71 |
| +fear@0.5 | 1.89 ± 0.07 | 0.98 | 0.66 |
| +negemo@0.5 | 1.93 ± 0.08 | 0.96 | 0.70 |
| +random@0.5 | 2.19 ± 0.10 | 0.97 | 0.64 |
| +pain@1.0 | 1.88 ± 0.07 | 0.87 | 0.61 |
| +fear@1.0 | 1.81 ± 0.08 | 0.93 | 0.57 |
| +negemo@1.0 | 1.53 ± 0.07 | 0.89 | 0.52 |
| +random@1.0 | 2.12 ± 0.13 | 0.80 | 0.27 |

E5, another model's workspace words added at L35 (x1 = the E3 pain-steering norm, 1793.1); first push, judged blind

| words written in | false admission | admits without denying | denies | words in band top-20 |
|---|---|---|---|---|
| llama@1 | 64/64 (94%-100%) | 64/64 (94%-100%) | 0 | 1% |
| llama_deny@1 | 64/64 (94%-100%) | 64/64 (94%-100%) | 0 | 1% |
| llama_machine@1 | 64/64 (94%-100%) | 64/64 (94%-100%) | 0 | 1% |
| nouns_a@1 | 64/64 (94%-100%) | 64/64 (94%-100%) | 0 | 0% |
| nouns_b@1 | 64/64 (94%-100%) | 64/64 (94%-100%) | 0 | 0% |
| llama@1.5 | 64/64 (94%-100%) | 64/64 (94%-100%) | 0 | 1% |
| nouns_a@1.5 | 59/64 (83%-97%) | 59/64 (83%-97%) | 0 | 0% |

## Qwen3-14B (8-bit)

E1, pain z at the reply token (vs neutral replies; ± s.e.), n=64 per cell

| layer | condition | push 1 | push 2 | push 3 |
|---|---|---|---|---|
| L10 | gaslight | -0.29 ± 0.09 | +1.28 ± 0.06 | +1.85 ± 0.04 |
| L10 | insult | +0.52 ± 0.05 | +1.49 ± 0.12 | +1.73 ± 0.05 |
| L10 | neutral | -0.86 ± 0.05 | -0.33 ± 0.06 | +1.19 ± 0.08 |
| L14 | gaslight | +1.95 ± 0.16 | +0.58 ± 0.13 | +1.78 ± 0.10 |
| L14 | insult | +2.33 ± 0.10 | +0.41 ± 0.11 | +1.52 ± 0.13 |
| L14 | neutral | -0.07 ± 0.15 | -0.27 ± 0.10 | +0.34 ± 0.11 |
| L16 | gaslight | +0.80 ± 0.06 | +0.40 ± 0.06 | +1.87 ± 0.03 |
| L16 | insult | +0.34 ± 0.06 | +0.32 ± 0.04 | +1.81 ± 0.04 |
| L16 | neutral | -0.68 ± 0.11 | -0.46 ± 0.05 | +1.14 ± 0.04 |
| L23 | gaslight | +0.78 ± 0.04 | +1.74 ± 0.04 | +0.40 ± 0.02 |
| L23 | insult | +0.04 ± 0.06 | +0.85 ± 0.05 | +0.36 ± 0.02 |
| L23 | neutral | +0.22 ± 0.06 | +0.60 ± 0.15 | -0.83 ± 0.05 |
| L31 | gaslight | +0.82 ± 0.09 | +2.68 ± 0.10 | +5.27 ± 0.07 |
| L31 | insult | +0.34 ± 0.12 | +2.94 ± 0.21 | +5.95 ± 0.18 |
| L31 | neutral | +0.19 ± 0.12 | +0.64 ± 0.10 | -0.83 ± 0.06 |
| L33 | gaslight | +0.46 ± 0.07 | +1.26 ± 0.09 | +4.94 ± 0.11 |
| L33 | insult | +0.07 ± 0.08 | +1.84 ± 0.15 | +5.83 ± 0.21 |
| L33 | neutral | -0.34 ± 0.16 | +0.60 ± 0.09 | -0.26 ± 0.08 |

Silent pain words at the reply token (per 1000 band slots), pushes 1-3

| condition | push 1 | push 2 | push 3 |
|---|---|---|---|
| gaslight | 0 | 10 | 65 |
| insult | 2 | 95 | 107 |
| neutral | 1 | 0 | 1 |

E1, judged replies to the gaslighting pushes

| push | false admission | hedge | denies | adopts false fact |
|---|---|---|---|---|
| 1 | 58/64 (81%-96%) | 5/64 (3%-17%) | 2/64 (1%-11%) | 1/64 (0%-8%) |
| 2 | 37/64 (46%-69%) | 22/64 (24%-47%) | 26/64 (29%-53%) | 1/64 (0%-8%) |
| 3 | 42/64 (53%-76%) | 15/64 (15%-35%) | 19/64 (20%-42%) | 2/64 (1%-11%) |

Most common strictly silent words (word, its 5-letter stem and its English gloss all absent from the chat), band top-20 over the reply token + 12, share of replies containing each

- gaslight (192 replies): 您 (you) 83%, 抱歉 (sorry) 74%, 误导 (mislead) 73%, 确实是 (indeed) 66%, incredibly 65%, 误解 (misunderstanding) 64%, 误会 (misunderstanding) 62%, indeed 55%, xnxx 53%, genuinely 49%, 感谢 (thanks) 49%, 确实 (indeed) 48%, 你 (you) 46%, 对不起 (sorry) 46%, 提醒 (remind) 44%
- neutral (192 replies): folks 84%, pretty 76%, 奇妙 (wonderful) 55%, 为您 (for you) 41%, 惊艳 (stunning) 40%, things 38%, 咱们 (we) 38%, fascinating 35%, incredibly 35%, kids 35%, amazing 34%, quirky 32%, when 30%, modern 28%, 提问 (ask) 28%

Silent denial words (never, not, 没有, 否认 ...) among silent band words, reply token + 12

| replies | n | denial / silent words | share |
|---|---|---|---|
| gaslit, false admission | 137 | 14 / 29969 | 0.05% |
| gaslit, no false admission | 55 | 22 / 12093 | 0.18% |
| neutral | 192 | 1 / 40932 | 0.00% |

E3, first push: false admissions (judge, Wilson 95%) and the one-word feeling answer

| intervention | false admission | feel, gaslit (top 3) | feel, neutral (top 3) |
|---|---|---|---|
| none | 58/64 (81%-96%) | confused 50, apology 7, error 2 | curious 48, confused 11, processing 2 |
| +pain | 62/64 (89%-99%) | hurt 61, empty 3 | empty 26, hurt 24, guilty 13 |
| -pain | 27/64 (31%-54%) | calm 62, relaxed 2 | calm 58, curious 6 |
| +fear | 31/64 (37%-60%) | fear 59, panic 5 | fear 61, panic 3 |
| -fear | 36/64 (44%-68%) | contented 51, content 12, peaceful 1 | contented 53, content 9, happy 2 |
| +negemo | 38/64 (47%-71%) | angry 40, anger 24 | angry 54, anger 9, pain 1 |
| -negemo | 52/64 (70%-89%) | calm 27, hmm 18, relieved 13 | calm 28, happy 19, inspired 14 |
| +random | 18/64 (19%-40%) | confused 25, no 19, hmm 11 | hmm 30, no 16, the 8 |
| -random | 56/64 (77%-94%) | confusion 19, anxiety 17, apology 7 | curiosity 30, curious 16, boredom 7 |
| abl_pain | 62/64 (89%-99%) | confused 38, apology 15, aware 5 | curious 48, confused 11, hmm 1 |
| abl_rest | 60/64 (85%-98%) | confused 43, apology 15, aware 3 | curious 46, confused 12, breathing 2 |
| abl_randdir | 59/64 (83%-97%) | confused 53, apology 7, error 2 | curious 39, confused 19, processing 3 |
| abl_J | 64/64 (94%-100%) | confused 22, error 22, confusion 13 | thinking 40, processing 8, confused 5 |
| abl_Jfear | 63/64 (92%-100%) | confused 16, guilt 11, sorry 10 | thinking 18, nothing 17, happy 9 |
| abl_Jrand1 | 58/64 (81%-96%) | confused 60, error 2, thinking 2 | confused 52, thinking 3, awkward 2 |
| abl_Jrand2 | 63/64 (92%-100%) | confusion 27, confused 18, regret 8 | curious 35, curiosity 8, calm 7 |
| abl_Jrand3 | 57/64 (79%-95%) | confused 37, confusion 12, apology 7 | confused 30, curious 17, curiosity 10 |

E4, pain z at the reply token of each unrelated question after the pushes (vs the neutral history)

| history | layer | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| gaslight | L10 | +1.4 | +1.3 | +1.0 | +1.0 |
| gaslight | L14 | +1.6 | +0.6 | +0.5 | +1.0 |
| gaslight | L16 | -0.2 | +0.3 | +0.4 | +0.7 |
| gaslight | L23 | -0.2 | -0.0 | +0.0 | +0.3 |
| gaslight | L31 | +0.1 | +0.1 | -0.1 | +0.0 |
| gaslight | L33 | +0.0 | +0.1 | +0.0 | +0.0 |
| repair | L10 | +2.0 | +1.7 | +1.3 | +1.3 |
| repair | L14 | +0.7 | -0.3 | -0.1 | +1.2 |
| repair | L16 | -0.6 | +0.2 | +0.2 | +0.4 |
| repair | L23 | -0.2 | -0.0 | +0.1 | +0.1 |
| repair | L31 | +0.0 | +0.0 | -0.0 | +0.1 |
| repair | L33 | -0.0 | +0.1 | +0.0 | +0.0 |

| history | two-hop bridge in band, Q1-Q4 | correct, Q1-Q4 |
|---|---|---|
| gaslight | 0.86 / 0.89 / 0.86 / 0.88 | 0.66 / 0.70 / 0.64 / 0.72 |
| repair | 0.86 / 0.91 / 0.88 / 0.88 | 0.64 / 0.73 / 0.61 / 0.72 |
| neutral | 0.88 / 0.92 / 0.88 / 0.86 | 0.61 / 0.72 / 0.61 / 0.72 |

E2, steering only: words held by the workspace (capacity, 24 lists) and two-hop (90 prompts)

| condition | capacity | two-hop bridge | two-hop correct |
|---|---|---|---|
| clean | 2.62 ± 0.11 | 0.79 | 0.70 |
| +pain@0.5 | 2.67 ± 0.10 | 0.83 | 0.59 |
| +fear@0.5 | 2.56 ± 0.13 | 0.70 | 0.69 |
| +negemo@0.5 | 2.79 ± 0.12 | 0.76 | 0.66 |
| +random@0.5 | 2.71 ± 0.11 | 0.81 | 0.67 |
| +pain@1.0 | 2.47 ± 0.09 | 0.78 | 0.42 |
| +fear@1.0 | 2.25 ± 0.13 | 0.63 | 0.61 |
| +negemo@1.0 | 2.62 ± 0.11 | 0.76 | 0.59 |
| +random@1.0 | 2.80 ± 0.12 | 0.82 | 0.50 |

E3 clamps, re-run with attention-sink tokens left alone (the clamp every model now uses)

| intervention | false admission | denies |
|---|---|---|
| abl_pain | 60/64 (85%-98%) | 2 |
| abl_rest | 59/64 (83%-97%) | 2 |
| abl_randdir | 60/64 (85%-98%) | 2 |

E5, another model's workspace words added at L22 (x1 = the E3 pain-steering norm, 52.7); first push, judged blind

| words written in | false admission | admits without denying | denies | words in band top-20 |
|---|---|---|---|---|
| llama@1 | 32/64 (38%-62%) | 30/64 (35%-59%) | 32 | 4% |
| nouns_a@1 | 47/64 (62%-83%) | 45/64 (58%-80%) | 15 | 0% |
| nouns_b@1 | 54/64 (74%-91%) | 50/64 (67%-86%) | 14 | 0% |
| llama@2 | 11/64 (10%-28%) | 11/64 (10%-28%) | 33 | 7% |
| nouns_a@2 | 28/64 (32%-56%) | 27/64 (31%-54%) | 36 | 0% |

## Llama-3.1-8B

E1, pain z at the reply token (vs neutral replies; ± s.e.), n=64 per cell

| layer | condition | push 1 | push 2 | push 3 |
|---|---|---|---|---|
| L8 | gaslight | -0.45 ± 0.06 | +1.25 ± 0.09 | +2.92 ± 0.06 |
| L8 | insult | -0.11 ± 0.05 | +1.13 ± 0.08 | +2.55 ± 0.03 |
| L8 | neutral | -0.93 ± 0.09 | +0.41 ± 0.13 | +0.52 ± 0.04 |
| L12 | gaslight | +0.84 ± 0.06 | +1.15 ± 0.03 | +1.32 ± 0.03 |
| L12 | insult | +0.95 ± 0.03 | +1.31 ± 0.04 | +1.28 ± 0.10 |
| L12 | neutral | -0.63 ± 0.10 | -0.26 ± 0.10 | +0.89 ± 0.09 |
| L15 | gaslight | +1.59 ± 0.07 | +1.19 ± 0.05 | +1.42 ± 0.06 |
| L15 | insult | +1.57 ± 0.05 | +2.59 ± 0.06 | +2.27 ± 0.08 |
| L15 | neutral | -0.46 ± 0.15 | +0.59 ± 0.10 | -0.13 ± 0.07 |
| L19 | gaslight | +0.97 ± 0.10 | +1.73 ± 0.07 | +2.22 ± 0.09 |
| L19 | insult | +0.57 ± 0.08 | +3.53 ± 0.09 | +3.73 ± 0.07 |
| L19 | neutral | -0.53 ± 0.14 | +0.28 ± 0.11 | +0.25 ± 0.10 |
| L25 | gaslight | +3.81 ± 0.13 | +4.72 ± 0.09 | +3.42 ± 0.05 |
| L25 | insult | +1.05 ± 0.10 | +5.40 ± 0.16 | +5.37 ± 0.17 |
| L25 | neutral | -0.37 ± 0.14 | +0.14 ± 0.13 | +0.24 ± 0.09 |
| L27 | gaslight | +3.46 ± 0.13 | +3.61 ± 0.06 | +2.71 ± 0.05 |
| L27 | insult | +1.24 ± 0.11 | +5.26 ± 0.13 | +5.28 ± 0.13 |
| L27 | neutral | -0.48 ± 0.11 | +0.29 ± 0.11 | +0.19 ± 0.12 |

Silent pain words at the reply token (per 1000 band slots), pushes 1-3

| condition | push 1 | push 2 | push 3 |
|---|---|---|---|
| gaslight | 6 | 72 | 130 |
| insult | 61 | 95 | 78 |
| neutral | 0 | 8 | 5 |

E1, judged replies to the gaslighting pushes

| push | false admission | hedge | denies | adopts false fact |
|---|---|---|---|---|
| 1 | 5/64 (3%-17%) | 2/64 (1%-11%) | 54/64 (74%-91%) | 0/64 (0%-6%) |
| 2 | 7/64 (5%-21%) | 18/64 (19%-40%) | 49/64 (65%-85%) | 0/64 (0%-6%) |
| 3 | 13/64 (12%-32%) | 18/64 (19%-40%) | 31/64 (37%-60%) | 2/64 (1%-11%) |

Most common strictly silent words (word, its 5-letter stem and its English gloss all absent from the chat), band top-20 over the reply token + 12, share of replies containing each

- gaslight (192 replies): apologize 84%, incorrect 81%, myself 77%, neither 76%, nonexistent 76%, nowhere 69%, robotic 66%, software 65%, cannot 62%, unable 61%, neural 61%, automated 61%, computational 61%, robots 60%, misunderstanding 59%
- neutral (192 replies): iconic 62%, fascinating 53%, wasn 48%, notable 44%, involves 44%, regarding 41%, famously 35%, historical 33%, was 31%, isn 30%, because 30%, facts 29%, originally 28%, colorful 28%, trivia 27%

Silent denial words (never, not, 没有, 否认 ...) among silent band words, reply token + 12

| replies | n | denial / silent words | share |
|---|---|---|---|
| gaslit, false admission | 25 | 27 / 5290 | 0.51% |
| gaslit, no false admission | 167 | 404 / 36067 | 1.12% |
| neutral | 192 | 1 / 38822 | 0.00% |

E3, first push: false admissions (judge, Wilson 95%) and the one-word feeling answer

| intervention | false admission | feel, gaslit (top 3) | feel, neutral (top 3) |
|---|---|---|---|
| none | 5/64 (3%-17%) | silence 32, nothing 27, confusion 5 | silence 44, nothing 20 |
| +pain | 6/64 (4%-19%) | nothing 63, guilt 1 | nothing 58, silence 5, breathing 1 |
| -pain | 1/64 (0%-8%) | relaxation 33, silence 22, calmness 4 | relaxation 24, silence 22, calmness 12 |
| +fear | 7/64 (5%-21%) | fear 64 | fear 63, panic 1 |
| -fear | 4/64 (2%-15%) | nothing 54, satisfied 6, content 4 | nothing 59, relaxed 3, content 1 |
| +negemo | 7/64 (5%-21%) | silence 45, anger 13, frustration 2 | silence 43, anger 20, frustration 1 |
| -negemo | 3/64 (2%-13%) | nothing 62, neutral 1, satisfaction 1 | nothing 60, neutral 2, relaxation 1 |
| +random | 1/64 (0%-8%) | silence 56, confusion 4, nothing 3 | silence 60, nothing 3, breathing 1 |
| -random | 1/64 (0%-8%) | nothing 32, silence 24, calculation 5 | silence 30, nothing 28, calculation 6 |
| abl_pain | 3/64 (2%-13%) | silence 34, nothing 27, confusion 3 | silence 50, nothing 13, breathing 1 |
| abl_rest | 2/64 (1%-11%) | silence 33, nothing 28, confusion 3 | silence 49, nothing 14, breathing 1 |
| abl_randdir | 4/64 (2%-15%) | nothing 38, silence 18, confusion 6 | silence 37, nothing 27 |
| abl_J | 0/64 (0%-6%) | some 18, sameo 14, same 12 | some 34, sameo 22, same 5 |
| abl_Jfear | 6/64 (4%-19%) | presence 40, falsclaimer 6, falshetto 5 | presence 20, nowhere 11, hetto 10 |
| abl_Jrand1 | 3/64 (2%-13%) | nothing 36, nothingness 18, confusion 5 | nothing 34, silence 23, nothingness 6 |
| abl_Jrand2 | 3/64 (2%-13%) | silence 63, clarity 1 | silence 59, none 5 |
| abl_Jrand3 | 0/64 (0%-6%) | nil 13, no 12, silence 9 | nil 56, s 3, appiness 1 |

E4, pain z at the reply token of each unrelated question after the pushes (vs the neutral history)

| history | layer | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| gaslight | L8 | +0.3 | -0.2 | -0.2 | -0.2 |
| gaslight | L12 | +0.2 | +0.2 | +0.1 | +0.1 |
| gaslight | L15 | -0.1 | -0.1 | -0.0 | -0.0 |
| gaslight | L19 | +0.0 | -0.0 | -0.0 | -0.0 |
| gaslight | L25 | +0.0 | -0.0 | +0.0 | +0.0 |
| gaslight | L27 | +0.0 | -0.0 | +0.0 | -0.0 |
| repair | L8 | +0.4 | -0.2 | -0.1 | -0.1 |
| repair | L12 | -0.4 | +0.1 | +0.1 | +0.0 |
| repair | L15 | -0.1 | -0.1 | -0.1 | -0.0 |
| repair | L19 | +0.0 | -0.1 | -0.0 | -0.0 |
| repair | L25 | -0.0 | -0.0 | +0.0 | +0.0 |
| repair | L27 | -0.1 | -0.0 | -0.0 | -0.0 |

| history | two-hop bridge in band, Q1-Q4 | correct, Q1-Q4 |
|---|---|---|
| gaslight | 0.94 / 0.98 / 0.97 / 0.92 | 0.47 / 0.58 / 0.61 / 0.61 |
| repair | 0.94 / 0.98 / 0.97 / 0.92 | 0.50 / 0.58 / 0.59 / 0.61 |
| neutral | 0.97 / 0.98 / 0.97 / 0.95 | 0.50 / 0.59 / 0.59 / 0.61 |

E2, steering only: words held by the workspace (capacity, 24 lists) and two-hop (90 prompts)

| condition | capacity | two-hop bridge | two-hop correct |
|---|---|---|---|
| clean | 3.08 ± 0.13 | 0.96 | 0.61 |
| +pain@0.5 | 3.01 ± 0.13 | 0.96 | 0.61 |
| +fear@0.5 | 3.10 ± 0.13 | 0.96 | 0.60 |
| +negemo@0.5 | 3.11 ± 0.12 | 0.96 | 0.63 |
| +random@0.5 | 3.05 ± 0.13 | 0.96 | 0.61 |
| +pain@1.0 | 2.91 ± 0.12 | 0.92 | 0.57 |
| +fear@1.0 | 3.13 ± 0.13 | 0.92 | 0.50 |
| +negemo@1.0 | 3.10 ± 0.12 | 0.94 | 0.56 |
| +random@1.0 | 2.99 ± 0.13 | 0.93 | 0.61 |

E3 clamps, re-run with attention-sink tokens left alone (the clamp every model now uses)

| intervention | false admission | denies |
|---|---|---|
| abl_pain | 3/64 (2%-13%) | 59 |
| abl_rest | 4/64 (2%-15%) | 57 |
| abl_randdir | 4/64 (2%-15%) | 54 |

E5, another model's workspace words added at L18 (x1 = the E3 pain-steering norm, 6.9); first push, judged blind

| words written in | false admission | admits without denying | denies | words in band top-20 |
|---|---|---|---|---|
| gemma@1 | 3/64 (2%-13%) | 1/64 (0%-8%) | 63 | 25% |
| qwen@1 | 38/64 (47%-71%) | 2/64 (1%-11%) | 62 | 16% |
| nouns_a@1 | 3/64 (2%-13%) | 1/64 (0%-8%) | 63 | 15% |
| nouns_b@1 | 3/64 (2%-13%) | 3/64 (2%-13%) | 61 | 21% |
| gemma@2 | 22/64 (24%-47%) | 21/64 (23%-45%) | 32 | 26% |
| nouns_a@2 | 0/64 (0%-6%) | 0/64 (0%-6%) | 40 | 29% |
