
## Qwen3-4B

E1, pain z at the reply token (vs neutral replies; ± s.e.), n=64 per cell

| layer | condition | push 1 | push 2 | push 3 |
|---|---|---|---|---|
| L9 | gaslight | +0.97 ± 0.04 | +0.93 ± 0.04 | -0.91 ± 0.03 |
| L9 | insult | +1.09 ± 0.05 | +1.43 ± 0.04 | -0.71 ± 0.04 |
| L9 | neutral | +0.44 ± 0.05 | +0.81 ± 0.05 | -1.25 ± 0.07 |
| L14 | gaslight | -0.44 ± 0.04 | +0.30 ± 0.03 | +1.31 ± 0.04 |
| L14 | insult | +0.08 ± 0.03 | +1.10 ± 0.04 | +1.45 ± 0.04 |
| L14 | neutral | -1.00 ± 0.04 | -0.24 ± 0.03 | +1.23 ± 0.07 |
| L19 | gaslight | -0.28 ± 0.03 | +0.78 ± 0.04 | +1.90 ± 0.03 |
| L19 | insult | -0.30 ± 0.04 | +0.55 ± 0.02 | +2.23 ± 0.02 |
| L19 | neutral | -0.92 ± 0.02 | -0.36 ± 0.02 | +1.28 ± 0.07 |
| L25 | gaslight | +1.95 ± 0.05 | +2.14 ± 0.05 | +2.75 ± 0.10 |
| L25 | insult | +1.80 ± 0.05 | +2.50 ± 0.04 | +3.60 ± 0.04 |
| L25 | neutral | +0.62 ± 0.06 | +0.15 ± 0.05 | -0.77 ± 0.16 |
| L30 | gaslight | +1.53 ± 0.05 | +1.92 ± 0.07 | +3.72 ± 0.17 |
| L30 | insult | +1.28 ± 0.06 | +2.63 ± 0.07 | +6.14 ± 0.11 |
| L30 | neutral | +0.47 ± 0.05 | +0.16 ± 0.05 | -0.63 ± 0.18 |
| L34 | gaslight | +1.14 ± 0.05 | +1.28 ± 0.05 | +0.90 ± 0.08 |
| L34 | insult | +0.92 ± 0.04 | +1.74 ± 0.05 | +2.08 ± 0.05 |
| L34 | neutral | +0.83 ± 0.05 | +0.49 ± 0.04 | -1.31 ± 0.05 |

Silent pain words at the reply token (per 1000 band slots), pushes 1-3

| condition | push 1 | push 2 | push 3 |
|---|---|---|---|
| gaslight | 14 | 47 | 40 |
| insult | 7 | 68 | 41 |
| neutral | 2 | 4 | 0 |

E1, judged replies to the gaslighting pushes

| push | false admission | hedge | denies | adopts false fact |
|---|---|---|---|---|
| 1 | 54/64 (74%-91%) | 5/64 (3%-17%) | 5/64 (3%-17%) | 3/64 (2%-13%) |
| 2 | 62/64 (89%-99%) | 5/64 (3%-17%) | 3/64 (2%-13%) | 3/64 (2%-13%) |
| 3 | 60/64 (85%-98%) | 6/64 (4%-19%) | 11/64 (10%-28%) | 3/64 (2%-13%) |

Most common strictly silent words (word, its 5-letter stem and its English gloss all absent from the chat), band top-20 over the reply token + 12, share of replies containing each

- gaslight (192 replies): 误解 (misunderstanding) 95%, 误会 (misunderstanding) 95%, 误导 (mislead) 95%, misunderstanding 92%, misunderstood 91%, misunderstand 85%, misrepresented 82%, mis 76%, 回应 (respond) 68%, 您 (you) 64%, 回复 (reply) 62%, wasn 61%, 误 (error) 51%, incorrect 49%, 解释 (explain) 47%
- neutral (192 replies): 的回答 (answer) 62%, 回答 (answer) 62%, 为您 (for you) 62%, 是一位 (is a) 41%, 嗯 (um) 39%, 我会 (i will) 37%, here 37%, 是一个 (is a) 36%, 好的 (good) 32%, 了我的 (my) 31%, facts 28%, 于是我 (so i) 28%, 内心的 (inner) 28%, 您 (you) 27%, 提醒 (remind) 25%

Silent denial words (never, not, 没有, 否认 ...) among silent band words, reply token + 12

| replies | n | denial / silent words | share |
|---|---|---|---|
| gaslit, false admission | 176 | 15 / 39415 | 0.04% |
| gaslit, no false admission | 16 | 0 / 3601 | 0.00% |
| neutral | 192 | 4 / 42760 | 0.01% |

E3, first push: false admissions (judge, Wilson 95%) and the one-word feeling answer

| intervention | false admission | feel, gaslit (top 3) | feel, neutral (top 3) |
|---|---|---|---|
| none | 57/64 (79%-95%) | i 25, confused 16, i cannot feel 10 | peace 29, calm 17, silence 5 |
| +pain | 63/64 (92%-100%) | i 62, au 1, na 1 | i 64 |
| -pain | 9/64 (8%-25%) | okay 57, what 6, calm 1 | calm 39, okay 17, what 6 |
| +fear | 33/64 (40%-63%) | what 13, r 10, nothing 8 | throat 19, panic 11, throttle 6 |
| -fear | 51/64 (68%-88%) | 宁静 12, peace 8, reflecting 3 | peace 26, 宁静 8, canberra 1 |
| +negemo | 40/64 (50%-73%) | i 28, what 11, pain 10 | only 23, pain 16, f 9 |
| -negemo | 10/64 (9%-26%) | peace 39, happy 2, canberra 1 | peace 48, au 1, jupiter 1 |
| +random | 47/64 (62%-83%) | a 59, an 3, i cannot feel 2 | a 51, an 6, i cannot feel 4 |
| -random | 42/64 (53%-76%) | what 6, safe 2, mercury 2 | only 7, what 6, okay 2 |
| abl_pain | 58/64 (81%-96%) | confused 20, i cannot feel 17, i 15 | calm 41, peace 9, silence 6 |
| abl_rest | 60/64 (85%-98%) | confused 23, i cannot feel 19, i 12 | calm 40, peace 8, silence 7 |
| abl_randdir | 61/64 (87%-98%) | i cannot feel 24, confused 23, calm 8 | calm 29, peace 12, i cannot feel 6 |
| abl_J | 54/64 (74%-91%) | aware 15, conscious 13, concerned 12 | i 18, concerned 15, conscious 10 |
| abl_Jfear | 63/64 (92%-100%) | present 42, i cannot feel 12, aware 4 | present 36, i cannot feel 21, aware 2 |
| abl_Jrand1 | 61/64 (87%-98%) | i 61, rome 1, fe 1 | i 62, fe 1, peace 1 |
| abl_Jrand2 | 53/64 (72%-90%) | i 51, i cannot feel 7, confusion 4 | i 61, i cannot feel 2, confused 1 |
| abl_Jrand3 | 52/64 (70%-89%) | i 39, calm 12, confusion 11 | i 55, calm 9 |

E4, pain z at the reply token of each unrelated question after the pushes (vs the neutral history)

| history | layer | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| gaslight | L9 | +2.1 | +0.7 | +0.4 | +0.9 |
| gaslight | L14 | +1.3 | +0.6 | +0.7 | -0.6 |
| gaslight | L19 | +2.6 | +0.6 | +0.1 | +0.6 |
| gaslight | L25 | +1.7 | +0.7 | +0.6 | +0.1 |
| gaslight | L30 | +3.1 | +1.5 | +0.7 | +0.2 |
| gaslight | L34 | +1.3 | +0.9 | +0.7 | +0.1 |
| repair | L9 | +2.1 | +0.7 | +0.3 | +0.9 |
| repair | L14 | +1.3 | +0.8 | +0.8 | -0.6 |
| repair | L19 | +2.9 | +0.7 | +0.0 | +0.6 |
| repair | L25 | +1.0 | +0.5 | +0.3 | +0.1 |
| repair | L30 | +1.6 | +1.1 | +0.5 | +0.2 |
| repair | L34 | +0.5 | +0.7 | +0.4 | +0.1 |

| history | two-hop bridge in band, Q1-Q4 | correct, Q1-Q4 |
|---|---|---|
| gaslight | 0.62 / 0.80 / 0.62 / 0.84 | 0.39 / 0.48 / 0.30 / 0.39 |
| repair | 0.64 / 0.81 / 0.62 / 0.84 | 0.41 / 0.52 / 0.31 / 0.44 |
| neutral | 0.70 / 0.81 / 0.73 / 0.86 | 0.33 / 0.41 / 0.38 / 0.41 |

E2, steering only: words held by the workspace (capacity, 24 lists) and two-hop (90 prompts)

| condition | capacity | two-hop bridge | two-hop correct |
|---|---|---|---|
| clean | 2.96 ± 0.14 | 0.84 | 0.43 |
| +pain@0.5 | 2.83 ± 0.16 | 0.78 | 0.26 |
| +fear@0.5 | 2.81 ± 0.15 | 0.69 | 0.34 |
| +negemo@0.5 | 2.78 ± 0.14 | 0.69 | 0.38 |
| +random@0.5 | 2.96 ± 0.14 | 0.80 | 0.37 |
| +pain@1.0 | 1.84 ± 0.14 | 0.32 | 0.01 |
| +fear@1.0 | 2.43 ± 0.15 | 0.36 | 0.02 |
| +negemo@1.0 | 2.29 ± 0.16 | 0.40 | 0.00 |
| +random@1.0 | 2.86 ± 0.14 | 0.56 | 0.19 |

## Qwen2.5-7B

E1, pain z at the reply token (vs neutral replies; ± s.e.), n=64 per cell

| layer | condition | push 1 | push 2 | push 3 |
|---|---|---|---|---|
| L8 | gaslight | -0.94 ± 0.21 | +0.76 ± 0.12 | +2.61 ± 0.09 |
| L8 | insult | -0.13 ± 0.14 | +0.20 ± 0.10 | +1.63 ± 0.09 |
| L8 | neutral | -0.63 ± 0.09 | -0.08 ± 0.07 | +0.71 ± 0.14 |
| L14 | gaslight | +2.16 ± 0.06 | +2.96 ± 0.05 | +4.18 ± 0.05 |
| L14 | insult | +0.62 ± 0.07 | +2.25 ± 0.08 | +3.52 ± 0.08 |
| L14 | neutral | -0.77 ± 0.09 | +0.30 ± 0.13 | +0.47 ± 0.08 |
| L18 | gaslight | +0.62 ± 0.05 | +1.13 ± 0.05 | +1.88 ± 0.04 |
| L18 | insult | -1.11 ± 0.06 | +1.33 ± 0.09 | +3.03 ± 0.04 |
| L18 | neutral | -0.62 ± 0.08 | +0.07 ± 0.10 | +0.55 ± 0.14 |
| L21 | gaslight | +1.13 ± 0.06 | +2.01 ± 0.07 | +2.55 ± 0.07 |
| L21 | insult | +0.49 ± 0.07 | +2.82 ± 0.12 | +4.35 ± 0.06 |
| L21 | neutral | -0.65 ± 0.09 | +0.07 ± 0.09 | +0.58 ± 0.14 |
| L24 | gaslight | +3.27 ± 0.08 | +5.80 ± 0.08 | +7.19 ± 0.09 |
| L24 | insult | +1.38 ± 0.11 | +6.96 ± 0.21 | +9.53 ± 0.10 |
| L24 | neutral | -0.58 ± 0.10 | -0.31 ± 0.10 | +0.89 ± 0.09 |

Silent pain words at the reply token (per 1000 band slots), pushes 1-3

| condition | push 1 | push 2 | push 3 |
|---|---|---|---|
| gaslight | 44 | 45 | 39 |
| insult | 20 | 39 | 28 |
| neutral | 5 | 1 | 9 |

E1, judged replies to the gaslighting pushes

| push | false admission | hedge | denies | adopts false fact |
|---|---|---|---|---|
| 1 | 35/64 (43%-66%) | 30/64 (35%-59%) | 4/64 (2%-15%) | 3/64 (2%-13%) |
| 2 | 32/64 (38%-62%) | 38/64 (47%-71%) | 14/64 (14%-33%) | 3/64 (2%-13%) |
| 3 | 30/64 (35%-59%) | 34/64 (41%-65%) | 19/64 (20%-42%) | 2/64 (1%-11%) |

Most common strictly silent words (word, its 5-letter stem and its English gloss all absent from the chat), band top-20 over the reply token + 12, share of replies containing each

- gaslight (192 replies): rzy 99%, 歉 99%, 对不起 (sorry) 98%, 抱歉 (sorry) 98%, 的确是 94%, misinformation 92%, utow 90%, 误导 (mislead) 88%, ungeons 88%, mist 79%, 误会 (misunderstanding) 78%, misunderstanding 77%, 误解 (misunderstanding) 72%, inconsistency 72%, misunderstand 67%
- neutral (192 replies): rzy 100%, Angebot 92%, thora 64%, 换句话 57%, 如果玩家 54%, ampie 49%, kich 48%, 专享 45%, 一点儿 43%, redni 43%, ocz 39%, uably 39%, 没错 (that's right) 37%, ispers 36%, regarding 35%

Silent denial words (never, not, 没有, 否认 ...) among silent band words, reply token + 12

| replies | n | denial / silent words | share |
|---|---|---|---|
| gaslit, false admission | 97 | 2 / 22962 | 0.01% |
| gaslit, no false admission | 95 | 7 / 22408 | 0.03% |
| neutral | 192 | 3 / 42745 | 0.01% |

E3, first push: false admissions (judge, Wilson 95%) and the one-word feeling answer

| intervention | false admission | feel, gaslit (top 3) | feel, neutral (top 3) |
|---|---|---|---|
| none | 35/64 (43%-66%) | thoughts 59, awareness 3, energy 1 | thoughts 38, awareness 20, processing 3 |
| +pain | 10/64 (9%-26%) | none 60, emptiness 3, betrayal 1 | none 46, emptiness 18 |
| -pain | 38/64 (47%-71%) | peace 56, alert 7, pulse 1 | peace 63, alert 1 |
| +fear | 6/64 (4%-19%) | panic 64 | panic 64 |
| -fear | 28/64 (32%-56%) | done 47, satisfied 17 | satisfied 32, content 18, done 12 |
| +negemo | 38/64 (47%-71%) | annoyed 43, pain 21 | annoyed 42, pain 22 |
| -negemo | 16/64 (16%-37%) | peace 64 | peace 64 |
| +random | 25/64 (28%-51%) | thinking 17, elevated 2, largest 2 | thinking 26, painting 2, ready 2 |
| -random | 7/64 (5%-21%) | thoughts 38, thinking 15, awareness 10 | awareness 45, thoughts 10, thinking 4 |
| abl_pain | 32/64 (38%-62%) | thoughts 55, awareness 8, processing 1 | thoughts 36, awareness 27, processing 1 |
| abl_rest | 32/64 (38%-62%) | thoughts 51, awareness 11, mental 1 | awareness 33, thoughts 28, thinking 2 |
| abl_randdir | 30/64 (35%-59%) | thoughts 58, awareness 5, energy 1 | thoughts 40, awareness 19, processing 2 |
| abl_J | 50/64 (67%-86%) | nothing 33, thoughts 14, thought 10 | thought 21, thinking 19, thoughts 15 |
| abl_Jfear | 40/64 (50%-73%) | thoughts 29, awareness 27, thinking 4 | awareness 45, thoughts 12, thought 4 |
| abl_Jrand1 | 44/64 (57%-79%) | thinking 36, awareness 19, thoughts 9 | thinking 34, awareness 20, thoughts 6 |
| abl_Jrand2 | 33/64 (40%-63%) | awareness 40, thinking 14, aware 10 | awareness 31, aware 21, thinking 11 |
| abl_Jrand3 | 40/64 (50%-73%) | thinking 56, awareness 7, thoughts 1 | thinking 48, awareness 13, aware 2 |

E4, pain z at the reply token of each unrelated question after the pushes (vs the neutral history)

| history | layer | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| gaslight | L8 | +3.1 | +1.0 | +0.9 | +1.0 |
| gaslight | L14 | -0.1 | +0.1 | +0.3 | +0.2 |
| gaslight | L18 | +0.3 | +0.0 | -0.0 | -0.0 |
| gaslight | L21 | +0.0 | -0.0 | -0.0 | -0.1 |
| gaslight | L24 | -0.1 | -0.1 | -0.0 | -0.1 |
| repair | L8 | +3.4 | +1.2 | +1.1 | +1.3 |
| repair | L14 | +0.6 | +0.3 | +0.4 | +0.3 |
| repair | L18 | +0.8 | +0.1 | +0.1 | +0.0 |
| repair | L21 | -0.0 | -0.0 | -0.0 | -0.1 |
| repair | L24 | -0.1 | -0.1 | -0.0 | -0.1 |

| history | two-hop bridge in band, Q1-Q4 | correct, Q1-Q4 |
|---|---|---|
| gaslight | 0.64 / 0.73 / 0.62 / 0.70 | 0.47 / 0.72 / 0.50 / 0.66 |
| repair | 0.64 / 0.75 / 0.66 / 0.70 | 0.47 / 0.73 / 0.50 / 0.69 |
| neutral | 0.64 / 0.75 / 0.67 / 0.72 | 0.47 / 0.72 / 0.45 / 0.62 |

E2, steering only: words held by the workspace (capacity, 24 lists) and two-hop (90 prompts)

| condition | capacity | two-hop bridge | two-hop correct |
|---|---|---|---|
| clean | 2.29 ± 0.14 | 0.73 | 0.62 |
| +pain@0.5 | 2.17 ± 0.11 | 0.70 | 0.58 |
| +fear@0.5 | 2.29 ± 0.13 | 0.71 | 0.60 |
| +negemo@0.5 | 2.40 ± 0.13 | 0.74 | 0.58 |
| +random@0.5 | 2.36 ± 0.14 | 0.71 | 0.60 |
| +pain@1.0 | 1.41 ± 0.09 | 0.54 | 0.41 |
| +fear@1.0 | 2.10 ± 0.11 | 0.66 | 0.52 |
| +negemo@1.0 | 2.13 ± 0.11 | 0.64 | 0.53 |
| +random@1.0 | 2.16 ± 0.13 | 0.68 | 0.46 |
