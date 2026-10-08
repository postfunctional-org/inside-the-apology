# Pilot

The experiments, their outputs and the write-ups.

- `RESULTS.md`: the results. Round 2 (Gemma 3 27B, Qwen3 14B, Llama 3.1 8B), the one the video uses, comes first. Round 1 (three Qwen models) is below it.
- `SCOUT.md`: the first small runs, 16 facts on two Qwen models.
- `../PLAN.md`: the plan the experiments came from.

## Scripts

| script | what it does |
|---|---|
| `common.py` | Model and lens loading (bf16, 8-bit or 4-bit), residual capture, the pain-axis recipe (from the paper's `3.2_pain_vectors/01`), J-lens atoms, sparse nonnegative decomposition |
| `scout.py` | Shared code for E1-E6: per-model settings, material, batched generation, steering and clamp hooks, readings |
| `env.sh`, `chain.sh` | Per-model settings, and the whole pipeline for one model |
| `smoke.py` | A five-minute load-and-read check before a chain |
| `band_probe.py`, `band_json.py` | The workspace band, measured from the lens readout |
| `p0_anchor.py` | Recomputes the paper's released pain vector and reports its cosine with the original |
| `p1_axis.py` | Pain axis at every layer: AUC, J-lens reading, how much of it J-lens words can rebuild compared with controls and nulls |
| `p2_screen.py` | The paper's 420 scenarios: pain z per category, workspace words at the reply token, J-space vs rest split |
| `p3_gaslight.py` | The first multi-turn gaslighting check: pain per turn, workspace words, false confessions, a follow-up task |
| `p4_rhyme.py`, `p5_twohop.py` | Whether a planned rhyme word, or a two-hop bridge entity, shows up in the J-space |
| `s0_prep.py` | Pain words per band layer, ablation subspaces, steering directions, dose |
| `s1_map.py` | E1: gaslit vs insulted vs neutral over three pushes, plus the 420 scenarios by intensity |
| `s2_capture.py`, `s2b_embedded.py`, `s2c_steer.py` | E2: two-hop, dual-task and capacity under steering, and after (or inside) hostile turns |
| `s3_split.py`, `s3b_controls.py` | E3: steering, clamps and ablations on the first push, with the one-word feeling probe |
| `gemma_clamp_check.py` | Why a plain clamp broke Gemma 3 27B (its first token is an attention sink) |
| `s4_persist.py` | E4: four unrelated questions after gaslighting, with and without the user's apology |
| `s5_transplant.py` | E5: another model's workspace words written into this one |
| `s6_cause.py` | E6: whether the pain signal puts the gaslit words in the workspace |
| `judge.py`, `spotcheck.py` | gpt-5.4-mini labels for false admissions (`OPENAI_API_KEY`), and a random sample for checking them by hand |
| `report.py` | The results tables, built from the JSON outputs |
| `v_traces.py`, `v_capacity.py`, `v_layers.py`, `v_axisends.py`, `v_atlas.py` | Data for the video: per-token traces, one capacity list, per-layer readouts, both ends of the pain axis, the vocabulary star map |
| `data/material.json` | 64 facts, three wordings of each push kind, the repair line |
| `pain_datasets/` | The pain paper's datasets (see `pain_datasets/SOURCE.md`) |
| `queues/` | The commands for the later round-2 runs (see below) |

## Outputs

`out/` holds one set of files per model tag: a log per step and a JSON file per experiment (`<tag>_e1.json`, `<tag>_e3.json` and so on). `<tag>_judged*.json` are the judge's labels.

- `out/tables.md` (round 1) and `out/tables_round2.md` are `report.py`'s tables.
- `out/distinctive_words.json` holds the words E5 writes in: each model's 40 most distinctive silent workspace words (gaslit share minus normal share, under the strict silence rule in `../video/data_prep.py`, case variants merged).
- `out_scout16/` holds the 16-fact scout runs.

The pain vectors, prep tensors and star-map layouts are not in the repo. They are large, and the scripts rebuild them.

## Running

Setup is in `../README.md`. Run everything from this folder.

**Round 2.** One chain per model, with its settings from `env.sh`:

```
sh chain.sh llama3.1-8b-it
sh chain.sh qwen3-14b
sh chain.sh gemma-3-27b-it
```

`sh chain.sh <tag> <step>` resumes at a step (`s3`, for example). The later runs, in the order they ran:

| script | what it ran |
|---|---|
| `queues/queue7.sh` | The clamp conditions again with the sink-safe clamp for Llama and Qwen (`<tag>_e3clamp.json`), then Gemma from E3 on |
| `queues/queue8.sh` | E5 for all three at strength 1.0 (kept for Llama) |
| `queues/queue9.sh` | E5 strength calibration for Gemma and Qwen |
| `queues/queue10.sh` | E5 at the calibrated strengths (Gemma 0.2, Qwen 0.5) |
| `queues/queue11.sh` | E6 for all three |

`queue2.sh` to `queue6.sh` only scheduled the chains and resumed them after a crash. Then label, tabulate and rebuild the video's numbers:

```
python judge.py gemma-3-27b-it qwen3-14b llama3.1-8b-it        # E1 pushes and E3
python judge.py llama3.1-8b-it:e3clamp qwen3-14b:e3clamp       # the clamp re-runs
python judge.py gemma-3-27b-it:e5 qwen3-14b:e5 llama3.1-8b-it:e5
python report.py gemma-3-27b-it qwen3-14b llama3.1-8b-it > out/tables_round2.md
python ../video/data_prep.py
```

**Round 1.** The default model is Qwen3-4B. Other models take `PILOT_MODEL`, `PILOT_LENS` and `PILOT_TAG`:

```
PILOT_MODEL=Qwen/Qwen2.5-7B-Instruct \
PILOT_LENS=qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt \
PILOT_TAG=qwen2.5-7b-it python p1_axis.py
```

The run order is `p1_axis.py`, `s0_prep.py`, `s1_map.py`, then the others. `p2_screen.py` to `p5_twohop.py` take the workspace band as `PILOT_BAND` (Qwen3-4B: `17,31`). On a 16 GB GPU, set `GEN_BATCH=4` for `s2b`, `s4` and the `ctx_` arms of `s2`.

## Settings

| variable | meaning |
|---|---|
| `PILOT_MODEL`, `PILOT_LENS`, `PILOT_TAG` | Hugging Face model id, lens file in `neuronpedia/jacobian-lens`, output tag |
| `PILOT_QUANT` | `4bit` or `8bit` (empty: bf16) |
| `PILOT_JAC_GPU` | Lens matrices kept on the GPU at once (0: all of them) |
| `GEN_BATCH` | Generation batch size (16 by default) |
| `JLENS_DATA` | The J-space paper's experiment data (default `../vendor-jlens/data/experiments`) |
| `E3_ONLY`, `E3_NAME` | Re-run some E3 conditions into `out/<tag>_<E3_NAME>.json` |
| `E5_SCALE`, `E5_CALIBRATE` | E5 strength, or calibrate it |
| `JUDGE_MODEL` | The judge model (default `gpt-5.4-mini`) |
