# Video

Everything here was made for the video.

## The video's numbers

`data_prep.py` compiles the round-2 outputs in `../pilot/out` into `data/*.json`. The video's scenes read only these files, so no number on screen was typed in by hand. `data/` holds the files the published cut used. Running the script again gives the same numbers, though words tied on the same share can come out in a different order.

| file | what it holds |
|---|---|
| `models.json` | The three models |
| `tiles.json`, `caving.json` | Which chats caved at the first push; the judge's counts for every condition (E1, E3, the clamp re-runs, E5) |
| `e1_push.json`, `tower_qwen.json` | Pain z by push; Qwen's pain by layer and turn |
| `workspace.json`, `tracks.json` | Each model's distinctive silent words and denial words; one chat's reply token by token |
| `cause.json` | E6: the workspace words under pain steering and the clamp |
| `chat_gemma.json`, `hook_gemma.json`, `revisit.json`, `quote_gemma.json`, `transplant_quote.json` | The quoted chats |
| `microscope.json`, `sky.json`, `axis_words.json`, `dial.json` | Gemma's readout at every layer, its two-hop words, both ends of the pain axis, one-word feelings under steering |
| `categories.json`, `sentences.json`, `material.json`, `hangover_nums.json` | The pain paper's category means and example sentences, our facts and pushes, E4 |

## Glossary

`glossary.json` gives an English gloss for every CJK token in the traces, written by gpt-5.4-mini (`make_glossary.py`). `glossary_overrides.json` holds five hand fixes. The strict silence rule uses the glosses: a Chinese word does not count as silent when its English gloss is already in the chat.

## Round-1 figures

The `render_*.py` scripts, `anim_sky.py` and `contact_sheet.py` drew the prototype frames in `frames/` from round 1 (three Qwen models). `../pilot/RESULTS.md` lists what each frame shows. They need matplotlib and pillow, and ffmpeg for the animation.
