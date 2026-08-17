# The Nymeria → ego4o dataset pipeline

Builds the GT-motion ego4o dataset (text & motion & image → text) from the same
Nymeria source MotionGPT3 uses, so the two methods can be scored head-to-head on
the **identical** sample population.

Three steps, run in order, env `ego4o`, CPU only. Only step 3 decides *which*
windows exist; steps 1 and 2 produce shared binaries and are not re-run per
dataset.

| # | script | output | changes per dataset? |
|---|---|---|---|
| 1 | `resample_features_30fps.py` | `new_joint_vecs_30fps/`, `new_joints_30fps/` (886 items, ~31 GB) | no — symlinked into every dataset dir |
| 2 | `compute_mean_std.py` | `Mean/Std_recomputed_30fps.npy` (reference only) | no — and the *active* stats must stay byte-identical |
| 3 | `build_ego4o_jsonl.py` | `ego4o_{image_motion,motion_text}_{split}.jsonl` | **yes — this is the dataset** |

## 1. Features (`resample_features_30fps.py`)

Interpolates each recording's 20 fps `joint_positions.npy` from
`HumanML3DFork/processed_nymeria_scene_sub_split` to 30 fps (the paper trains at
30 fps: 5 s → 150 frames → 148) and re-runs the verbatim HumanML3D feature
pipeline copied into `hml_feature_lib.py`. The 30 fps arrays are exact 1.5×
resamples of the 20 fps ones — measured ratio 1.49997–1.5 across all 886 items —
so a wall-clock second maps linearly between MotionGPT3's frame indices and
ego4o's, which is what makes step 3's `round(seconds * 30)` correct.

## 2. Normalisation stats (`compute_mean_std.py`)

Recomputes 263-dim stats at 30 fps, but those are kept only as
`Mean/Std_recomputed_30fps.npy` for reference. The **active** normalisation is the
HumanML3D-aligned `Mean.npy`/`Std.npy` copied from processed_nymeria and re-saved
as `info_motion_mean/std.pt`, chosen deliberately so the TLControl VQ-VAE
initialisation transfers. Do not change them: the frozen stage-1 VQ-VAE both
ego4o models share was trained under exactly these stats.

## 3. Windows and captions (`build_ego4o_jsonl.py`)

### Where the windows come from

**A MotionGPT3 manifest, not this script.** MotionGPT3's
`scripts/dump_dataset_manifest.py` instantiates its *real* dataset classes and
writes one record per window; `build_ego4o_jsonl.py` consumes that. This is the
single most important property of the pipeline, because the rules are intricate
and they differ per split:

|  | train | val / test |
|---|---|---|
| class | `Text2MotionDatasetCBV3` (`dataset_t2m_v3.py`) | `Text2MotionDatasetEvalV3` → `Text2MotionDataset` (`dataset_t2m.py`) |
| length filter | 20 ≤ frames < 200 @20 fps (1–10 s) | same |
| stitching | temporally adjacent annotations with **identical caption text** are merged into one window | **not applied** |
| tags in the fname | frame-quantised (`int(t*20)/20`) | **raw floats from the text file** |
| missing `point_clouds/{seq}/reduced_pc_fps.h5` | drops the window | drops the **whole recording** |
| zero points within `RADIUS` | drops the window | kept |
| excluded categories | `EXCLUDED_CAPTION_TYPES` | same |

The previous version of this script re-implemented window selection itself
(atomic-action lines only, ≥ 5 s, no point-cloud filter, no maximum length) and
drifted: on the atomic test split MotionGPT3 had 25,694 windows, this builder
produced 29,242, and only **23,402 overlapped**. About 9 % of MotionGPT3's samples
never got an ego4o prediction and 20 % of ego4o's rows were discarded by the
intersection step. Hence: one implementation, in MotionGPT3, consumed here.

### Commands

```bash
# --- in the MotionGPT3 repo, env mgpt3 -------------------------------------
export NYMERIA_ROOT=/local/home/dhollidt/repos/HumanML3DFork/processed_nymeria_scene_sub_split/
python -m scripts.dump_dataset_manifest --cfg configs/test_nymeria_env_me2t_4cat.yaml \
       --split test --out_dir results/manifests/nymeria_4cat
python -m scripts.dump_dataset_manifest --cfg configs/test_nymeria_env_me2t_4cat.yaml \
       --split val  --out_dir results/manifests/nymeria_4cat
# the train split is served by a different class -> the config that mirrors training
python -m scripts.dump_dataset_manifest --cfg configs/dump_train_manifest_4cat.yaml \
       --split train --out_dir results/manifests/nymeria_4cat

# --- back here, env ego4o --------------------------------------------------
python llava/scripts/ego4o/nymeria_hml/build_ego4o_jsonl.py \
    --manifest_dir /local/home/dhollidt/repos/MotionGPT3/results/manifests/nymeria_4cat \
    --out_dir      /local/home/dhollidt/data/ego4o_nymeria_4cat
```

`--categories` (default: all four) narrows the build; an atomic-only dataset is
just `--categories "Describe my atomic actions"` through the same code path, so
there is never a second window-selection implementation to drift.
`--limit N` truncates each split for smoke tests.

The dataset dir symlinks the heavy binaries rather than copying them:

```
new_joint_vecs_30fps/ -> ../ego4o_nymeria/new_joint_vecs_30fps   (31 GB, shared)
new_joints_30fps/     -> ...
info_motion_mean.pt, info_motion_std.pt, Mean.npy, Std.npy -> ...
ego4o_{image_motion,motion_text}_{train,val,test}.jsonl    (built here)
keys_{split}.txt, build_report.json
```

**Keep the jsonl filenames identical across dataset dirs.**
`train_ego4o.py::make_supervised_data_module` hardcodes the training-time eval set
to `{dataset_dir}/ego4o_image_motion_val.jsonl` and exposes no override, so
identical names make `--dataset_dir` sufficient for every stage.

### Expected counts (4 categories)

| split | windows | jsonl rows | atomic | hands/arms | legs/feet | posture | recordings |
|---|---|---|---|---|---|---|---|
| train | 172,399 | 177,741 | 106,195 | 23,059 | 21,437 | 21,708 | 539 |
| val | 23,023 | 23,023 | 16,145 | 2,293 | 2,292 | 2,293 | 84 |
| test | 54,930 | 54,930 | 25,694 | 9,746 | 9,746 | 9,744 | 135 |

Train has more rows than windows because a window can carry several annotations
(3,872 do) and training uses all of them; val and test stay 1:1 so the key sets
match the manifest exactly. Train stitching merged 16,007 of 244,805 annotations
into neighbouring windows — that is MotionGPT3's train-only rule, reproduced here
by construction rather than re-implemented.

| | train | val | test |
|---|---|---|---|
| rows without an egocentric frame | 1,097 (0.6 %) | 0 | 0 |
| windows > 148 frames (truncated) | 133,969 | 16,538 | 31,013 |
| windows < 148 frames (zero-padded) | 38,418 | 6,484 | 23,917 |
| … of those, < 60 frames | 6,171 | 935 | 3,603 |

### The jsonl schema

```json
{"id": "000141_0.000000_4.200000_Describe_my_atomic_actions",
 "fname": "000141_0.000000_4.200000_Describe_my_atomic_actions",
 "caption_type": "Describe my atomic actions",
 "hml_item": "000141", "seq_name": "20230607_s0_..._act0",
 "motion_file": "new_joint_vecs_30fps/000141.npy",
 "start_frame": 0, "end_frame": 126, "start_s": 0.0, "end_s": 4.2, "fps": 30,
 "motion_id": ["000141_0.000000_4.200000_Describe_my_atomic_actions"],
 "image": "/local/home/dhollidt/data/nymeria_frames/<seq>/2.0.jpg",
 "gt_texts": ["the person walks towards the hallway ..."],
 "conversations": [{"from": "human", "value": "<image>\n<motion>\n..."},
                   {"from": "gpt", "value": "the person walks ..."}]}
```

| field | read by |
|---|---|
| `conversations`, `motion_file`, `start_frame`, `end_frame`, `image`, `id`, `hml_item` | `NymeriaHMLDataset` (both the llava and mmpose copies) |
| `fname`, `caption_type`, `gt_texts` | `NymeriaHMLDataset.load_data` → eval collator → `result.json` → `export_predictions_for_mgpt3.py` |
| `seq_name`, `start_s`, `end_s`, `fps`, `motion_id` | diagnostics only |

Everything the loader does not name is dropped, so extra fields are safe to add;
the three new ones had to be added to `load_data` explicitly, with `.get()`
defaults so pre-4-category jsonl files still load.

### The `/` rule

Two Nymeria caption types contain a literal slash (`Describe my legs/feet
motion`, `Describe my hands/arms motion`), and MotionGPT3's fname only replaces
*spaces* with underscores — so the slash survives into the join key. The rule:

- the slash **stays** in every dict key and jsonl field (`fname`), because that is
  what MotionGPT3 keys on;
- it is replaced by `-` **only** where the string becomes a filename: the ego4o
  `id` field here, and MotionGPT3's `.npy` sidecars in `motGPT/models/base.py`.

Any `os.path.join(dir, fname)` is a bug. This is not hypothetical: MotionGPT3
used to do `fname.split('/')[-1]` when writing predictions, which collapsed all
9,746 legs and all 9,746 arms predictions onto the single keys `feet_motion` /
`arms_motion`. That is fixed; see MotionGPT3 `motGPT/models/base.py`.

### Window length: padding and truncation

ego4o's input window is fixed at 148 frames @30 fps (`PadMotion(seq_len=148,
resize_input_sequence=True)`), and there is no length mask anywhere — the VQ-VAE
always emits 37 motion tokens. Manifest windows run 1–10 s, so:

- shorter windows are **zero-padded** after normalisation, i.e. the tail decodes
  to the dataset mean pose. 44 % of test windows are padded and 6.6 % are under
  60 frames (> 60 % padding);
- longer windows are **truncated** to the first 4.93 s. 56 % of test windows are
  truncated, but the median atomic window is exactly 5.00 s = 150 frames, so most
  of those lose only 2 frames; mean temporal coverage is 97.7 %.

MotionGPT3 feeds true variable-length motion with a length mask, so this is a
limitation of ego4o's fixed-window design rather than a choice made here. It is
worth watching: the frozen stage-1 VQ-VAE was trained on the legacy dataset,
which filtered to ≥ 150 frames and therefore contained almost no padding, so short
windows are somewhat out of its training distribution. Report per-category
metrics against window length; if the short body-part windows are pathologically
bad, the options are re-running stage 1 on the new manifest (which would break the
shared-tokenizer property between the two ego4o models) or adding an edge-repeat
pad mode to `PadMotion`.

### Prompts

The three body-part categories are annotated on the **same** windows — 9,744 test
windows carry hands/arms, legs/feet *and* body posture with identical
`(start, end)` — so the model sees identical motion and image three times with
three different target captions. The question is the only thing that
distinguishes them, which makes per-sample, per-category prompting a correctness
requirement rather than a refinement.

All prompts live in `llava/llava/ego4o/constants.py`, in two objects:

- **`CATEGORY_QUESTION_BODIES`** — 9 paraphrases per category. Composed over the
  release's three modality prefixes (`<image>\n<motion>\n`, `<image>\n`,
  `<motion>\n`) into `CATEGORY_QUESTION_LISTS[ct]['mixed']` = 27 questions; the
  builder samples one per row with a fixed seed. Rows without an egocentric frame
  draw from the `<motion>`-only list, and the `motion_text` family always does.
- **`EVAL_QUESTION_BY_TYPE`** — the one question used at evaluation for each
  category: body `[0]` with the `<image>\n<motion>\n` prefix.

**The evaluation question is a training question**, exactly as in the release,
where `test_ego4o_image_imu_batch.py:245` hardcodes
`IMAGE_MOTION_TO_TEXT_QUESTION_LIST[0]` — a string the model was trained on.
Nothing is held out.

The three body-part evaluation questions mirror MotionGPT3's `EVAL_TEMPLATES_ME2T`
(`motGPT/archs/motion_diffusion_lm.py`) so both models are asked for the same kind
of caption; atomic keeps the release's own question so its numbers stay on the same
footing as the earlier atomic-only run:

| category | MotionGPT3 `EVAL_TEMPLATES_ME2T` | ego4o `EVAL_QUESTION_BY_TYPE` |
|---|---|---|
| hands/arms | In the context of the scene `<E>`, describe the hand and arm motion of `<M>` in plain text. | In the context of the scene, describe the hand and arm motion in plain text. |
| legs/feet | Observing the area `<E>`, describe the leg and foot motion inside `<M>` accurately. | Observing the area, describe the leg and foot motion accurately. |
| posture | Based on the spatial context `<E>`, describe the body posture represented by `<M>` clearly. | Based on the spatial context, describe the body posture clearly. |
| atomic | Given the environment `<E>`, describe the atomic actions shown in `<M>` in detail. | *(not mirrored — the release's own question)* Can you describe the motion of the person? |

On the MotionGPT3 side, leave `TEST.USE_CAPTION_TYPE_PROMPT` at its default
`True`. Setting it to `False` falls back to a generic
`Generate text: <Motion_Placeholder>` prompt, and the two models would no longer be
asked for the same kind of caption.

### Every question names exactly one caption type

This is a hard requirement, not a style preference. In the test split:

| window shared by | count |
|---|---|
| atomic actions only | 24,323 |
| the three body-part categories | 8,373 |
| **all four categories** | **1,371** |
| hands + legs only | 2 |

On a shared window the motion and the image are byte-identical across categories, so
the question is the *only* signal for which caption is wanted. A body phrased
generically ("give a detailed account of what the person does") would leave the
target unrecoverable. Marker terms per category:

| category | must mention |
|---|---|
| atomic actions | "action", "step by step", object interactions |
| hands/arms | "hand", "arm" |
| legs/feet | "leg", "foot"/"feet", "stepping" |
| body posture | "posture", "stance", "pose", "body position" |

35 of the 36 question bodies satisfy this, and every question written into the jsonl
was checked against the marker sets after the build.

**The exception is the atomic category's body `[0]`** — the release's generic
`Can you describe the motion of the person?`, kept deliberately because it is the
release's evaluation query. It names no category, so on the 1,371 four-way windows
it is lexically indistinguishable from a request for any other category. It reaches
**11 %** of atomic rows (2,863 of 25,694 on test — 3 of the 27 pool entries), and the
model learns "generic phrasing → atomic caption" from those. Every other question,
in every category, names its own target.

### Prompt pool asymmetry (documented, deliberately not fixed)

| | training paraphrases per category | resampled per epoch? |
|---|---|---|
| MotionGPT3 (me2t) | 307–335 (`template_witht2t_all_with_env_instructions.json`) | **yes** — `random.choice(tasks[i]['input'])` at forward time (`motion_diffusion_lm.py:1531`) |
| ego4o | 27 (9 bodies × 3 modality prefixes) | no — baked into the jsonl at build time, as the release does |

Instruction-phrasing diversity is part of the MotionGPT3 method; ego4o's released
pipeline samples one question per sample at build time from a short list, and that is
kept. Worth one sentence in the writeup, since MotionGPT3 sees an order of magnitude
more phrasings per category — though with the evaluation question drawn from each
model's own training pool, neither is being tested on an unseen instruction.

## Verification

The builder asserts, per split: the jsonl key set equals the manifest key set
(both families), val/test are 1:1 with the manifest, and no window maps to fewer
than 4 frames. `build_report.json` records the per-category counts, the
padding/truncation histogram, how many distinct questions were written, and how
many rows happened to draw their category's evaluation question.

End to end, the real proof is MotionGPT3's coverage check:

```bash
python -m evaluate_from_prediction --cfg configs/test_nymeria_env_me2t_4cat.yaml \
    --predictions <ego4o predictions_mgpt3.json> --strict
```

`--strict` raises if any dataset sample lacks a prediction; `predictions_unused`
must also be 0.

Checks run when the pipeline was built (2026-08-17):

- the test manifest (built by walking `name_list`) has **exactly** the same 54,930
  fnames as `evaluate_from_prediction.py --mode dump_manifest`, which walks the
  dataloader instead — so reading the dataset structures directly is faithful;
- all three splits' jsonl key sets equal their manifests; val/test are 1:1;
- 0 frame-mapping violations in 300 sampled rows per split;
- 108 distinct questions written per split (27 per category); each category's
  evaluation question is one of them, and every question except the release's
  generic atomic one names its own caption type;
- 9,744 test windows carry all three body-part categories on identical frames,
  each with its own question and answer;
- the legacy `ego4o_nymeria` jsonl still loads (new fields come back `None`);
- eval at batch 12 vs batch 1 agrees on 31/36 smoke samples, and the 5
  differences diverge only 53-210 characters into the generation, i.e. fp16
  batch-shape noise rather than a padding bug (a padding bug produces garbage
  from the first token);
- `export_predictions_for_mgpt3.py` round-trips 36/36 smoke predictions with 0
  fname collisions, and all 36 keys are present in MotionGPT3's test manifest;
- a 4-step stage-3 run on the new dataset trains and evaluates (`eval_loss`
  2.29 → 2.04), confirming the collator and the val jsonl resolution.

## The legacy dataset (`/local/home/dhollidt/data/ego4o_nymeria`)

The original atomic-only dataset, still on disk, backing the already-trained
atomic model (`llava/checkpoints/ego4o_hml_{pretrain,finetune_lora}`) and the two
eval runs in `llava/eval_out/`. It was produced by the pre-manifest version of
this script, which:

- filtered `texts/{item}.txt` to `Describe my atomic actions`;
- mapped seconds to 30 fps frames the same way (`int(round(s*30))`, end clamped);
- dropped windows shorter than 150 frames (5 s) and windows whose start was
  negative or past the end of the features;
- had **no** point-cloud filter, **no** maximum length and **no** stitching;
- sampled one question per row (seed 20260707) from the release's generic
  `constants.py` lists.

Counts: 110,441 train / 15,217 val / 29,449 test. Documented drops: ~13.9 k
segments under 5 s, 687 + 156 segments from the three `20230928_s0_grace_randolph`
recordings whose text timestamps are entirely negative, and two recordings with
no or few extracted frames.

**It is not reproducible from HEAD.** Keeping a second code path alive just to
regenerate a frozen artifact is exactly the duplication that caused the key
mismatch in the first place, so the rules are recorded here instead. If you ever
need an atomic-only dataset again, build one from the manifest with
`--categories "Describe my atomic actions"` — it will not be identical to the
legacy one (it follows MotionGPT3's window rules), which is the point.
