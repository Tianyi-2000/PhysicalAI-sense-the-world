# Urchin Monitoring Tool

Code associated with the TNC-USF 2025 practicum project to build a data
pipeline to monitor purple urchins from underwater videos.

---

## Pipeline overview

The pipeline has two stages, run in sequence:

1. **Detection** (`code/detections.py`):
   run a Faster R-CNN model on every video frame and save per-frame
   bounding boxes to a JSON file. Uses GPU if available, falls back to
   CPU. Writes checkpoints every 50 frames so the run can be resumed.

2. **Tracking & counting** (`code/pathway_kalman.py`):
   link per-frame detections into trajectories using a constant-velocity
   Kalman filter and Hungarian assignment, then count confirmed tracks as
   unique urchins.

3. **Estimated density** (`code/density.py`):
   convert the unique urchin count into estimated density (urchins/m²)
   using the belt-transect formula: `count / (transect_length × swath_width)`.

The tracking stage can optionally render an annotated video with bounding
boxes and track IDs for visual QA.

```
video.MP4 ──► code/detections.py ──► detection_result.json ──► code/pathway_kalman.py ──► count + pathway_result.json
                                                                                   ├─► pathway_kalman_summary.json
                                                                                   └─► (optional) annotated_video.mp4

pathway_kalman_summary.json ──► code/density.py ──► estimated_density.json
```

---

## Quick start

### 0. Get the model files and at least one video

The model weights and the GoPro videos are **not** stored in this Git
repository (they are large binary files). Before you can run the
pipeline, place these files locally:

| What | Where to put it | Where to get it |
|---|---|---|
| `UrchinfRCNN_TNC2.pt` (model object) | `models/UrchinfRCNN_TNC2.pt` | Project Box folder / S3 bucket `s3://urchins/models/` |
| `UrchinfRCNN_TNC2_best.pt` (weights) | `models/UrchinfRCNN_TNC2_best.pt` | Same as above |
| At least one transect video, e.g. `06-16-2025_Transect1.MP4` | `videos/06-16-2025_Transect1.MP4` | Project Box folder / S3 bucket `s3://urchins/videos/` |

If you only have the AWS S3 access set up, you can pull them with:

```bash
mkdir -p models videos
aws s3 cp s3://urchins/models/UrchinfRCNN_TNC2.pt      models/
aws s3 cp s3://urchins/models/UrchinfRCNN_TNC2_best.pt models/
aws s3 cp s3://urchins/videos/06-16-2025_Transect1.MP4 videos/
```

If you only have Box access, download manually into the same folders.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.9+ is required.

### 2. Run detection on a video

```bash
python code/detections.py \
  --input_video  videos/06-16-2025_Transect1.MP4 \
  --model_path   models/UrchinfRCNN_TNC2.pt \
  --best_ckpt    models/UrchinfRCNN_TNC2_best.pt
```

By default the detection JSON is written to
`outputs/06-16-2025_Transect1/detection_result.json` (the folder
name is taken from the input video's filename without its extension).
You can override that with `--output_dir <dir>` or `--output_json
<path>` if you need a specific location.

This can take a while. Detection checkpoints the JSON every 50 frames by
default, so you can resume the same command if it is interrupted. Add
`--amp` on a CUDA GPU for ~2× speedup; `--no_resume` to ignore any
existing output file and restart from frame 0.

### 3. Track and count

```bash
python code/pathway_kalman.py \
  --input_json   outputs/06-16-2025_Transect1/detection_result.json \
  --write_summary
```

By default the pathway JSON and the summary JSON land in the same
folder as the input detection JSON — so following the Step 2 example,
they end up in `outputs/06-16-2025_Transect1/`. Pass `--output_dir`
to override.

The script prints the headline number:

```
Unique urchin IDs (hits >= 50.0): 391   [total candidate tracks: 2358]
```

It also writes a summary JSON `pathway_kalman_summary.json` next to
the pathway JSON (with the parameters used and the count).

#### Overriding parameters

Any tracking parameter (see the **Parameters** section below) can be
overridden on the command line. Just append `--<flag> <value>`:

```bash
python code/pathway_kalman.py \
  --input_json   outputs/06-16-2025_Transect1/detection_result.json \
  --output_dir   outputs/06-16-2025_Transect1_minhits70 \
  --write_summary \
  --min_hits 70 \
  --score_thresh 0.15
```

Tips:

- **Use a distinct `--output_dir` name per run** (e.g. include the
  parameter and value, like `_minhits70`) so previous results aren't
  overwritten.
- **Always pass `--write_summary`** when sweeping parameters — the
  summary JSON records which parameter values produced which count.
- **Change one parameter at a time** so you can attribute the count
  change to the parameter you moved.

For which parameter to change when (e.g. count is too low / too
high, dark substrate, fast diver, dense clusters), see the
**Quick tuning guide** further below.

#### Running on 30 fps / out-of-distribution video (Big River profile)

The default `code/pathway_kalman.py` is calibrated for the format the
detector was trained on: **1920×1080, 16:9, 60 fps**. Videos shot in a
different format — most notably the Big River survey videos at
**5312×4648, 8:7, 30 fps** — require a different profile to avoid
severe undercount caused by track fragmentation at the lower frame
rate. For these videos we provide a separate script,
**`code/pathway_kalman_distancefallback.py`**, which adds an opt-in
`--distance_fallback` flag and is otherwise identical to the default
tracker.

Recommended command for a 30 fps Big River–style video:

```bash
python code/pathway_kalman_distancefallback.py \
  --input_json   outputs/<video_stem>/detection_result.json \
  --write_summary \
  --distance_fallback \
  --score_thresh 0.10 \
  --min_hits  25 \
  --max_age   20 \
  --max_step  160
```

What each change does, briefly:

- `--distance_fallback`: a code-level tracker option (off by default)
  that rescues track-detection pairs whose boxes do not overlap but
  whose centers are still within `max_step`. Without this flag the
  matcher rejects all zero-overlap pairs, which is the dominant cause
  of fragmentation at 30 fps. Only available in
  `pathway_kalman_distancefallback.py`. See METHODS §5.9.
- `--score_thresh 0.10` (vs 0.30 default): the detector under-scores
  small / distant urchins in out-of-distribution footage; lowering
  the gate admits those weak detections.
- `--min_hits 25`, `--max_step 160`: frame-rate-scaled values that
  preserve the same physical time threshold (0.83 s) and per-second
  motion budget at 30 fps that the defaults give at 60 fps.
- `--max_age 20`: empirical, set higher than a strict FPS scaling
  would suggest, to bridge the detector's intermittent "flicker" on
  small urchins.

For a fuller account of why this profile exists and the diagnostic
that led to it, see METHODS §6.b.

>  **60 fps in-distribution videos**: stick with the default
> `code/pathway_kalman.py` and do **not** pass these flags. The
> default parameter set is correct and validated for that format.

### 4. (Optional) render an annotated video for QA

```bash
python code/pathway_kalman.py \
  --input_json   outputs/06-16-2025_Transect1/detection_result.json \
  --input_video  videos/06-16-2025_Transect1.MP4 \
  --write_video --write_summary
```

The annotated MP4 lands at `outputs/06-16-2025_Transect1/annotated.mp4`
alongside the JSON outputs.

### 5. (Optional) compute estimated density

```bash
python code/density.py \
  --summary_json    outputs/06-16-2025_Transect1/pathway_kalman_summary.json \
  --transect_length 30 \
  --swath_width     1.5 2.0
```

The estimated-density JSON lands at
`outputs/06-16-2025_Transect1/estimated_density.json`.

`--swath_width` accepts one or more values; passing two (e.g. `1.5 2.0`)
produces a sensitivity range across plausible camera widths.

> **Note**: `--summary_json` expects the **summary** JSON
> (`pathway_kalman_summary_*.json`), which contains the
> `unique_urchin_count` field, not the per-frame pathways JSON
> (`pathway_result_*.json`). These two filenames look similar but
> hold very different content.
---

## Parameters (`code/pathway_kalman.py`)

Defaults were tuned empirically on TNC North Coast kelp-forest videos
(1920×1080 @ 60 fps GoPro). Any parameter can be overridden from the CLI.

| Parameter | Default | What it does |
|---|---|---|
| `--score_thresh` | `0.20` | Minimum detector confidence. Lower rescues weak detections of small / low-contrast urchins. |
| `--nms_iou` | `0.50` | IoU threshold for non-maximum suppression on duplicate boxes within the same frame. |
| `--match_iou` | `0.30` | Minimum IoU required to match a track's predicted box to a new detection. |
| `--max_age` | `6` | Max consecutive unmatched frames before a track is killed. |
| `--min_hits` | `50` | Number of matched detections required to confirm a track. At 60 fps, 50 ≈ 0.83 s. |
| `--max_step` | `80` | Max allowed box-center movement (pixels) between frames. |
| `--keep_labels` | `None` | Optional class filter, e.g. `--keep_labels 1`. |

Kalman-filter parameters (`--kalman_dt`, `--kalman_process_var`,
`--kalman_measurement_var`) rarely need to be changed. See the Methods
document for the full parameter tuning rationale.

### Quick tuning guide

**Before touching any parameter, watch a few seconds of the video.**
Five seconds of visual inspection saves hours of blind sweeping. Most
parameter mistakes come from tuning a video you haven't actually
looked at also different transects have different conditions, and the
right parameters depend on what's in the frame.

#### Pre-tuning visual checklist

Open the raw video (or an annotated run with default parameters) and
note the following:

| What to look for | Why it matters | Implication |
|---|---|---|
| **Substrate brightness** — is the seabed bright sand or dark purple-rocky reef? | Dark substrate hides low-contrast urchins from the detector. | If dark → expect undercount, plan to lower `--score_thresh`. |
| **Urchin density** — sparse field or dense clumps? | Dense clumps cause overlap and within-frame duplicates. | If dense → keep `--nms_iou` at 0.50 (do not lower it). |
| **Kelp / occlusion frequency** — does kelp drift across the frame? | Frequent occlusions fragment tracks. | If frequent → consider raising `--max_age`. |
| **Diver speed** — is the diver hovering or swimming briskly? | Fast motion increases per-frame box displacement. | If fast → consider raising `--max_step` (80 → 120). |
| **Urchin size in frame** — large and clear, or small specks? | Small urchins get low detector scores. | If small → lower `--score_thresh`, accept some noise. |
| **Transit time per urchin** — does each urchin stay in view 2 s or 10 s? | Short transits get filtered by `min_hits`. | If short → lower `--min_hits` (50 → 30). |
| **Visible noise** — bubbles, particulates, light flashes? | These spawn short false-positive tracks. | If noisy → keep `--min_hits` high (50 or 70). |

After this 5 second scan you usually already know which 1–2 parameters
to touch. The tables below tell you which direction to push them.

If the count looks wrong on a new video, change parameters one at a
time and rerun tracking only (no need to rerun detection).

**If the count is too LOW** (real urchins missing):

| Symptom | Try |
|---|---|
| Small / dark / low-contrast urchins not picked up | `--score_thresh` lower (0.20 → 0.15 or 0.10) |
| Urchin briefly occluded by kelp, then re-counted as a new ID | `--max_age` higher (6 → 10 or 15) |
| Urchins that pass through the frame quickly are dropped | `--min_hits` lower (50 → 30) |
| Adjacent urchins merged into one box | `--nms_iou` higher (0.50 → 0.60) |
| Fast diver motion, valid matches rejected as "teleports" | `--max_step` higher (80 → 120) |

**If the count is too HIGH** (noise / duplicates inflating count):

| Symptom | Try |
|---|---|
| Bubbles or kelp flashes counted as urchins | `--score_thresh` higher (0.20 → 0.30) |
| Same urchin counted twice across an occlusion | `--max_age` higher (paradoxical but reduces re-ID splits) |
| Many short noise tracks | `--min_hits` higher (50 → 70) |
| Same urchin double-counted from overlapping boxes | `--nms_iou` lower (0.50 → 0.35) |

**Tuning order**: `score_thresh` first (largest effect), then
`min_hits` (second-largest), then everything else as fine-tuning.

**Always change one parameter at a time** 

Changing several at once makes it impossible to tell which one moved
the count. Save each run with a distinct `--output_json` name and
always pass `--write_summary` so the parameter values are recorded
alongside the count.

Example sweep over `min_hits`:

```bash
for h in 30 50 70 100; do
  python code/pathway_kalman.py \
    --input_json outputs/06-16-2025_Transect1/detection_result.json \
    --output_dir "outputs/06-16-2025_Transect1_minhits${h}" \
    --write_summary \
    --min_hits $h
done
```

---

## Output files

| File | Produced by | Contents |
|---|---|---|
| `detection_result.json` | `code/detections.py` | Per-frame boxes, scores, labels from the detector. |
| `pathway_result.json` | `code/pathway_kalman.py` | Per-frame `[box, track_id]` entries for confirmed tracks only. |
| `pathway_kalman_summary.json` | `code/pathway_kalman.py` (with `--write_summary`) | Parameter set + unique urchin count for the video. |
| `annotated.mp4` | `code/pathway_kalman.py` (with `--write_video`) | Source video overlaid with boxes and track IDs. |
| `estimated_density.json` | `code/density.py` | Count + transect length + swath width(s) + estimated density (urchins/m²) per swath assumption. |

All outputs land in `outputs/<video_stem>/` by default, so a single
video's products stay together.

The headline metric is `unique_urchin_count` in the summary JSON (also
printed to stdout as `Unique urchin IDs (hits >= N)`).

---

## Known limitations

- **Small / low-contrast urchins are undercounted.** The detector assigns
  them low confidence and they fall below `--score_thresh`. Lowering the
  threshold partially recovers them but increases the chance of false
  positives. This is the dominant source of residual error.
- **Ground-truth counts are themselves human estimates** and have their own
  error. Validation figures should be read with that in mind.
- **Count is not density.** To convert a count into urchins/m², multiply
  the transect length by the camera swath width to get the surveyed area:
  `estimated density = count / (transect_length × swath_width)`. 
- **Tracker does not perform re-identification.** A single urchin that
  leaves the frame for longer than `max_age` frames and re-enters will be
  counted twice. In practice this is rare on linear transect passes.

---

## Developer setup

### GitHub — clone the repo

```bash
git clone https://github.com/tnc-ca-geo/urchin.git
```

### Activate the pre-commit hook for Jupyter notebooks

```bash
pip install pre-commit
pre-commit install
```

After that you will get a warning whenever you have not removed outputs
from your Notebooks. You can remove them with:

```bash
nbstripout filename.ipynb
```

**Note.** Please don't push (large) binary data to the repository. In
particular, store video files in S3.

---

## Amazon Web Services (AWS)

We use AWS as the place where we deploy the final model. Falk will work
with you toward the final deployment. Currently, the provided user account
allows for S3 access only; permissions will be extended as needed.

Ideally the resources we use would be defined by an AWS CloudFormation
script or the AWS CDK, which would make our workflows reproducible — see
https://docs.aws.amazon.com/cdk/v2/guide/work-with-cdk-python.html.

### Cost monitoring

Costs are monitored on the account. Please make sure that you shut down
unused Notebook kernels, particularly in SageMaker.

### User setup

Credentials for a shared `devuser` account will be provided separately.

We recommend https://github.com/99designs/aws-vault for credential
management. Otherwise follow the AWS CLI directions using the most
elementary form of a credential file — "long-term credentials" in this
tutorial: https://docs.amazonaws.cn/en_us/cli/latest/userguide/cli-configure-files.html.

### S3 bucket

A bucket has been created with the name `s3://urchins`.

Install the AWS CLI:

```bash
pip install aws-cli
```

With credentials in place you should be able to copy files to the bucket:

```bash
aws s3 cp a_file.txt s3://urchins/
```
