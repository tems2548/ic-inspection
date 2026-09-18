# IC Inspection

Automated visual inspection of integrated circuits: **surface and package defect
detection** with PatchCore, and **marking verification** with a CRNN reader.
Both run from a desktop application or from the command line, and both produce
a PASS/FAIL verdict with the evidence drawn back onto the photo.

| Check | Model | Question it answers | Training data |
|---|---|---|---|
| **Defects** | PatchCore (anomalib 2.x, `wide_resnet50_2`) | Is the package damaged? Where, how much, and what kind? | good parts only |
| **Marking** | CRNN + CTC (PyTorch) | What does the top marking say? Is it the expected part? | synthetic crops, then real crops harvested from photos |

Developed and validated on NE555P DIP-8 parts photographed on a fixed fixture
(marking `NE555P` / `93M` / `DNJW26 G4`). The code is generic; the numbers
quoted in this document are from that dataset.

---

## Contents

1. [Quick start](#1-quick-start)
2. [The desktop application](#2-the-desktop-application)
3. [Defect detection](#3-defect-detection)
4. [Marking recognition](#4-marking-recognition)
5. [Command-line reference](#5-command-line-reference)
6. [Dataset layout](#6-dataset-layout)
7. [Report columns](#7-report-columns)
8. [Project layout](#8-project-layout)
9. [Tests](#9-tests)
10. [Validation record](#10-validation-record)
11. [Operating notes and known limitations](#11-operating-notes-and-known-limitations)

---

## 1. Quick start

### Requirements

- Windows 10/11 with conda (Miniconda or Anaconda)
- Python 3.10–3.12 inside the environment (anomalib does not support 3.13)
- NVIDIA GPU recommended. RTX 50-series (Blackwell, `sm_120`) needs the
  CUDA 12.8 PyTorch wheels; the setup script installs them.

### Install

```powershell
cd %USERPROFILE%\Desktop\ic-inspection
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1      # add -Cpu on a machine without a GPU
conda activate icinspect
python scripts\check_env.py                                   # Python, torch/CUDA kernels, anomalib, data on disk
```

`check_env.py` runs a real matmul on the GPU rather than comparing version
strings, so a torch build without `sm_120` kernels is caught before training.

### Create the dataset tree

```powershell
python scripts\init_dataset.py
```

The dataset lives **outside the repository** at `%USERPROFILE%\Desktop\ic\ic-dataset`
(see [§6](#6-dataset-layout)), so photos can never be committed by accident.
Move it by editing `dataset_root` in [configs/paths.yaml](configs/paths.yaml)
or, for one session, `set IC_DATASET_ROOT=D:\other-dataset`.

### Open the application

```powershell
python -m icinspect.gui
```

or double-click **`IC Inspection.cmd`**, which activates the conda environment
first.

### Typical workflow

1. **Setup tab** — confirm the dataset path and the counts on disk.
2. Copy good parts into `anomaly\ic\train\good`, held-out good parts into
   `anomaly\ic\test\good`, defective parts into `anomaly\ic\test\defect`.
3. **Train tab** — generate synthetic markings, train the reader, train the
   detector. Then **Harvest crops** from a folder of good photos and re-train
   the reader with **fine-tune** ticked.
4. **Inspect tab** — press **Calibrate** once so the part-only cutoff is
   fitted to your test set, then inspect photos or folders.
5. **Stats tab** — the reporting table, confusion matrix, IoU and the training curves.

---

## 2. The desktop application

Four tabs, in the order a new machine needs them:

| Tab | Purpose |
|---|---|
| **Setup** | Dataset location, folder creation, torch/CUDA check, counts of images and labels on disk. **Save as default** writes `dataset_root` back to `configs/paths.yaml`. |
| **Train** | Synthetic marking generation, crop harvesting, reader training, detector training. Each runs as a child process with the exact command echoed to the log and a **progress bar with time remaining**. |
| **Inspect** | Inspect a photo or a folder, choose the checkpoints, run both checks, see the verdict drawn on the photo. |
| **Stats** | The reporting table (escape and overkill rates with confidence intervals, PRO, detection limit in mm, repeatability, latency), score and confidence distributions, confusion matrix and IoU against the labels, reader training curves, and every hyperparameter both models carry. |

Only one job runs at a time, on a background thread. **Stop** cancels between
photos or terminates a training run. Models stay loaded between runs: the
second photo takes about a second rather than ten. Inspecting one photo out of
a folder scores only that photo — it is hard-linked into a scratch folder
first, because anomalib scores whole directories.

### 2.1 Inspect tab

Each result is drawn back onto the source photo:

| Colour | Meaning |
|---|---|
| **blue** box | the package the detector found |
| **green** box | a marking segment that was read |
| **grey dashed** box | a segment ignored as a logo or an unreadable mark |
| **violet dashed** box | the region PatchCore scored |
| **red** outline | the area the model calls defective |
| **coloured** blob boxes | the defect blobs the shape analysis measured, the largest labelled with its kind |

Click a row in the results list to highlight its box; double-click to zoom to
it. When a read looks wrong, the picture usually shows why — the cut was wrong,
not the model.

**Views** (switchable on a finished result):

| View | Content |
|---|---|
| `photo only` | the capture with the marking boxes, no map |
| `heat over photo` *(default)* | the anomaly map as a yellow → orange → red ramp, quiet where nothing is happening |
| `anomalib map (JET)` | anomalib's own JET rendering across the whole frame at 50% |
| `anomalib panels` | the visualizer grid: Image · Image + Anomaly Map · Anomaly Map · Predicted Mask |

The ramp is absolute rather than stretched per photo: anomalib normalises
scores so the threshold is always 0.50, so a spotless part stays clear and two
photos can be compared by colour alone. A labelled legend beside the controls
maps colour back to a number. **Sensitivity** sets where the colour starts
(*defects only* 0.42, *near threshold* 0.35, *all activity* 0.12); it is greyed
out under JET, which paints the whole range. **Save view…** writes what is on
screen at the source photo's full resolution.

**Choosing a model.** Each check has a checkpoint picker. anomalib keeps every
training run in its own `vN` folder, so the picker lists them all, newest
first; the reader lists `crnn_best.pt` and `crnn_last.pt`. Switching reloads on
the next run and re-points the Stats tab. Left alone, the window uses exactly
what the CLI would: the newest detector and `crnn_best.pt`.

The **Checks** box carries the per-run settings: the anomaly cutoff (**score
over**), the scored region, the expected part, the minimum OCR confidence and
how to find the marking (`auto` / `half` / `none`). **More options** adds the
line cap, the per-character confidence, the tolerated CER against the expected
part, whether to find the package first, whether to save line crops, and
**fail over __% of the part damaged**, which adds `damage-over-limit` as a
failure reason.

### 2.2 Train tab progress

Every training script prints `[progress] done/total text` lines
([progress.py](src/icinspect/progress.py)); the window turns them into the
bar, a percentage and a time-remaining estimate, and lightning's own
`Fitting ━━━ 12/44` bars are understood too. The ETA restarts at each stage
(PatchCore: extracting features → coreset selection → validation → test).
Run from PowerShell, the same lines appear as text.

---

## 3. Defect detection

PatchCore builds a memory bank of patch features from good parts and scores a
new photo by how far each patch is from that bank. It scores every patch, so
the output is both an image score and a per-pixel map. Configuration is in
[configs/patchcore.yaml](configs/patchcore.yaml); `max_epochs: 1` is correct —
PatchCore is a one-pass method.

### 3.1 Scoring the part, not the fixture

anomalib's image score is the hottest patch anywhere in the frame. On a real
capture that is usually the halo around the package or the lead row, not the
moulding. **Ignore the background** (on by default) restricts both the score
and the map to the package. The map shown is the map that was counted.

**Where the region comes from.** Measuring the package on the photo being
inspected fails on the case that matters most: a part with a chunk broken off
reports a smaller body, and the damage falls outside the scored region. On this
dataset that happens on 45 of 220 defective photos (body shrunk to as little as
40 % of a normal part). So the region is **learned from the good parts** — the
median body box over `train/good`, cached in `results/patchcore/part_region.json`
— and damage cannot move it. Across 60 good captures the box varies by about
0.03 of the frame.

The **region** selector offers three sources:

| Mode | Region |
|---|---|
| `normal part` *(default)* | the box learned from `train/good` |
| `this photo` | the body detector on each photo |
| `drawn box` | a box you drew (below) |

**Tools → Relearn the scored region** discards the learned box after the camera
or fixture moves (changing the dataset root does this automatically).

**Drawing the box.** **Draw box** under the photo lets you drag a box on the
viewer — corners and edges resize, the inside moves, right-drag pans. **Apply
box** saves it to `results/patchcore/manual_region.json` and switches the
region to `drawn box`. In that mode the box is used twice: as the region
PatchCore scores **and** as the package the marking reader crops to instead of
its own package search — useful when the detector keeps landing on a bevel or a
shadow, or a part is too worn to find. **Tools → Forget the drawn box** returns
to the learned region.

### 3.2 Calibrating the cutoff

Scoring the part changes what is measured, so anomalib's 0.50 line no longer
applies: fitted with the background in the picture, it sits above everything a
part alone reaches and would pass defects in silence (57 of 220 on this
dataset). Until a part-only cutoff exists, the part score is displayed but the
verdict still comes from anomalib's whole-frame decision, and the line under
the checkbox says so.

**Calibrate** scores every image in `test/good` and `test/defect` part-only and
picks the cutoff with the best F1 — the same criterion anomalib's adaptive
threshold uses, refitted for the new measurement. It offers to write the value
to `threshold.part_score` in `configs/patchcore.yaml`. The current value is
**0.4393**, fitted for detector `v19`. Typing a number into **score over**
overrides everything for that run.

Four ways of choosing what to score, measured on detector `v8` (46 good, 220
defective):

| Scored region | good parts | defective | margin | cutoff |
|---|---|---|---|---|
| anomalib image score (whole frame) | 0.0103–0.4897 | 0.5000–1.0000 | 0.010 | 0.50 |
| map peak over the whole frame | 0.1059–0.4667 | 0.4980–0.8784 | 0.031 | 0.4824 |
| part measured on each photo | 0.0275–0.3137 | 0.4431–0.7961 | 0.129 | 0.3784 |
| **part learned from good parts** | 0.0235–0.3255 | 0.4549–0.8000 | **0.129** | **0.3902** |

All four reach AUROC 1.00 on that set; the margin between the classes is the
difference, and the fixed region wins on the broken parts (0.4980–0.8000
against a window that stays full size).

Two caveats. The detector finds the **moulded body**, so a bent or damaged
**lead** is outside the scored region — if lead defects matter, do not leave
this on. When no region can be learned and none is found in the photo, that
photo falls back to whole-frame scoring and the line under the image says so.

### 3.3 Damage as a share of the part

Every result carries the share of the **part** that is damaged (`damage` in
the results table, `damage_ratio` in the CSV). Two decisions make it usable:

- It is a share of the part, not the frame: the same broken corner reads 4.93 %
  of the frame and **26.4 %** of the chip.
- It is measured at the cutoff that decides the verdict, not at anomalib's
  pixel mask. That mask is calibrated over whole frames and is empty on a part
  that fails at 0.46 — on `P00049` it gives 0.00 % where the deciding cutoff
  gives 23.4 %. Measured this way the number can never contradict the verdict.

| Photo | part score | damage |
|---|---|---|
| good parts | 0.19–0.20 | 0.0 % |
| `P00049` (surface) | 0.4627 | 23.4 % |
| `P00244` (chunk broken off) | 0.8000 | 40.0 % |
| `P00218` (chunk broken off) | 0.6824 | 58.2 % |

### 3.4 What kind of defect it is

PatchCore says whether and where; [shape.py](src/icinspect/anomaly/shape.py)
says what. The damage map is cut into connected blobs; each is measured — size
as a share of the part, elongation (major/minor axis, corrected for the map's
square pixels on a 16:9 photo), extent (how well it fills its box), whether it
touches the part's edge, whether the fixture backdrop shows through it — and a
few readable rules name the kind:

| Kind | Rule | Typical cause |
|---|---|---|
| `chunk missing` | backdrop visible through a blob (≥ 20 % of it), **or** the package outline is under 85 % of a normal part's with damage on the edge | broken corner or side |
| `surface wear` | ≥ 30 % of the part over the cutoff | abrasion, heavy scratching, soiling |
| `scratch` / `scratches` | a blob ≥ 3× longer than wide (two or more → `scratches`) | scratches |
| `spot` | compact blob (elongation < 2, extent ≥ 45 %) under 5 % of the part | droplet, dent, contamination |
| `patch` | an irregular blob over 1 % of the part | dirt, discolouration, worn area |
| `specks` | only tiny blobs (under 0.4 % of the part each) | dust, pitting |

Not counted: blobs under 0.05 % of the part (map noise) and the thin strip the
lit bevel leaves along the region border, which PatchCore scores as anomalous
but which is not damage. Rules rather than a classifier because there are no
kind labels to train on, and a rule can be checked against the photo. On the
364 photos in `test/defect` the 37 whose body measures small are exactly the 37
labelled `chunk missing`. The detail line gives the reason ("3.5x longer than
wide, 65% of the part's diagonal"), the log ends with a count per kind, and the
CSV carries `defect_kind`, `defect_detail`, `defect_blobs`. Thresholds are
constants at the top of `shape.py`.

### 3.5 Was the window right — confusion matrix and IoU

**Confusion matrix** (Stats tab): each verdict against the folder the photo
came from — `test/defect` is defective, `test/good` and `train/good` are good,
folder names read from `dataset:` in `configs/patchcore.yaml`. Photos from
anywhere else have no truth and are counted nowhere; the matrix says how many
of the run it covers. Cells are named for what they mean on a line (a **missed
defect** ships, a **false alarm** scraps a good part), with accuracy,
precision, recall, specificity and F1.

**IoU**: intersection over union of the reported damaged area against a mask
in `ground_truth/defect` (same file name as the photo, or `<name>_mask.png`,
white = defect). Masks are resampled to the map's grid with nearest-neighbour.
A photo without a mask is left out of the mean; a clean part with an empty mask
scores 1.0. IoU is measured against damage *inside the scored region*, so a
mask marking a broken lead counts against the score — the honest reading.
Both metrics reach the CSV as `truth` and `iou`.

### 3.6 The reporting table

The Stats tab opens with the seven numbers a qualification report asks for,
computed from the inspection on screen ([report.py](src/icinspect/gui/report.py)).
Each row carries its value, the count and interval it rests on, or — when it
cannot be computed — what would fill it. **Copy reporting table** puts it on
the clipboard as Markdown.

| Metric | Definition | Needs |
|---|---|---|
| Image AUROC | rank-based, from the run's deciding scores against the folder labels; for comparing detectors, not judging one | labelled photos of both kinds (falls back to the training summary) |
| PRO score @ τ | per-region overlap: the share of each connected region of the mask that the found damage covers, averaged over regions — a pinhole counts as much as a broken corner | masks under `ground_truth/defect` |
| Escape rate @ τ | missed defects / defective parts, with a Wilson 95 % interval and n | photos from `test/defect` |
| Overkill rate @ τ | false alarms / good parts, Wilson 95 % interval and n | photos from `test/good` or `train/good` |
| **Detection limit** | the equivalent diameter (√(4A/π)) in mm of the smallest masked defect caught that is larger than every defect missed | masks, the background ignored, and `report.body_mm` |
| Score repeatability σ | Gage R&R repeatability: the pooled within-part standard deviation of the score over repeated shots of the same part, also as a share of the cutoff | repeated shots, grouped by `report.repeats_pattern` (`A17_1.jpg`, `A17_2.jpg`) or a `repeats.csv` (`image,part`) under `anomaly/ic` |
| Inference latency | median wall time per part with the models warm: its share of the batched PatchCore pass plus its own marking read and shape analysis; the hardware is named | any inspection |

τ is the cutoff the run was judged against. Wilson rather than the normal
approximation because these rates sit at zero, where the normal interval
collapses to [0, 0] and Wilson gives the honest upper bound (0 escapes in 220
is "under 1.7 %"). The millimetre scale comes from the scored region: it is
the moulded body pulled in by `segment.chip_inset`, so `body_mm` (length ×
width as the part lies in the photo, `[9.5, 6.35]` for a PDIP-8) converts a
mask's pixel area to mm² with a separate scale per axis, since the map's
pixels are not square on a 16:9 frame. The `report:` block of
`configs/patchcore.yaml` holds these settings.

---

## 4. Marking recognition

A CRNN reads **one line of text per crop**. Photos of a whole part go through
package detection and line segmentation first ([§4.3](#43-whole-part-photos)).

### 4.1 Training

Bootstrap on synthetic markings, then fine-tune on real crops:

```powershell
python scripts\gen_synth_ocr.py --count 20000 --split
python -m icinspect.ocr.train
python -m icinspect.ocr.train --labels train.txt --resume results\ocr\crnn_best.pt --epochs 20 --lr 3e-4
```

The generator renders part numbers, date codes and lot codes with Windows
fonts, then adds defocus, uneven lighting, package grain, rotation and noise.
Metrics are **character accuracy** (1 − CER) and **sequence accuracy** (whole
string exactly right); sequence accuracy is the one that matters for part
verification.

Label format — one line per crop, tab-separated, path relative to the label
file. Characters the model can emit are set by `charset.symbols` in
[configs/ocr.yaml](configs/ocr.yaml); labels using anything else are dropped
with a warning.

```
images/u17_top.png	STM32F407VGT6
images/u18_top.png	LM358N
```

### 4.2 Harvesting real crops without hand labelling

The reader fails on lighting it has not seen — a batch with a fainter etch
reads `NED55` where the earlier batch read `NE555P` — and nothing at inference
time fixes that (per-crop contrast stretching and histogram equalisation were
both measured and both made it worse). The model has to see such crops.

When every photo in a folder carries the same marking, the label of each
segment follows from its position. **Harvest crops** on the Train tab (or
[`icinspect.ocr.harvest`](src/icinspect/ocr/harvest.py)) segments the photos
with the same code the window uses, labels the segments by position — lines
separated by `|`, segments by spaces — writes the crops to
`ocr/images/harvest/` and appends them to `train.txt` under a
`# harvested from …` header. Re-running never duplicates a label.

```powershell
python -m icinspect.ocr.harvest --input <folder> --expect "NE555P | 93M | DNJW26 G4"
```

- The reads are used only to tell the logo from the text where a line has more
  segments than the marking says. A photo whose lines do not match the marking
  is skipped, not guessed at; a wrong label is worse than no crop.
- Segments the reader gave up on are still harvested — those are the crops the
  training set is missing.
- **Only from parts that pass the defect check** (on by default) runs
  PatchCore first, so a scratched-through `NE555P` is never labelled `NE555P`.
- Crops are named after their two parent folders (`test_good_P00001_l0p0.png`)
  because capture sessions reuse photo numbers.
- Truncated JPEGs are skipped and logged rather than aborting the run.

On this dataset the harvest yields **2 132 crops from 533 parts** — 533 each of
`NE555P`, `93M`, `DNJW26` and `G4`, the faint batch included:

| Source | Photos used | Crops |
|---|---|---|
| `anomaly/ic/train/good` | 348 of 350 | 1 392 |
| `anomaly/ic/test/good` | 98 of 114 | 392 |
| external batch 1 | 21 of 24 | 84 |
| external batch 2 | 66 of 69 | 264 |

Skipped photos are those where `DNJW26` and `G4` run together at the crop's
edge. `train.txt` now holds 18 000 synthetic + 80 original + 2 132 harvested
lines. After harvesting: Train tab → tick **fine-tune** → **Train reader**.

### 4.3 Whole-part photos

```powershell
python -m icinspect.ocr.infer --input "%USERPROFILE%\Desktop\ic\ic-dataset\raw" --split auto
python -m icinspect.ocr.infer --input chip.png --split half              # exactly two lines, cut at the midline
python -m icinspect.ocr.infer --input chip.png --split auto --max-lines 1
python -m icinspect.ocr.infer --input crop.png --split auto --no-chip-crop  # already a tight package crop
```

| `--split` | Behaviour | Use when |
|---|---|---|
| `none` *(default)* | one read per file | inputs are single-line crops |
| `auto` | find the package, then every text line inside it | line count unknown |
| `half` | find the package, cut at the midline, tighten each strip onto its ink | exactly top + bottom |

Output is **one CSV row per readable segment** with both the package box and
the segment box in source-photo pixels:

```
image,line,part,chip,box,crop,text,confidence,raw_text,dropped
P00001.jpg,0,1,"1155,487,2640,1515","1331,487,2492,882",...\0000_P00001_l0p1.png,NE555P,1.0,NE555P,0
P00001.jpg,1,2,"1155,487,2640,1515","1928,783,2465,1102",...\0000_P00001_l1p2.png,93M,1.0,93M,0
P00001.jpg,2,0,"1155,487,2640,1515","1406,1105,2208,1365",...\0000_P00001_l2p0.png,DNJW26,1.0,DNJW26,0
```

The package crop and every line crop go to `results\ocr\crops\` (`--crops-dir`,
`--no-crops`). With `--labels`, one label covers the whole marking, so the lines
are scored joined by a space.

**Finding the package.** Everything that differs from the frame border — which
the fixture keeps empty — is foreground; its widest run bounds the whole part,
leads included. Inside that box only the rows and columns the foreground nearly
fills are the moulded body (`chip_body_frac`: a row of leads covers ~55 % of
the width, the body ~97 %), inset by `chip_inset` to clear the moulding edge.
A result under 2 % or over 98 % of the frame is rejected and the whole frame is
used.

**Finding the lines.** Each row of the package crop is scored by how much of
it is *ink* — deviation from a heavily blurred copy of itself, so uneven
lighting cancels and laser-etched marking reads the same whether lighter or
darker than the plastic. The ink map is binarised (Otsu, floored at
`min_contrast`), then a stroke-density mask (`stroke_frac`, `stroke_density`)
keeps stroke-like ink and drops isolated grain. The row profile is smoothed and
cut at a level from its own Otsu scaled by `line_threshold`, with hysteresis
(`line_hysteresis`) so a band grows to its full height once found. The level
is measured on the interior of the profile so a lit bevel at the crop border
cannot lift it above the real lines, and a band at the border much brighter
than the interior is dropped as that bevel. Faint lines that surface as a thin
slice through their characters are grown to a full line height
(`thin_line_frac`). Each line is then cut into segments at gaps wider than
`split_gap_frac` line heights on a square-root-compressed column profile
(stems and bowls of a `D` stay together), and segments narrower than
`min_part_frac` of the line height — or `edge_mark_frac` at the crop border —
are dropped as a speck of moulding, the pin-1 dimple or the bevel.

Knobs, all under `segment:` in [configs/ocr.yaml](configs/ocr.yaml):

| Knob | Default | Raise it when | Lower it when |
|---|---|---|---|
| `line_threshold` | 0.80 | two lines merge into one band | a faint line is missed |
| `line_hysteresis` | 0.85 | a line swallows the one beside it | a line comes back cropped short |
| `smooth_frac` | 0.01 | speckle bridges two lines | a short line is smoothed away |
| `min_contrast` | 0.05 | grain is read as text | the marking is low-contrast |
| `stroke_density` | 0.15 | grain survives into the mask | thin strokes drop out |
| `min_line_frac` | 0.05 | a stray band splits off a line | a small date code is dropped |
| `thin_line_frac` | 0.70 | a faint line returns as a slice | a genuinely smaller line is padded out |
| `merge_gap_frac` | 0.05 | one line breaks into two bands | two lines merge into one |
| `split_gap_frac` | 0.35 | a spaced part number is cut in two | a logo stays attached to the text |
| `min_part_frac` | 0.50 | a speck is read as a digit | a lone narrow character (1, I) is dropped |
| `edge_mark_frac` | 0.75 | the pin-1 dimple or bevel is read as a character | a narrow character at the crop edge is dropped |
| `pad_frac` | 0.25 | crops pull in the neighbouring line | ascenders/descenders are clipped |
| `background_blur` | 0.06 | lighting is very uneven | strokes come out hollow |
| `chip_inset` | 0.08 | the package edge shows as an extra line | characters near the edge are clipped |
| `chip_body_frac` | 0.75 | leads are kept in the crop | the body itself is rejected |

Set `chip_detect: false` (or `--no-chip-crop`) for inputs that are already
tight crops. If no lines stand out, the crop is read as one line so a photo
with no readable marking still produces a row with a low confidence to reject
on.

### 4.4 Rejecting unsure characters and logos

Every decoded character carries its own confidence — the strongest CTC frame
behind it — and anything below `decode.min_char_confidence` (**0.95**) is
dropped rather than guessed. On a real marking that is the difference between
`7QO793M` and `93M`: the TI logo decodes to junk scored 0.35–0.91 while the real
characters come back at ~1.0. The unfiltered read is kept in `raw_text`;
`confidence` is the mean over the characters actually reported.

A segment where at least `decode.max_unsure_frac` (0.5) of its characters were
unsure is thrown away whole, which is what makes the reader skip the
manufacturer logo and marks it has no training data for:

```
NE555P            <- read
[TI logo]  93M    <- logo ignored, "93M" read
DNJW26  G4        <- both read (after harvesting; before, "G4" was ignored as an unknown mark)
```

Splitting matters as much as rejection: without it the logo shares a crop with
`93M`, steals width when the crop is squeezed to the model's 160 px input, and
drags the whole segment under the threshold (`93M` survives on 2/94 photos
rather than 90/94). `--keep-unreadable` reports the ignored segments; the run
always prints how many it ignored and what they decoded to.

Measured over 94 real captures, the reader trained on synthetic data only:

| `min_char_confidence` | `NE555P` | `93M` | `DNJW26` | synthetic val, seq acc |
|---|---|---|---|---|
| 0 (off) | 94/94 | 0/94 | 0/94 | 0.9985 |
| 0.90 | 94/94 | 84/94 | 68/94 | 0.9930 |
| **0.95** | **94/94** | **92/94** | **86/94** | **0.9910** |
| 0.99 | 88/94 | 90/94 | 94/94 | — |

When measuring model quality with `--labels`, pass `--min-char-confidence 0`
so the score reflects the model, not the filter.

---

## 5. Command-line reference

Everything the window does is a module with a `--help`:

```powershell
# defect detector
python -m icinspect.anomaly.train_patchcore
python -m icinspect.anomaly.infer_patchcore --input "%USERPROFILE%\Desktop\ic\ic-dataset\raw"

# marking reader
python scripts\gen_synth_ocr.py --count 20000 --split
python -m icinspect.ocr.train [--labels train.txt --resume results\ocr\crnn_best.pt --epochs 20 --lr 3e-4]
python -m icinspect.ocr.harvest --input <folder> --expect "NE555P | 93M | DNJW26 G4" [--out DIR] [--labels train.txt] [--ckpt PATH] [--no-chip-crop]
python -m icinspect.ocr.infer --input <file-or-folder> [--split auto|half|none] [--labels test.txt] [--min-char-confidence 0.95] [--keep-unreadable]

# both checks, PASS/FAIL report, exit code 1 if anything failed
python -m icinspect.pipeline --input "%USERPROFILE%\Desktop\ic\ic-dataset\raw"
python -m icinspect.pipeline --input .\board.png --expect STM32F407VGT6 --roi 0.15 0.35 0.85 0.60
```

`icinspect.pipeline` writes `results\inspect\inspection_report.csv` with the
verdict, failure reasons, anomaly score, decoded marking and OCR confidence.
`--skip-anomaly` / `--skip-ocr` run one check alone; `--min-confidence` rejects
uncertain reads; `--max-cer` allows N % character slack against `--expect`.

`infer_patchcore` writes `results\patchcore\predictions.csv`. Without a test
set anomalib's adaptive threshold has nothing to calibrate against — pin
`threshold.image_score` in `configs/patchcore.yaml` once the score distribution
is known. PatchCore tuning knobs, in the order worth trying:
`coreset_sampling_ratio` (0.01–0.25, lower = faster and smaller bank),
`image.size`, `layers` (`layer2`+`layer3` for texture defects, `layer1` for fine
scratches), `backbone` (`resnet18` if VRAM is tight).

---

## 6. Dataset layout

```
%USERPROFILE%\Desktop\ic\ic-dataset\
├── anomaly\ic\
│   ├── train\good\           defect-free images  -> PatchCore memory bank    (350 here)
│   ├── test\good\            defect-free, held out                             (114)
│   ├── test\defect\          defective images                                  (364)
│   └── ground_truth\defect\  optional masks, white = defect, for IoU and PRO
├── ocr\
│   ├── images\               original marking crops, one text line each        (81)
│   ├── images\harvest\       crops written by the harvester                    (2 132)
│   ├── synth\                generated crops + labels.txt                      (20 000)
│   ├── train.txt             "<relative/path.png><TAB><LABEL>"
│   ├── val.txt               optional (else train.txt is split)
│   └── test.txt              optional held-out set
└── raw\                      unlabelled captures / staging                     (64)
```

Folder names under `anomaly/ic` come from `dataset:` in
`configs/patchcore.yaml`. Photo numbers are reused between capture sessions;
files are matched by content (MD5) when folders are merged, and truncated JPEGs
are skipped with a log line by every reader.

---

## 7. Report columns

The window's **Save report as CSV…** writes `icinspect.pipeline`'s columns
first and in its order, so a reader written against that CSV keeps working,
followed by the window's own:

| Column | Meaning |
|---|---|
| `image`, `verdict`, `reasons` | photo, PASS/FAIL, `;`-joined failure reasons (`anomaly`, `anomaly-missing`, `damage-over-limit`, `marking-mismatch`, `marking-unreadable`, `low-confidence`) |
| `anomaly_score`, `anomaly_label` | anomalib's whole-frame score and decision |
| `marking`, `marking_lines`, `ocr_confidence`, `expected` | the read, its lines joined by ` \| `, mean confidence, the expected part |
| `chip` | package box in source pixels |
| `peak_score`, `defect_area` | hottest pixel; share of the frame flagged by anomalib's mask |
| `part_score`, `part`, `decided_on` | part-only score; scored region as frame fractions; `part` or `frame` |
| `damage_ratio` | share of the part over the deciding cutoff |
| `truth`, `iou` | label from the folder; IoU against the mask, blank without one |
| `defect_kind`, `defect_detail`, `defect_blobs` | shape analysis: kind, reason, blob count |

---

## 8. Project layout

```
configs/         paths.yaml (dataset location), patchcore.yaml, ocr.yaml
scripts/         init_dataset.py, gen_synth_ocr.py, check_env.py
setup_env.ps1    conda environment + CUDA 12.8 torch + anomalib
IC Inspection.cmd  double-click launcher
src/icinspect/
  config.py            config loading and path resolution
  pipeline.py          combined PASS/FAIL inspection
  progress.py          [progress] line protocol shared by the trainers and the window
  anomaly/
    common.py          anomalib glue, lightning progress callback
    shape.py           blob measurement and defect-kind rules            (no torch)
    train_patchcore.py / infer_patchcore.py
  ocr/
    charset.py         charset + CTC decode with per-character scores    (no torch)
    dataset.py         label files, normalisation, augmentation
    model.py           CRNN
    segment.py         photo -> package crop -> line crops -> segments   (no torch)
    synth.py           synthetic marking generator                       (no torch)
    harvest.py         position-labelled crops from photos of one marking
    train.py / infer.py
  gui/
    app.py             the window: four tabs over one worker thread
    report.py          the reporting table: rates with intervals, PRO, detection
                       limit in mm, repeatability, latency                 (no torch)
    inspect_tab.py     photos -> verdicts -> boxes, heat map and blobs on the photo
    train_tab.py       synthetic data, harvesting, both trainings as child processes
    setup_tab.py       dataset location, folder creation, counts on disk
    stats_tab.py / stats.py / charts.py   tiles, charts, model fact table
    heatmap.py         anomaly map -> heat overlay, JET view, panel grid   (no torch)
    service.py         inspection engine: models kept loaded, regions, calibration
    viewer.py          zoom/pan canvas, detection boxes, box drawing
    runner.py          background thread, log queue, child processes, progress streaming
    widgets.py         shared Tk widgets
tests/
```

`results/`, `*.ckpt` and `*.pt` are gitignored; models are reproducible from
the dataset.

---

## 9. Tests

```powershell
pytest                  # everything — 251 tests
pytest -m "not slow"    # skip the CTC overfit smoke test
```

| File | Covers |
|---|---|
| `test_ocr_segment.py` | package detection and line/segment segmentation on synthetic markings: faint lines, bevel streaks, dimples, specks, thin bands |
| `test_ocr_harvest.py` | position labelling, logo dropping, skip rules, crop naming, no duplicate labels |
| `test_ocr_decode.py`, `test_charset.py`, `test_ocr_data.py`, `test_crnn.py` | CTC decode with per-character scores, charset, label files, the model |
| `test_anomaly_shape.py` | blob measurement and every defect-kind rule |
| `test_anomaly_glue.py` | anomalib API tolerance |
| `test_progress.py` | the progress protocol, lightning bar parsing, child-output streaming, the PatchCore callback on a tiny lightning run |
| `test_gui_service.py`, `test_gui_region.py`, `test_gui_background.py`, `test_gui_damage.py`, `test_gui_metrics.py`, `test_gui_drawn_box.py` | the window's decisions: reads, report columns, one-photo staging, scored region, calibration, damage share, confusion matrix and IoU, the drawn box |
| `test_gui_report.py` | Wilson intervals, AUROC, per-region overlap, pooled σ, the detection-limit rule, the mm scale, repeat grouping, and every row of the table |
| `test_gui_stats.py`, `test_gui_models.py` | heat ramp, charts (headless Agg), JET, checkpoint picker |
| `test_gui_viewer.py` | canvas regressions incl. box drawing — needs a display, skips without one |
| `test_config.py` | path resolution |

Torch-dependent tests skip cleanly when torch is missing; everything marked
"no torch" above runs in any environment.

---

## 10. Validation record

Environment: RTX 5060 Ti, Python 3.11.16, torch 2.11.0+cu128, anomalib 2.6.0.

| Date | Result |
|---|---|
| 2026-09-08 | End-to-end on synthetic data: OCR 99.5 % char / 97.3 % sequence accuracy on 400 held-out crops; PatchCore image AUROC 1.00 on generated ICs; pipeline 15 PASS / 15 FAIL, exit code 1 as designed. |
| 2026-09-08 | 94 real NE555P captures: package found 94/94, three marking lines 94/94, `NE555P` read 94/94. With character rejection and segment splitting, **90/94** read exactly `NE555P` / `93M` / `DNJW26`; the remaining four read `9M`. |
| 2026-09-10 | GUI: one photo out of a 56-image folder in 10.4 s cold, 1.3 s warm. Anomaly map, all four views and sensitivity levels, **Save view…** at full 3840×2160. Checkpoint picker: `v6` vs `v7` on the same defect, 0.5587 vs 0.5977. |
| 2026-09-11 | Part-only scoring (detector `v8`, 46 good + 220 defective): cutoff 0.3784, F1 1.00, AUROC 1.00, 220/220 caught, 0/46 false alarms, package found 266/266. Against anomalib's 0.50 line only 163/220. Fixed region learned from 60 good parts `(0.3008, 0.2255, 0.6875, 0.7046)`: broken parts 0.68–0.80 vs good 0.19–0.20; refitted cutoff 0.3902. |
| 2026-09-11 | Damage share: good 0.0 %, `P00049` 23.4 %, `P00244` 40.0 %, `P00218` 58.2 % (anomalib's own mask: 0.00 / 0.00 / 4.93 / 7.09 %). |
| 2026-09-14 | Confusion matrix over the 266 labelled test photos: TP 220, FN 0, FP 0, TN 46. A fit, not a forecast — the cutoff was calibrated on these photos. |
| 2026-09-16 | Segmentation rework for a new capture batch (lit bevel streak, fainter etching): `test/good` read fully 59 → 98 of 114, the 64 `raw/` photos benchmarked, no regression on `train/good`. Drawn box: `P00324`, unreadable with the detector's own crop, reads fully with a drawn box. |
| 2026-09-17 | Shape analysis on the 364 `test/defect` photos: kinds come out in contiguous runs of photo numbers, and the 37 body-shrunk photos are exactly the 37 `chunk missing`. |
| 2026-09-18 | Harvest: 2 132 labelled crops from 533 parts across four sources, 533 per label, no duplicates; the six truncated JPEGs on the external drive skipped and logged. Detector `v19`: anomalib whole-frame image AUROC 0.951 / F1 0.967 on `test/good` + `test/defect`; part-only cutoff calibrated at **0.4393**. Train-tab progress bar verified against the OCR trainer, the synthetic generator and a PatchCore run. |
| 2026-09-18 | Reporting table over 20 `test/good` + 40 `test/defect` photos (detector `v19`, cutoff 0.4393): AUROC 1.0000; escape rate 0.00 % (95 % CI 0.00–8.76 %, n = 40); overkill rate 0.00 % (95 % CI 0.00–16.11 %, n = 20); latency 572 ms/unit warm on the RTX 5060 Ti. PRO and detection limit blank: `ground_truth/defect` holds no masks yet. |
| 2026-09-18 | `pytest`: 265 passed. |

The early PatchCore figures were on generated images and say nothing about
real parts; the real-part figures are on one fixture and one part type.

---

## 11. Operating notes and known limitations

- **Fixture stability beats model tuning.** PatchCore compares against the
  training images; if lighting or working distance drifts, everything scores
  as anomalous. Fix camera, lens, distance and illumination first.
- **Package detection reads the frame border** to learn what background looks
  like. Keep the fixture's backdrop clear to the edges of the capture — it is
  the one assumption that, if broken, makes the crop wrong rather than
  imprecise. The drawn box is the fallback when it is.
- **Marking lines must be roughly horizontal.** A couple of degrees is
  tolerated; a visibly rotated part needs deskewing first. Nothing in the
  pipeline rotates images.
- **Leads are outside the scored region** when the background is ignored. Lead
  defects need whole-frame scoring or a drawn box that includes them.
- **The reader learns pixel levels, not just shapes.** A new lighting setup
  needs harvested crops and a fine-tune; inference-time normalisation was
  measured and rejected.
- `num_workers: 0` in both configs — Windows spawns rather than forks, and
  workers need a guarded entry point. Raise it once throughput is verified.
- The CRNN width sets the CTC frame budget (`T = width // 4 - 1`); training
  aborts with a clear message if a label is longer than the frames available.
- CTC loss runs in fp32 under AMP; fp16 log-softmax produces `inf`.
- The BiLSTM stack is built from single-layer `nn.LSTM` modules on purpose: a
  multi-layer cuDNN LSTM with `dropout > 0` crashes at interpreter shutdown
  (`0xC0000409`) on this machine, which would poison the pipeline's exit code.
- Settings in the window override the config files for that run only. Nothing
  is written back except **Save as default** (Setup) and the cutoff that
  **Calibrate** offers to write.
