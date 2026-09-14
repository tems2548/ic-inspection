# IC Inspection — PatchCore anomaly detection + marking OCR

Two independent checks for integrated circuits, both in Python:

| Check | Model | Answers |
|---|---|---|
| **Anomaly** | PatchCore (anomalib) | Is the package/surface defective? Trains on **good parts only** |
| **Marking** | CRNN + CTC (PyTorch) | What does the top marking say? Is it the expected part? |

`icinspect.pipeline` runs both and emits a PASS/FAIL report.

Code lives here. **The dataset lives outside this folder**, at
`%USERPROFILE%\Desktop\ic-dataset`, so images never end up in version control.

---

## The window

Everything below can be driven from a GUI instead of the command line:

```powershell
python -m icinspect.gui
```

or double-click **`IC Inspection.cmd`** in this folder — it activates the conda
environment first, so nothing has to be typed at all.

Four tabs, in the order a new machine needs them:

| Tab | What it does |
|---|---|
| **Setup** | point at the dataset folder, create the tree, check torch/CUDA, and see how many images and labels are actually on disk |
| **Train** | generate synthetic markings, train the reader, train the detector — the same modules as below, with the exact command echoed into the log |
| **Inspect** | pick a photo or a folder, choose which trained model to use, run both checks, and see the result drawn on the photo |
| **Stats** | this run's score and confidence distributions, its confusion matrix and IoU against the labels, the reader's training curves, and every hyperparameter both models carry |

The Inspect tab is the point of it. Each read is drawn back onto the source
photo: **blue** is the package the detector found, **green** is a marking
segment that was read, **grey dashed** is a segment ignored as a logo or an
unreadable mark. Click a row in the reads list to highlight its box, or
double-click to zoom to it. When a read looks wrong, that picture usually
answers why — the cut was wrong, not the model.

### The anomaly map

PatchCore does not only score a photo, it scores every patch of it, and the
window draws that map over the part: **where** it thinks the defect is, not just
how bad. anomalib normalises its scores so the threshold always lands at 0.50,
which is why the colour scale is absolute rather than stretched per photo — a
spotless part stays clear instead of being painted in the same fierce colours as
a cracked one, and two photos can be compared by colour alone.

The scale runs yellow → orange → red and is drawn as a labelled strip beside the
controls, so a colour reads back to a number. The red outline is the model's own
decision: the region it calls defective, outlined rather than filled so the
pixels underneath stay visible. Under the photo: the image score, the hottest
pixel, and how much of the frame was flagged.

**Sensitivity** moves where the colour starts. Every pixel of a moulded package
carries some score — a real capture reads 0.2–0.4 across the body — so the
default (*near threshold*, 0.35) starts just under the decision boundary and
shows what actually drove it. *Defects only* is stricter; *all activity* walks
the floor down to 0.12 to see the whole field.

Four **views**, switchable while looking at a result:

| View | What you get |
|---|---|
| `photo only` | the capture with the marking boxes, no map |
| `heat over photo` *(default)* | the ramp above: quiet where nothing is happening |
| `anomalib map (JET)` | anomalib's own look — JET across the whole frame at 50% |
| `anomalib panels` | the visualizer grid: Image · Image + Anomaly Map · Anomaly Map · Predicted Mask, each captioned |

The last two are the pictures anomalib's own visualizer produces, so a
screenshot from here lines up with the figures in its docs and in the PatchCore
papers. JET is a rainbow, and a rainbow is the wrong default for reading
magnitude — its neighbouring hues are not ordered, so 0.3 and 0.5 are not
tellable apart without the legend — which is why the window's own ramp stays the
default and the legend is drawn either way. Sensitivity greys out under JET: it
paints the whole range, so there is no floor to move.

**Save view…** writes what is on screen at the source photo's full resolution —
the annotated capture, or the panel grid — which is the picture worth putting in
a report.

### Choosing a model

Each check has a checkpoint picker beside it. anomalib keeps every training run
in its own `vN` folder, so the box lists them all, newest first, by version and
date; the reader lists `crnn_best.pt` and `crnn_last.pt`. Switching reloads on
the next run and re-points the Stats tab, so comparing a retrain against the run
it replaced is two clicks and no file juggling — on the captures here, `v7`
scores a defect 0.5977 where `v6` scores it 0.5587. Leave them alone and the
window uses exactly what the CLI would: the newest detector and `crnn_best.pt`.

### Scoring the part, not the fixture

PatchCore scores every patch of the frame, and the image score is whatever the
hottest patch anywhere in it reads. On a real capture that is usually **not the
chip**: the halo around the package and the lead row light up more than the
moulding does, and none of it says whether the part is defective.

**"Ignore the background"** (on by default) restricts both the score and the
picture to the package, using the same body detector the marking reader uses.
The map you see is the map that was counted; the background is not dimmed for
looks, it is excluded.

**Where the region comes from matters more than it looks.** The obvious answer —
find the package in the photo being inspected — is wrong for the case that
matters most: a part with a chunk broken off. The body detector measures what is
actually there, so it reports a smaller body, and the damage falls outside the
region being scored. The worse the break, the less of it counts. On this dataset
that happens on **45 of the 220 defective photos**, where the detected body
shrinks to as little as **40% of a normal part**.

So the region is learned from the **good parts instead** — the median body box
over `train/good`, cached in `results/patchcore/part_region.json`. Damage cannot
move it, because the parts it was measured from have none. It works because the
fixture is stable: across 60 good captures the box varies by about 0.03 of the
frame, which is the assumption this project already rests on. Pick `this photo`
in the **region** box to measure per-photo instead; **Tools → Relearn the scored
region** throws the learned one away after the camera or the fixture is moved
(changing the dataset root does that for you).

That changes what is being measured, so **the threshold has to be refitted**.
anomalib's line of 0.50 was calibrated with the background still in the picture;
against a part-only score it sits above everything, and judging by it would pass
defects in silence — on the captures here it would have missed **57 of 220**.
So the window will not use it: until a cutoff is fitted, the part score is
measured and displayed, but PASS/FAIL still comes from anomalib's whole-frame
decision, and the line under the checkbox says so.

**Calibrate** fits it: every image in `test/good` and `test/defect` is scored
part-only, and the cutoff with the best F1 is chosen — the same quantity
anomalib's own adaptive threshold maximises, refitted for what is now measured.
It offers to write the result to `threshold.part_score` in
[configs/patchcore.yaml](configs/patchcore.yaml) so the next session starts
there. Typing a number into **score over** overrides everything for that run.

Measured on this dataset (46 good, 220 defective, detector `v8`), four ways of
choosing what to score — the difference is how much room each leaves:

| Scored region | good parts | defective | gap between them | cutoff |
|---|---|---|---|---|
| anomalib's image score (whole frame) | 0.0103–0.4897 | 0.5000–1.0000 | 0.010 | 0.50 |
| map peak over the whole frame | 0.1059–0.4667 | 0.4980–0.8784 | 0.031 | 0.4824 |
| part measured on each photo | 0.0275–0.3137 | 0.4431–0.7961 | 0.129 | 0.3784 |
| **part from the good parts (default)** | 0.0235–0.3255 | 0.4549–0.8000 | **0.129** | **0.3902** |

All four score 1.00 AUROC and F1 on this set; the gap is what separates them. On
the whole frame a good part sits one hundredth below the line, so a lighting
wobble on the backdrop is enough to fail it — scoring the part leaves twelve
times that margin. The two part-only rows discriminate identically, and the
fixed region wins on the broken parts: those 45 photos score **0.4980–0.8000**
against a window that stays full size, rather than 0.4824–0.7961 against a
window their own damage shrank.

Two things to know. The detector finds the **moulded body**, inset off the
edge — a bent or damaged **lead** is outside the scored region, so if lead
defects matter to you, this is the wrong setting to leave on (every defect in
this test set is still caught, but that is this dataset, not a guarantee). And
when no region can be learned and none can be found in the photo, that photo
falls back to whole-frame scoring and the line under the image says so.

The window that was scored is drawn on the photo as a dashed violet box, and
outside it the photo is left untouched rather than tinted, so a score always
comes with the region it came from. The report CSV gains
`part_score`, the `part` box as fractions of the frame, and `decided_on`
(`part` or `frame`) — so a row always says which number the verdict came from.

### How much of the part is damaged

Alongside the score, every result carries **the share of the part that is
damaged** — the number to grade by, and the one the results table shows as
`damage`.

Two decisions make it mean something:

*It is a share of the part, not of the frame.* The frame is mostly fixture, so
the same broken corner reads 4.93% of the frame and **26.4% of the chip**. The
chip is what is being judged.

*It is measured at the cutoff that decides the verdict*, not at anomalib's pixel
mask. That mask is thresholded at the pixel line calibrated over whole frames,
and it is empty on a part that fails at 0.46 — so a failing part would report
nothing damaged. On `P00049` anomalib's mask gives 0.00% where the deciding
cutoff gives 23.4%. Measured this way the number can never contradict the
verdict: damage above zero is exactly what failing means.

Measured on this dataset:

| photo | part score | damage |
|---|---|---|
| good parts | 0.19–0.20 | 0.0% |
| `P00049` (surface) | 0.4627 | 23.4% |
| `P00244` (chunk broken off) | 0.8000 | 40.0% |
| `P00218` (chunk broken off) | 0.6824 | 58.2% |

The outline drawn on the photo is that same area, so the picture and the
percentage under it are the same thing. **fail over __% of the part damaged**
in More options adds it as its own reason to fail (`damage-over-limit`), for
when a part is only scrap past a certain amount of damage rather than at the
first sign of it. The report CSV carries it as `damage_ratio`.

### Scoring the window itself

Two more panels on the Stats tab answer the question the others cannot: **was
the window right**.

**The confusion matrix** compares each verdict against the folder the photo came
from — `test/defect` is a defective part, `test/good` and `train/good` are good
ones, and the names come from `dataset:` in
[configs/patchcore.yaml](configs/patchcore.yaml) rather than being assumed.
Photos from anywhere else (`raw/`, a one-off capture) have no truth to compare
with and are counted nowhere, so the matrix always says how many of the run it
covers. The four cells are named for what they mean on a line — a **missed
defect** ships, a **false alarm** scraps a good part — and accuracy, precision,
recall, specificity and F1 are listed beside them. Two tiles carry the summary:
accuracy, and defects caught as `caught/total`, which turns red the moment
anything is missed.

**IoU** answers *where* rather than *whether*. A part can be failed correctly
with the damage found in the wrong place, and nothing else on the tab notices.
Per photo it is the intersection over union of the damaged area the window
reports — the same area the `damage` percentage counts — against a hand-drawn
mask in `ground_truth/defect`: the same file name as the photo (or
`<name>_mask.png`), white where the defect is. Masks are resampled to the
anomaly map's grid with nearest-neighbour, because a mask is a decision per
pixel and a blurred edge would invent half-marked ones. A photo with no mask is
left out of the mean rather than counted as zero, and a clean part with an empty
mask scores 1.0 — nothing marked and nothing found is agreement.

IoU is measured against the damage *inside the scored region*, so a mask that
marks a broken lead — outside the moulded body — counts against the score. That
is the honest reading: the window did not find what was marked.

`ground_truth/defect` is empty on this machine, so the IoU panel says what to
put there and the column stays blank; everything else works without it. Both
metrics reach the report CSV as `truth` and `iou`.

### The Stats tab

Eight tiles across the top (photos inspected, pass rate, failures, accuracy
against the labels, defects caught, mean IoU, detector AUROC, reader sequence
accuracy), six charts, and a table of every number behind them:

| Chart | Reads from |
|---|---|
| anomaly scores, with the threshold marked | the inspection currently on screen |
| confusion matrix of verdict against truth | the same run, labelled by folder |
| IoU of the damage found against the damage marked | the same run + `ground_truth/defect` |
| OCR confidence per photo, with the reject line | the same run |
| reader accuracy per epoch (character + sequence) | `results/ocr/metrics.json` |
| reader training loss per epoch (log scale) | the same file |

The table lists what each model was *trained* with — backbone, layers, coreset
ratio, neighbours, input size, test AUROC/F1 for the detector; parameters, CTC
frames, charset, best epoch and validation accuracy for the reader — read from
the checkpoints themselves rather than from today's config, which is the first
thing to check when a number looks wrong. **Save charts as PNG…** exports the
grid. The charts need matplotlib; without it the tab still shows every number.

Models stay loaded between runs, so the second photo takes about a second
rather than ten. Inspecting one photo out of a folder scores only that photo:
it is hardlinked into a scratch folder first, because anomalib scores whole
directories and would otherwise run the other 55 captures to answer for one.

Results export with `icinspect.pipeline`'s columns first and in its order, so a
reader written against that CSV keeps working, followed by two the pipeline has
no data for: `peak_score` (the hottest pixel), `defect_area` (the share of the
frame flagged), and the part-only columns described above. Settings in the window override
the config files for that run only — nothing is written back except
**Save as default** on the Setup tab, which rewrites `dataset_root` in
[configs/paths.yaml](configs/paths.yaml) and leaves its comments alone.

Only one job runs at a time, on a background thread; **Stop** cancels between
photos, or terminates a training run.

---

## 1. Setup

Anomalib supports Python 3.10–3.12 (your base is 3.13), and the RTX 5060 Ti is
Blackwell/`sm_120`, which needs CUDA 12.8 PyTorch wheels. The setup script handles both:

```powershell
cd %USERPROFILE%\Desktop\ic-inspection
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
conda activate icinspect
```

Verify at any time:

```powershell
python scripts\check_env.py
```

It checks the Python version, that torch actually has `sm_120` kernels (a matmul is
run, not just a version string comparison), anomalib, and how much data you have.

CPU-only machine: `.\setup_env.ps1 -Cpu`.

## 2. Create the dataset folders

```powershell
python scripts\init_dataset.py
```

Builds this tree on the Desktop, next to the project:

```
%USERPROFILE%\Desktop\ic-dataset\
├── anomaly\ic\
│   ├── train\good\           defect-free images  -> PatchCore memory bank
│   ├── test\good\            defect-free, held out
│   ├── test\defect\          defective images
│   └── ground_truth\defect\  optional masks (white = defect) for pixel AUROC
├── ocr\
│   ├── images\               marking crops, one text line each
│   ├── train.txt             "<relative/path.png><TAB><LABEL>"
│   ├── val.txt               optional (else train.txt is split)
│   ├── test.txt              optional held-out set
│   └── synth\                generated crops + labels.txt
└── raw\                      unlabelled captures / staging
```

Move the dataset anywhere by editing `dataset_root` in [configs/paths.yaml](configs/paths.yaml),
or per-run with `set IC_DATASET_ROOT=D:\other-dataset`.

## 3. PatchCore — anomaly detection

Copy 30–300 photos of **good** ICs into `anomaly\ic\train\good`, and defective ones
into `anomaly\ic\test\defect`. Then:

```powershell
python -m icinspect.anomaly.train_patchcore
```

PatchCore is a one-pass memory-bank method — `max_epochs: 1` is correct, not a typo.
Training takes seconds to a couple of minutes. Results land in `results\patchcore\`.

Score new images:

```powershell
python -m icinspect.anomaly.infer_patchcore --input "%USERPROFILE%\Desktop\ic-dataset\raw"
```

Writes `results\patchcore\predictions.csv` with a score per image. Without a test set,
anomalib's adaptive threshold has nothing to calibrate against — pin one yourself in
`threshold.image_score` ([configs/patchcore.yaml](configs/patchcore.yaml)) once you have
seen the score distribution.

Tuning knobs, in the order worth trying:
`coreset_sampling_ratio` (0.01–0.25: lower = faster, smaller bank),
`image.size`, `layers` (`layer2`+`layer3` catches texture defects; add `layer1` for fine
scratches), `backbone` (`resnet18` is much faster if 8 GB VRAM gets tight).

## 4. OCR — marking recognition

Bootstrap with synthetic markings so you can train before labelling anything real:

```powershell
python scripts\gen_synth_ocr.py --count 20000 --split
python -m icinspect.ocr.train
```

The generator renders part numbers, date codes and lot codes with Windows fonts, then
adds defocus, uneven lighting, package grain, rotation and noise. Then fine-tune on real
crops (a few hundred goes a long way):

```powershell
python -m icinspect.ocr.train --labels train.txt --resume results\ocr\crnn_best.pt --epochs 20 --lr 3e-4
```

Read images:

```powershell
python -m icinspect.ocr.infer --input "%USERPROFILE%\Desktop\ic-dataset\ocr\images"
python -m icinspect.ocr.infer --input ... --labels test.txt   # reports char/seq accuracy
```

That reads **one line of text per file**. For photos of a whole chip, see
[4.1](#41-whole-chip-photos--chip-detection--line-splitting) — it finds the package and
cuts its marking into lines first.

Metrics are **character accuracy** (1 − CER) and **sequence accuracy** (whole string
exactly right). Sequence accuracy is the one that matters for part verification.

**Unsure characters are left out.** Every decoded character carries its own confidence —
the strongest CTC frame behind it — and anything below `decode.min_char_confidence`
([configs/ocr.yaml](configs/ocr.yaml), **0.95**) is dropped rather than guessed at. On a
real marking that is the difference between `7QO793M` and `93M`: the TI logo shares a
band with the date code and decodes to junk the model scores 0.35–0.91, while the real
characters come back at ~1.0. The unfiltered read is kept in a `raw_text` column, and
`confidence` is the mean over the characters actually reported, so it sits on the same
scale as the threshold.

```powershell
python -m icinspect.ocr.infer --input chip.png --split auto --min-char-confidence 0.99
python -m icinspect.ocr.infer --input chip.png --split auto --min-char-confidence 0   # keep everything
```

Measured over the 94 real captures — 0.95 is the knee, and 0.99 starts eating real
characters:

| `min_char_confidence` | `NE555P` (line 0) | `93M` (line 1) | `DNJW26` (line 2) | synthetic val, seq acc |
|---|---|---|---|---|
| 0 (off) | 94/94 | 0/94 | 0/94 | 0.9985 |
| 0.90 | 94/94 | 84/94 | 68/94 | 0.9930 |
| **0.95** | **94/94** | **92/94** | **86/94** | **0.9910** |
| 0.99 | 88/94 | 90/94 | 94/94 | — |

That last column is the cost: on clean synthetic crops, where the model is right anyway,
filtering throws away a few correct characters. When measuring model quality with
`--labels`, pass `--min-char-confidence 0` so the score reflects the model, not the
filter.

**Logos and unreadable marks are ignored.** A marking line is cut into *segments* at any
gap wider than `segment.split_gap_frac` line heights (0.40 by default — a word space is
about 0.3, the gap around a logo about 0.5), and each segment is read on its own. A
segment where at least `decode.max_unsure_frac` of the decoded characters were unsure is
then thrown away whole, instead of being reported as the one character that happened to
score well. That is what makes the reader skip a manufacturer logo, and the stamped
foundry mark the model has no training data for:

```
NE555P            <- read
[TI logo]  93M    <- logo ignored, "93M" read
DNJW26  [C4]      <- "DNJW26" read, the stamped mark ignored
```

Splitting matters as much as the rejection: without it the logo shares a crop with
`93M`, steals width when the crop is squeezed to the model's 160 px input, and drags the
whole segment under the unsure threshold — `93M` is then lost along with the logo (it
survives on 2/94 photos rather than 90/94).

`--keep-unreadable` reports the ignored segments instead of skipping them; the run always
prints how many it ignored and what they decoded to.

Measured over the 94 real captures, reading each photo down to exactly
`NE555P` / `93M` / `DNJW26`:

| `split_gap_frac` | photos read exactly right | note |
|---|---|---|
| 0 (no splitting) | 2/94 | the logo drags `93M` down with it |
| 0.30 | 90/94 | |
| **0.40** | **90/94** | default — widest gap that still scores 90, so the least likely to cut real text |
| 0.50 | 86/94 | the stamped mark starts staying attached to `DNJW26` |
| 0.60 | 84/94 | |

All 94 give exactly three segments. The four imperfect ones read `9M` for `93M` — a
recognition slip, not a segmentation one.

Label format — one line per crop, tab-separated, path relative to the label file:

```
images/u17_top.png	STM32F407VGT6
images/u18_top.png	LM358N
```

Characters the model can emit are set by `charset.symbols` in
[configs/ocr.yaml](configs/ocr.yaml); labels using anything else are dropped with a
warning rather than silently corrupting training.

### 4.1 Whole-chip photos — chip detection + line splitting

Point the reader at a photo of the part and it does two things before reading:
**finds the package** in the frame, then **splits its marking into text lines**.
The CRNN reads one line at a time, and a real top marking is two or three stacked
lines (part number, then date/lot code) surrounded by background and leads.

```powershell
# a photo of one chip on the fixture
python -m icinspect.ocr.infer --input "%USERPROFILE%\Desktop\ic-dataset\raw\P00001.jpg" --split auto

# a whole folder of captures
python -m icinspect.ocr.infer --input "%USERPROFILE%\Desktop\ic-dataset\raw" --split auto

# exactly two lines: cut at the midline, top and bottom
python -m icinspect.ocr.infer --input chip.png --split half

# part number only (the strongest line)
python -m icinspect.ocr.infer --input chip.png --split auto --max-lines 1

# already a tight crop of the package -- skip the chip search
python -m icinspect.ocr.infer --input crop.png --split auto --no-chip-crop
```

| `--split` | What it does | Use when |
|---|---|---|
| `none` *(default)* | one read per file, no cropping or cutting | the input is already single-line crops |
| `auto` | finds the package, then the text lines inside it | you don't know how many lines there are |
| `half` | finds the package, cuts at its midline, tightens each strip onto its ink | you know it is exactly top + bottom |

`--max-lines N` caps `auto` at the N strongest lines (most ink), still reported top to
bottom; under `half` it is the number of strips (default 2, so `--max-lines 3` gives
thirds). The output is **one CSV row per readable segment**, carrying both the package box and
the segment's own box, in source-photo pixels so they can be drawn back onto the original:

```
image,line,part,chip,box,crop,text,confidence,raw_text,dropped
P00001.jpg,0,1,"1155,487,2640,1515","1331,487,2492,882",...\0000_P00001_l0p1.png,NE555P,1.0,NE555P,0
P00001.jpg,1,2,"1155,487,2640,1515","1928,783,2465,1102",...\0000_P00001_l1p2.png,93M,1.0,93M,0
P00001.jpg,2,0,"1155,487,2640,1515","1406,1105,2208,1365",...\0000_P00001_l2p0.png,DNJW26,1.0,DNJW26,0
```

One row per *segment*: `line` is the text line, `part` the segment within it (a logo
ignored earlier in the line leaves a gap in the numbering, which is the point).
`raw_text` and `dropped` appear only when `min_char_confidence` is above 0.

The package crop (`..._chip.png`) and every line crop go to `results\ocr\crops\`
(`--crops-dir` to move them, `--no-crops` to skip) — when a read looks wrong, open the
crops first: the fault is usually the cut, not the model. With `--labels`, one label
covers the whole marking, so the lines are scored joined by a space.

**Finding the package.** Everything that differs from the *frame border* — which the
fixture keeps empty — is foreground; its widest run bounds the whole part, leads
included. Inside that box, only the rows and columns the foreground nearly fills are the
moulded body: a row of four leads covers about 40% of the width, the body covers all of
it, so `chip_body_frac` separates them with no shape modelling. The result is then
inset by `chip_inset` to clear the moulding edge. If the result is not plausibly a chip
(under 2% or over 98% of the frame) the search is abandoned and the whole frame is used.

**Finding the lines.** Each row of the package crop is scored by how much of it is
*ink* — pixel deviation from a heavily blurred copy of itself, so uneven lighting
cancels out and laser-etched marking reads the same whether it is lighter or darker than
the plastic. That map is binarised (Otsu, floored at `min_contrast`) and the row
coverages are smoothed and cut at a level taken from the profile's own Otsu, scaled by
`line_threshold`. The level has to be adaptive: a moulded package is speckled, so on a
real photo every row carries some ink — text rows just carry more (~0.30 coverage
against ~0.15 for blank package) — while a synthetic crop has a hard-zero background.

Knobs, all under `segment:` in [configs/ocr.yaml](configs/ocr.yaml):

| Knob | Raise it when | Lower it when |
|---|---|---|
| `line_threshold` | two lines merge into one band | a faint line (date code) is missed |
| `smooth_frac` | speckle bridges two lines together | a short line is smoothed away |
| `min_contrast` | grain is being read as text | the marking is low-contrast |
| `min_line_frac` | a stray band splits off a line | a small date code is dropped |
| `merge_gap_frac` | one line breaks into two bands | two lines merge into one |
| `pad_frac` | crops pull in the line above or below | ascenders/descenders are clipped |
| `background_blur` | lighting is very uneven | strokes come out hollow |
| `chip_inset` | the package edge shows up as an extra line | characters near the edge are clipped |
| `chip_body_frac` | leads are kept in the crop | the body itself is being rejected |
| `split_gap_frac` | a spaced part number is being cut in two | a logo stays attached to the text |
| `min_char_confidence` (under `decode:`) | junk characters still get through | real characters are being dropped |
| `max_unsure_frac` (under `decode:`) | a partly-worn marking is being thrown away | a logo is still reported |

Set `chip_detect: false` (or pass `--no-chip-crop`) when the inputs are already tight
crops. Two things it will not do: a marking rotated more than a couple of degrees needs
deskewing first, and a bright specular streak straight across the package has more row
coverage than any text line and will pull the threshold above the real lines — raise
`chip_inset` to cut such an edge out of the crop.

If no lines stand out, the crop is read as one line rather than failing, so a photo with
no readable marking still produces a row with a low confidence to reject on.

## 5. End-to-end test

```powershell
# both checks, whole folder
python -m icinspect.pipeline --input "%USERPROFILE%\Desktop\ic-dataset\raw"

# verify a specific part, reading only the marking region of each image
python -m icinspect.pipeline --input .\board.png --expect STM32F407VGT6 --roi 0.15 0.35 0.85 0.60
```

Writes `results\inspect\inspection_report.csv` with verdict, failure reasons, anomaly
score, decoded marking and OCR confidence. Exit code is 1 if anything failed, so it drops
straight into a test runner or CI. `--skip-anomaly` / `--skip-ocr` run one check alone;
`--min-confidence` rejects uncertain reads; `--max-cer` allows N% character slack against
`--expect`.

## 6. Tests

```powershell
pytest                  # everything
pytest -m "not slow"    # skip the CTC overfit smoke test
```

`tests/test_gui_service.py` covers the window's decision-making — what counts as
a read, what the report says, and the staging that keeps a one-photo inspection
from scoring its whole folder. `tests/test_gui_stats.py` covers the heat ramp
(quiet pixels stay unpainted, the ramp gets hotter with the score, the mask is
outlined not filled) and the charts, drawn on a headless Agg figure and checked
structurally: the threshold is marked, two series carry a legend, an empty run
says so, and no chart grows a second y-axis. `tests/test_gui_models.py` covers
the JET ramp and the checkpoint picker. Those three open no window, so they run
headless like the rest; `tests/test_gui_viewer.py` does need a display and skips
without one — it holds the canvas regressions, including switching out of the
panel grid.

Torch-dependent tests skip cleanly if torch is missing, so the charset, config, label
parsing, line-segmentation and synthetic-data tests run in any environment.

---

## Verified on this machine

Everything below was run end to end on 2026-09-08 (RTX 5060 Ti, torch 2.11.0+cu128,
anomalib 2.6.0, Python 3.11.16):

- `pytest` — 100 passed
- OCR: 4000 synthetic crops, 20 epochs at ~5.6 s/epoch → **99.5% char / 97.3% sequence
  accuracy** on 400 held-out crops
- PatchCore: 60 good / 15 good-test / 15 defect synthetic ICs → **image AUROC 1.00**,
  clean score separation (good 0.04–0.12, defect 0.50–1.00)
- `icinspect.pipeline` — 15 PASS / 15 FAIL, exit code 1 as designed
- OCR chip detection + line splitting, on **94 real NE555P captures** (the 47 in
  `raw\` and 47 of the 48 in `anomaly\ic\train\good\`): package found on 94/94,
  exactly 3 marking lines on 94/94, and line 0 read as `NE555P` on **94/94**. Line 2
  comes back `DNJW266Q`/`DNJW26SQ` against a true `DNJW26 C4`, and line 1 carries the
  TI logo into the crop — both are recognition errors on a model trained only on
  synthetic markings, not segmentation errors; the crops are correct
- Dropping unsure characters (`min_char_confidence: 0.95`) costs ~0.7% sequence accuracy
  on the 2000 clean synthetic validation crops (0.9985 → 0.9910)
- With segment splitting and whole-mark rejection on top, every one of the 94 photos
  yields exactly three segments, and **90/94 read exactly `NE555P` / `93M` / `DNJW26`** —
  the TI logo and the stamped `C4` mark ignored. The four others read `9M` for `93M`
- `anomaly\ic\train\good\P00048.jpg` is truncated and cannot be opened by PIL at all
- GUI, on 2026-09-10: the window opens on all three tabs, and inspecting
  `raw\P00001.jpg` through it gives PASS with `NE555P 93M DNJW26`, the package
  box found and the three segments drawn on the photo. One photo out of the
  56-image `raw\` folder takes **10.4 s** cold and **1.3 s** with the models
  already loaded — it was ~19 s per photo before only the picked file was
  scored
- Anomaly map + Stats tab, on 2026-09-10: 8 defective and 8 good captures
  inspected through the window — 3 PASS / 13 FAIL, mean score 0.407, scores
  0.0272–0.6743. The map draws over the photo (defect `P00049`: score 0.5977,
  hottest pixel 0.518, 0.04% of the frame outlined), all three sensitivity
  levels repaint cleanly, and **Save view…** wrote the annotated photo at the
  full 3840x2160. The Stats tab tiles read 16 / 19% / 13 / 0.407 / AUROC 0.9972
  / sequence accuracy 0.9985, with the reader's 31 training epochs charted from
  `results\ocr\metrics.json`
- Views and model choice, on 2026-09-10: all four views render and switch on a
  live result, with sensitivity correctly greyed out under JET. The picker lists
  the five PatchCore runs still on disk (`v2`–`v7`) and both readers; selecting
  `v6` and re-running scored the same defect 0.5587 against `v7`'s 0.5977, and
  the Stats tab's fact table followed the selection
- Fixed 2026-09-10: leaving the `anomalib panels` view left the grid on
  screen, because the canvas treated "same photo" as "same picture already
  loaded" and skipped re-reading the file. It now keys that cache on what is
  actually loaded
- Scoring the part only, on 2026-09-11 (detector `v8`, 46 good + 220 defective
  test parts): calibration fits the cutoff at **0.3784** with **F1 1.00 and
  AUROC 1.00**, catching 220/220 defects with 0/46 good parts failed, and the
  package was found on 266/266 photos. Against anomalib's 0.50 line the same
  part-only scores would have caught only 163/220. Whole-frame scoring also
  scores 1.00/1.00 on this set, at a cutoff of 0.4949 — the difference is the
  margin: 0.129 between the classes part-only against 0.010 whole-frame. The
  fitted value is pinned in `configs/patchcore.yaml`
- Scoring a **fixed** region, on 2026-09-11: on 45 of the 220 defective photos
  the per-photo body detector shrinks to 40-64% of a normal part — the damage
  cutting itself out of its own scored region. The region learned from 60 good
  parts, `(0.3008, 0.2255, 0.6875, 0.7046)`, does not move: those same broken
  parts then score 0.68-0.80 against a 0.3902 cutoff (good parts 0.19-0.20), and
  one of them reads `flagged area 4.93%` with the break outlined. Refitted on
  the full test set the fixed region gives F1 1.00 / AUROC 1.00 at **0.3902**,
  now pinned in the config
- Damage as a share of the part, on 2026-09-11: good parts 0.0%, `P00049`
  23.4%, `P00244` 40.0%, `P00218` 58.2% — against 0.00%, 0.00%, 4.93% and 7.09%
  for anomalib's own mask over the whole frame, which is empty on `P00049`
  despite it failing. A 20% limit adds `damage-over-limit` to the three
  defective photos and leaves the good ones passing. Fixed at the same time: the
  JET view painted the excluded background dark blue (JET's colour for zero)
  instead of leaving it alone
- Confusion matrix, on 2026-09-14: the whole labelled test set through the
  window -- 266 photos, all 266 labelled by folder -- gives **TP 220, FN 0,
  FP 0, TN 46**: accuracy, precision, recall, specificity and F1 all 1.0000.
  Read that as a fit, not a forecast: `threshold.part_score` was calibrated on
  these same photos, so the matrix confirms the cutoff was applied correctly
  rather than estimating how it will do on parts it has not seen. IoU stayed
  blank -- `ground_truth/defect` holds no masks on this machine -- and the panel
  says so instead of drawing an empty chart. `pytest` -- 192 passed

The PatchCore numbers come from throwaway generated images used only to exercise the
code path; they say nothing about accuracy on real parts. The OCR numbers are on
synthetic markings — expect lower on real photos until you fine-tune on real crops.

## Layout

```
configs/       paths.yaml (dataset location), patchcore.yaml, ocr.yaml
src/icinspect/
  config.py            config loading + path resolution (single source of truth)
  pipeline.py          combined PASS/FAIL inspection
  anomaly/
    common.py          anomalib glue, tolerant of 1.x/2.x API differences
    train_patchcore.py
    infer_patchcore.py
  gui/
    app.py             the window: four tabs over one worker thread
    inspect_tab.py     pick photos -> verdicts -> boxes on the photo
    train_tab.py       data generation and both trainings, as child processes
    setup_tab.py       dataset location, folder creation, what is on disk
    stats_tab.py       tiles, charts and the model fact table
    stats.py           training history + model facts, read off disk  (no torch needed)
    charts.py          the four matplotlib charts, one per function
    heatmap.py         anomaly map -> heat overlay, anomalib's JET view, the
                       panel grid, the part-only restriction, legends (no torch)
    service.py         the GUI's inspection engine (models kept loaded)
    viewer.py          zoom/pan photo canvas with the detection boxes
    runner.py          background thread + log queue + child processes
  ocr/
    charset.py         charset + CTC decode w/ per-char scores    (no torch)
    dataset.py         label files, image normalisation, augmentation
    model.py           CRNN
    segment.py         photo -> package crop -> one crop per segment (no torch)
    synth.py           synthetic marking generator                (no torch)
    train.py  infer.py
scripts/       init_dataset.py, gen_synth_ocr.py, check_env.py
tests/
```

## Practical notes

- **Fixture stability beats model tuning.** PatchCore compares against your training
  images; if lighting or working distance drifts, everything scores as anomalous.
  Fix camera, lens, distance and illumination first.
- `num_workers: 0` is the default in both configs — Windows spawns (not forks) workers,
  and >0 needs the entry point guarded. Raise it once you have verified throughput.
- The CRNN width sets the CTC frame budget: `T = width // 4 - 1`. Training aborts with a
  clear message if a label is longer than the available frames.
- `--split` assumes the marking lines are roughly **horizontal**. It tolerates a couple
  of degrees; a visibly rotated chip needs deskewing first, or the projection profile
  smears the lines together. Nothing in the pipeline rotates images for you.
- Chip detection reads the **frame border** to learn what background looks like, so keep
  the fixture's backdrop clear to the edges of the capture. It is the one assumption
  that, if broken, makes the crop wrong rather than merely imprecise.
- CTC loss runs in fp32 even under AMP; fp16 log-softmax makes it produce `inf`.
- The BiLSTM stack is built from single-layer `nn.LSTM` modules on purpose. A
  multi-layer cuDNN LSTM with `dropout > 0` crashes this machine at interpreter
  shutdown (`0xC0000409`) — results are correct but every run exits nonzero, which
  would silently poison the pipeline's PASS/FAIL exit code. Don't collapse it back.
- `results/` and `*.ckpt`/`*.pt` are gitignored — models are reproducible from the dataset.
