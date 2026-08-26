# Face Matching — Experiment Report

**Target Image:** `./test_data/new_target.jpeg`
**Matcher:** `face_recognition` &nbsp;|&nbsp; **Version:** `2.0` (vectorized euclidean distance)

Lower distance = more similar. Genuine (same person) references are `{1, 2, 3, 4}`;
impostors (different people) are `{5, 6, 7, 8, 9, 10}`.

---

## 1. Configuration Matrix

Each run is **internally consistent**: the target image is detected + encoded with the
**same detector (hog/cnn)** and the **same landmark/encoding model (small/large)** used to
generate the reference encodings it is compared against.

| Run | Detector | Encoding model | Reference folder | Tolerance |
|-----|----------|----------------|------------------|-----------|
| A | cnn | large | `.../face_recognition/cnn/large` | 0.55 |
| B | hog | large | `.../face_recognition/hog/large` | 0.55 |
| C | cnn | small | `.../face_recognition/cnn/small` | 0.55 |
| D | hog | small | `.../face_recognition/hog/small` | 0.55 |

---

## 2. Raw Results

### Run A — cnn + large  (tolerance 0.55)
Reference Encodings: `./store/encodings/face_recognition/cnn/large`
```
No match: 3.npy  with distance 0.569191569769326
No match: 5.npy  with distance 0.7949859162091971
No match: 9.npy  with distance 0.7714525649526007
No match: 7.npy  with distance 0.7651441686563272
No match: 8.npy  with distance 0.7775319404030934
No match: 2.npy  with distance 0.56220451546217
No match: 1.npy  with distance 0.5911058544916225
No match: 4.npy  with distance 0.5651366711438991
No match: 10.npy with distance 0.7320710721223203
No match: 6.npy  with distance 0.7611751506822101
Best match: 2.npy with distance 0.56220451546217
```
Execution time: **40.367 s**

### Run B — hog + large  (tolerance 0.55)
Reference Encodings: `./store/encodings/face_recognition/hog/large`
```
Match found: 3.npy with distance 0.5256710372484781
No match:    5.npy with distance 0.7950114204944887
No match:    9.npy with distance 0.8298662368527373
No match:    7.npy with distance 0.8476140917495466
No match:    8.npy with distance 0.8499533214762602
No match:    2.npy with distance 0.5753520827870336
Match found: 1.npy with distance 0.5447985227276336
Match found: 4.npy with distance 0.5216730898949169
No match:   10.npy with distance 0.799691419246417
No match:    6.npy with distance 0.7655771048716873
Best match: 4.npy with distance 0.5216730898949169
```
Execution time: **2.230 s**

### Run C — cnn + small  (tolerance 0.55)
Reference Encodings: `./store/encodings/face_recognition/cnn/small`
```
Match found: 3.npy with distance 0.5197239508704347
No match:    5.npy with distance 0.7488545247416423
No match:    9.npy with distance 0.8050054631082901
No match:    7.npy with distance 0.7913227794437746
No match:    8.npy with distance 0.8235300780180879
Match found: 2.npy with distance 0.5465484085114635
No match:    1.npy with distance 0.564372212341738
No match:    4.npy with distance 0.5518029067526157
No match:   10.npy with distance 0.7556717990280808
No match:    6.npy with distance 0.755620089782519
Best match: 3.npy with distance 0.5197239508704347
```
Execution time: **40.241 s**

### Run D — hog + small  (tolerance 0.55)
Reference Encodings: `./store/encodings/face_recognition/hog/small`
```
Match found: 3.npy with distance 0.5474764633533991
No match:    5.npy with distance 0.8068023623411404
No match:    9.npy with distance 0.7630447917169145
No match:    7.npy with distance 0.7758889676410194
No match:    8.npy with distance 0.8146253395947689
No match:    2.npy with distance 0.5552618926225074
No match:    1.npy with distance 0.5809585124275329
No match:    4.npy with distance 0.5601511386318543
No match:   10.npy with distance 0.7227902771726751
No match:    6.npy with distance 0.8051016600874911
Best match: 3.npy with distance 0.5474764633533991
```
Execution time: **2.247 s**

---

## 3. Analysis

### 3.1 The right metric: separation gap, not match count

For a search application, the number of files that crossed an arbitrary 0.55 threshold is
**not** the quality indicator. What matters is the **separation gap** = distance between the
*worst genuine* match and the *best impostor*. A larger gap means the threshold can be placed
safely and the system is robust to small variations.

| Run | Worst genuine (max of 1–4) | Best impostor (min of 5–10) | **Separation gap** | Time |
|-----|-----------------------------|------------------------------|--------------------|------|
| A — cnn + large | 1 → 0.591 | 10 → 0.732 | 0.141 | 40.4 s |
| **B — hog + large** | 2 → 0.575 | 6 → 0.766 | **0.190** ✅ | 2.2 s |
| C — cnn + small | 1 → 0.564 | 6 → 0.756 | **0.192** ✅ | 40.2 s |
| D — hog + small | 1 → 0.581 | 10 → 0.723 | 0.142 | 2.2 s |

### 3.2 Key findings

1. **All four runs cleanly separate genuine from impostor.** In every run the 4 genuine faces
   sit below ~0.59 and every impostor is above ~0.72 — there is **no overlap**. The earlier
   impression that lower-tolerance runs were "better" was a *threshold artifact*: tolerances of
   0.50 / 0.55 cut *into* the genuine cluster and silently dropped true matches.

2. **`hog + large` and `cnn + small` give the widest gaps** (~0.19) and are effectively tied on
   quality.

3. **CNN provides no accuracy advantage here** — `cnn + large` actually has the *narrowest* gap
   while costing **~18× the time** (40 s vs 2.2 s). On CPU (no GPU), CNN is pure cost for these
   frontal faces. CNN's real value is *detecting* hard faces (profile / small / rotated) that HOG
   misses — not producing better encodings for faces HOG already finds.

4. **`large` vs `small`** are comparable in separation on this data; `large` (68 landmarks) is
   generally more robust to pose/alignment variation in the wild, at negligible extra cost.

### 3.3 Choosing the threshold

Across all valid runs:

- All **genuine** distances ≤ **0.591**
- All **impostor** distances ≥ **0.723**

So the safe threshold band is **~0.60 – 0.68**. The classic dlib default of **0.6** sits right at
the bottom of that band and separates this data perfectly. Tolerance 0.55 was too strict (it drops
genuine `2` at 0.575); 0.50 was much too strict.

---

## 4. Recommendation

**Use `hog` + `large` encoding with tolerance `0.6`** for the face-search application.

- Best separation gap (0.190), tied for best overall quality.
- ~2.2 s vs ~40 s — **18× faster** than any CNN run on CPU.
- `large` encodings are more robust for real-world pose/lighting variation.
- At 0.6 it captures all four genuine matches and rejects every impostor.

```bash
python3 one_target__multiple_reference_encodings.py \
    --target_image ./test_data/new_target.jpeg \
    --reference_encodings ./store/encodings/face_recognition/hog/large \
    --matcher face_recognition \
    --face_recognition_model hog --face_recognition_encoding_model large \
    --tolerance 0.6
```

**Reserve CNN** for the fallback case where HOG fails to detect a face at all
(`No faces found`), ideally only when a GPU is available. A good production strategy: try HOG
first, and retry with CNN only when detection fails.

### Caveats
- This is **one target vs 10 references** — a very small sample. The 0.6 threshold is supported by
  both this data and dlib's published calibration (~99.38% on LFW), but validate on a larger,
  diverse set (varied lighting, pose, age, camera) and pick the threshold from a
  genuine-vs-impostor distance histogram / ROC curve before production.
- For a real **search** app, return a **ranked list** sorted by ascending distance with the score
  shown, applying 0.6 as a filter — more useful than binary match/no-match and lets a human
  adjudicate borderline hits.

---

## Appendix — Earlier exploratory run (invalid baseline)

An earlier `hog` run at tolerance **0.5** did not record a matching reference-encoding folder and
mixed configurations, so it is **not directly comparable** to Runs A–D. Kept only for history.

```
No match: 3.npy  with distance 0.5134747626433757
No match: 5.npy  with distance 0.7762915864406831
No match: 9.npy  with distance 0.8099223872202613
No match: 7.npy  with distance 0.8301887695891219
No match: 8.npy  with distance 0.837085799150806
No match: 2.npy  with distance 0.5609600379072229
No match: 1.npy  with distance 0.5304909719528655
Match found: 4.npy with distance 0.4993126640937045
No match: 10.npy with distance 0.7854664060160621
No match: 6.npy  with distance 0.758777527183283
Best match: 4.npy with distance 0.4993126640937045
```
Execution time: 1.116 s