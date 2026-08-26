# Face Matching

## One Face - Multiple Targets


<details>
  <summary>Package Requirements</summary>

  <!-- This blank line is required! -->
  ```bash
  Package                Version
  ---------------------- ---------
  deepface               0.0.100
  face-recognition       1.3.0
  opencv-python-headless 4.14.0.94
  pip                    24.0
  PySocks                1.7.1
  tf_keras               2.21.0
  ```
</details>

<br>

When you have one face to be matched against multiple faces

### [one_target__multiple_reference_encodings.py](./one_target__multiple_reference_encodings.py)

- does comparative analysis on a solution where we compare one image against multiple encodings present at a location
- we demonstrate
    - simple iterative approach
    - optimized approach - vectorised multiple encodings into one single encoding
    - various distance calculations ( cosine & euclidian L2)
    - compare `face_recognition` and `Deepface`

#### Script Explaination

- The script contains two Matchers : `FaceRecognitionMatching` and `DeepFaceMatching`
- Both the matchers have a `match` function with multiple versions , each demonstrating a different feature :
    - FaceRecognitionMatching
        > NOTE : the `face_recognition` library is clibrated to perform with Euclidian L2 Norm Distance only .
        > NOTE : the match threshold for a ideal match is **0.6** (Dlib's calibrated default; validated on this project's data in `report.md`). Lower values like 0.5 are stricter and tend to drop true matches.
        - v1.0 :
            - Iterative Approach
            - once source and target encodings are prepared , we will loop over all the source encodings and match each with target encoding and print result
        - v2.0 :
            - Vectorized Approach
            - once source and target encodings are prepared , we will flatten all the source encodings and fit into one single vector of all the encodings.

    - DeepFaceMatching
        - v1.0
            - Iterative Approach
            - once source and target encodings are prepared , we will loop over all the source encodings and match each with target encoding and print result
            - using `Euclidean Distance Calculation`
        - v1.1
            - Iterative Approach
            - once source and target encodings are prepared , we will loop over all the source encodings and match each with target encoding and print result
            - using `Cosine Distance Calculation` (`cosine_distance = 1 - cosine_similarity`, so **lower = more similar**, consistent with euclidean distance)
        - v1.2
            - Iterative Approach
            - once source and target encodings are prepared , we will loop over all the source encodings and match each with target encoding and print result
            - using `Cosine Distance Calculation` WITH `deepface.verify` call
        - v1.3
            - Iterative Approach
            - once source and target encodings are prepared , we will loop over all the source encodings and match each with target encoding and print result
            - using `Euclidean Distance Calculation` WITH `deepface.verify` call
        - v2.0
            - Vectorized Approach
            - once source and target encodings are prepared , we will flatten all the source encodings and fit into one single vector of all the encodings.
            - using `Cosine Distance Calculation` (`cosine_distance = 1 - cosine_similarity`, so **lower = more similar**; best match uses `argmin`)
    > NOTE : when you run the script , you get to choose the version with which you wish to execute .
    > NOTE : all deepface distance versions are now **distance-based** (lower = better match), just like `face_recognition`. So `--tolerance` always caps a **distance** — a match is found when the score is **below** the tolerance.
    
**Cosine Similarity Formula:**

$$
\text{Cosine Similarity} = \frac{A \cdot B}{\|A\|_2 \|B\|_2} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}
$$

Sample Run
- Let's find the similarity between Vector A (3, 4) and Vector B (4, 3):
    - Step 1: Get the top part (Dot Product)
        > $(3 \times 4) + (4 \times 3) = 12 + 12 = \mathbf{24}$
    - Step 2: Get the length of Vector A
        > $\sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = \mathbf{5}$
    - Step 3: Get the length of Vector B
        > $\sqrt{4^2 + 3^2} = \sqrt{16 + 9} = \sqrt{25} = \mathbf{5}$
    - Step 4: Multiply the lengths (Bottom part)
        > $5 \times 5 = \mathbf{25}$
    - Step 5: Divide top by bottom
        > $24 \div 25 = \mathbf{0.96}$
- The vectors point in almost the exact same direction because 0.96 is very close to 1.

Cosine Similarity - Formula Breakdown :

$$
\cos({θ}) = \frac{A \cdot B}{\|A\|_2 \|B\|_2} = (\frac{A}{\|A\|_2})\cdot(\frac{B}{\|B\|_2}) = \frac{1}{\|A\|_2 \|B\|_2}(A \cdot B)
$$

**Cosine Distance** — the script uses cosine **distance** (not similarity) so that, like Euclidean distance, a **lower** score means a **closer** match:

$$
\text{Cosine Distance} = 1 - \text{Cosine Similarity} = 1 - \frac{A \cdot B}{\|A\|_2 \|B\|_2}
$$

- range is $[0, 2]$; `0` = identical direction (perfect match), `1` = orthogonal, `2` = opposite.
- a match is declared when `cosine_distance < tolerance` (so a smaller `--tolerance` is stricter).

**Euclidean Distance Formula:**
- 2D Space: $(d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2})$
- 3D Space: $(d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2})$
- N-Dimensional Space: $(d = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2})$

With L2 Normalization :
- $(L_{2})$ Norm (Length of a single vector $(x$)):
$$
\|{}x\|{}_{2}=\sqrt{x_{1}^{2}+x_{2}^{2}+\dots +x_{n}^{2}}
$$
- Euclidean Distance (Distance between vector $(p$) and vector $(q$)):
$$
d(p,q)=\|{}p-q\|{}_{2}=\sqrt{(p_{1}-q_{1})^{2}+(p_{2}-q_{2})^{2}+\dots +(p_{n}-q_{n})^{2}}
$$

Visualization $L_{2}$
```
        ▲ Y-axis
        │
        │       /| Vector x = (3, 4)
        │      / |
        │     /  |
        │    /   |  Length (L2 Norm) = √(3² + 4²) = 5
        │   /    |
        └────────────► X-axis
            3    4

```
Key Breakdown
- Vector Difference: You subtract one point from another to create a distance vector.
- Square Elements: You square each coordinate value to make them all positive.
- Sum Up: You add all the squared values together.
- Square Root: You take the final square root to get the physical distance.

#### Script Commands

first you must have encodings generated against which you will run the face match .

**Commands List :**

- with Deepface library

    ```sh
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e deepface --deepface_model Facenet512 --deepface_detector_backend retinaface --deepface_normalization Facenet2018 --subfolder
        
    python3 one_target__multiple_reference_encodings.py --target_image ./raw_images/target_image.jpg --reference_encodings ./encodings/deepface/Facenet512/retinaface/Facenet2018/ --tolerance 0.4 --version 2.0 --matcher deepface --deepface_model Facenet512 --deepface_detector_backend retinaface --deepface_normalization Facenet2018
    ```

> **IMPORTANT (deepface):** there are **three independent knobs**:
> - `--deepface_model` selects the **recognition model** (`Facenet512`/`ArcFace`/...) that produces the embedding.
> - `--deepface_detector_backend` selects the **face detector** (`retinaface`/`mtcnn`/`opencv`/...) used to locate the face before encoding. The default `opencv` Haar cascade is weak and fails on many images; `retinaface` is more robust. Use `skip` to bypass detection.
> - `--deepface_normalization` selects the **input normalization** (`Facenet2018`/`base`/`ArcFace`/...) applied to the face before embedding. Matching the model's expected normalization greatly improves embedding quality; `Facenet2018` is a good default for `Facenet`/`Facenet512`.
>
> When `--subfolder` is used, deepface encodings are stored under `deepface/<model>/<detector_backend>/<normalization>/` (e.g. `./encodings/deepface/Facenet512/retinaface/Facenet2018/`). While matching, the target **must** use the **same recognition model, detector backend, and normalization** used to generate the references.


- with face_recognition library
    ```sh
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e face_recognition --face_recognition_model cnn --face_recognition_encoding_model large --subfolder

    python3 one_target__multiple_reference_encodings.py --target_image ./raw_images/target_image.jpg --reference_encodings ./encodings/face_recognition/cnn/large/ --tolerance 0.6 --version 2.0 --matcher face_recognition --face_recognition_model cnn --face_recognition_encoding_model large
    ```

> **IMPORTANT (face_recognition):** there are **two independent knobs**:
> - `--face_recognition_model` selects the **detector** (`hog`/`cnn`) — used by `face_locations()`.
> - `--face_recognition_encoding_model` selects the **landmark/encoding** model (`small`/`large`) — used by `face_encodings()`.
>
> When `--subfolder` is used, encodings are stored under `<encoder>/<detector>/<encoding_model>/` (e.g. `./encodings/face_recognition/hog/large/`). While matching, the target **must** be run with the **same detector and the same encoding model** used to generate the references, otherwise the encodings are not comparable.

**Sample Commands**

Generation :

-   deepface
    ```sh
    python3 -m store.generate_encodings -s ./store/raw_images/ -t ./store/encodings/ -e deepface --deepface_model Facenet512 --deepface_detector_backend retinaface --deepface_normalization Facenet2018 --subfolder
    ```

-   deepface
    ```sh
    python3 -m store.generate_encodings -s ./store/raw_images/ -t ./store/encodings/ -e deepface --deepface_model Facenet --deepface_detector_backend retinaface --deepface_normalization Facenet2018 --subfolder
    ```

-   deepface
    ```sh
    python3 -m store.generate_encodings -s ./store/raw_images/ -t ./store/encodings/ -e deepface --deepface_model ArcFace --deepface_detector_backend retinaface --deepface_normalization ArcFace --subfolder
    ```

-   face_recognition
    ```sh
    python3 -m store.generate_encodings -s ./store/raw_images/ -t ./store/encodings/ -e face_recognition --face_recognition_model cnn --face_recognition_encoding_model large --subfolder
    ```
-   face_recognition
    ```sh
    python3 -m store.generate_encodings -s ./store/raw_images/ -t ./store/encodings/ -e face_recognition --face_recognition_model cnn --face_recognition_encoding_model small --subfolder
    ```
-   face_recognition
    ```sh
    python3 -m store.generate_encodings -s ./store/raw_images/ -t ./store/encodings/ -e face_recognition --face_recognition_model hog --face_recognition_encoding_model large --subfolder
    ```
-   face_recognition
    ```sh
    python3 -m store.generate_encodings -s ./store/raw_images/ -t ./store/encodings/ -e face_recognition --face_recognition_model hog --face_recognition_encoding_model small --subfolder
    ```

<br>

Matching :
- #1
    `deepface : Facenet512` | `detector : retinaface` | `normalization : Facenet2018` | `tolerance : 0.4` | `version : 2.0` | `scoring : cosine distance` |
    
    ```bash
    python3 one_target__multiple_reference_encodings.py --target_image ./store/raw_images/10.jpg --reference_encodings ./store/encodings/deepface/Facenet512/retinaface/Facenet2018/ --tolerance 0.4 --version 2.0 --matcher deepface --deepface_model Facenet512 --deepface_detector_backend retinaface --deepface_normalization Facenet2018
    ```
    - uses existing encodings : `./store/encodings/deepface/Facenet512/retinaface/Facenet2018/`
    - uses existing target : `./store/raw_images/10.jpg`
    - the target is detected with `retinaface` and normalized with `Facenet2018` — matching how the references were generated.
    - with version `2.0` , the **cosine distance** score is used , so `tolerance` caps the distance (lower = closer match) , not the similarity .
    
    <br>
    <br>
    
    
- #2
    `face_recognition : hog` | `encoding : large` | `tolerance : 0.6` | `version : 2.0` | `scoring : L_2 Norm Distance` |
    
    ```bash
    python3 one_target__multiple_reference_encodings.py --target_image ./store/raw_images/10.jpg --reference_encodings ./store/encodings/face_recognition/hog/large/ --tolerance 0.6 --version 2.0 --matcher face_recognition --face_recognition_model hog --face_recognition_encoding_model large
    ```
    - uses existing encodings : `./store/encodings/face_recognition/hog/large/`
    - uses existing target : `./store/raw_images/10.jpg`
    - the target is detected with `hog` and encoded with the `large` (68-landmark) predictor — matching how the references were generated.
    - for all face_recognition euclidian $(L_{2})$ Norm Distance score is used , so `tolerance` caps the distance score and not the similarity .
    

- #3
    `face_recognition : cnn` | `encoding : small` | `tolerance : 0.6` | `version : 2.0` | `scoring : L_2 Norm Distance` |
    
    ```bash
    python3 one_target__multiple_reference_encodings.py --target_image ./store/raw_images/10.jpg --reference_encodings ./store/encodings/face_recognition/cnn/small/ --tolerance 0.6 --version 2.0 --matcher face_recognition --face_recognition_model cnn --face_recognition_encoding_model small
    ```
    - uses existing encodings : `./store/encodings/face_recognition/cnn/small/`
    - uses existing target : `./store/raw_images/10.jpg`
    - the target is detected with `cnn` and encoded with the `small` (5-landmark) predictor — matching how the references were generated.
    - for all face_recognition euclidian $(L_{2})$ Norm Distance score is used , so `tolerance` caps the distance score and not the similarity .


---

<br>

## Point to Note :

**Question 1** : You will notice that the test run with both `face_recognition:hog` and `face_recognition:cnn` generate exact same results , but if `cnn` is better tha `hog` then there must be difference in reults , why not so ??

<details>
  <summary>Answer</summary>

  <!-- This blank line is required! -->
There were actually **two** things going on here.

**(a) A bug in the original scripts.** The old code passed `model="hog"` / `model="cnn"`
to `face_recognition.face_encodings()`. But that function's `model` argument only accepts the
**landmark predictor** values `"small"`/`"large"` — it has nothing to do with hog/cnn. So both
runs silently fell back to the default predictor **and** the default detector, producing byte-for-byte
identical encodings. This is now fixed: the scripts first call
`face_recognition.face_locations(image, model=<hog|cnn>)` (the correct place for the detector) and
then `face_recognition.face_encodings(image, known_face_locations=..., model=<small|large>)`.

**(b) Even when wired correctly, hog and cnn only affect *detection*.** The choice between hog and
cnn selects the face **detector** (finding the bounding box). It does not change the recognition
model that produces the 128-d embedding — that is always Dlib's ResNet-based CNN.

The exact reasons matching results can still look identical for easy images:

- **One Universal Encoding Model** : No matter which detector you use, the `face_recognition.face_encodings()` function always routes the cropped face patch through the exact same deep-learning network (a *ResNet-based CNN* pre-trained by Dlib). Because the model calculating the 128 numbers is identical, the embeddings will be near-identical.

- **Identical Bounding Boxes** : If the face in your image is clear, front-facing, and well-lit, both the `hog` and `cnn` detectors will crop the exact same pixel coordinates for the face. Passing the exact same pixel data into the same encoding model produces identical matching percentages.


When will the results actually differ?

- You will only notice a difference between the two models in more challenging scenarios:

    1. **Side profiles or tilted angles** : The hog model will likely fail to find the face entirely (returning an empty list), while the cnn model will successfully locate it and generate a match.

    2. **Slightly shifted crops** : If an image is blurry or dark, `cnn` and `hog` might calculate slightly different bounding box sizes. The minor variation in the cropped face pixels will lead to slightly different embedding numbers, resulting in marginally different matching distances.
</details>

<br>

**Question 2** : What is the `--face_recognition_encoding_model` (`small`/`large`) option, and does it matter?

<details>
  <summary>Answer</summary>

  <!-- This blank line is required! -->
It selects the **landmark predictor** used by `face_encodings()` before computing the 128-d vector:

- `small` → 5 facial landmarks (faster).
- `large` → 68 facial landmarks (slightly slower, generally more robust to pose/alignment).

This is **independent** of the hog/cnn detector. The one hard rule: the target must be encoded with
the **same** `encoding_model` that was used to generate the references it is compared against — a
`large` reference and a `small` target produce vectors that are not directly comparable.

Because of this, when `--subfolder` is used the encodings are namespaced as
`<encoder>/<detector>/<encoding_model>/` (e.g. `hog/large/`, `cnn/small/`) so the two configurations
never overwrite each other.

**Empirical recommendation (see `report.md`):** on clear frontal faces, `hog + large` with a
`tolerance` of **0.6** gave the best speed/quality trade-off (~18× faster than cnn on CPU with equal
or better separation). Reserve `cnn` for cases where `hog` fails to detect a face at all, ideally on
a GPU.
</details>

<br>

**Question 3** : What is the `--deepface_detector_backend` (`retinaface`/`mtcnn`/`opencv`/...) option, and why did deepface throw `FaceNotDetected` / `module 'cv2' has no attribute 'CascadeClassifier'`?

<details>
  <summary>Answer</summary>

  <!-- This blank line is required! -->
deepface has **two** independent stages: a **detector** (finds the face box) and a **recognition
model** (produces the embedding). `--deepface_model` sets the recognition model; the new
`--deepface_detector_backend` sets the detector.

- The default detector is `opencv` (Haar cascade) — fast but weak; it frequently raises
  `FaceNotDetected` on non-frontal / small / dimly-lit faces. Switching to `retinaface` (default in
  this project now) or `mtcnn` detects those faces. `skip` bypasses detection entirely.
- When `--subfolder` is used, deepface encodings are namespaced as
  `deepface/<model>/<detector_backend>/` (e.g. `Facenet512/retinaface/`). As with face_recognition,
  the target should be matched with the **same** model + detector backend used for the references.

**Environment gotchas encountered (and fixes):**

- `deepface module not found` while it is clearly installed → you're running with the **wrong Python
  interpreter**. Use the project venv: `source venv3-12/bin/activate` (or call
  `./venv3-12/bin/python ...` directly). The system `python3` has none of these packages.
- `No module named 'tensorflow.python'` → a **corrupted/incomplete TensorFlow install**. Fix:
  `./venv3-12/bin/pip install --no-cache-dir --force-reinstall --no-deps tensorflow==2.21.0`.
- `module 'cv2' has no attribute 'CascadeClassifier'` → **two conflicting OpenCV packages** were
  installed (`opencv-python 5.x` **and** `opencv-python-headless 4.x`); the v5 pre-release shadowed
  `CascadeClassifier`. Fix: uninstall both, then install a single
  `pip install "opencv-python-headless<5"`.
</details>

<br>

**Question 4** : What is the `--deepface_normalization` (`Facenet2018`/`base`/`ArcFace`/...) option, and why did deepface (cosine) look so much *worse* than `face_recognition`?

<details>
  <summary>Answer</summary>

  <!-- This blank line is required! -->
Two separate issues were making the earlier comparison misleading:

**(a) Missing / wrong normalization.** deepface applies an **input normalization** to the face crop
*before* running it through the recognition model. Each model was trained with a specific
normalization, and skipping it (the old `base` default) produces low-quality embeddings — genuine
pairs came out with abnormally low cosine similarity (~0.19–0.42 instead of the expected ~0.6–0.85).
The new `--deepface_normalization` flag (default `Facenet2018`) fixes this. **Rule:** the target must
be encoded with the **same** normalization used for the references, so when `--subfolder` is used the
encodings are namespaced as `deepface/<model>/<detector_backend>/<normalization>/`.

**(b) A metric direction mismatch (now fixed).** deepface's cosine versions used to report cosine
**similarity** (higher = better), while `face_recognition` uses Euclidean **distance** (lower =
better). Comparing a "> tolerance" rule against a "< tolerance" rule made the two look inconsistent.
The scripts now use cosine **distance** (`1 - cosine_similarity`) everywhere, so **every** matcher is
distance-based: **lower = closer**, and a match is declared when the score is **below** `--tolerance`
(v2.0's best match uses `argmin`). This makes deepface and `face_recognition` directly comparable.

> Note: cosine distance and Euclidean L2 distance still live on **different scales** (cosine distance
> ∈ [0, 2]; Dlib L2 ≈ [0, 1] with a 0.6 threshold), so pick a `--tolerance` appropriate to the metric
> (e.g. ~0.3–0.4 for deepface cosine, ~0.6 for face_recognition L2). Don't reuse one threshold across
> both.
</details>



