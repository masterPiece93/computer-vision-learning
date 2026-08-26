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
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e deepface --deepface_model Facenet512 --subfolder
        
    python3 one_target__multiple_reference_encodings.py --target_image ./raw_images/target_image.jpg --reference_encodings ./encodings/ --tolerance 0.4 --version 1.0 --matcher deepface --deepface_model Facenet512
    ```


- with face_recognition library
    ```sh
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e face_recognition --face_recognition_model cnn --subfolder

    python3 one_target__multiple_reference_encodings.py --target_image ./raw_images/target_image.jpg --reference_encodings ./encodings/ --tolerance 0.6 --version 2.0 --matcher face_recognition --face_recognition_model cnn
    ```

**Sample Commands**

Generation :

-   deepface
    ```sh
    python3 -m store.generate_encodings -s ./store/raw_images/ -t ./store/encodings/ -e deepface --deepface_model Facenet512 --subfolder
    ```

-   deepface
    ```sh
    python3 -m store.generate_encodings -s ./store/raw_images/ -t ./store/encodings/ -e deepface --deepface_model Facenet --subfolder
    ```

-   deepface
    ```sh
    python3 -m store.generate_encodings -s ./store/raw_images/ -t ./store/encodings/ -e deepface --deepface_model ArcFace --subfolder
    ```

-   face_recognition
    ```sh
    python3 -m store.generate_encodings -s ./store/raw_images/ -t ./store/encodings/ -e face_recognition --face_recognition_model cnn --subfolder
    ```
-   face_recognition
    ```sh
    python3 -m store.generate_encodings -s ./store/raw_images/ -t ./store/encodings/ -e face_recognition --face_recognition_model hog --subfolder
    ```

<br>

Matching :
- #1
    `deepface : Facenet512` | `tolerance : 0.4` | `version : 2.0` | `scoring : cosine similarity` |
    
    ```bash
    python3 one_target__multiple_reference_encodings.py --target_image ./store/raw_images/10.jpg --reference_encodings ./store/encodings/deepface/Facenet512/ --tolerance 0.4 --version 2.0 --matcher deepface --deepface_model Facenet512
    ```
    - uses existing encodings : `./store/encodings/face_recognition/hog/`
    - uses existing target : `./store/raw_images/10.jpg`
    - with version `2.0` , the similarity score is used , so `tolerance` caps the similarity score not distance .
    
    <br>
    <br>
    
    
- #2
    `face_recognition : hog` | `tolerance : 0.5` | `version : 2.0` | `scoring : L_2 Norm Distance` |
    
    ```bash
    python3 one_target__multiple_reference_encodings.py --target_image ./store/raw_images/10.jpg --reference_encodings ./store/encodings/face_recognition/hog/ --tolerance 0.5 --version 2.0 --matcher face_recognition --face_recognition_model hog
    ```
    - uses existing encodings : `./store/encodings/face_recognition/hog/`
    - uses existing target : `./store/raw_images/10.jpg`
    - for all face_recognition euclidian $(L_{2})$ Norm Distance score is used , so `tolerance` caps the distance score and not the similarity .
    

- #3
    `face_recognition : cnn` | `tolerance : 0.5` | `version : 2.0` | `scoring : L_2 Norm Distance` |
    
    ```bash
    python3 one_target__multiple_reference_encodings.py --target_image ./store/raw_images/10.jpg --reference_encodings ./store/encodings/face_recognition/cnn/ --tolerance 0.4 --version 2.0 --matcher face_recognition --face_recognition_model cnn
    ```
    - uses existing encodings : `./store/encodings/face_recognition/cnn/`
    - uses existing target : `./store/raw_images/10.jpg`
    - for all face_recognition euclidian $(L_{2})$ Norm Distance score is used , so `tolerance` caps the distance score and not the similarity .


---

<br>

## Point to Note :

**Question 1** : You will notice that the test run with both `face_recognition:hog` and `face_recognition:cnn` generate exact same results , but if `cnn` is better tha `hog` then there must be difference in reults , why not so ??

<details>
  <summary>Answer</summary>

  <!-- This blank line is required! -->
It makes perfect sense that your results were exactly the same.

In the face_recognition library, the choice between hog and cnn only affects face detection (finding the bounding box of the face). It does not change the face recognition model (generating the 128-dimensional embedding vector).

The exact reasons for identical matching results include:

- **One Universal Encoding Model** : No matter which detector you use, the `face_recognition.face_encodings()` function always routes the cropped face patch through the exact same deep-learning network (a *ResNet-based CNN* pre-trained by Dlib). Because the model calculating the 128 numbers is identical, the embeddings will be near-identical.

- **Identical Bounding Boxes** : If the face in your image is clear, front-facing, and well-lit, both the `hog` and `cnn` detectors will crop the exact same pixel coordinates for the face. Passing the exact same pixel data into the same encoding model produces identical matching percentages.


When will the results actually differ?

- You will only notice a difference between the two models in more challenging scenarios:

    1. **Side profiles or tilted angles** : The hog model will likely fail to find the face entirely (returning an empty list), while the cnn model will successfully locate it and generate a match.

    2. **Slightly shifted crops** : If an image is blurry or dark, `cnn` and `hog` might calculate slightly different bounding box sizes. The minor variation in the cropped face pixels will lead to slightly different embedding numbers, resulting in marginally different matching distances.
</details>
