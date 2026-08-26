"""
reads all images from a folder, encodes them and stores the encodings in a .npy file in a target folder. The .npy file is named after the image name. 

Sample usage:
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e face_recognition
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e deepface

    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e deepface --subfolder
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e face_recognition --subfolder

    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e deepface --deepface_model Facenet512 --subfolder
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e deepface --deepface_model VGG-Face --subfolder
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e deepface --deepface_model ArcFace --subfolder

    # deepface: choose the DETECTOR backend via --deepface_detector_backend (default retinaface).
    # With --subfolder, deepface encodings are stored under: deepface/<model>/<detector_backend>/
    #   e.g. ./encodings/deepface/Facenet512/retinaface/
    # For best results, generate references with the SAME backend used at match time.
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e deepface --deepface_model Facenet512 --deepface_detector_backend retinaface --subfolder
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e deepface --deepface_model Facenet512 --deepface_detector_backend mtcnn --subfolder

    # face_recognition: choose the DETECTOR (hog/cnn) via --face_recognition_model
    # and the LANDMARK/ENCODING model (small/large) via --face_recognition_encoding_model.
    # With --subfolder, encodings are stored under: <encoder>/<detector>/<encoding_model>/
    #   e.g. ./encodings/face_recognition/hog/large/
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e face_recognition --face_recognition_model hog --face_recognition_encoding_model small --subfolder
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e face_recognition --face_recognition_model hog --face_recognition_encoding_model large --subfolder
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e face_recognition --face_recognition_model cnn --face_recognition_encoding_model small --subfolder
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e face_recognition --face_recognition_model cnn --face_recognition_encoding_model large --subfolder
"""

import argparse
import pathlib
import enum
import logging

from common_utils import StandardColorizer, log_entry_and_exit

__LOGGING_LEVEL__ = logging.INFO

logging.basicConfig(level=__LOGGING_LEVEL__, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(__LOGGING_LEVEL__)

c = StandardColorizer()


class SupportedEncoders(str, enum.Enum):
    """
    A class to represent supported encoders for face recognition.
    """
    FACE_RECOGNITION = 'face_recognition'
    DEEPFACE = 'deepface'
    # Add more encoders here if needed


class DeepFaceModelOptions(str, enum.Enum):
    """
    A class to represent model options for face recognition.
    """
    FACENET = 'Facenet'
    FACENET512 = 'Facenet512'
    VGGFACE = 'VGG-Face'
    ARCFACE = 'ArcFace'


class DeepFaceDetectorBackendOptions(str, enum.Enum):
    """
    A class to represent the face *detector* backend used by deepface to locate a
    face before computing its embedding. This is independent of the recognition
    model (DeepFaceModelOptions).
        - opencv    : Haar cascade. Fast but weak; fails on many angles/lighting.
        - ssd       : Single-Shot Detector. Fast, reasonable accuracy.
        - mtcnn     : Accurate, slower.
        - retinaface: Most accurate, slowest. Good default for hard images.
        - mediapipe / yolov8 / yunet / dlib : other supported backends.
        - skip      : bypass detection entirely (uses the whole image).
    NOTE: For best results, the detector backend used to generate the reference
    encodings should match the one used at match time.
    """
    OPENCV = 'opencv'
    SSD = 'ssd'
    MTCNN = 'mtcnn'
    RETINAFACE = 'retinaface'
    MEDIAPIPE = 'mediapipe'
    YOLOV8 = 'yolov8'
    YUNET = 'yunet'
    DLIB = 'dlib'
    SKIP = 'skip'


class DeepFaceNormalizationOptions(str, enum.Enum):
    """
    A class to represent the input normalization applied by deepface before the
    recognition model computes an embedding. Each recognition model expects a
    specific normalization; using the wrong one degrades the embedding quality.
        - base        : deepface default (simple /255 style). Sub-optimal for Facenet.
        - raw         : no normalization.
        - Facenet     : normalization tuned for the original Facenet weights.
        - Facenet2018 : normalization tuned for Facenet/Facenet512 (recommended).
        - VGGFace / VGGFace2 : for VGG-Face models.
        - ArcFace     : for ArcFace models.
    NOTE: The normalization used to generate the reference encodings MUST match
    the one used at match time, otherwise embeddings are not comparable.
    """
    BASE = 'base'
    RAW = 'raw'
    FACENET = 'Facenet'
    FACENET2018 = 'Facenet2018'
    VGGFACE = 'VGGFace'
    VGGFACE2 = 'VGGFace2'
    ARCFACE = 'ArcFace'


class FaceRecognitionModelOptions(str, enum.Enum):
    """
    A class to represent model options for face_recognition library.
    """
    HOG = 'hog'
    CNN = 'cnn'


class FaceRecognitionEncodingModelOptions(str, enum.Enum):
    """
    A class to represent the landmark/encoding model used by face_recognition's
    face_encodings(). This is INDEPENDENT of the HOG/CNN *detector* choice.
        - "small": faster, computes 5 facial landmarks.
        - "large": slower but more accurate, computes 68 facial landmarks.
    """
    SMALL = 'small'
    LARGE = 'large'

@log_entry_and_exit(logger=logger, level="info", logging_tag="FaceRecognitionEncodingGeneration")
def generate_encodings_via_face_recognition(source_folder: pathlib.Path, target_folder: pathlib.Path, model: str = FaceRecognitionModelOptions.HOG.value, encoding_model: str = FaceRecognitionEncodingModelOptions.SMALL.value):
    """
    Generate encodings for images in the source folder using the face_recognition library and store them in the target folder.

    Args:
        model (str): The *detector* model, "hog" or "cnn" (used by face_locations()).
        encoding_model (str): The *landmark/encoding* model, "small" or "large" (used by face_encodings()).
    """

    try:
        import face_recognition
    except ImportError as e:
        import sys
        print(f"face_recognition import failed ({e}). Using interpreter: {sys.executable}")
        print("If face_recognition is installed in a venv (e.g. ./venv3-12), run the script with that venv's python,")
        print("  e.g. ./venv3-12/bin/python -m store.generate_encodings ...  (or 'source venv3-12/bin/activate' first).")
        print("Otherwise install it via 'pip install face_recognition'")
        print("NOTE : setuptools<81   →   setuptools 80.10.2  (still bundles pkg_resources) which is required by face_recognition")
        return

    try:
        import numpy as np
    except ImportError:
        print("numpy module not found. Please install it via 'pip install numpy'")
        return

    print(c.blue(f"Generating encodings with \n\t- source_folder={source_folder} \n\t- target_folder={target_folder} \n\t- model={model} \n\t- encoding_model={encoding_model}"))

    for image_path in source_folder.glob('*.*'):
        if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue  # Skip non-image files

        # Load the image
        image = face_recognition.load_image_file(image_path)

        # NOTE: In the face_recognition library, HOG vs CNN is a *face detection*
        # choice, controlled by face_locations(model=...). The `model` argument of
        # face_encodings() only accepts "small"/"large" (the landmark predictor),
        # so passing "hog"/"cnn" there is ignored and yields identical results.
        # Therefore we first detect face locations with the chosen detector, then
        # compute encodings from those locations.
        face_locations = face_recognition.face_locations(image, model=model)

        # Get the face encodings for the image using the detected face locations.
        # The `model` here is the landmark predictor ("small"/"large"), which is
        # separate from the hog/cnn detector used above.
        encodings = face_recognition.face_encodings(image, known_face_locations=face_locations, model=encoding_model)

        if len(encodings) == 0:
            print(f"No faces found in {image_path.name}. Skipping.")
            continue

        # Assuming one face per image, take the first encoding
        encoding = encodings[0]

        # Save the encoding to a .npy file in the target folder
        target_file = target_folder / f"{image_path.stem}.npy"
        np.save(target_file, encoding)
        print(f"Saved encoding for {image_path.name} to {target_file}")

@log_entry_and_exit(logger=logger, level="info", logging_tag="DeepFaceEncodingGeneration")
def generate_encodings_via_deepface(source_folder: pathlib.Path, target_folder: pathlib.Path, model: str = DeepFaceModelOptions.FACENET.value, detector_backend: str = DeepFaceDetectorBackendOptions.RETINAFACE.value, normalization: str = DeepFaceNormalizationOptions.FACENET2018.value):
    """
    Generate encodings for images in the source folder using the deepface library and store them in the target folder.

    Args:
        model (str): The deepface recognition model (e.g. Facenet512, ArcFace).
        detector_backend (str): The face detector backend used to locate the face
            before encoding (e.g. retinaface, mtcnn, opencv). Should match the
            backend used at match time.
        normalization (str): The input normalization applied before the recognition
            model (e.g. Facenet2018 for Facenet/Facenet512). Should match the
            normalization used at match time.
    """

    try:
        from deepface import DeepFace
    except ImportError as e:
        import sys
        print(f"deepface import failed ({e}). Using interpreter: {sys.executable}")
        print("If deepface is installed in a venv (e.g. ./venv3-12), run the script with that venv's python,")
        print("  e.g. ./venv3-12/bin/python -m store.generate_encodings ...  (or 'source venv3-12/bin/activate' first).")
        print("Otherwise install it via 'pip install deepface'")
        print("NOTE : deepface requires tensorflow, keras, opencv-python, numpy, pandas, gdown, tqdm, mtcnn, retina-face, and other dependencies.")
        print("Please run : pip install tf-keras")
        print("Please run : pip install opencv-python-headless<5")
        return

    try:
        import numpy as np
    except ImportError:
        print("numpy module not found. Please install it via 'pip install numpy'")
        return

    print(c.blue(f"Generating encodings with \n\t- source_folder={source_folder} \n\t- target_folder={target_folder} \n\t- model={model} \n\t- detector_backend={detector_backend} \n\t- normalization={normalization}"))

    for image_path in source_folder.glob('*.*'):
        if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue  # Skip non-image files

        # Analyze the image to get the embedding.
        # detector_backend selects how deepface locates the face before encoding;
        # a stronger backend (e.g. retinaface / mtcnn) detects faces the default
        # opencv Haar cascade misses. Use 'skip' to bypass detection entirely.
        # normalization must match the recognition model (e.g. Facenet2018 for
        # Facenet/Facenet512); the wrong normalization degrades embedding quality.
        try:
            embedding = DeepFace.represent(img_path=str(image_path), model_name=model, detector_backend=detector_backend, normalization=normalization)[0]["embedding"]
            # NOTE : deepface has a bug in the latest version where it raises an error if no face is detected. To avoid this, we can use enforce_detection=False to skip images without faces.
            # embedding = DeepFace.represent(img_path=str(image_path), model_name=model, detector_backend=detector_backend, normalization=normalization, enforce_detection=False)[0]["embedding"]
        except Exception as e:
            print(f"Error processing {image_path.name} with detector_backend='{detector_backend}': {e}. Skipping.")
            continue

        # Save the embedding to a .npy file in the target folder
        target_file = target_folder / f"{image_path.stem}.npy"
        np.save(target_file, embedding)
        print(f"Saved encoding for {image_path.name} to {target_file}")

def generate_encodings(source_folder: pathlib.Path, target_folder: pathlib.Path, encoder: SupportedEncoders = SupportedEncoders.FACE_RECOGNITION, model: str = None, encoding_model: str = None, detector_backend: str = None, normalization: str = None):
    """
    Generate encodings for images in the source folder and store them in the target folder.

    Args:
        source_folder (pathlib.Path): Path to the source folder containing images.
        target_folder (pathlib.Path): Path to the target folder where encodings will be stored.
        encoder (SupportedEncoders): The encoder to use for generating encodings. Default is FACE_RECOGNITION.
        model (str): For face_recognition, the detector model ("hog"/"cnn"); for deepface, the model name.
        encoding_model (str): For face_recognition, the landmark/encoding model ("small"/"large").
        detector_backend (str): For deepface, the face detector backend ("retinaface"/"mtcnn"/"opencv"/...).
        normalization (str): For deepface, the input normalization applied before embedding ("Facenet2018"/"base"/...).
    """
    match encoder:
        case SupportedEncoders.FACE_RECOGNITION:
            generate_encodings_via_face_recognition(source_folder, target_folder, model=model if model else FaceRecognitionModelOptions.HOG.value, encoding_model=encoding_model if encoding_model else FaceRecognitionEncodingModelOptions.SMALL.value)
        case SupportedEncoders.DEEPFACE:
            generate_encodings_via_deepface(source_folder, target_folder, model=model if model else DeepFaceModelOptions.FACENET.value, detector_backend=detector_backend if detector_backend else DeepFaceDetectorBackendOptions.RETINAFACE.value, normalization=normalization if normalization else DeepFaceNormalizationOptions.FACENET2018.value)
        case _:
            raise ValueError(f"Encoder {encoder} is not supported yet.")

# entrypoint
if __name__ == "__main__":
    print('\n')
    parser = argparse.ArgumentParser(description='Generate encodings for images in a folder and store them in a target folder.')
    parser.add_argument('-s', '--source_folder', type=str, required=True, help='Path to the source folder containing images.')
    parser.add_argument('-t', '--target_folder', type=str, required=True, help='Path to the target folder where encodings will be stored.')
    parser.add_argument('-e', '--encoder', type=str, choices=[e.value for e in SupportedEncoders], default=SupportedEncoders.FACE_RECOGNITION.value, help='The encoder to use for generating encodings. Default is face_recognition.')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0', default='1.0', help='Show program version and exit.')
    # parser argument for storing encodings in a sub folder named after the encoder
    parser.add_argument('--subfolder', action='store_true', help='Store encodings in a subfolder named after the encoder. Default is False.')
    parser.add_argument('--deepface_model', type=str, choices=[m.value for m in DeepFaceModelOptions], default=DeepFaceModelOptions.FACENET.value, help='The model to use for generating encodings. Default is Facenet.')
    parser.add_argument('--deepface_detector_backend', type=str, choices=[b.value for b in DeepFaceDetectorBackendOptions], default=DeepFaceDetectorBackendOptions.RETINAFACE.value, help='The deepface face detector backend used to locate the face before encoding (only used when -e deepface). A stronger backend detects faces the default opencv Haar cascade misses. Use "skip" to bypass detection. Default is retinaface.')
    parser.add_argument('--deepface_normalization', type=str, choices=[n.value for n in DeepFaceNormalizationOptions], default=DeepFaceNormalizationOptions.FACENET2018.value, help='The deepface input normalization applied before embedding (only used when -e deepface). Matching the model\'s expected normalization greatly improves embedding quality. Default is Facenet2018.')
    parser.add_argument('--face_recognition_model', type=str, choices=[m.value for m in FaceRecognitionModelOptions], default=FaceRecognitionModelOptions.HOG.value, help='The detector model to use for face detection (hog/cnn). Default is hog.')
    parser.add_argument('--face_recognition_encoding_model', type=str, choices=[m.value for m in FaceRecognitionEncodingModelOptions], default=FaceRecognitionEncodingModelOptions.SMALL.value, help='The landmark/encoding model used by face_encodings (small/large). Default is small.')
    
    args = parser.parse_args()
    
    source_folder = pathlib.Path(args.source_folder)
    target_folder = pathlib.Path(args.target_folder)
    encoder = SupportedEncoders(args.encoder)

    if not source_folder.exists() or not source_folder.is_dir():
        raise ValueError(f"Source folder {source_folder} does not exist or is not a directory.")

    if not target_folder.exists() or not target_folder.is_dir():
        raise ValueError(f"Target folder {target_folder} does not exist or is not a directory.")

    if args.subfolder:
        target_folder = target_folder / encoder.value / (args.deepface_model if encoder == SupportedEncoders.DEEPFACE else args.face_recognition_model)
        if encoder == SupportedEncoders.FACE_RECOGNITION:
            target_folder = target_folder / args.face_recognition_encoding_model
        elif encoder == SupportedEncoders.DEEPFACE:
            target_folder = target_folder / args.deepface_detector_backend / args.deepface_normalization
        target_folder.mkdir(parents=True, exist_ok=True)
    
    # print important cli values
    print("--- Selected CLI values ---")
    print(f"Source folder: {source_folder}")
    print(f"Target folder: {target_folder}")
    print(f"Encoder: {encoder}")
    print(f"Model: {args.deepface_model if encoder == SupportedEncoders.DEEPFACE else args.face_recognition_model}")
    if encoder == SupportedEncoders.FACE_RECOGNITION:
        print(f"Encoding model (landmark predictor): {args.face_recognition_encoding_model}")
    if encoder == SupportedEncoders.DEEPFACE:
        print(f"Detector backend: {args.deepface_detector_backend}")
        print(f"Normalization: {args.deepface_normalization}")
    print(f"Subfolder: {args.subfolder}")
    print(f"Version: {args.version}")
    print("Generating encodings...")

    print("---------------------------")

    generate_encodings(source_folder, target_folder, encoder, model=args.deepface_model if encoder == SupportedEncoders.DEEPFACE else args.face_recognition_model, encoding_model=args.face_recognition_encoding_model, detector_backend=args.deepface_detector_backend, normalization=args.deepface_normalization)

    if target_folder.exists() and target_folder.is_dir():
        print(f"Encodings generated and stored in {target_folder}")
        print(c.B(c.green("Generated encodings:")))
        for encoding_file in target_folder.glob('*.npy'):
            print(c.green(f" - {encoding_file.name}"))
    else:
        print(f"Target folder {target_folder} does not exist or is not a directory. No encodings were generated.")
