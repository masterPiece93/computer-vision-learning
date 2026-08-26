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
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e face_recognition --face_recognition_model hog --subfolder
    python3 -m store.generate_encodings -s ./raw_images/ -t ./encodings/ -e face_recognition --face_recognition_model cnn --subfolder
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


class FaceRecognitionModelOptions(str, enum.Enum):
    """
    A class to represent model options for face_recognition library.
    """
    HOG = 'hog'
    CNN = 'cnn'

@log_entry_and_exit(logger=logger, level="info", logging_tag="FaceRecognitionEncodingGeneration")
def generate_encodings_via_face_recognition(source_folder: pathlib.Path, target_folder: pathlib.Path, model: str = FaceRecognitionModelOptions.HOG.value):
    """
    Generate encodings for images in the source folder using the face_recognition library and store them in the target folder.
    """

    try:
        import face_recognition
    except ImportError:
        print("face_recognition module not found. Please install it via 'pip install face_recognition'")
        print("NOTE : setuptools<81   →   setuptools 80.10.2  (still bundles pkg_resources) which is required by face_recognition")
        return

    try:
        import numpy as np
    except ImportError:
        print("numpy module not found. Please install it via 'pip install numpy'")
        return

    print(c.blue(f"Generating encodings with \n\t- source_folder={source_folder} \n\t- target_folder={target_folder} \n\t- model={model}"))

    for image_path in source_folder.glob('*.*'):
        if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue  # Skip non-image files

        # Load the image
        image = face_recognition.load_image_file(image_path)

        # Get the face encodings for the image
        encodings = face_recognition.face_encodings(image, model=model)

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
def generate_encodings_via_deepface(source_folder: pathlib.Path, target_folder: pathlib.Path, model: str = DeepFaceModelOptions.FACENET.value):
    """
    Generate encodings for images in the source folder using the deepface library and store them in the target folder.
    """

    try:
        from deepface import DeepFace
    except ImportError:
        print("deepface module not found. Please install it via 'pip install deepface'")
        print("NOTE : deepface requires tensorflow, keras, opencv-python, numpy, pandas, gdown, tqdm, mtcnn, retina-face, and other dependencies.")
        print("Please run : pip install tf-keras")
        print("Please run : pip install opencv-python-headless<5")
        return

    try:
        import numpy as np
    except ImportError:
        print("numpy module not found. Please install it via 'pip install numpy'")
        return

    for image_path in source_folder.glob('*.*'):
        if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue  # Skip non-image files

        # Analyze the image to get the embedding
        try:
            embedding = DeepFace.represent(img_path=str(image_path), model_name=model)[0]["embedding"]
            # NOTE : deepface has a bug in the latest version where it raises an error if no face is detected. To avoid this, we can use enforce_detection=False to skip images without faces.
            # embedding = DeepFace.represent(img_path=str(image_path), model_name=model, enforce_detection=False)[0]["embedding"]
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}. Skipping.")
            continue

        # Save the embedding to a .npy file in the target folder
        target_file = target_folder / f"{image_path.stem}.npy"
        np.save(target_file, embedding)
        print(f"Saved encoding for {image_path.name} to {target_file}")

def generate_encodings(source_folder: pathlib.Path, target_folder: pathlib.Path, encoder: SupportedEncoders = SupportedEncoders.FACE_RECOGNITION, model: str = None):
    """
    Generate encodings for images in the source folder and store them in the target folder.

    Args:
        source_folder (pathlib.Path): Path to the source folder containing images.
        target_folder (pathlib.Path): Path to the target folder where encodings will be stored.
        encoder (SupportedEncoders): The encoder to use for generating encodings. Default is FACE_RECOGNITION.
    """
    match encoder:
        case SupportedEncoders.FACE_RECOGNITION:
            generate_encodings_via_face_recognition(source_folder, target_folder, model=model if model else FaceRecognitionModelOptions.HOG.value)
        case SupportedEncoders.DEEPFACE:
            generate_encodings_via_deepface(source_folder, target_folder, model=model if model else DeepFaceModelOptions.FACENET.value)
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
    parser.add_argument('--face_recognition_model', type=str, choices=[m.value for m in FaceRecognitionModelOptions], default=FaceRecognitionModelOptions.HOG.value, help='The model to use for generating encodings. Default is hog.')
    
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
        target_folder.mkdir(parents=True, exist_ok=True)
    
    # print important cli values
    print("--- Selected CLI values ---")
    print(f"Source folder: {source_folder}")
    print(f"Target folder: {target_folder}")
    print(f"Encoder: {encoder}")
    print(f"Model: {args.deepface_model if encoder == SupportedEncoders.DEEPFACE else args.face_recognition_model}")
    print(f"Subfolder: {args.subfolder}")
    print(f"Version: {args.version}")
    print("Generating encodings...")

    print("---------------------------")

    generate_encodings(source_folder, target_folder, encoder, model=args.deepface_model if encoder == SupportedEncoders.DEEPFACE else args.face_recognition_model)

    if target_folder.exists() and target_folder.is_dir():
        print(f"Encodings generated and stored in {target_folder}")
        print(c.B(c.green("Generated encodings:")))
        for encoding_file in target_folder.glob('*.npy'):
            print(c.green(f" - {encoding_file.name}"))
    else:
        print(f"Target folder {target_folder} does not exist or is not a directory. No encodings were generated.")
