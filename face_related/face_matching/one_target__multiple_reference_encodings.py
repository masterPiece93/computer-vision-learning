import logging
import pathlib
import enum
import numpy as np
from typing import *
from common_utils import *

__LOGGING_LEVEL__ = logging.INFO

logging.basicConfig(level=__LOGGING_LEVEL__, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(__LOGGING_LEVEL__)

Colors.register("ORANGE_BOLD", "\033[38;5;208m\033[1m")
Colors.register("PURPLE_UNDERLINE", "\033[38;5;129m\033[4m")

c = StandardColorizer()
c.green("This text will be green")
c.red("This text will be red")
c.B("This text will be bold")
c.U("This text will be underlined")
c.I("This text will be italic")

# ---

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


class FaceRecognitionMatching:
    """
    A class to handle face recognition matching using the face_recognition library.
    """

    def __init__(self, reference_encodings_folder: pathlib.Path, tolerance: float = 0.6, model: FaceRecognitionModelOptions = FaceRecognitionModelOptions.HOG.value ):
        
        self.reference_encodings_folder = reference_encodings_folder
        self.tolerance = tolerance
        self.model = model
        print(c.blue(f"Initialized FaceRecognitionMatching with \n\t- reference_encodings_folder={self.reference_encodings_folder} \n\t- tolerance={self.tolerance} \n\t- model={self.model}"))

    def get_target_encoding(self, image_path: pathlib.Path):
        """
        Generate encoding for target image using the face_recognition library.
        """
        try:
            import face_recognition
        except ImportError:
            print("face_recognition module not found. Please install it via 'pip install face_recognition'")
            return None

        # Load the target image
        target_image = face_recognition.load_image_file(image_path)

        # Get the face encodings for the target image
        target_encodings = face_recognition.face_encodings(target_image, model=self.model)

        if len(target_encodings) == 0:
            print(f"No faces found in {image_path.name}.")
            return None

        # Assuming one face per image, take the first encoding
        return target_encodings[0]
    
    @log_entry_and_exit(logger=logger, level="info", logging_tag="FaceRecognitionMatching")
    def match(self, target_image_path: pathlib.Path, version: str):
        """
        Match a target encoding against multiple reference encodings using the face_recognition library.
        """
        try:
            import face_recognition
        except ImportError:
            print("face_recognition module not found. Please install it via 'pip install face_recognition'")
            return

        try:
            import numpy as np
        except ImportError:
            print("numpy module not found. Please install it via 'pip install numpy'")
            return

        # Load the target encoding
        target_encoding = self.get_target_encoding(target_image_path)

        if target_encoding is None or len(target_encoding) == 0:
            print(c.red("No target encoding could be generated. Exiting."))
            return

        # Ensure the reference encodings folder exists and contains .npy files
        if not self.reference_encodings_folder.exists() or not self.reference_encodings_folder.is_dir():
            print(c.red(f"Reference encodings folder does not exist: {self.reference_encodings_folder}"))
            return

        if not any(self.reference_encodings_folder.glob('*.npy')):
            print(c.red(f"No reference encodings (.npy) found in {self.reference_encodings_folder}"))
            return

        match str(float(version)) :
            case "1.0":
                # -------------------------- #
                # Simple Iterative Approach
                # -------------------------- #
                # Iterate through all reference encodings in the folder
                for reference_encoding_path in self.reference_encodings_folder.glob('*.npy'):
                    reference_encoding = np.load(reference_encoding_path)

                    # Compare the target encoding with the reference encoding
                    matches = face_recognition.compare_faces([reference_encoding], target_encoding, tolerance=self.tolerance)

                    if matches[0]:
                        print(f"Match found: {reference_encoding_path.name}")
                    else:
                        print(f"No match: {reference_encoding_path.name}")
            case "2.0":
                # -------------------------- #
                # Vectorized batch Approach
                # Distance Calculation : using euclidean distance
                # Approach : raw calculation of euclidean distance between target and reference encodings
                # -------------------------- #
                # Load all reference encodings into a single numpy array
                reference_encodings = []
                reference_names = []
                for reference_encoding_path in self.reference_encodings_folder.glob('*.npy'):
                    reference_encodings.append(np.load(reference_encoding_path).flatten())
                    reference_names.append(reference_encoding_path.name) # this is done to reference the name of the file in the output
                
                reference_encodings = np.array(reference_encodings)

                # Compute euclidean distances for all reference encodings at once
                distances = np.linalg.norm(reference_encodings - target_encoding, axis=1)

                for name, distance in zip(reference_names, distances):
                    if distance < self.tolerance:  # You can adjust this threshold based on your needs
                        print(f"{c.green('Match found')}: {name} with distance {c.I(c.U(distance))}")
                    else:
                        print(f"{c.red('No match')}: {name} with distance {c.I(c.U(distance))}")

                best_match_index = np.argmin(distances)
                best_match_name = reference_names[best_match_index]
                best_match_distance = distances[best_match_index]
                print(f"{c.B('Best match')}: {best_match_name} with distance {c.I(c.U(best_match_distance))}")
            case _:
                raise NotImplementedError(f"Version {version} is not implemented. Please use '1.0' or '2.0'.")

        
class DeepFaceMatching:
    """
    A class to handle face recognition matching using the deepface library.
    """

    def __init__(self, reference_encodings_folder: pathlib.Path, tolerance: float = 0.4, model: DeepFaceModelOptions = DeepFaceModelOptions.FACENET512.value):
        self.reference_encodings_folder = reference_encodings_folder
        self.tolerance = tolerance
        self.model = model
        print(c.blue(f"Initialized FaceRecognitionMatching with \n\t- reference_encodings_folder={self.reference_encodings_folder} \n\t- tolerance={self.tolerance} \n\t- model={self.model}"))

    def get_target_encoding(self, image_path: pathlib.Path):
        """
        Generate encoding for target image using the deepface library.
        """
        try:
            from deepface import DeepFace
        except ImportError:
            print("deepface module not found. Please install it via 'pip install deepface'")
            return None

        # Generate the target encoding
        target_encoding = DeepFace.represent(img_path=str(image_path), model_name=self.model, enforce_detection=True)[0]["embedding"]
        return target_encoding

    @log_entry_and_exit(logger=logger, level="info", logging_tag="DeepFaceMatching")
    def match(self, target_image_path: pathlib.Path, version: str = "1.0"):
        """
        Match a target encoding against multiple reference encodings using the deepface library.
        """
        try:
            from deepface import DeepFace
        except ImportError:
            print("deepface module not found. Please install it via 'pip install deepface'")
            return

        # Generate the target encoding
        target_encoding = self.get_target_encoding(target_image_path)

        if target_encoding is None or len(target_encoding) == 0:
            print(c.red("No target encoding could be generated. Exiting."))
            return

        # Ensure the reference encodings folder exists and contains .npy files
        if not self.reference_encodings_folder.exists() or not self.reference_encodings_folder.is_dir():
            print(c.red(f"Reference encodings folder does not exist: {self.reference_encodings_folder}"))
            return

        if not any(self.reference_encodings_folder.glob('*.npy')):
            print(c.red(f"No reference encodings (.npy) found in {self.reference_encodings_folder}"))
            return

        match str(float(version)) :
            case "1.0":
                # -------------------------- #
                # Simple Iterative Approach
                # Distance Calculation : using Euclidean Distance
                # Approach : raw calculation of euclidean distance between target and reference encodings
                # -------------------------- #
                # Iterate through all reference encodings in the folder
                for reference_encoding_path in self.reference_encodings_folder.glob('*.npy'):
                    reference_encoding = np.load(reference_encoding_path)

                    # Compare the target encoding with the reference encoding
                    distance = np.linalg.norm(target_encoding - reference_encoding)
                    if distance < self.tolerance:  # You can adjust this threshold based on your needs
                        print(f"{c.green('Match found')}: {reference_encoding_path.name} with distance {c.I(c.U(distance))}")
                    else:
                        print(f"{c.red('No match')}: {reference_encoding_path.name} with distance {c.I(c.U(distance))}")
            case "1.1":
                # -------------------------- #
                # Simple Iterative Approach
                # Distance Calculation : using Cosine Similarity
                # Approach : raw calculation of cosine similarity between target and reference encodings
                # -------------------------- #
                # Iterate through all reference encodings in the folder
                for reference_encoding_path in self.reference_encodings_folder.glob('*.npy'):
                    reference_encoding = np.load(reference_encoding_path)

                    # Compute cosine similarity
                    cosine_similarity = np.dot(target_encoding, reference_encoding) / (np.linalg.norm(target_encoding) * np.linalg.norm(reference_encoding))
                    if cosine_similarity > self.tolerance:  # You can adjust this threshold based on your needs
                        print(f"{c.green('Match found')}: {reference_encoding_path.name} with cosine similarity {c.I(c.U(cosine_similarity))}")
                    else:
                        print(f"{c.red('No match')}: {reference_encoding_path.name} with cosine similarity {c.I(c.U(cosine_similarity))}")
            case "1.2":
                # -------------------------- #
                # Simple Iterative Approach
                # Distance Calculation : using Cosine Similarity
                # Approach : using deepface.verify to compare embeddings
                # -------------------------- #
                # Iterate through all reference encodings in the folder
                for reference_encoding_path in self.reference_encodings_folder.glob('*.npy'):
                    reference_encoding = np.load(reference_encoding_path)

                    # Use deepface.verify to compare embeddings
                    result = DeepFace.verify(target_encoding, reference_encoding, model_name=self.model, enforce_detection=False, distance_metric="cosine")
                    if result["verified"]:
                        print(f"{c.green('Match found')}: {reference_encoding_path.name} with distance {c.I(c.U(result['distance']))}")
                    else:
                        print(f"{c.red('No match')}: {reference_encoding_path.name} with distance {c.I(c.U(result['distance']))}")
            case "1.3":
                # -------------------------- #
                # Simple Iterative Approach
                # Distance Calculation : using Euclidean Distance with normalization
                # Approach : using deepface.verify to compare embeddings
                # -------------------------- #
                # euclidian distance with normalization : using deepface.verify with different distance metrics
                for reference_encoding_path in self.reference_encodings_folder.glob('*.npy'):
                    reference_encoding = np.load(reference_encoding_path)

                    # Use deepface.verify to compare embeddings
                    result = DeepFace.verify(target_encoding, reference_encoding, model_name=self.model, enforce_detection=False, distance_metric="euclidean")
                    if result["verified"]:
                        print(f"{c.green('Match found')}: {reference_encoding_path.name} with distance {c.I(c.U(result['distance']))}")
                    else:
                        print(f"{c.red('No match')}: {reference_encoding_path.name} with distance {c.I(c.U(result['distance']))}")
            case "2.0":
                # -------------------------- #
                # Vectorized batch Approach
                # Distance Calculation : using Cosine Similarity
                # Approach : raw calculation of cosine similarity between target and reference encodings
                # -------------------------- #
                # Load all reference encodings into a single numpy array
                reference_encodings = []
                reference_names = []
                for reference_encoding_path in self.reference_encodings_folder.glob('*.npy'):
                    reference_encodings.append(np.load(reference_encoding_path).flatten())
                    reference_names.append(reference_encoding_path.name) # this is done to reference the name of the file in the output
                
                reference_encodings = np.array(reference_encodings)
                # Compute cosine similarity for all reference encodings at once
                cosine_similarities = np.dot(reference_encodings, target_encoding) / (np.linalg.norm(reference_encodings, axis=1) * np.linalg.norm(target_encoding))

                for name, similarity in zip(reference_names, cosine_similarities):
                    if similarity > self.tolerance:  # You can adjust this threshold based on your needs
                        print(f"{Colors.GREEN.colorize('Match found')}: {name} with cosine similarity {c.I(c.U(similarity))}")
                    else:
                        print(f"{Colors.FAIL.colorize('No match')}: {name} with cosine similarity {c.I(c.U(similarity))}")

                best_match_index = np.argmax(cosine_similarities)
                best_match_name = reference_names[best_match_index]
                best_match_similarity = cosine_similarities[best_match_index]
                print(f"{c.B('Best match')}: {best_match_name} with cosine similarity {c.I(c.U(best_match_similarity))}")
            case _:
                raise NotImplementedError(f"Version {version} is not implemented. Please use '1.0' or '2.0'.")


FaceRecognitionMatching.__doc__ = "A class to handle face recognition matching using the face_recognition library."
DeepFaceMatching.__doc__ = "A class to handle face recognition matching using the deepface library."

class MatchingFactory(str, enum.Enum):
    """
    A factory class to create instances of face recognition matching classes based on the encoder type.
    """
    FACE_RECOGNITION: Final[str] = 'face_recognition'
    DEEPFACE: Final[str] = 'deepface'

    _DEFAULT_FACE_RECOGNITION_MODEL: Final[str] = FaceRecognitionModelOptions.HOG.value
    _DEFAULT_DEEPFACE_MODEL: Final[str] = DeepFaceModelOptions.FACENET512.value

    @staticmethod
    def create_matching(encoder: SupportedEncoders, reference_encodings_folder: pathlib.Path, tolerance: float = 0.6, deepface_model: DeepFaceModelOptions = None, face_recognition_model: FaceRecognitionModelOptions = None):
        if encoder == SupportedEncoders.FACE_RECOGNITION:
            return FaceRecognitionMatching(reference_encodings_folder, tolerance, model=face_recognition_model if face_recognition_model else MatchingFactory._DEFAULT_FACE_RECOGNITION_MODEL)
        elif encoder == SupportedEncoders.DEEPFACE:
            return DeepFaceMatching(reference_encodings_folder, tolerance, model=deepface_model if deepface_model else MatchingFactory._DEFAULT_DEEPFACE_MODEL)
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")

if __name__ == '__main__':

    class Defaults:
        """Default values for the cli args of matching script."""
        TOLERANCE: Final[float] = 0.6
        VERSION: Final[str] = "2.0"
        MATCHER: Final[str] = MatchingFactory.FACE_RECOGNITION.value
        DEEPFACE_MODEL: Final[str] = DeepFaceModelOptions.FACENET512.value
        FACE_RECOGNITION_MODEL: Final[str] = FaceRecognitionModelOptions.HOG.value

    import argparse
    parser = argparse.ArgumentParser(description='Match a target image against multiple reference encodings.')
    parser.add_argument('--target_image', type=str, required=True, help='Path to the target image file.')
    parser.add_argument('--reference_encodings', type=str, required=True, help='Path to the reference encodings folder.')
    parser.add_argument('--tolerance', type=float, default=Defaults.TOLERANCE, help='Closeness threshold . Weather using similarity or distance , it will act as threshold of tolerance for matching.')
    # version argument for matching
    parser.add_argument('--version', type=str, default=Defaults.VERSION, help='Version of the matching algorithm to use. Default is 2.0.')
    # matcher option argument for choosing the matching algorithm
    parser.add_argument('--matcher', type=str, choices=[e.value for e in MatchingFactory], default=Defaults.MATCHER, help='The matching algorithm to use. Default is face_recognition.')
    # model selection argument for deepface and face_recognition
    parser.add_argument('--deepface_model', type=str, choices=[m.value for m in DeepFaceModelOptions], default=Defaults.DEEPFACE_MODEL, help='The model to use for generating encodings. Default is Facenet.')
    parser.add_argument('--face_recognition_model', type=str, choices=[m.value for m in FaceRecognitionModelOptions], default=Defaults.FACE_RECOGNITION_MODEL, help='The model to use for generating encodings. Default is hog.')
    
    args = parser.parse_args()

    # print important cli values
    print(f"Target Image: {args.target_image}")
    print(f"Reference Encodings: {args.reference_encodings}")
    print(f"Tolerance: {args.tolerance}")
    print(f"Version: {args.version}")
    print(f"Matcher: {args.matcher}")
    print(f"Model: {args.deepface_model if args.matcher == SupportedEncoders.DEEPFACE.value else args.face_recognition_model}")
    print("Matching encodings...")

    matcher = MatchingFactory.create_matching(
        encoder=SupportedEncoders(args.matcher),
        reference_encodings_folder=pathlib.Path(args.reference_encodings),
        tolerance=args.tolerance,
        deepface_model=args.deepface_model if args.matcher == MatchingFactory.DEEPFACE.value else None,
        face_recognition_model=args.face_recognition_model if args.matcher == MatchingFactory.FACE_RECOGNITION.value else None
    )
    matcher.match(target_image_path=pathlib.Path(args.target_image), version=args.version)
