import cv2
import logging
import dataclasses
import functools
import easyocr
import functions_framework
from flask import jsonify, request, redirect
from typing import List, NewType, Callable, Tuple, Optional, Final
import warnings
from environs import Env

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ======================
# OCR Extraction Engine
# ======================
CV2Image = NewType('CV2Image', 'numpy.ndarray')

def _ocr_extract_text(image: CV2Image, reader: easyocr.Reader, confidence_threshold: float, calculate_roi_callback: Optional[Callable[[CV2Image,], CV2Image]]=None, text_cleaning_callback: Callable[[str], str] = lambda x: x) -> List[str]:
    """

    i want to draw the roi on the image to see what part of the image is being cropped. The roi is the bottom left corner of the image. The roi is defined by the following coordinates:

    (0, 0)
    ________________________________________________________________
    |                                                              |
    |                                                              |
    |                                                              |
    |                                                              |
    ----------------------------------------------------------------
                                                                    (width * left_crop_ratio, height * (1 - bottom_crop_ratio))
    """
    # 1. Read the screenshot

    height, width, _ = image.shape

    roi = image  # If no callback is provided, use the entire image

    # 2. Crop the ROI using the provided callback function
    if calculate_roi_callback is not None:
        roi = calculate_roi_callback(image)
    

    # --- SAVE THE ROI TO A FILE ---
    # cv2.imwrite("cropped_roi.png", roi)
    # print(f"Saved cropped ROI to: cropped_roi.png")

    # 3. Optional: Pre-process image to boost contrast
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 4. Run EasyOCR on the cropped region
    results = reader.readtext(gray)

    # 5. Process extracted text
    names = []
    for bbox, text, confidence in results:

        # print(f"Detected text: {text} with confidence: {confidence}")
        cleaned = text_cleaning_callback(text)
        if confidence > confidence_threshold and len(cleaned) > 1:
            names.append(cleaned)

    return names

__DEFAULT_CONFIDENCE_THRESHOLD__ = 0.5
__DEFAULT_BOTTOM_CROP_RATIO__ = 0.15
__DEFAULT_LEFT_CROP_RATIO__ = 0.30
__DEFAULT_TOP_CROP_RATIO__ = 0.15
__DEFAULT_RIGHT_CROP_RATIO__ = 0.30

class OCRService:

    @dataclasses.dataclass(frozen=True)
    class OCROptions:
        use_gpu: bool = False
        confidence_threshold: float = __DEFAULT_CONFIDENCE_THRESHOLD__
        bottom_crop_ratio: float = __DEFAULT_BOTTOM_CROP_RATIO__
        left_crop_ratio: float = __DEFAULT_LEFT_CROP_RATIO__
        right_crop_ratio: float = __DEFAULT_RIGHT_CROP_RATIO__
        top_crop_ratio: float = __DEFAULT_TOP_CROP_RATIO__
        text_cleaning_callback: Callable[[str], str] = lambda x: x
        calculate_roi_callback: Optional[Callable[[CV2Image,], CV2Image]] = None

    def __init__(self, ocr_engine: str, options: OCROptions, logger: logging.Logger = None, logger_level=logging.INFO):

        self.ocr_engine = ocr_engine
        if ocr_engine == "easyocr":
            self.reader = easyocr.Reader(['en'], gpu=options.use_gpu)
        else:
            raise ValueError(f"Unsupported OCR engine: {ocr_engine}")
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logger_level)
        else:
            self.logger = logger
        self.options = options
        self.logger.info("EasyOCR reader initialized.")
        self.logger.info(f"Using GPU: {options.use_gpu}")
        self.logger.info(f"OCR Confidence threshold: {options.confidence_threshold}")
        self.logger.info(f"OCR Bottom crop ratio: {options.bottom_crop_ratio}")
        self.logger.info(f"Left crop ratio: {options.left_crop_ratio}")


    def perform_ocr(self, image: CV2Image, extract_type: str, **kwargs) -> list:
        """Perform OCR on the provided image and return extracted text."""
        if extract_type == "bottom_left":
            # Set default values if not provided
            if "confidence_threshold" not in kwargs:
                kwargs.setdefault('confidence_threshold', self.options.confidence_threshold)
            if "bottom_crop_ratio" not in kwargs:
                kwargs.setdefault('bottom_crop_ratio', self.options.bottom_crop_ratio)
            if "left_crop_ratio" not in kwargs:
                kwargs.setdefault('left_crop_ratio', self.options.left_crop_ratio)
            self.logger.debug(f"Performing OCR with parameters: {kwargs}")
            return self.extract_bottom_left_name(image, OCRService.OCROptions(**kwargs))
        elif extract_type == "bottom_right":
            if "confidence_threshold" not in kwargs:
                kwargs.setdefault('confidence_threshold', self.options.confidence_threshold)
            if "bottom_crop_ratio" not in kwargs:
                kwargs.setdefault('bottom_crop_ratio', self.options.bottom_crop_ratio)
            if "right_crop_ratio" not in kwargs:
                kwargs.setdefault('right_crop_ratio', self.options.right_crop_ratio)
            self.logger.debug(f"Performing OCR with parameters: {kwargs}")
            return self.extract_bottom_right_name(image, OCRService.OCROptions(**kwargs))
        elif extract_type == "top_left":
            if "confidence_threshold" not in kwargs:
                kwargs.setdefault('confidence_threshold', self.options.confidence_threshold)
            if "top_crop_ratio" not in kwargs:
                kwargs.setdefault('top_crop_ratio', self.options.top_crop_ratio)
            if "left_crop_ratio" not in kwargs:
                kwargs.setdefault('left_crop_ratio', self.options.left_crop_ratio)
            self.logger.debug(f"Performing OCR with parameters: {kwargs}")
            return self.extract_top_left_name(image, OCRService.OCROptions(**kwargs))
        elif extract_type == "top_right":
            if "confidence_threshold" not in kwargs:
                kwargs.setdefault('confidence_threshold', self.options.confidence_threshold)
            if "top_crop_ratio" not in kwargs:
                kwargs.setdefault('top_crop_ratio', self.options.top_crop_ratio)
            if "right_crop_ratio" not in kwargs:
                kwargs.setdefault('right_crop_ratio', self.options.right_crop_ratio)
            self.logger.debug(f"Performing OCR with parameters: {kwargs}")
            return self.extract_top_right_name(image, OCRService.OCROptions(**kwargs))
        else:
            self.logger.error(f"Unsupported extract_type: {extract_type}")
            raise ValueError(f"Unsupported extract_type: {extract_type}")

    def extract_bottom_left_name(self, image: CV2Image, options: OCROptions) -> List[str]:
        if self.ocr_engine == "easyocr":
            return _ocr_extract_text(image,
                self.reader, options.confidence_threshold,
                calculate_roi_callback=options.calculate_roi_callback if options.calculate_roi_callback else functools.partial(self._calculate_roi, extract_type="bottom_left"),
                text_cleaning_callback=options.text_cleaning_callback
            )
        raise ValueError(f"Unsupported OCR engine: {self.ocr_engine}")

    def extract_bottom_right_name(self, image: CV2Image, options: OCROptions) -> List[str]:
        if self.ocr_engine == "easyocr":
            return _ocr_extract_text(image,
                self.reader, options.confidence_threshold,
                calculate_roi_callback=options.calculate_roi_callback if options.calculate_roi_callback else functools.partial(self._calculate_roi, extract_type="bottom_right"),
                text_cleaning_callback=options.text_cleaning_callback
            )
        raise ValueError(f"Unsupported OCR engine: {self.ocr_engine}")
    
    def extract_top_left_name(self, image: CV2Image, options: OCROptions) -> List[str]:
        if self.ocr_engine == "easyocr":
            return _ocr_extract_text(image,
                self.reader, options.confidence_threshold,
                calculate_roi_callback=options.calculate_roi_callback if options.calculate_roi_callback else functools.partial(self._calculate_roi, extract_type="top_left"),
                text_cleaning_callback=options.text_cleaning_callback
            )
        raise ValueError(f"Unsupported OCR engine: {self.ocr_engine}")
    
    def extract_top_right_name(self, image: CV2Image, options: OCROptions) -> List[str]:
        if self.ocr_engine == "easyocr":
            return _ocr_extract_text(image,
                self.reader, options.confidence_threshold,
                calculate_roi_callback=options.calculate_roi_callback if options.calculate_roi_callback else functools.partial(self._calculate_roi, extract_type="top_right"),
                text_cleaning_callback=options.text_cleaning_callback
            )
        raise ValueError(f"Unsupported OCR engine: {self.ocr_engine}")
    
    def _calculate_roi(self, image: CV2Image, extract_type: str) -> CV2Image:
        """Calculate the region of interest (ROI) for the ocr extraction.
        The ROI is defined by the following coordinates:
        (0, 0)
        ________________________________________________________________
        |                                                              |
        |                                                              |
        |                                                              |
        |                                                              |
        ----------------------------------------------------------------
                                                                    (width * left_crop_ratio, height * (1 - bottom_crop_ratio))
        """

        height, width, _ = image.shape
        
        if extract_type == "bottom_left":
            required_height_proportion = 1 - self.options.bottom_crop_ratio
            required_width_proportion = self.options.left_crop_ratio
            crop_y1 = int(height * required_height_proportion)
            crop_y2 = height
            crop_x1 = 0
            crop_x2 = int(width * required_width_proportion)
        elif extract_type == "bottom_right":
            required_height_proportion = 1 - self.options.bottom_crop_ratio
            required_width_proportion = 1 - self.options.left_crop_ratio
            crop_y1 = int(height * required_height_proportion)
            crop_y2 = height
            crop_x1 = int(width * required_width_proportion)
            crop_x2 = width
        elif extract_type == "top_left":
            required_height_proportion = self.options.bottom_crop_ratio
            required_width_proportion = self.options.left_crop_ratio
            crop_y1 = 0
            crop_y2 = int(height * required_height_proportion)
            crop_x1 = 0
            crop_x2 = int(width * required_width_proportion)
        elif extract_type == "top_right":
            required_height_proportion = self.options.bottom_crop_ratio
            required_width_proportion = 1 - self.options.left_crop_ratio
            crop_y1 = 0
            crop_y2 = int(height * required_height_proportion)
            crop_x1 = int(width * required_width_proportion)
            crop_x2 = width
        else:
            raise ValueError(f"Unsupported extract_type: {extract_type}")
        return image[crop_y1:crop_y2, crop_x1:crop_x2]

# ===========================================


# ===============
# Utilities Class
# ===============

class Utilities:

    @staticmethod
    def fetch_image(image_url: str) -> CV2Image:
        """Fetch the image from the provided URL and return the local path."""
        # Implement logic to download the image from the URL and save it locally
        # Return the local file path of the downloaded image
        # TODO: add logic for bucket image download
        img = cv2.imread(image_url)
        if img is None:
            raise FileNotFoundError("Image file not found.")
        return img  # Return the image as a numpy array

# ===========================================


# ===========================================
# Candidate Name Extractor Class
# - Extracts candidate names from frames using OCR.
# ===========================================
class CandidateNameExtractor:
    """Class to extract candidate names from images using OCR.
    
    args:
        ocr_processing_confidence_threshold: float - The confidence threshold for OCR processing.
        frame_bottom_percentage: float - The percentage of the bottom of the frame to consider for OCR.
        frame_left_percentage: float - The percentage of the left of the frame to consider for OCR.
        utilities: Utilities - An instance of the Utilities class for image fetching and processing.
    """

    def __init__(self,
        ocr_processing_confidence_threshold: float,
        frame_bottom_percentage: float,
        frame_left_percentage: float,
        utilities: Utilities,
        gpu_enabled: bool = False,
        logger: logging.Logger = None,
        logger_level=logging.INFO
        ):
        self._ocr_service = OCRService(
            ocr_engine="easyocr",
            options=OCRService.OCROptions(
                use_gpu=gpu_enabled,
                confidence_threshold=ocr_processing_confidence_threshold,
                bottom_crop_ratio=frame_bottom_percentage/100.0,
                left_crop_ratio=frame_left_percentage/100.0,
                text_cleaning_callback=lambda text: text.replace("(You)", "").strip(),
                calculate_roi_callback=None,
            ),
            logger_level=logger_level
        )
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logger_level)
        else:
            self.logger = logger
        self._extract_type = "bottom_left"
        self._utilities = utilities
        self._text_cleaning_callback = lambda text: text.replace("(You)", "").strip()

    def extract(self, image_url: str) -> List[str]:
        """Extract candidate names from the image at the provided URL."""
        # Fetch the image from the URL
        image_path = self._utilities.fetch_image(image_url)

        # Perform OCR on the fetched image
        extracted_names: List[str] = self._ocr_service.perform_ocr(image_path,
            extract_type=self._extract_type,
        )

        # return the extracted names
        return extracted_names

# ===========================================

# =============
# ENV VARIABLES
# =============

# Initialize the environment reader
env = Env()
env.read_env() # Reads .env file in the current directory
# Set default values for environment variables if they are not set
DEBUG: Final[bool] = env.bool("DEBUG", False)                       # Optional
__default_log_level__ = "DEBUG" if DEBUG else "INFO"                # Optional
LOG_LEVEL: Final[str] = env.str("LOG_LEVEL", __default_log_level__) # Optional
GPU_ENABLED: Final[bool] = env.bool("GPU_ENABLED")                  # Required
ENV: Final[str] = env.str("ENV")                                    # Required
ECI__OCR_CONFIDENCE_THRESHOLD: Final[float] = env.float("ECI__OCR_CONFIDENCE_THRESHOLD")    # Required
ECI__FRAME_BOTTOM_PERCENTAGE: Final[float] = env.float("ECI__FRAME_BOTTOM_PERCENTAGE")      # Required
ECI__FRAME_LEFT_PERCENTAGE: Final[float] = env.float("ECI__FRAME_LEFT_PERCENTAGE")          # Required
__supported_envs__ = ('dev', 'qa', 'stage', 'prod', )

logger.setLevel(LOG_LEVEL)

def redirect_to_https():
    if request.is_secure:
        return # Already on HTTPS, do nothing
    
    # Check for X-Forwarded-Proto header if behind a proxy/load balancer
    # This header indicates the original protocol of the request
    if request.headers.get('X-Forwarded-Proto') == 'http':
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301) # Use 301 for permanent redirect
    
    # Fallback for direct HTTP requests without X-Forwarded-Proto
    if request.url.startswith('http://'):
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)
    
# Entry point for the Cloud Function
@functions_framework.http
def ocr(request):
    """HTTP Cloud Function that performs OCR on an image."""

    if ENV in __supported_envs__[2:]:
        value = redirect_to_https()
        if value:
            return value
        
    # Parse data from GET query parameters or JSON POST body
    request_json = request.get_json(silent=True)
    image_url = request.args.get('image_url') or (request_json.get('image_url') if request_json else None)
    print(f"Received image URL: {image_url}", image_url=='x.png')
    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    ocr_confidence_threshold = float(request.args.get('ocr_confidence_threshold', ECI__OCR_CONFIDENCE_THRESHOLD))
    frame_bottom_percentage = float(request.args.get('frame_bottom_percentage', ECI__FRAME_BOTTOM_PERCENTAGE))
    frame_left_percentage = float(request.args.get('frame_left_percentage', ECI__FRAME_LEFT_PERCENTAGE))

    if not GPU_ENABLED:
        # Silence the specific PyTorch pin_memory warning
        warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")

    candidate_name_extractor: CandidateNameExtractor = CandidateNameExtractor(
        ocr_processing_confidence_threshold=ocr_confidence_threshold,
        frame_bottom_percentage=frame_bottom_percentage,
        frame_left_percentage=frame_left_percentage,
        utilities=Utilities(),
        gpu_enabled=GPU_ENABLED,
        logger_level=LOG_LEVEL
    )

    extracted_text = candidate_name_extractor.extract(image_url)

    # Return a JSON response
    return jsonify({"extracted_text": extracted_text}), 200


