"""
OCR on selected sections images using the EasyOCR library.
"""
import sys
import cv2
import easyocr
import pathlib

sys.path.append("/home/ubuntu/Documents/personal/computer-vision-learning/text_related")
from utilities.colored import cprint, Styleit, Palette
from utilities.profiler import easy_profiler

@easy_profiler(num_runs=3) # Comment this line to disable profiling
def extract_bottom_left_name(image_path,
    confidence_threshold=0.5,
    bottom_crop_ratio=0.15,
    left_crop_ratio=0.30,
    save_to_filename=None,
    debug_mode=False
    ):
    """

    i want to draw the roi on the image to see what part of the image is being cropped. 
    The roi is the bottom left corner of the image. The roi is defined by the following coordinates:

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
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image file not found.")

    height, width, _ = img.shape

    # 2. Crop the bottom-left corner
    # Height: bottom 15% of the image (0.85 to 1.0)
    # Width: left 30% of the image (0.0 to 0.3)

    required_height_proportion = 1 - bottom_crop_ratio  # 0.85
    required_width_proportion = left_crop_ratio  # 0.30
    crop_y1 = int(height * required_height_proportion)
    crop_y2 = height
    crop_x1 = 0
    crop_x2 = int(width * required_width_proportion)

    roi = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # --- SAVE THE ROI TO A FILE ---
    if save_to_filename:
        cv2.imwrite(save_to_filename, roi)
        print(f"Saved cropped ROI to: {save_to_filename}").style(Palette.YELLOW, B=True)

    # 3. Optional: Pre-process image to boost contrast
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 4. Run EasyOCR on the cropped region
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(gray)

    # 5. Process extracted text
    names = []
    for i, (bbox, text, confidence) in enumerate(results):

        # print(f"Detected text: {text} with confidence: {confidence}")
        cleaned = text.replace("(You)", "").strip()
        if debug_mode:
            print(f"Result {i}:").style(Palette.YELLOW, B=True)
            print(f"\t - Detected text: {Palette.BOLD.colorize(text)} with confidence: {Palette.BOLD.colorize(str(confidence))}").style(Palette.YELLOW, I=True)
            print(f"\t - Cleaned text: {cleaned}").style(Palette.YELLOW, I=True)
            print(f"\t - Bounding box: {bbox}").style(Palette.YELLOW, I=True)
            print(f"\t - Image dimensions: {width}x{height}").style(Palette.YELLOW, I=True)
        
        if confidence > confidence_threshold and len(cleaned) > 1:
            names.append(cleaned)
            

    return names


if __name__ == "__main__":

    print = cprint  # Override print to use colored output
    __DEFAULT_IMAGE__ = pathlib.Path(__file__).parent / "test_data" / "image01.png"

    # Example usage

    # Input : image path
    image_path: str = input(Styleit(Palette.BLACK, B=True) >> "Enter the path to your image : ")
    image_path = image_path.strip() or str(__DEFAULT_IMAGE__)
    image_path = pathlib.Path(image_path)
    if not image_path.exists():
        print(f"Error: Selected file {Palette.ITALIC.colorize(str(image_path))} does not exist.").style(Palette.RED, B=True)
        sys.exit(0)
    else:
        print(f"Selected Image: {Palette.ITALIC.colorize(str(image_path))}").style(Palette.GREEN)

    # Input : debug mode
    debug_mode: bool = input(Styleit(Palette.BLACK, B=True) >> "Enable debug mode? (y/n): ").strip().lower() in ('y', 'yes', str(1))
    print(f"Debug mode is {'enabled' if debug_mode else 'disabled'}").style(Palette.GREEN)

    # Input : bottom width and height percentages
    __DEFAULT_BOTTOM_WIDTH_PERCENT__ = 30
    __DEFAULT_BOTTOM_HEIGHT_PERCENT__ = 15
    bottom_width_percent: float = float(input(Styleit(Palette.BLACK, B=True) >> "Enter bottom width percentage (e.g., 30 for 30%): ") or __DEFAULT_BOTTOM_WIDTH_PERCENT__)
    print(f"Set bottom width percentage is {bottom_width_percent}%").style(Palette.GREEN)
    bottom_height_percent: float = float(input(Styleit(Palette.BLACK, B=True) >> "Enter bottom height percentage (e.g., 15 for 15%): ") or __DEFAULT_BOTTOM_HEIGHT_PERCENT__)
    print(f"Set bottom height percentage is {bottom_height_percent}%").style(Palette.GREEN)

    # Input : confidence threshold
    __DEFAULT_CONFIDENCE_THRESHOLD__ = 0.7
    confidence_threshold: float = float(input(Styleit(Palette.BLACK, B=True) >> "Enter confidence threshold (e.g., 0.7 for 70%): ") or __DEFAULT_CONFIDENCE_THRESHOLD__)
    print(f"Set confidence threshold is {confidence_threshold}").style(Palette.GREEN)

    # Input : save to filename
    save_to_filename: str = input(Styleit(Palette.BLACK, B=True) >> "Enter filename to save cropped ROI (or leave blank to skip saving): ").strip()
    if not save_to_filename:
        save_to_filename = None
    
    image_name: str = image_path.stem
    image_extension: str = image_path.suffix

    # Execute the OCR extraction
    print(extract_bottom_left_name(
        image_path=image_path,
        confidence_threshold=0.7,
        bottom_crop_ratio=bottom_height_percent / 100,
        left_crop_ratio=bottom_width_percent / 100,
        save_to_filename=f"roi_{image_name}{image_extension}",
        debug_mode=debug_mode
    )).style(Palette.CYAN, B=True)
