import sys
import cv2
import easyocr
import pathlib

sys.path.append("/home/ubuntu/Documents/personal/computer-vision-learning/text_related")
from utilities.colored import cprint, Styleit, Palette

def extract_names_from_image(image_path, language='en', confidence_threshold=0.5, debug=False):
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    reader = easyocr.Reader([language], gpu=False)
    results = reader.readtext(img)
    
    names = []
    for i, (bbox, text, conf) in enumerate(results):
        if debug:
            print(f"Result {i}:").style(Palette.YELLOW, B=True)
            print(f"\t - Detected text: {Palette.BOLD.colorize(text)} with confidence: {Palette.BOLD.colorize(str(conf))}").style(Palette.YELLOW, I=True)
            print(f"\t - Bounding box: {bbox}").style(Palette.YELLOW, I=True)
            print(f"\t - Image dimensions: {width}x{height}").style(Palette.YELLOW, I=True)
        # Google Meet names are usually mixed-case words (e.g., "Jane Doe")
        if conf > confidence_threshold and any(c.isalpha() for c in text):
            names.append(text)
            
    return names

# Entry point for testing the function
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

    # Execute the OCR extraction
    print("\nStarting OCR extraction...\n").style(Palette.CYAN, B=True)
    names = extract_names_from_image(image_path, debug=debug_mode)
    print("Extracted names:", names).style(Palette.CYAN, B=True)
