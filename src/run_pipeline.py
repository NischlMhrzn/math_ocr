import argparse
import cv2
from src.detection.detect import MathDetector, ArgStub
import numpy as np
from src.process.extract_crops import get_bbox_crops, get_equation_removed
from src.text_ocr.main import ocr
from src.img2latex.main import call_model, initialize
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    print("Running Pipeline")
    parser = argparse.ArgumentParser(description="Use model", add_help=False)
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.333,
        help="Softmax sampling frequency",
    )
    parser.add_argument(
        "-c", "--config", type=str, default="config/img2latex_config.yaml"
    )
    parser.add_argument(
        "-m", "--checkpoint", type=str, default="models/img2latex/weights.pth"
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Show the rendered predicted latex code",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="Predict LaTeX code from image file instead of clipboard",
    )
    parser.add_argument(
        "-k",
        "--katex",
        action="store_true",
        help="Render the latex code in the browser",
    )
    parser.add_argument("--no-cuda", action="store_true", help="Compute on CPU")
    parser.add_argument(
        "--no-resize", action="store_true", help="Resize the image beforehand"
    )
    parser.add_argument(
        "--img_path", type=str, default=None, help="path of image for OCR"
    )
    arguments = parser.parse_args()

    args, *objs = initialize(arguments)

    md = MathDetector("./models/AMATH512_e1GTDB.pth", ArgStub())
    image = cv2.imread(arguments.img_path, cv2.IMREAD_COLOR)
    bbox, scores = md.DetectAny(0.4, np.array(image))

    crops = get_bbox_crops(image, bbox[0])
    text_img = get_equation_removed(image, bbox[0])
    plt.imshow(text_img)
    plt.show()
    print(ocr(text_img))

    for crop in crops:
        img = Image.fromarray(crop)
        plt.imshow(img)
        plt.show()
        pred = call_model(args, *objs, img=img)
        print(pred)
