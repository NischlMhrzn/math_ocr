import argparse
import cv2
from src.detection.detect import MathDetector, ArgStub
import numpy as np
from src.process.extract_crops import (
    get_bbox_crops,
    get_equation_removed,
    pad_images,
    pad_bbox2_img,
)
from src.text_ocr.main import ocr
from src.img2latex.main import call_model, initialize
from src.process.process_bbox import process_bboxes
from src.ranking.main import rank
import matplotlib.pyplot as plt
from PIL import Image
from src.equation_postprocess.latex2sympy import rank_equation
from sympy.parsing.latex import parse_latex

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
    parser.add_argument(
        "--detect_thresh",
        type=str,
        default=0.2,
        help="threshold on scores for detecting bounding box for mathematical equation",
    )
    parser.add_argument("--csv_path", type=str, default=None, help="path to the csv")
    arguments = parser.parse_args()

    args, *objs = initialize(arguments)

    md = MathDetector("./models/AMATH512_e1GTDB.pth", ArgStub())
    image = cv2.imread(arguments.img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pad_image, ratio = pad_images(image, 512)
    plt.imshow(pad_image)
    plt.show()
    pad_bbox, scores = md.DetectAny(arguments.detect_thresh, np.array(pad_image))
    img_bbox = pad_bbox2_img(pad_bbox[0], ratio)
    # print("Bboxes:", img_bbox)
    # print("Scores:", scores)
    processed_bboxes = process_bboxes(image, img_bbox)
    # print(processed_bboxes)
    crops = get_bbox_crops(image, processed_bboxes)
    text_img = get_equation_removed(image, processed_bboxes)
    plt.imshow(text_img)
    plt.show()
    ocr_text = " ".join(ocr(text_img))
    print(ocr_text)
    print(rank([ocr_text], arguments.csv_path))

    for crop in crops:
        plt.imshow(crop)
        plt.show()
        img = Image.fromarray(np.array(crop))
        pred = call_model(args, *objs, img=img)
        print(pred)
        rank_equation(pred)
        
