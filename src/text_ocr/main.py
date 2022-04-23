import pytesseract
import cv2
import argparse


def ocr(image):
    # configurations
    config = "-l eng --oem 1 --psm 6 tessconfig"
    # pytessercat
    text = pytesseract.image_to_string(image, config=config)
    # print text
    text = text.split("\n")
    return text[:-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path", type=str, default=None, help="path of image for OCR"
    )
    args = parser.parse_args()
    print(ocr(args.img_path))
