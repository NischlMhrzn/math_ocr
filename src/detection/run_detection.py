import torch
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import shutil
from math import sqrt as sqrt
from itertools import product as product
import argparse
from src.detection.detect import MathDetector, ArgStub
from src.process.extract_crops import pad_images


def parse_args():
    parser = argparse.ArgumentParser(description="Single Shot MultiBox Detection")
    parser.add_argument(
        "--trained_model",
        default="weights/ssd300_GTDB_990.pth",
        type=str,
        help="Trained state_dict file path to open",
    )
    parser.add_argument(
        "--save_folder", default="eval/", type=str, help="Dir to save results"
    )
    parser.add_argument(
        "--visual_threshold", default=0.6, type=float, help="Final confidence threshold"
    )
    parser.add_argument(
        "--cuda", default=False, type=bool, help="Use cuda to train model"
    )
    parser.add_argument(
        "--dataset_root", default="GTDB_ROOT", help="Location of VOC root directory"
    )
    parser.add_argument("--test_data", default="testing_data", help="testing data file")
    parser.add_argument("--verbose", default=False, type=bool, help="plot output")
    parser.add_argument(
        "--suffix",
        default="_10",
        type=str,
        help="suffix of directory of images for testing",
    )
    parser.add_argument(
        "--exp_name",
        default="SSD",
        help="Name of the experiment. Will be used to generate output",
    )
    parser.add_argument(
        "--model_type",
        default=512,
        type=int,
        help="Type of ssd model, ssd300 or ssd512",
    )
    parser.add_argument(
        "--use_char_info",
        default=False,
        type=bool,
        help="Whether or not to use char info",
    )
    parser.add_argument(
        "--limit", default=-1, type=int, help="limit on number of test examples"
    )
    parser.add_argument(
        "--cfg",
        default="hboxes512",
        type=str,
        help="Type of network: either gtdb or math_gtdb_512",
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers used in data loading",
    )
    parser.add_argument(
        "--kernel",
        default="3 3",
        type=str,
        nargs="+",
        help="Kernel size for feature layers: 3 3 or 1 5",
    )
    parser.add_argument(
        "--padding",
        default="1 1",
        type=str,
        nargs="+",
        help="Padding for feature layers: 1 1 or 0 2",
    )
    parser.add_argument(
        "--neg_mining",
        default=True,
        type=bool,
        help="Whether or not to use hard negative mining with ratio 1:3",
    )
    parser.add_argument(
        "--log_dir", default="logs", type=str, help="dir to save the logs"
    )
    parser.add_argument(
        "--stride", default=0.1, type=float, help="Stride to use for sliding window"
    )
    parser.add_argument("--window", default=1200, type=int, help="Sliding window size")
    parser.add_argument(
        "--img_path", default=None, type=str, help="image path to infer on"
    )

    parser.add_argument(
        "-f",
        default=None,
        type=str,
        help="Dummy arg so we can load in Jupyter Notebooks",
    )

    args = parser.parse_args()

    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if os.path.exists(os.path.join(args.save_folder, args.exp_name)):
        shutil.rmtree(os.path.join(args.save_folder, args.exp_name))

    return args


def get_freer_gpu():
    """
    Find which gpu is free
    """
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return int(np.argmax(memory_available))


if __name__ == "__main__":
    args = parse_args()

    gpu_id = 0
    if args.cuda:
        gpu_id = get_freer_gpu()
        torch.cuda.set_device(gpu_id)

    md = MathDetector("./models/AMATH512_e1GTDB.pth", ArgStub())
    image = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
    pad_image, ratio = pad_images(image, 512)
    plt.imshow(pad_image)
    plt.show()
    pad_bbox, scores = md.DetectAny(0.2, np.array(pad_image))
    print("Bboxes:", pad_bbox)
    print("Scores:", scores)
