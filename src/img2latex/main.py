from src.img2latex.dataset import test_transform
import cv2
import pandas.io.clipboard as clipboard
from PIL import ImageGrab
from PIL import Image
import os
import sys
import argparse
import logging
import yaml
import re

import numpy as np
import torch
from torchvision import transforms
from munch import Munch
from transformers import PreTrainedTokenizerFast
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame

# from dataset.latex2png import tex2pil
from src.img2latex.model import get_model
from utils.img2latex.utils import *
from src.img2latex.get_latest_checkpoint import download_checkpoints

last_pic = None


def minmax_size(img, max_dimensions=None, min_dimensions=None):
    if max_dimensions is not None:
        ratios = [a / b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size) // max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        if any([s < min_dimensions[i] for i, s in enumerate(img.size)]):
            padded_im = Image.new("L", min_dimensions, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img


def initialize(arguments=None):
    if arguments is None:
        arguments = Munch(
            {
                "config": "config/img2latex_config.yaml",
                "checkpoint": "models/img2latex/weights.pth",
                "no_cuda": True,
                "no_resize": False,
            }
        )
    logging.getLogger().setLevel(logging.FATAL)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    with open(arguments.config, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params))
    args.update(**vars(arguments))
    # args.wandb = False
    # args.device = "cpu"
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    if not os.path.exists(args.checkpoint):
        download_checkpoints()
    model = get_model(args)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    if (
        "image_resizer.pth" in os.listdir(os.path.dirname(args.checkpoint))
        and not arguments.no_resize
    ):
        image_resizer = ResNetV2(
            layers=[2, 3, 3],
            num_classes=max(args.max_dimensions) // 32,
            global_pool="avg",
            in_chans=1,
            drop_rate=0.05,
            preact=True,
            stem_type="same",
            conv_layer=StdConv2dSame,
        ).to(args.device)
        image_resizer.load_state_dict(
            torch.load(
                os.path.join(os.path.dirname(args.checkpoint), "image_resizer.pth"),
                map_location=args.device,
            )
        )
        image_resizer.eval()
    else:
        image_resizer = None
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    return args, model, image_resizer, tokenizer


def call_model(args, model, image_resizer, tokenizer, img=None):
    global last_pic
    encoder, decoder = model.encoder, model.decoder
    if type(img) is bool:
        img = None
    if img is None:
        if last_pic is None:
            print("Provide an image.")
            return ""
        else:
            img = last_pic.copy()
    else:
        last_pic = img.copy()
    img = minmax_size(pad(img), args.max_dimensions, args.min_dimensions)
    if image_resizer is not None and not args.no_resize:
        with torch.no_grad():
            input_image = img.convert("RGB").copy()
            r, w, h = 1, input_image.size[0], input_image.size[1]
            for _ in range(10):
                h = int(h * r)  # height to resize
                img = pad(
                    minmax_size(
                        input_image.resize(
                            (w, h), Image.BILINEAR if r > 1 else Image.LANCZOS
                        ),
                        args.max_dimensions,
                        args.min_dimensions,
                    )
                )
                t = test_transform(image=np.array(img.convert("RGB")))["image"][
                    :1
                ].unsqueeze(0)
                w = (image_resizer(t.to(args.device)).argmax(-1).item() + 1) * 32
                logging.info(r, img.size, (w, int(input_image.size[1] * r)))
                if w == img.size[0]:
                    break
                r = w / img.size[0]
    else:
        img = np.array(pad(img).convert("RGB"))
        t = test_transform(image=img)["image"][:1].unsqueeze(0)
    im = t.to(args.device)

    with torch.no_grad():
        model.eval()
        device = args.device
        encoded = encoder(im.to(device))
        dec = decoder.generate(
            torch.LongTensor([args.bos_token])[:, None].to(device),
            args.max_seq_len,
            eos_token=args.eos_token,
            context=encoded.detach(),
            temperature=args.get("temperature", 0.25),
        )
        pred = post_process(token2str(dec, tokenizer)[0])
    try:
        clipboard.copy(pred)
    except:
        pass
    return pred


if __name__ == "__main__":
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
    arguments = parser.parse_args()

    args, *objs = initialize(arguments)

    if args.file:
        img = Image.open(args.file)
    pred = call_model(args, *objs, img=img)
    print(pred)
