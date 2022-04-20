import torch.nn as nn
from src.detection.ssd import SSD


base = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "C",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
]


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "C":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers


def add_extras(cfg, size, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False

    extras = cfg["extras"][str(size)]

    for k, v in enumerate(extras):
        if in_channels != "S":
            if v == "S":
                layers += [
                    nn.Conv2d(
                        in_channels,
                        extras[k + 1],
                        kernel_size=(1, 3)[flag],
                        stride=2,
                        padding=1,
                    )
                ]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v

    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))

    return layers


def multibox(args, vgg, extra_layers, cfg, size, num_classes):
    loc_layers = []
    conf_layers = []

    vgg_source = [21, -2]

    for k, v in enumerate(vgg_source):
        loc_layers += [
            nn.Conv2d(
                vgg[v].out_channels, cfg[k] * 4, kernel_size=(1, 5), padding=(0, 2)
            )
        ]
        conf_layers += [
            nn.Conv2d(
                vgg[v].out_channels,
                cfg[k] * num_classes,
                kernel_size=(1, 5),
                padding=(0, 2),
            )
        ]

    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [
            nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=(1, 5), padding=(0, 2))
        ]
        conf_layers += [
            nn.Conv2d(
                v.out_channels, cfg[k] * num_classes, kernel_size=(1, 5), padding=(0, 2)
            )
        ]

    return vgg, extra_layers, (loc_layers, conf_layers)


def build_ssd(args, phase, cfg, gpu_id, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return

    base_, extras_, head_ = multibox(
        args,
        vgg(base, 3, False),
        add_extras(cfg, size, 1024),
        cfg["mbox"][str(size)],
        size,
        num_classes,
    )

    return SSD(args, phase, cfg, size, base_, extras_, head_, num_classes, gpu_id)
