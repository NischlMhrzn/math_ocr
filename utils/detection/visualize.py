import cv2
import matplotlib.pyplot as plt


def plot_bbox(image, bbox):
    img = cv2.rectangle(
        image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255, 0, 0), thickness=1
    )
    plt.imshow(img)
    plt.show()
