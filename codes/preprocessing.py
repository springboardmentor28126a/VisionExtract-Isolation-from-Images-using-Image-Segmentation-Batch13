import cv2
import numpy as np
import random

IMG_SIZE = 256

def resize(image, mask):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return image, mask


def normalize(image):
    image = image / 255.0
    return image.astype(np.float32)


def horizontal_flip(image, mask):
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask


def random_brightness(image):
    if random.random() > 0.5:
        factor = 1.0 + random.uniform(-0.2, 0.2)
        image = np.clip(image * factor, 0, 255)
    return image


def convert_to_binary(mask):
    mask = (mask > 0).astype(np.uint8)
    return mask


def preprocess(image, mask, augment=False):

    if augment:
        image, mask = horizontal_flip(image, mask)
        image = random_brightness(image)

    image, mask = resize(image, mask)

    mask = convert_to_binary(mask)
    image = normalize(image)

    mask = np.expand_dims(mask, axis=-1)

    return image, mask