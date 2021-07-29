from PIL import Image
import imutils
import numpy as np
import cv2


def process_image(image_array, x_image, y_image, keep_aspect=False):
    if keep_aspect:
        return imutils.resize(image_array, width=y_image)
    return cv2.cvtColor(cv2.resize(image_array, (x_image, y_image)), cv2.COLOR_BGR2RGB)


def crop_face_from_frame(frame, box):
    x, y, x_, y_ = box
    return frame[int(y):int(abs(y_)), int(x):int(abs(x_)), :]


def standard_scaling_image(image_array: np.ndarray):
    image_array = image_array.astype('float32')
    mean, std = image_array.mean(), image_array.std()
    image_array = (image_array - mean) / std
    return image_array
