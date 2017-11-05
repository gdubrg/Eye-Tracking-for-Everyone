import numpy as np
import random


# normalize all data
def normalize(data):

    print("Data normalization...")
    shape = data.shape
    data = np.reshape(data, (shape[0], -1))
    # scaling
    data = data.astype('float32') / 255.
    # normalizing
    data = data - np.mean(data, axis=0)
    print("Done.")
    return np.reshape(data, shape)


# normalize a single image
def image_normalization(img):

    img = img.astype('float32') / 255.
    img = img - np.mean(img)

    return img


# prepare all data (npz version)
def prepare_data(data):
    print("Data preparing...")
    eye_left, eye_right, face, face_mask, y = data
    eye_left = normalize(eye_left)
    eye_right = normalize(eye_right)
    face = normalize(face)
    face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
    y = y.astype('float32')
    print("Done.")
    return [eye_left, eye_right, face, face_mask, y]


# shuffle data
def shuffle_data(data):

    idx = np.arange(data[0].shape[0])
    np.random.shuffle(idx)
    for i in list(range(len(data))):
        data[i] = data[i][idx]
    return data
