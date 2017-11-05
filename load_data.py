import numpy as np
import cv2
import os
import glob
from os.path import join
import json
from data_utility import image_normalization


# load data directly from the npz file (small dataset, 48k and 5k for train and test)
def load_data_from_npz(file):

    print("Loading dataset from npz file...", end='')
    npzfile = np.load(file)
    train_eye_left = npzfile["train_eye_left"]
    train_eye_right = npzfile["train_eye_right"]
    train_face = npzfile["train_face"]
    train_face_mask = npzfile["train_face_mask"]
    train_y = npzfile["train_y"]
    val_eye_left = npzfile["val_eye_left"]
    val_eye_right = npzfile["val_eye_right"]
    val_face = npzfile["val_face"]
    val_face_mask = npzfile["val_face_mask"]
    val_y = npzfile["val_y"]
    print("Done.")

    return [train_eye_left, train_eye_right, train_face, train_face_mask, train_y], [val_eye_left, val_eye_right, val_face, val_face_mask, val_y]


# load a batch with data loaded from the npz file
def load_batch(data, img_ch, img_cols, img_rows):

    # useful for debug
    save_images = False

    # if save images, create the related directory
    img_dir = "images"
    if save_images:
        if not os.path.exists(img_dir):
            os.makedir(img_dir)

    # create batch structures
    left_eye_batch = np.zeros(shape=(data[0].shape[0], img_ch, img_cols, img_rows), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(data[0].shape[0], img_ch, img_cols, img_rows), dtype=np.float32)
    face_batch = np.zeros(shape=(data[0].shape[0], img_ch, img_cols, img_rows), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(data[0].shape[0], 1, 25, 25), dtype=np.float32)
    y_batch = np.zeros((data[0].shape[0], 2), dtype=np.float32)

    # load left eye
    for i, img in enumerate(data[0]):
        img = cv2.resize(img, (img_cols, img_rows))
        if save_images:
            cv2.imwrite(join(img_dir, "left" + str(i) + ".png"), img)
        img = image_normalization(img)
        left_eye_batch[i] = img.transpose(2, 0, 1)

    # load right eye
    for i, img in enumerate(data[1]):
        img = cv2.resize(img, (img_cols, img_rows))
        if save_images:
            cv2.imwrite("images/right" + str(i) + ".png", img)
        img = image_normalization(img)
        right_eye_batch[i] = img.transpose(2, 0, 1)

    # load faces
    for i, img in enumerate(data[2]):
        img = cv2.resize(img, (img_cols, img_rows))
        if save_images:
            cv2.imwrite("images/face" + str(i) + ".png", img)
        img = image_normalization(img)
        face_batch[i] = img.transpose(2, 0, 1)

    # load grid faces
    for i, img in enumerate(data[3]):
        if save_images:
            cv2.imwrite("images/grid" + str(i) + ".png", img)
        face_grid_batch[i] = img.reshape((1, img.shape[0], img.shape[1]))

    # load labels
    for i, labels in enumerate(data[4]):
        y_batch[i] = labels

    return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch


# create a list of all names of images in the dataset
def load_data_names(path):

    seq_list = []
    seqs = sorted(glob.glob(join(path, "0*")))

    for seq in seqs:

        file = open(seq, "r")
        content = file.read().splitlines()
        for line in content:
            seq_list.append(line)

    return seq_list


# load a batch given a list of names (all images are loaded)
def load_batch_from_names(names, path, img_ch, img_cols, img_rows):

    save_img = False

    # data structures for batches
    left_eye_batch = np.zeros(shape=(len(names), img_ch, img_cols, img_rows), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(len(names), img_ch, img_cols, img_rows), dtype=np.float32)
    face_batch = np.zeros(shape=(len(names), img_ch, img_cols, img_rows), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(len(names), 1, 25, 25), dtype=np.float32)
    y_batch = np.zeros((len(names), 2), dtype=np.float32)

    for i, img_name in enumerate(names):

        # directory
        dir = img_name[:5]

        # frame name
        frame = img_name[6:]

        # index of the frame inside the sequence
        idx = int(frame[:-4])

        # open json files
        face_file = open(join(path, dir, "appleFace.json"))
        left_file = open(join(path, dir, "appleLeftEye.json"))
        right_file = open(join(path, dir, "appleRightEye.json"))
        dot_file = open(join(path, dir, "dotInfo.json"))
        grid_file = open(join(path, dir, "faceGrid.json"))

        # load json content
        face_json = json.load(face_file)
        left_json = json.load(left_file)
        right_json = json.load(right_file)
        dot_json = json.load(dot_file)
        grid_json = json.load(grid_file)

        # open image
        img = cv2.imread(join(path, dir, "frames", frame))

        # debug stuff
        # if img is None:
        #     print("Error opening image: {}".format(join(path, dir, "frames", frame)))
        #     continue
        #
        # if int(face_json["X"][idx]) < 0 or int(face_json["Y"][idx]) < 0 or \
        #     int(left_json["X"][idx]) < 0 or int(left_json["Y"][idx]) < 0 or \
        #     int(right_json["X"][idx]) < 0 or int(right_json["Y"][idx]) < 0:
        #     print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
        #     continue

        # get face
        tl_x_face = int(face_json["X"][idx])
        tl_y_face = int(face_json["Y"][idx])
        w = int(face_json["W"][idx])
        h = int(face_json["H"][idx])
        br_x = tl_x_face + w
        br_y = tl_y_face + h
        face = img[tl_y_face:br_y, tl_x_face:br_x]

        # get left eye
        tl_x = tl_x_face + int(left_json["X"][idx])
        tl_y = tl_y_face + int(left_json["Y"][idx])
        w = int(left_json["W"][idx])
        h = int(left_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        left_eye = img[tl_y:br_y, tl_x:br_x]

        # get right eye
        tl_x = tl_x_face + int(right_json["X"][idx])
        tl_y = tl_y_face + int(right_json["Y"][idx])
        w = int(right_json["W"][idx])
        h = int(right_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        right_eye = img[tl_y:br_y, tl_x:br_x]

        # get face grid (in ch, cols, rows convention)
        face_grid = np.zeros(shape=(1, 25, 25), dtype=np.float32)
        tl_x = int(grid_json["X"][idx])
        tl_y = int(grid_json["Y"][idx])
        w = int(grid_json["W"][idx])
        h = int(grid_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        face_grid[0, tl_y:br_y, tl_x:br_x] = 1

        # get labels
        y_x = dot_json["XCam"][idx]
        y_y = dot_json["YCam"][idx]

        # resize images
        face = cv2.resize(face, (img_cols, img_rows))
        left_eye = cv2.resize(left_eye, (img_cols, img_rows))
        right_eye = cv2.resize(right_eye, (img_cols, img_rows))

        # save images (for debug)
        if save_img:
            cv2.imwrite("images/face.png", face)
            cv2.imwrite("images/right.png", right_eye)
            cv2.imwrite("images/left.png", left_eye)
            cv2.imwrite("images/image.png", img)

        # normalization
        face = image_normalization(face)
        left_eye = image_normalization(left_eye)
        right_eye = image_normalization(right_eye)

        ######################################################

        # transpose images
        face = face.transpose(2, 0, 1)
        left_eye = left_eye.transpose(2, 0, 1)
        right_eye = right_eye.transpose(2, 0, 1)

        # check data types
        face = face.astype('float32')
        left_eye = left_eye.astype('float32')
        right_eye = right_eye.astype('float32')

        # add to the related batch
        left_eye_batch[i] = left_eye
        right_eye_batch[i] = right_eye
        face_batch[i] = face
        face_grid_batch[i] = face_grid
        y_batch[i][0] = y_x
        y_batch[i][1] = y_y

    return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch


# load a batch of random data given the full list of the dataset
def load_batch_from_names_random(names, path, batch_size, img_ch, img_cols, img_rows):

    save_img = False

    # data structures for batches
    left_eye_batch = np.zeros(shape=(batch_size, img_ch, img_cols, img_rows), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(batch_size, img_ch, img_cols, img_rows), dtype=np.float32)
    face_batch = np.zeros(shape=(batch_size, img_ch, img_cols, img_rows), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(batch_size, 1, 25, 25), dtype=np.float32)
    y_batch = np.zeros((batch_size, 2), dtype=np.float32)

    # counter for check the size of loading batch
    b = 0

    while b < batch_size:

        # lottery
        i = np.random.randint(0, len(names))

        # get the lucky one
        img_name = names[i]

        # directory
        dir = img_name[:5]

        # frame name
        frame = img_name[6:]

        # index of the frame into a sequence
        idx = int(frame[:-4])

        # open json files
        face_file = open(join(path, dir, "appleFace.json"))
        left_file = open(join(path, dir, "appleLeftEye.json"))
        right_file = open(join(path, dir, "appleRightEye.json"))
        dot_file = open(join(path, dir, "dotInfo.json"))
        grid_file = open(join(path, dir, "faceGrid.json"))

        # load json content
        face_json = json.load(face_file)
        left_json = json.load(left_file)
        right_json = json.load(right_file)
        dot_json = json.load(dot_file)
        grid_json = json.load(grid_file)

        # open image
        img = cv2.imread(join(path, dir, "frames", frame))

        # if image is null, skip
        if img is None:
            # print("Error opening image: {}".format(join(path, dir, "frames", frame)))
            continue

        # if coordinates are negatives, skip (a lot of negative coords!)
        if int(face_json["X"][idx]) < 0 or int(face_json["Y"][idx]) < 0 or \
            int(left_json["X"][idx]) < 0 or int(left_json["Y"][idx]) < 0 or \
            int(right_json["X"][idx]) < 0 or int(right_json["Y"][idx]) < 0:
            # print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
            continue

        # get face
        tl_x_face = int(face_json["X"][idx])
        tl_y_face = int(face_json["Y"][idx])
        w = int(face_json["W"][idx])
        h = int(face_json["H"][idx])
        br_x = tl_x_face + w
        br_y = tl_y_face + h
        face = img[tl_y_face:br_y, tl_x_face:br_x]

        # get left eye
        tl_x = tl_x_face + int(left_json["X"][idx])
        tl_y = tl_y_face + int(left_json["Y"][idx])
        w = int(left_json["W"][idx])
        h = int(left_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        left_eye = img[tl_y:br_y, tl_x:br_x]

        # get right eye
        tl_x = tl_x_face + int(right_json["X"][idx])
        tl_y = tl_y_face + int(right_json["Y"][idx])
        w = int(right_json["W"][idx])
        h = int(right_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        right_eye = img[tl_y:br_y, tl_x:br_x]

        # get face grid (in ch, cols, rows convention)
        face_grid = np.zeros(shape=(1, 25, 25), dtype=np.float32)
        tl_x = int(grid_json["X"][idx])
        tl_y = int(grid_json["Y"][idx])
        w = int(grid_json["W"][idx])
        h = int(grid_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        face_grid[0, tl_y:br_y, tl_x:br_x] = 1

        # get labels
        y_x = dot_json["XCam"][idx]
        y_y = dot_json["YCam"][idx]

        # resize images
        face = cv2.resize(face, (img_cols, img_rows))
        left_eye = cv2.resize(left_eye, (img_cols, img_rows))
        right_eye = cv2.resize(right_eye, (img_cols, img_rows))

        # save images (for debug)
        if save_img:
            cv2.imwrite("images/face.png", face)
            cv2.imwrite("images/right.png", right_eye)
            cv2.imwrite("images/left.png", left_eye)
            cv2.imwrite("images/image.png", img)

        # normalization
        face = image_normalization(face)
        left_eye = image_normalization(left_eye)
        right_eye = image_normalization(right_eye)

        ######################################################

        # transpose images
        face = face.transpose(2, 0, 1)
        left_eye = left_eye.transpose(2, 0, 1)
        right_eye = right_eye.transpose(2, 0, 1)

        # check data types
        face = face.astype('float32')
        left_eye = left_eye.astype('float32')
        right_eye = right_eye.astype('float32')

        # add to the related batch
        left_eye_batch[b] = left_eye
        right_eye_batch[b] = right_eye
        face_batch[b] = face
        face_grid_batch[b] = face_grid
        y_batch[b][0] = y_x
        y_batch[b][1] = y_y

        # increase the size of the current batch
        b += 1

    return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch


if __name__ == "__main__":

    # debug
    seq_list = load_data_names("/cvgl/group/GazeCapture/test")

    batch_size = len(seq_list)
    dataset_path = "/cvgl/group/GazeCapture/gazecapture"
    img_ch = 3
    img_cols = 64
    img_rows = 64

    test_batch = load_batch_from_names_random(seq_list, dataset_path, batch_size, 3, 64, 64)

    print("Loaded: {} data".format(len(test_batch[0][0])))
