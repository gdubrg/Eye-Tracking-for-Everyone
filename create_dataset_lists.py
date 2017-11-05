import json
import os
from os.path import join
import glob
import argparse


def create_dataset_lists(args):

    # read args from main (input and output)
    dataset_path = args.input
    output_root = args.output

    # read all sequence directories (starting with a number)
    dirs = sorted(glob.glob(join(dataset_path, "0*")))

    # count how many frames are finally selected
    tot_valid_frame = 0

    # main loop
    for dir in dirs[:]:

        print("analyzing {}".format(dir))

        # open json files
        face_file = open(join(dataset_path, dir, "appleFace.json"))
        left_file = open(join(dataset_path, dir, "appleLeftEye.json"))
        right_file = open(join(dataset_path, dir, "appleRightEye.json"))
        frames_file = open(join(dataset_path, dir, "frames.json"))
        info_file = open(join(dataset_path, dir, "info.json"))

        # read json content
        face_json = json.load(face_file)
        left_json = json.load(left_file)
        right_json = json.load(right_file)
        frames_json = json.load(frames_file)
        info_json = json.load(info_file)

        # divide in train, test and validation
        if info_json["Dataset"] == "train":
            output = open(join(output_root, "train", os.path.basename(dir)), "w")

        if info_json["Dataset"] == "test":
            output = open(join(output_root, "test", os.path.basename(dir)), "w")

        if info_json["Dataset"] == "val":
            output = open(join(output_root, "validation", os.path.basename(dir)), "w")

        # as reported in the original paper, a sanity check is conducted
        for i in range(0, int(info_json["TotalFrames"])):
            if left_json["IsValid"][i] and right_json["IsValid"][i] and face_json["IsValid"][i]:
                output.write(os.path.basename(dir) + "/" + frames_json[i])
                output.write("\n")
                # increase the number of valid frame
                tot_valid_frame += 1

        # close the file
        output.close()

    # number of total valid frame reported by the original paper is 1490959
    print("Total valid frames: {}".format(tot_valid_frame))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create dataset names for train, validation and test.")

    parser.add_argument('-i', '-input', type=str, required=True, help="Directory wich contains the unzipped dataset.")
    parser.add_argument('-o','-output', type=str, required=True, help="Ouptut directory")
    args = parser.parse_args()

    create_dataset_lists(args)

