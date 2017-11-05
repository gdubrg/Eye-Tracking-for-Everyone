# Eye Tracking for Everyone
This is an implementation in **Keras** and **Tensorflow** backend  of the famous paper published in CVPR 2016 by Torralba et al.

## How To
### Configuration
The model has been tested with the following configuration:
- Python 3.4.3
- OpenCV 3.3.0
- Keras 2.0.8
- TensorFlow 1.3.0

### Dataset
In order to run the code, it is necessary download the dataset from [here](http://gazecapture.csail.mit.edu/) and following these steps:
- Unzip the dataset in a certain folder
- Run the script create_dataset_lists to create the same dataset used in the original paper. In particular, this script divide the dataset in train, test and validation and delete all non valid detections of subjects' faces. The total number of selected files should be 1490959.

The code can run also on a subset of the original dataset (about 180 GB of data), click [here](http://hugochan.net/download/eye_tracker_train_and_val.npz) to download it. It consists of 48k train images and 5k test samples. In this case, it is sufficient only download the npz file.

### Train

## Acknowledgements
This work is partially inspired by the work of [hugochan](https://github.com/hugochan) 
