# Eye Tracking for Everyone
This is an implementation in **Keras** and **Tensorflow** backend  of the famous paper published in CVPR 2016 by *Torralba et al*.

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
- Run the script [create_dataset_lists](create_dataset_lists.py) to create the same dataset used in the original paper. In particular, this script divides the dataset in train, test and validation sets and delete all non valid detections of subjects' faces. The total number of selected files should be 1490959.

The code can run also on a subset of the original dataset, that is about 180 GB of data, click [here](http://hugochan.net/download/eye_tracker_train_and_val.npz) to download it. It consists of 48k train images and 5k test samples. In this case, it is sufficient only download the npz file.

### Train
The entry point of the code is the [main](main.py) script. 
The command is
```
python3 main.py
```
You can pass the following arguments:
- ```-h```: manual
- ```-train```: train algorithm
- ```-test```: test algorithm (not yet available)
- ```-data```: what dataset, ```big``` for the original dataset, ```small``` for the subset from the npz file
- ```-dev```: what device, ```-1``` for the CPU, ```0...n``` for your n GPU devices
- ```-max_epoch```: max number of epochs
- ```-batch_size```: size fo the batch size, to specify accordingly to your GPU memory
- ```-patience```: early stopping

## Results
Results are obtained on two types of input images: 64x64 and 224x224 (as in the original paper) and are expressed in term of *Mean Absolute Error* (MAE) and *Standard Deviation* (STD). The network has been tested on both datasets.

### Small dataset 
This dataset consist of 48k train images and 5k test images. All face detections are valid and all coordinates are positives.

| Input Size   | MAE            | STD           |
| :---:         |     :---:         |             :---: |
|64x64         | 1.00, 1.10        | 1.21, 1.28       |
|224x224       | 1.42, 1.47        | 1.48, 1.55       |

### Original dataset

| Input Size   | MAE            | STD           |
| :---:         |     :---:     |          :---: |
|64x64         | 1.45, 1.67     | 1.43, 1.62    |
|224x224       | -              | -             |

## Notes
In the original paper, authors wrote that the toal of train, test and validation images is about 1.5 M, but I found that a lot of images are annotated with negative coordinates of the face's bounding box, so I can train and test my network on a lower amount of data. Waiting for the authors' response.

## Acknowledgements
This work is partially inspired by the work of [hugochan](https://github.com/hugochan).
