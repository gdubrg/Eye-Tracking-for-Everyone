# Eye Tracking for Everyone
This is an implementation in **Keras** and **Tensorflow** of the famous paper published in **CVPR 2016** by *Torralba et al*.

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
- Run the script [create_dataset_lists](create_dataset_lists.py) to create the same dataset used in the original paper. In particular, this script divides the dataset in train, test and validation sets and delete all non valid detections of subjects' faces. 
The total number of selected files should be **1490959**.

The code can run also on a subset of the original dataset, that is about 180 GB of data, click [here](http://hugochan.net/download/eye_tracker_train_and_val.npz) to download it. It consists of **48k** train images and **5k** test samples. In this case, it is sufficient only download the npz file.

### Train and Test
The entry point of the code is the [main](main.py) script. 
The command is
```
python3 main.py
```
You can pass the following arguments:
- ```-h```: manual
- ```-train```: **train** the network
- ```-test```: **test** the network
- ```-data```: what dataset, ```big``` for the original dataset, ```small``` for the subset from the npz file
- ```-dev```: what device, ```-1``` for the CPU, ```0...n``` for your n GPU devices
- ```-max_epoch```: max number of epochs
- ```-batch_size```: size fo the batch size, to specify accordingly to your GPU memory
- ```-patience```: early stopping

## Results
Results are obtained on two types of input images: 224x224 (as in the original paper) and 64x64 (useful to fast training and testing, with low GPU memory requirements).
Results are expressed in term of *Mean Absolute Error* (MAE) and *Standard Deviation* (STD). 
Finally, the network has been tested on both datasets.

### Small dataset 
This dataset consist of 48k train images and 5k test images. All face detections are valid and all coordinates are positives.

| Input Size   | MAE               | STD              | Loss MSE  |
| :---:        |     :---:         |            :---: | :---:     |
|64x64         | 1.00, 1.10        | 1.21, 1.28       |  2.662    |
|224x224       | 1.42, 1.47        | 1.48, 1.55       |  4.410    |

### Original dataset

| Input Size   | MAE            | STD           |Loss MSE  |
| :---:         |     :---:     |         :---: |:---:     |
|64x64         | 1.45, 1.67     | 1.43, 1.62    |    3.942 |
|224x224       | -              | -             |          |

## Notes
### Data
Even after the sanity check about the detection validity, a lot of detections have negative coordinates.
Authors' response: *maybe* they used the last data about detections extracted.
So, I implemented two solutions. The first is called ```load_batch_from_names_random```, that loads a batch of data randomly, discarding negative coordinates. The second is ```load_batch_from_names_fixed```, that retrieves data from the last valid detection. I have used the first one for the experiments.

### Path
In the code are present some hardcoded paths, at the beginning of ```train``` and ```test``` scripts.

## Acknowledgements
This work is partially inspired by the work of [hugochan](https://github.com/hugochan).
