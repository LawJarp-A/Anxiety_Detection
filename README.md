# Tensorflow 2.0 Realtime Multi-Person Pose Estimation

<<<<<<< HEAD
This repo contains a new upgraded version of the **keras_Realtime_Multi-Person_Pose_Estimation** project. It is ready for the new Tensorflow 2.0.

I added a new model based on MobileNetV2 for mobile devices.
You can train it from scratch in the same way as the CMU model. There is still room for performance improvement, like quantization training, which I will add as a next step.
=======
## Anxiety detection model

This repo uses code from the original github repo-https://github.com/ildoonet/tf-pose-estimation.git
The pose model estimation is used is the program to detect movement.
Further Anxiety and confidence score is based on the irregular movemnet of the candidate. Further models include trcaking eye movement and voice modulation to predict the Anxiety-Confidence score.

## Progress uptil Now
Score predictor using movement is being implemented

## TO-DO
1. Upgrade code to be compatible with tensorflow 2.0
2. Add eye tracking module
3. Add Voice modulation module
4. Collect a good amount of datat to train on
5. Improve speed and efficiency
>>>>>>> 3459ab9365133647039d0ecaae4f3497fb1d0f2a

[Download](https://www.dropbox.com/s/gif7s1qlie2xftd/best_pose_mobilenet_model.zip?dl=1) the model and checkpoints.

I didn't change much the augmentation process as the tensorpack does a good job. The only changes I have made are in fetching samples to the model. I added the interface Dataset as recommended by Tensorflow.

It is worth to mention that I purposely didn't use the Keras interface **model.compile, model.run** as I had problems with loss regularizers - I kept getting NaN after a few iterations. I suspect that the solution would be to add loss to the input tensor: *add_loss(tf.abs(tf.reduce_mean(x)))*. I will update the repo as soon as I get satisfactory results.

I added a visualization of final heatmaps and pafs in the Tensorboard.
Every 100 iterations, a single image is passed to the model. The predicted heatmaps and pafs are logged in the Tensorboard.
You can check this visual representation of prediction every few minutes/hours as it gives a good sense of how the training performs.

# Scripts and notebooks

This project contains the following scripts and jupyter notebooks:

**train_custom_loop.py** - training code for the CMU model. This is a new version of the training code from the old repo *keras_Realtime_Multi-Person_Pose_Estimation*. It has been upgraded to Tensorflow 2.0.

**train_custom_loop_mobilenet.py** - training code for smaller model. It is based on the MobilenetV2. Simplified model with just 2 stages.

<<<<<<< HEAD
**convert_to_tflite.py** - script used to create *TFLite* model based on checkpoint or keras h5 file.

**dataset_inspect.ipynb** - helper notebook to get more insights into what is generated from the dataset.

**test_pose_mobilenet.ipynb** - helper notebook to preview the predictions from the mobilenet-based model.

**test_pose_vgg.ipynb** - helper notebook to preview the predictions from the original vgg-based model.
=======
```bash
$ git clone https://github.com/LawJarp-A/Anxiety_Detection.git
$ cd tf-pose-estimation
$ pip3 install -r requirements.txt
```

Build c++ library for post processing. 
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```
>>>>>>> 3459ab9365133647039d0ecaae4f3497fb1d0f2a

**test_tflite_model.ipynb** - helper notebook to verify exported *TFLite* model.

<<<<<<< HEAD
**estimation_example/** - This is an example demonstrating the estimation algorithm. Here you will find sample heatmaps and pafs dumped into numpy arrays (*.npy) and some scripts: *coordinates.py*, *connections.py*, *estimators.py* containing the code for each step of the estimation algorithm. You can run these scripts separately to better understand each step. In addition, there is the script: *example.py* that shows all the steps together. This script creates an output image with the connections.  
=======
After the above steps, install it as python package
>>>>>>> 3459ab9365133647039d0ecaae4f3497fb1d0f2a

# Installation

## Prerequisites

* download [dataset and annotations](http://cocodataset.org/#download) into a separate folder datasets, outside of this project:
```bash
<<<<<<< HEAD
    ├── datasets
    │   └── coco_2017_dataset
    │       ├── annotations
    │       │   ├── person_keypoints_train2017.json
    │       │   └── person_keypoints_val2017.json
    │       ├── train2017/*
    │       └── val2017/*
    └── tensorflow_Realtime_Multi-Person_Pose_Estimation/*
```                
* install [CUDNN](https://developer.nvidia.com/cudnn) and [CUDA](https://developer.nvidia.com/cuda-downloads)

    If you use Anaconda, there is a simpler way. Just install the precompiled libs:
```bash    
    conda install -c anaconda cudatoolkit==10.0.130-0
```

## How to install (with tensorflow-gpu)


**Virtualenv**

```bash
pip install virtualenv
virtualenv venv
source venv/bin/activate

pip install -r requirements.txt
# ...or
pip install -r requirements_all.txt # completely frozen environment with all dependent libraries
=======
$ git clone https://github.com/LawJarp-A/Anxiety_Detection.git
$ cd tf-pose-estimation
$ python setup.py install  # Or, `pip install -e .`
```

### Usage of pose_model

This model detects 6 keypoints of upper body
Usage:

```bash
import pose_model
pose_model.get_points("Image path")
>>>>>>> 3459ab9365133647039d0ecaae4f3497fb1d0f2a
```
It returns a dictionary.

**Anaconda**

```bash
conda create --name tf_pose_estimation_env
conda activate tf_pose_estimation_env

bash requirements_conda.txt
```
