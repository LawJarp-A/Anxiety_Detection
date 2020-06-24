# POSE model

This repo uses code from the github repo-https://github.com/michalfaber/tensorflow_Realtime_Multi-Person_Pose_Estimation.git
The pose model estimation is used is the program to detect movement.
Further Anxiety and confidence score is based on the irregular movemnet of the candidate. Further models include trcaking eye movement and voice modulation to predict the Anxiety-Confidence score.
**Two machine learning models are used** to predict eye movement and predicts keypoinst on upper body. These data will be used to calculate the Anxiety/Confidence score.
NOTE: We are using a mathematical aproach to get the score as we have not yet collected enough credible data to train a model to do it.

# Progress uptil Now
Score predictor using movement is being implemented. For efficieny mobilenet model is being used


# Pre-requisites

You need to have tensorflow 2.0+ installed. And the dependencies listed in the requirments.txt
[Download](https://www.dropbox.com/s/gif7s1qlie2xftd/best_pose_mobilenet_model.zip?dl=1) the model and copy the weights.best.mobilenet.h5 file to the master directory of this repo.
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
```

**Anaconda**

```bash
conda create --name tf_pose_estimation_env
conda activate tf_pose_estimation_env

bash requirements_conda.txt
```
Note: These instructions are also available in the originl repo from which these models and code was borrowed and further trained on.
# Usage

**To find required keypoints on a image**

In your python console or program, move to the Anxiety_Detection directory and 
```bash
import pose_model
import cv2
img = cv2.imread("Imagepath")
keypoints_required = pose_model.get_points(img)
```
**Score predictor**

Run the score.py and enter the path of the video when prompted. 
Output is a list of scores based on contonuos movements such as tapping or shaking of elbows. Futher improvements are due.


# Score prediction algorithm:


**Pose based score estimation:**

1.  Shoulders : 
                   Almost no movement to - 2           0.7-0.9 (Stiff shoulders =  reduced 
                     times a minute,                                             response)
                     
2. Elbows :
                    
                          Currently Indeterminate. No Co-relation

3. Wrist :
                  Constant movement (tapping)           0.6 - 0.9
                   >100                                                  0.9
                   >80 and <100                                    0.8
                   >40 and <80                                      0.4 - 0.7
                   <40                                                    0.2- 0.4



(*note- the distributions are not made even intentionally)
