# Anxiety detection model

This repo uses code from the github repo-https://github.com/michalfaber/tensorflow_Realtime_Multi-Person_Pose_Estimation.git
The pose model estimation is used is the program to detect movement.
Further Anxiety and confidence score is based on the irregular movemnet of the candidate. Further models include trcaking eye movement and voice modulation to predict the Anxiety-Confidence score.

# Progress uptil Now
Score predictor using movement is being implemented. For efficieny mobilenet model is being used

# TO-DO
1. Upgrade code to be compatible with tensorflow 2.0 [DONE]
2. Add eye tracking module
3. Add Voice modulation module
4. Collect a good amount of datat to train on
5. Improve speed and efficiency

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
NOTE: The score predictor is yet to be updated.



#Score prediction algorithm:

Eye Motion score prediction:

1. Normalize (between zero and 1) avoiding eye contact (in this case, looking straight/center) for 15-60 (Subject to change) seconds.

2. Measure changes in position of the eye in 1 second time interval. Assign score as table below if possible:
                 <70 times per second                       0.2 (or as low as possible)
                   70-100 times per second                0.4 - 0.6   (This is perfectly normal 
                                                                                movement of eye)
                   >100 times per second                   0.7 - 0.9                

3. Measure changes in position of the eye in 1 second time interval. Assign score as table below if possible:
                 Statistical approach, Deviation from central axis, refer sumukh

4. DO not assign 0 or 1 in any case


Pose based score estimation:

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
