# Anxiety detection model

The pose model estimation is used is the program to detect movement.
Further Anxiety and confidence score is based on the irregular movemnet of the candidate. Further models include trcaking eye movement and voice modulation to predict the Anxiety-Confidence score.
**Two machine learning models are used** to predict eye movement and predicts keypoinst on upper body. These data will be used to calculate the Anxiety/Confidence score.
NOTE: We are using a mathematical aproach to get the score as we have not yet collected enough credible data to train a model to do it.

# Progress uptil Now
Score predictor using movement is being implemented. For efficieny mobilenet model is being used

# TO-DO
1. Upgrade code to be compatible with tensorflow 2.0 [DONE]
2. Add eye tracking module [DONE]
3. Add Voice modulation module
4. Collect a good amount of datat to train on
5. Improve speed and efficiency

# 1)POSE MODEL:
# Pre-requisites

You need to have tensorflow 2.0+ installed. And the dependencies listed in the requirments.txt
[Download](https://www.dropbox.com/s/gif7s1qlie2xftd/best_pose_mobilenet_model.zip?dl=1) the model and copy the weights.best.mobilenet.h5 file to the folder.
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


# 2)EYE-TRACKING Model:

The primary model of eye gaze is due to the courtesy of https://github.com/antoinelame/GazeTracking

Follow the following steps to run the program:

1. pip install -r requirements.txt.
NOTE:If some problem arises installing dlib, use dlib-19.19.0.tar.gz(in the ey_moedl folder) and refer https://www.youtube.com/watch?v=pHrgi8QLcKk&feature=youtu.be (in windows 10) 
or refer to https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/

2. Run the python file 'run_this.py' to get input from the webcam.For a video input , change the 0 in run_this.py to the videoname.mp4,
i.e. webcam = cv2.VideoCapture(0) to 
webcam = cv2.VideoCapture('videoname.mp4') 
where,videoname is the name of the video file

3. Press 'esc' on the keyboard to interept the input.

## Synopsis for eye model:

1. Eye frame is extracted from the face,using the particular landmarks defined using the pre-trained model shape_predictor_68_face_landmarks.dat.

2. It is converted into a grayscale image and uses the pupil detection algorithm by finding the best binarization threshold value for the person and the webcam(thanks to https://github.com/antoinelame/GazeTracking).

3. The pupil is extracted using contours and the center is found out using cenroid method of that contour i.e.
Cx=M10/M00
Cy=M01/M00
where,M is Image Moment which is a particular weighted average of image pixel intensities, with the help of which we can find some specific properties of an image, like radius, area, centroid etc

4. The video is converted into many frames and centroid of the pupil is found of both eyes of each frame.

## Synopsis for Anxiety Score:A score between 0 to 1 is assigned based on the following:-

1.The coordinates of the eye looking at the center is found by mean and the amount of distraction is found by calculating the spread of the coordinates from the mean.

2.Before finding the mean,outliers are removed using quartile method as to avoid the reduce the effect of the outliers on the mean.Even the data of certain coordinates which couldn't be recorded are filled by the mean.

3.Scores normalized between zero and 1 w.r.t avoiding eye contact (in this case, looking straight/center) for 15-60 seconds.

4.Changes in position of the eye in measured every frame. Scores are assigned as following:

               0-70 times per minute.                     0.0 to 0.2 

               70-100 times per minute.                   0.4 to 0.6     
   
               100-150 times per minute.                  0.6 to 1.0  

5.Mean of all above scores and of both eyes is the final score.


(*note- the distributions are not made even intentionally)


**NOTE: This repo uses some code from the github repo-https://github.com/michalfaber/tensorflow_Realtime_Multi-Person_Pose_Estimation.git**
