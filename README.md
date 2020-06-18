## Anxiety detection model

This repo uses code from the github repo-https://github.com/michalfaber/tensorflow_Realtime_Multi-Person_Pose_Estimation.git
The pose model estimation is used is the program to detect movement.
Further Anxiety and confidence score is based on the irregular movemnet of the candidate. Further models include trcaking eye movement and voice modulation to predict the Anxiety-Confidence score.

## Progress uptil Now
Score predictor using movement is being implemented. For efficieny mobilenet model is being used

## TO-DO
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

