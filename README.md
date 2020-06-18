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

# Installation

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
