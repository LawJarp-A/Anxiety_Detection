
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

## Install

### Dependencies

You need dependencies below.

- python3
- tensorflow 1.4.1, (Not yet tf-2.0 compatible)
- opencv3, protobuf, python3-tk
- slidingwindow
  - https://github.com/adamrehn/slidingwindow
  - I copied from the above git repo to modify few things.

### Pre-Install Jetson case

```bash
$ sudo apt-get install libllvm-7-ocaml-dev libllvm7 llvm-7 llvm-7-dev llvm-7-doc llvm-7-examples llvm-7-runtime
$ export LLVM_CONFIG=/usr/bin/llvm-config-7 
```

### Install

Clone the repo and install 3rd-party libraries.

```bash
$ git clone https://www.github.com/ildoonet/tf-pose-estimation
$ cd tf-pose-estimation
$ pip3 install -r requirements.txt
```

Build c++ library for post processing. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess
```
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

### Package Install

After the above steps, install it as python package

```bash
$ git clone https://www.github.com/ildoonet/tf-pose-estimation
$ cd tf-pose-estimation
$ python setup.py install  # Or, `pip install -e .`
```

### Usage of pose_model

This model detects 6 keypoints of upper body
Usage:

```bash
import pose_model
pose_model.get_points("Image path")
```
It returns a dictionary.


