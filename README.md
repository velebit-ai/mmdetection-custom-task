# MMDetection with separate Color Tagging task
Segmentation of fashion instances together with color tagging.
This repo was built on the [MMDetection] framework.
https://github.com/open-mmlab/mmdetection

# Preparation

## Installation

Tested on Python 3.9.9, Ubuntu 20.04, and CUDA 11.6.

Install dev environment:

```
Create a 3.9.9 python env
Optionally, update requirements.txt (pip install pip-tools==6.8.0, pip-compile requirements.in)
pip install -r requirements.txt
pip install -e . (or add project folder to PYTHONPATH)
Run this script inisde the new env to modify the default mmdet install: `fixes/fixlib.py`
```

## Example of training
`python tools/train.py configs/config.py --work-dir work_dirs/config`