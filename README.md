# vsrseg
Deep learning models for visual semantic role segmentation.

## Problem
This repository explores the joint labeling and identification of semantic roles in everyday images. It explores Visual Semantic Role Labeling (VSRL) and Situation Recognition, using the V-COCO and imSitu datasets.

## Setup
This project depends on:
- [V-COCO](https://github.com/s-gupta/v-coco). Follow the setup instructions to obtain the V-COCO dataset, then be sure to add the `v-coco/` and `v-coco/coco/PythonAPI/` directories to your `PYTHONPATH`.
- [Faster RCNN](https://github.com/longcw/pytorch-faster-rcnn). Be sure to symlink `faster_rcnn_pytorch/data/coco` to a working coco dataset.
