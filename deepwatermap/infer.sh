#!/bin/sh
TF_USE_LEGACY_KERAS=1 python inference.py --image_dir=../dataset/train/images --checkpoint_path=checkpoints/checkpoints/cp.135.ckpt
