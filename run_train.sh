#!/bin/sh




CUDA_VISIBLE_DEVICES=1 python main.py  --steps 20000 --lr 0.01  --lambda_CLA 0.05 --lambda_WAL 0.1  --thred 0.8 --T 0.5 --source PA_MultiLabel --target AP_MultiLabel


