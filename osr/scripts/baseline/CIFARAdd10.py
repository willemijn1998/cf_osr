import os

os.system('''CUDA_VISIBLE_DEVICES=2 python lvae_train.py --baseline --dataset CIFARAdd10 --epochs 100''')