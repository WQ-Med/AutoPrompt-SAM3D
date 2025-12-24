""" configurations for this project

author Yunli
"""
import os
from datetime import datetime
from ultralytics import YOLO

CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 100
step_size = 10
i = 1
MILESTONES = []
while i * 5 <= EPOCH:
    MILESTONES.append(i* step_size)
    i += 1

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().strftime("%F_%H-%M-%S.%f")

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10

# # 定义一个YOLO模型
# YOLO_MODEL = YOLO('checkpoints/half_label_small_stots_best.pt')  #引入YOLO模型，你先别急

# #冻结YOLO的参数
# for name, parameter in YOLO_MODEL.named_parameters():
#     parameter.requires_grad = False