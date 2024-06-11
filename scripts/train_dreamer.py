import warnings
import os
import hydra
import torch
import wandb
import envs
import utils
import numpy as np
from dm_env import specs
from pathlib import Path
from src.common.logger import Logger

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


@hydra.main(config_path='./', config_name='train_dreamer')
def main(cfg):
    print('is running')
    print(cfg)


if __name__ == '__main__':
    main()
