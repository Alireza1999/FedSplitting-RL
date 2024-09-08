import sys
from pathlib import Path

ROOT_DIR = Path.cwd().parent
sys.path.append(f"{ROOT_DIR}")
import random

import numpy as np
import torch

import utils
from RL_Algorithm.ActionNormalizer import ActionNormalizer
from RL_Algorithm.Agents.DDPG import DDPG
from SB3.environment.withBandwidth import CustomEnv
from config import ROOT_DIR

iotDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/iotDevices.csv",
                                       deviceType='iotDevice')
edgeDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/edges.csv", deviceType='edgeDevice')
cloud = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/cloud.csv")[0]

env = CustomEnv(iotDevices, edgeDevices, cloud, fraction=0.0, ep_length=100_000)
env = ActionNormalizer(env)


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


seed = 777
random.seed(seed)
np.random.seed(seed)
seed_torch(seed)

num_frames = 100_000
memory_size = 1_000_000
batch_size = 150
ou_noise_theta = 1.0
ou_noise_sigma = 0.1
initial_random_steps = 10_000

agent = DDPG(
    env,
    memory_size,
    batch_size,
    ou_noise_theta,
    ou_noise_sigma,
    initial_random_steps=initial_random_steps
)

agent.train(num_frames)
