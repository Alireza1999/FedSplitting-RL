import sys
from pathlib import Path

ROOT_DIR = Path.cwd().parent.parent
sys.path.append(f"{ROOT_DIR}")

import utils
from RL_Algorithm.DDPG.utils.ActionNormalizer import ActionNormalizer
from SB3.environment.withBandwidth import CustomEnv
from RL_Algorithm.PPO.Agent import PPO


iotDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/iotDevices.csv",
                                       deviceType='iotDevice')
edgeDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/edges.csv", deviceType='edgeDevice')
cloud = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/cloud.csv")[0]

env = CustomEnv(iotDevices, edgeDevices, cloud, fraction=0.0, ep_length=100_000)

# parameters
num_frames = 100000

agent = PPO(
    env,
    gamma=0.9,
    tau=0.8,
    batch_size=64,
    epsilon=0.2,
    epoch=64,
    rollout_len=2048,
    entropy_weight=0.005
)
