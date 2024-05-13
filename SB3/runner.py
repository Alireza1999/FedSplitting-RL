import sys

sys.path.append("/home/alireza_soleymani/UniversityWorks/Thesis/FedSplitting-RL/")

from stable_baselines3 import A2C
import Tensorforce.utils as utils

agent = 'AC'
fractions = 1.0
logger = utils.createLog(fileName=f"SB3_{agent}_{fractions}")

from SB3.environment.withBandwidth import CustomEnv
from Tensorforce import utils

iotDevices = utils.createDeviceFromCSV(csvFilePath="../envs_stats/iotDevices.csv",
                                       deviceType='iotDevice')
edgeDevices = utils.createDeviceFromCSV(csvFilePath="../envs_stats/edges.csv")
cloud = utils.createDeviceFromCSV(csvFilePath="../envs_stats/cloud.csv")[0]
FLEnergy, FLTrainingTime = utils.ClassicFLTrainingTime(iotDevices, edgeDevices, cloud)
# maxEnergy, minEnergy = minMaxAvgEnergy(iotDevices, edgeDevices, cloud)
rewardTuningParams = [FLEnergy, FLTrainingTime]
env = CustomEnv(rewardTuningParams, iotDevices, edgeDevices, cloud, fraction=1.0, ep_length=50)
# Define and Train the agent
model = A2C("MlpPolicy", env, verbose=1, device="cpu", learning_rate=0.00003)
model.learn(total_timesteps=200000)
