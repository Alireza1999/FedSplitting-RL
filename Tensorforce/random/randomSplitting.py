import numpy as np
from Tensorforce.random.customEnv import CustomEnvironment
import Tensorforce.utils as utils

iotDevices = utils.createDeviceFromCSV(csvFilePath="../../System/iotDevicesSmallScale.csv", deviceType='iotDevice')
edgeDevices = utils.createDeviceFromCSV(csvFilePath="../../System/edgesSmallScale.csv", deviceType='edge')
cloud = utils.createDeviceFromCSV(csvFilePath="../../System/cloud.csv")[0]

env = CustomEnvironment(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud)


def generateSplittingArray(deviceNum: int):
    splitting = np.random.uniform(low=0.0, high=1.0, size=(deviceNum * 2))
    print("----------------------------------------------------------")
    print(splitting)
    return splitting


reward_Of_timestep = list()
reward_Of_episode = list()
energy_Of_timestep = list()
trainingTime_Of_timestep = list()
x = list()

for i in range(50000):
    x.append(i)
    timestepReward, newState, trainingTime = env.rewardFun(generateSplittingArray(len(iotDevices)))
    reward_Of_timestep.append(timestepReward)
    energy_Of_timestep.append(newState[0])
    trainingTime_Of_timestep.append(trainingTime)

utils.draw_graph(title="Reward vs timestep",
                 xlabel="timestep",
                 ylabel="Reward",
                 figSizeX=15,
                 figSizeY=5,
                 x=x,
                 y=reward_Of_timestep,
                 savePath="Graphs/",
                 pictureName=f"Reward")

utils.draw_graph(title="energy vs timestep",
                 xlabel="timestep",
                 ylabel="energy",
                 figSizeX=15,
                 figSizeY=5,
                 x=x,
                 y=energy_Of_timestep,
                 savePath="Graphs/",
                 pictureName=f"Energy")

utils.draw_hist(title="energy hist",
                xlabel="training Time",
                x=energy_Of_timestep,
                savePath="Graphs/",
                pictureName=f"Energy_hist")

utils.draw_hist(title="reward hist",
                xlabel="training Time",
                x=reward_Of_timestep,
                savePath="Graphs/",
                pictureName=f"Reward_hist")

utils.draw_hist(title="trainingTime hist",
                xlabel="training Time",
                x=trainingTime_Of_timestep,
                savePath="Graphs/",
                pictureName=f"TrainingTime_hist")

utils.draw_scatter(title='Training Time VS Energy',
                   x=energy_Of_timestep,
                   y=trainingTime_Of_timestep,
                   xlabel="Energy",
                   ylabel="Training Time",
                   savePath='Graphs/',
                   pictureName='energy_trainingTime')