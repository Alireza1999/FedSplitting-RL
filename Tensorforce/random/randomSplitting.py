import numpy as np
from Tensorforce.random.customEnv import CustomEnvironment
import Tensorforce.utils as utils

iotDevices = utils.createDeviceFromCSV(csvFilePath="../System/iotDevices.csv", deviceType='iotDevice')
edgeDevices = utils.createDeviceFromCSV(csvFilePath="../System/edges.csv", deviceType='edge')
cloud = utils.createDeviceFromCSV(csvFilePath="../System/cloud.csv")[0]

env = CustomEnvironment(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud)


def generateSplittingArray(deviceNum: int):
    return np.random.uniform(low=0.0, high=1.0, size=(deviceNum * 2))


reward_Of_timestep = list()
reward_Of_episode = list()
energy_Of_timestep = list()
trainingTime_Of_timestep = list()
x = list()

for i in range(10000):
    x.append(i)
    timestepReward, newState,trainingTime = env.rewardFun(generateSplittingArray(len(iotDevices)))
    reward_Of_timestep.append(timestepReward)
    energy_Of_timestep.append(sum(newState[:len(iotDevices)]) / len(iotDevices))
    trainingTime_Of_timestep.append(trainingTime)

utils.draw_graph(title="Reward vs timestep",
                 xlabel="timestep",
                 ylabel="Reward",
                 figSizeX=15,
                 figSizeY=5,
                 x=x,
                 y=reward_Of_timestep,
                 savePath="random/Graphs",
                 pictureName=f"Random_Reward_timestep")

utils.draw_graph(title="energy vs timestep",
                 xlabel="timestep",
                 ylabel="energy",
                 figSizeX=15,
                 figSizeY=5,
                 x=x,
                 y=energy_Of_timestep,
                 savePath="random/Graphs/",
                 pictureName=f"Random_energy_timestep")

utils.draw_hist(title="energy hist",
                xlabel="training Time",
                figSizeX=15,
                figSizeY=5,
                x=energy_Of_timestep,
                savePath="random/Graphs/",
                pictureName=f"Random_energy_timestep_hist")

utils.draw_hist(title="reward hist",
                xlabel="training Time",
                figSizeX=15,
                figSizeY=5,
                x=reward_Of_timestep,
                savePath="random/Graphs/",
                pictureName=f"Random_reward_timestep_hist")

utils.draw_hist(title="trainingTime hist",
                xlabel="training Time",
                figSizeX=15,
                figSizeY=5,
                x=trainingTime_Of_timestep,
                savePath="random/Graphs/",
                pictureName=f"Random_trainingTime_timestep_hist")
