import logging
import sys

sys.path.append('/home/alireza_soleymani/UniversityWorks/Thesis/FedSplitting-RL')

from Tensorforce import config
from Tensorforce import utils
from Tensorforce.enviroments.customEnv import CustomEnvironment

# logging.basicConfig(filename="../simpleEnergy/Logs/AllPossibleAction/info.log",
#                     format='%(message)s',
#                     filemode='w')
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

iotDevices = utils.createDeviceFromCSV(csvFilePath="System/iotDevicesSmallScale.csv", deviceType='iotDevice')
edgeDevices = utils.createDeviceFromCSV(csvFilePath="System/edgesSmallScale.csv", deviceType='edge')
cloud = utils.createDeviceFromCSV(csvFilePath="System/cloud.csv")[0]

energy = []
trainingTime = []
reward = []

min_Energy = 10000
max_Energy = 0
min_Energy_TrainingTime = 0
max_Energy_TrainingTime = 0
min_energy_splitting = []
max_Energy_splitting = []

min_trainingTime = 10000
max_trainingTime = 0
min_trainingTime_energy = 0
max_trainingTime_energy = 0
min_trainingtime_splitting = []
max_trainingtime_splitting = []

splittingLayer = utils.allPossibleSplitting(modelLen=config.LAYER_NUM - 1, deviceNumber=len(iotDevices))

env = CustomEnvironment(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud, fraction=1)

for splitting in splittingLayer:
    splittingArray = list()
    for char in splitting:
        splittingArray.append(int(char))
    newReward, newState = env.rewardFun(splittingArray)

    if newState[0] < min_Energy:
        min_Energy = newState[0]
        min_energy_splitting = splittingArray
        min_Energy_TrainingTime = newState[1]
    if newState[0] > max_Energy:
        max_Energy = newState[0]
        max_Energy_splitting = splittingArray
        max_Energy_TrainingTime = newState[1]

    if newState[1] < min_trainingTime:
        min_trainingTime = newState[1]
        min_trainingtime_splitting = splittingArray
        min_trainingTime_energy = newState[0]
    if newState[1] > max_trainingTime:
        max_trainingTime = newState[1]
        max_trainingtime_splitting = splittingArray
        max_trainingTime_energy = newState[0]

    energy.append(newState[0])
    reward.append(newReward)
    trainingTime.append(newState[1])


def getMinMaxEnergy():
    return min_Energy, max_Energy


def getMinMaxTrainingTime():
    return min_trainingTime, max_trainingTime


print(f"------------------------------------------------")
print(f"MIN Energy : \n{min_Energy}")
print(f"MIN Energy Splitting: {min_energy_splitting}")
print(f"MIN Energy Trainingtime: {min_Energy_TrainingTime}")

print(f"------------------------------------------------")
print(f"MAX Energy : {max_Energy}")
print(f"MAX Energy Splitting: {max_Energy_splitting}")
print(f"MAX Energy Trainingtime: {max_Energy_TrainingTime}")

print(f"------------------------------------------------")
print(f"MIN TrainingTime : {min_trainingTime}")
print(f"MIN TrainingTime Splitting: {min_trainingtime_splitting}")
print(f"MIN TrainingTime Energy: {min_trainingTime_energy}")

print(f"------------------------------------------------")
print(f"MAX TrainingTime : {max_trainingTime}")
print(f"MAX TrainingTime Splitting: {max_trainingtime_splitting}")
print(f"MAX TrainingTime Energy: {max_trainingTime_energy}")

# logger.info(f"================================================")
# logger.info(f"MIN Energy : {min_Energy}")
# logger.info(f"MIN Energy Splitting: {min_energy_splitting}")
# logger.info(f"================================================")
# logger.info(f"Max Energy : {min_Energy}")
# logger.info(f"Max Energy Splitting: {min_energy_splitting}")
#
# logger.info(f"================================================")
# logger.info(f"MIN TrainingTime : {min_trainingTime}")
# logger.info(f"MIN TrainingTime Splitting: {min_trainingtime_splitting}")
# logger.info(f"================================================")
# logger.info(f"Max TrainingTime : {max_trainingTime}")
# logger.info(f"Max TrainingTime Splitting: {max_trainingtime_splitting}")

# utils.draw_hist(title='Avg Energy of IoT Devices [2 IOT device + 1 edge]',
#                 x=energy,
#                 xlabel="Average Energy",
#                 savePath='Graphs/AllPossibleAction',
#                 pictureName='AvgEnergy')
#
# utils.draw_hist(title='Reward of all possible action[2 IOT device + 1 edge]',
#                 x=reward,
#                 xlabel="Reward",
#                 savePath='Graphs/AllPossibleAction',
#                 pictureName='Reward')
#
# utils.draw_hist(title='Training Time of all possible action[2 IOT device + 1 edge]',
#                 x=trainingTime,
#                 xlabel="TrainingTime",
#                 savePath='Graphs/AllPossibleAction',
#                 pictureName='TrainingTime')

utils.draw_scatter(title='Training Time VS Energy',
                   x=energy,
                   y=trainingTime,
                   xlabel="Energy",
                   ylabel="Training Time",
                   savePath='Graphs/AllPossibleAction',
                   pictureName='energy_trainingTime')

utils.draw_scatter(title='Reward VS Energy',
                   x=reward,
                   y=energy,
                   xlabel="reward",
                   ylabel="Energy",
                   savePath='Graphs/AllPossibleAction',
                   pictureName='reward_energy')

utils.draw_scatter(title='Reward VS trainingtime',
                   x=reward,
                   y=trainingTime,
                   xlabel="reward",
                   ylabel="Training Time",
                   savePath='Graphs/AllPossibleAction',
                   pictureName='reward_trainingTime')

utils.draw_3dGraph(x=energy,
                   y=trainingTime,
                   z=reward,
                   xlabel="Energy",
                   ylabel="Training Time",
                   zlabel="reward")
