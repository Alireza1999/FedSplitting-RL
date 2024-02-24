import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from System.Device import Device
from Tensorforce import config
from Tensorforce.enviroments import customEnv_bandwidthState, customEnv, customEnvNoEdge, fedAdaptEnv
from Tensorforce.enviroments.Device_bandwidthState import Device as Device2
from Tensorforce.splittingMethods import FirstFit, PPO, TRPO, RandomAgent, NoSplitting, TensorforceAgent, AC


def createDeviceFromCSV(csvFilePath: str, deviceType: str = 'cloud') -> list[Device]:
    devices = list()
    with open(csvFilePath, 'r') as device:
        csvreader = csv.reader(device)
        for row in csvreader:
            if row[0] == 'FLOPS':
                continue
            if deviceType == 'iotDevice':
                device = Device2(deviceType=deviceType, FLOPS=int(row[0]), bandwidth=float(row[1]),
                                 edgeIndex=int(row[2]), maxPower=float(row[3]))
            else:
                device = Device2(deviceType=deviceType, FLOPS=int(row[0]), bandwidth=float(row[1]),
                                 maxPower=float(row[2]))
            devices.append(device)
    return devices


def draw_graph(figSizeX, figSizeY, x, y, title, xlabel, ylabel, savePath, pictureName, saveFig=True):
    # Create a plot
    plt.figure(figsize=(int(figSizeX), int(figSizeY)))  # Set the figure size
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if saveFig:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, pictureName))
    plt.close()
    # plt.show()


def draw_hist(x, title, xlabel, savePath, pictureName, saveFig=True):
    # Create a plot
    plt.hist(x, 10)
    plt.title(title)
    plt.xlabel(xlabel)
    if saveFig:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, pictureName))
    plt.close()
    # plt.show()


def draw_scatter(x, y, title, xlabel, ylabel, savePath, pictureName, saveFig=True):
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if saveFig:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, pictureName))
    plt.close()
    # plt.show()


def draw_3dGraph(x, y, z, xlabel, ylabel, zlabel):
    fig = go.Figure(data=[go.Mesh3d(x=x,
                                    y=y,
                                    z=z,
                                    opacity=0.7, )])

    fig.update_layout(scene=dict(xaxis_title=xlabel,
                                 yaxis_title=ylabel,
                                 zaxis_title=zlabel,
                                 xaxis_showspikes=False,
                                 yaxis_showspikes=False))

    fig.show()


def actionToLayer(splitDecision: list[float]) -> tuple[int, int]:
    """ It returns the offloading points for the given action ( op1 , op2 )"""

    totalWorkLoad = sum(config.COMP_WORK_LOAD[1:])

    op1: int
    op2: int = 0  # Offloading points op1, op2

    op1_workload = splitDecision[0] * totalWorkLoad
    for i in range(0, config.LAYER_NUM):
        difference = abs(sum(config.COMP_WORK_LOAD[:i + 1]) - op1_workload)
        temp2 = abs(sum(config.COMP_WORK_LOAD[:i + 2]) - op1_workload)
        if temp2 > difference:
            op1 = i
            break

    if splitDecision[1] != -1:
        remindedWorkLoad = sum(config.COMP_WORK_LOAD[op1 + 1:]) * splitDecision[1]

        for i in range(op1, len(config.COMP_WORK_LOAD)):
            difference = abs(sum(config.COMP_WORK_LOAD[op1 + 1:i + 1]) - remindedWorkLoad)
            temp2 = abs(sum(config.COMP_WORK_LOAD[op1 + 1:i + 2]) - remindedWorkLoad)
            if temp2 >= difference:
                op2 = i
                break
        if op2 == 0:
            op2 = op2 + 1
        if op1 == config.LAYER_NUM - 1:
            op2 = config.LAYER_NUM - 1

    return op1, op2


def sigmoidActivation(x: float) -> float:
    """ It returns 1/(1+exp(-x)). where the values lies between zero and one """

    return 1 / (1 + np.exp(-x))


def tanhActivation(x: float) -> float:
    """ It returns the value (1-exp(-2x))/(1+exp(-2x)) and the value returned will be lies in between -1 to 1."""

    return np.tanh(x)


def normalizeReward(maxAmount, minAmount, x, minNormalized, maxNormalized):
    P = [maxAmount, minNormalized]
    Q = [minAmount, maxNormalized]
    lineGradient = (P[1] - Q[1]) / (P[0] - Q[0])
    y = lineGradient * (x - Q[0]) + Q[1]
    return y


def normalizeReward_tan(x, turning_point):
    y = max(min(-pow(x - turning_point, 3) / pow(turning_point, 3), 1), -1)
    return y


def convert_To_Len_th_base(n, arr, modelLen, deviceNumber, allPossible):
    a: str = ""
    for i in range(deviceNumber * 2):
        a += str(arr[n % modelLen])
        n //= modelLen
    allPossible.append(a)


def randomSelectionSplitting(modelLen, deviceNumber) -> list[list[int]]:
    splittingForOneDevice = []
    for i in range(0, modelLen):
        for j in range(0, i + 1):
            splittingForOneDevice.append([j, i])

    result = []
    for i in range(deviceNumber):
        rand = random.randint(0, len(splittingForOneDevice) - 1)
        result.append(splittingForOneDevice[rand])
    return result


def allPossibleSplitting(modelLen, deviceNumber):
    arr = [i for i in range(0, modelLen + 1)]
    allPossible = list()
    for i in range(pow(modelLen, deviceNumber * 2)):
        # Convert i to Len th base
        convert_To_Len_th_base(i, arr, modelLen, deviceNumber, allPossible)
    result = list()
    for item in allPossible:
        isOk = True
        for j in range(0, len(item) - 1, 2):
            if int(item[j]) > int(item[j + 1]):
                isOk = False
        if isOk:
            result.append(item)
    return result


def ClassicFLTrainingTimeWithoutEdge(iotDevices, cloud):
    allTrainingTime = []
    maxTrainingTime = 0
    action = [config.LAYER_NUM - 1] * 1
    cloud.connectedDevice = 0

    iotRemainingFLOP = [iot.FLOPS for iot in iotDevices]
    cloudRemainingFLOP = cloud.FLOPS

    for i in range(0, len(iotDevices)):
        op = action[0]
        cloudRemainingFLOP -= sum(config.COMP_WORK_LOAD[op + 1:])
        iotRemainingFLOP[int(i)] -= sum(config.COMP_WORK_LOAD[0:op + 1])
        if sum(config.COMP_WORK_LOAD[op + 1:]) != 0:
            cloud.connectedDevice += 1

    for i in range(0, len(iotDevices)):
        # Mapping float number to Offloading points
        op = action[0]
        # computing training time of this action
        iotTrainingTime = iotDevices[int(i)].trainingTime(splitPoints=[op, op],
                                                          remainingFlops=iotRemainingFLOP[int(i)],
                                                          preTrain=True)
        cloudTrainingTime = cloud.trainingTime([op, op],
                                               remainingFlops=cloudRemainingFLOP,
                                               preTrain=True)

        totalTrainingTime = iotTrainingTime + cloudTrainingTime
        allTrainingTime.append(totalTrainingTime)

        if totalTrainingTime > maxTrainingTime:
            maxTrainingTime = totalTrainingTime
    return allTrainingTime


def ClassicFLTrainingTime(iotDevices, edgeDevices, cloud):
    offloadingPointsList = []
    allTrainingTime = []
    maxTrainingTime = 0
    totalEnergyConsumption = 0
    total_comp_e = 0
    total_comm_e = 0

    action = [[config.LAYER_NUM - 1, config.LAYER_NUM - 1]] * len(iotDevices)
    for i in range(len(action)):
        edgeDevices[iotDevices[i].edgeIndex].connectedDevice = 0
        cloud.connectedDevice = 0

    iotRemainingFLOP = [iot.FLOPS for iot in iotDevices]
    edgeRemainingFLOP = [edge.FLOPS for edge in edgeDevices]
    cloudRemainingFLOP = cloud.FLOPS

    for i in range(len(action)):
        op1 = action[i][0]
        op2 = action[i][1]
        cloudRemainingFLOP -= sum(config.COMP_WORK_LOAD[op2 + 1:])
        edgeRemainingFLOP[iotDevices[i].edgeIndex] -= sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1])
        iotRemainingFLOP[i] -= sum(config.COMP_WORK_LOAD[0:op1 + 1])

        if sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1]):
            edgeDevices[iotDevices[i].edgeIndex].connectedDevice += 1
        if sum(config.COMP_WORK_LOAD[op2 + 1:]) != 0:
            cloud.connectedDevice += 1

    for i in range(len(action)):
        # Mapping float number to Offloading points
        op1 = action[i][0]
        op2 = action[i][1]
        offloadingPointsList.append(op1)
        offloadingPointsList.append(op2)

        # computing training time of this action
        iot_comp_e, iot_comm_e, iot_comp_tt, iot_comm_tt = iotDevices[i].energy_tt(splitPoints=[op1, op2],
                                                                                   remainingFlops=iotRemainingFLOP[i])
        _, _, edge_comp_tt, edge_comm_tt = edgeDevices[iotDevices[i].edgeIndex] \
            .energy_tt(splitPoints=[op1, op2],
                       remainingFlops=edgeRemainingFLOP[iotDevices[i].edgeIndex])
        _, _, cloud_comp_tt, cloud_comm_tt = cloud.energy_tt([op1, op2], remainingFlops=cloudRemainingFLOP)

        totalTrainingTime = (iot_comm_tt + iot_comp_tt) + (edge_comm_tt + edge_comp_tt) + (
                cloud_comm_tt + cloud_comp_tt)
        allTrainingTime.append(totalTrainingTime)

        if totalTrainingTime > maxTrainingTime:
            maxTrainingTime = totalTrainingTime

        # computing energy consumption of iot devices
        total_comp_e += iot_comp_e
        total_comm_e += iot_comm_e

    totalEnergyConsumption = (total_comm_e + total_comp_e)
    avgCommE = total_comm_e / len(iotDevices)
    avgCompE = total_comp_e / len(iotDevices)
    averageEnergyConsumption = totalEnergyConsumption / len(iotDevices)
    return averageEnergyConsumption, maxTrainingTime


def minMaxAvgEnergy(iotDevices, edgeDevices, cloud):
    splittingLayer = allPossibleSplitting(modelLen=config.LAYER_NUM, deviceNumber=1)
    maxAvgEnergyOfOneDevice = 0
    minAvgEnergyOfOneDevice = 1.0e7
    maxEnergySplitting = []
    minEnergySplitting = []

    for splitting in splittingLayer:
        splittingArray = list()
        for char in splitting:
            splittingArray.append(int(char))

        avgEnergyOfOneDevice, trainingTimeOfOneDevice = preTrainEnv(iotDevices=iotDevices, edgeDevices=edgeDevices,
                                                                    cloud=cloud,
                                                                    action=splittingArray * len(iotDevices))
        if avgEnergyOfOneDevice > maxAvgEnergyOfOneDevice:
            maxAvgEnergyOfOneDevice = avgEnergyOfOneDevice
            maxEnergySplitting = splittingArray * len(iotDevices)
        if avgEnergyOfOneDevice < minAvgEnergyOfOneDevice:
            minAvgEnergyOfOneDevice = avgEnergyOfOneDevice
            minEnergySplitting = splittingArray * len(iotDevices)

    maxAvgEnergy, maxEnergyTrainingTime = preTrainEnv(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                                                      action=maxEnergySplitting)
    minAvgEnergy, minEnergyTrainingTime = preTrainEnv(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                                                      action=minEnergySplitting)
    print(f"Max Energy Splitting : {maxEnergySplitting}\nMin Energy Splitting : {minEnergySplitting}")
    print(f"Max Energy Training Time : {maxEnergyTrainingTime}\nMin Energy Training Time : {minEnergyTrainingTime}")
    return maxAvgEnergy, minAvgEnergy


def createEnv(iotDevices, edgeDevices, cloud, fraction, rewardTuningParams,
              envType=None, groupNum=1):
    if envType == 'default':
        return customEnv.CustomEnvironment(rewardTuningParams=rewardTuningParams, iotDevices=iotDevices,
                                           edgeDevices=edgeDevices, cloud=cloud, fraction=fraction)
    elif envType == "fedAdapt":
        return fedAdaptEnv.FedAdaptEnv(allTrainingTime=rewardTuningParams,
                                       iotDevices=iotDevices,
                                       cloud=cloud,
                                       groupNum=groupNum)
    elif envType == "defaultNoEdge":
        return customEnvNoEdge.CustomEnvironmentNoEdge(rewardTuningParams=rewardTuningParams,
                                                       iotDevices=iotDevices,
                                                       cloud=cloud)
    elif envType == "defaultWithBandwidth":
        return customEnv_bandwidthState.CustomEnvironment(rewardTuningParams=rewardTuningParams,
                                                          iotDevices=iotDevices,
                                                          edgeDevices=edgeDevices,
                                                          cloud=cloud,
                                                          fraction=fraction)
    else:
        raise "Invalid Environment Parameter. Valid option : default, fedAdapt, defaultNoEdge"


def createAgent(agentType, fraction, timestepNum, environment, saveSummariesPath, iotDevices=None, edgeDevices=None,
                cloud=None):
    if agentType == 'ppo':
        return PPO.create(fraction=fraction, environment=environment, timestepNum=timestepNum,
                          saveSummariesPath=saveSummariesPath)
    elif agentType == 'ac':
        return AC.create(fraction=fraction, environment=environment, timestepNum=timestepNum,
                         saveSummariesPath=saveSummariesPath)
    elif agentType == 'tensorforce':
        return TensorforceAgent.create(fraction=fraction, environment=environment,
                                       timestepNum=timestepNum, saveSummariesPath=saveSummariesPath)
    elif agentType == 'trpo':
        return TRPO.create(fraction=fraction, environment=environment,
                           timestepNum=timestepNum, saveSummariesPath=saveSummariesPath)
    elif agentType == 'random':
        return RandomAgent.RandomAgent(environment=environment)
    elif agentType == 'noSplitting':
        return NoSplitting.NoSplitting(environment=environment)
    elif agentType == 'firstFit':
        return FirstFit.FirstFit(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud)
    else:
        raise Exception('Invalid config select from [ppo, ac, tensorforce, random]')


def createLog(fileName):
    logging.basicConfig(filename=f"./Logs/{fileName}.log",
                        format='%(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


def preTrainEnv(iotDevices: list[Device], edgeDevices: list[Device], cloud: Device, action) -> tuple[float, float]:
    edgesConnectedDeviceNum = [0] * len(edgeDevices)

    for i in range(0, len(action), 2):
        edgeDevices[iotDevices[int(i / 2)].edgeIndex].connectedDevice = 0
        cloud.connectedDevice = 0

    totalEnergyConsumption = 0
    maxTrainingTime = 0
    offloadingPointsList = []

    iotRemainingFLOP = [iot.FLOPS for iot in iotDevices]
    edgeRemainingFLOP = [edge.FLOPS for edge in edgeDevices]
    cloudRemainingFLOP = cloud.FLOPS

    for i in range(0, len(action), 2):
        op1 = action[0]
        op2 = action[1]
        cloudRemainingFLOP -= sum(config.COMP_WORK_LOAD[op2 + 1:])
        edgeRemainingFLOP[iotDevices[int(i / 2)].edgeIndex] -= sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1])
        iotRemainingFLOP[int(i / 2)] -= sum(config.COMP_WORK_LOAD[0:op1 + 1])

        if sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1]):
            edgeDevices[iotDevices[int(i / 2)].edgeIndex].connectedDevice += 1
            edgesConnectedDeviceNum[iotDevices[int(i / 2)].edgeIndex] += 1
        if sum(config.COMP_WORK_LOAD[op2 + 1:]) != 0:
            cloud.connectedDevice += 1
    print(f"Action : {action}\nRemaining FLOP : {iotRemainingFLOP}\n{edgeRemainingFLOP}\n{cloudRemainingFLOP}")
    for i in range(0, len(action), 2):
        # Mapping float number to Offloading points
        op1 = action[0]
        op2 = action[1]
        offloadingPointsList.append(op1)
        offloadingPointsList.append(op2)

        # computing training time of this action
        iotEnergy, iotTrainingTime = iotDevices[int(i / 2)] \
            .energy_tt(splitPoints=[op1, op2], remainingFlops=iotRemainingFLOP[int(i / 2)], preTrain=True)

        _, edgeTrainingTime = edgeDevices[iotDevices[int(i / 2)].edgeIndex] \
            .energy_tt(splitPoints=[op1, op2], remainingFlops=edgeRemainingFLOP[iotDevices[int(i / 2)].edgeIndex],
                       preTrain=True)

        _, cloudTrainingTime = cloud.energy_tt([op1, op2], remainingFlops=cloudRemainingFLOP, preTrain=True)

        totalTrainingTime = iotTrainingTime + edgeTrainingTime + cloudTrainingTime
        if totalTrainingTime > maxTrainingTime:
            maxTrainingTime = totalTrainingTime

        # computing energy consumption of iot devices
        totalEnergyConsumption += iotEnergy
    averageEnergyConsumption = totalEnergyConsumption / len(iotDevices)

    return averageEnergyConsumption, maxTrainingTime


def preTrain(iotDevices, edgeDevices, cloud):
    rewardTuningParams = [0, 0, 0, 0]
    min_Energy = 1.0e7
    max_Energy = 0

    min_trainingTime = 1.0e7
    max_trainingTime = 0

    splittingLayer = allPossibleSplitting(modelLen=config.LAYER_NUM - 1, deviceNumber=len(iotDevices))

    for splitting in splittingLayer:
        splittingArray = list()
        for char in splitting:
            splittingArray.append(int(char))

        avgEnergy, trainingTime = preTrainEnv(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                                              action=splittingArray)
        if avgEnergy < min_Energy:
            min_Energy = avgEnergy
            rewardTuningParams[0] = min_Energy
            min_energy_splitting = splittingArray
            min_Energy_TrainingTime = trainingTime
        if avgEnergy > max_Energy:
            max_Energy = avgEnergy
            rewardTuningParams[1] = max_Energy
            max_Energy_splitting = splittingArray
            max_Energy_TrainingTime = trainingTime

        if trainingTime < min_trainingTime:
            min_trainingTime = trainingTime
            rewardTuningParams[2] = min_trainingTime
            min_trainingtime_splitting = splittingArray
            min_trainingTime_energy = avgEnergy
        if trainingTime > max_trainingTime:
            max_trainingTime = trainingTime
            rewardTuningParams[3] = max_trainingTime
            max_trainingtime_splitting = splittingArray
            max_trainingTime_energy = avgEnergy
    return rewardTuningParams
