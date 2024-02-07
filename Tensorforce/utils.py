import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from System.Device import Device
from Tensorforce import config


def createDeviceFromCSV(csvFilePath: str, deviceType: str = 'cloud') -> list[Device]:
    devices = list()
    with open(csvFilePath, 'r') as device:
        csvreader = csv.reader(device)
        for row in csvreader:
            if row[0] == 'FLOPS':
                continue
            if deviceType == 'iotDevice':
                device = Device(deviceType=deviceType, FLOPS=int(row[0]), bandwidth=float(row[1]),
                                edgeIndex=int(row[2]))
            else:
                device = Device(deviceType=deviceType, FLOPS=int(row[0]), bandwidth=float(row[1]))
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
        if temp2 >= difference:
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
