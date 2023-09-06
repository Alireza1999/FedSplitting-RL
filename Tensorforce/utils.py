import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from System.Device import Device
from Tensorforce import config


def createDeviceFromCSV(csvFilePath: str, deviceType: str = 'cloud') -> list[Device]:
    devices = list()
    with open(csvFilePath, 'r') as device:
        csvreader = csv.reader(device)
        for row in csvreader:
            if row[0] == 'CPU Core':
                continue
            if deviceType == 'iotDevice':
                device = Device(deviceType=deviceType, CPU=int(row[0]), bandwidth=float(row[1]), edgeIndex=int(row[2]))
            elif deviceType == 'edge':
                device = Device(deviceType=deviceType, CPU=int(row[0]), bandwidth=float(row[1]), capacity=int(row[2]))
            else:
                device = Device(deviceType=deviceType, CPU=int(row[0]), bandwidth=float(row[1]))

            devices.append(device)
    return devices


def draw_graph(figSizeX, figSizeY, x, y, title, xlabel, ylabel, savePath, pictureName, saveFig=True):
    # Create a plot
    plt.figure(figsize=(int(figSizeX), int(figSizeY)))  # Set the figure size
    plt.plot(x, y)  # Plot the data
    plt.title(title)  # Add a title
    plt.xlabel(xlabel)  # Add x-axis label
    plt.ylabel(ylabel)  # Add y-axis label

    if saveFig:
        if not os.path.exists(savePath):
            os.makedirs(savePath)  # Create the directory if it does not exist
        plt.savefig(os.path.join(savePath, pictureName))  # Save the plot as PNG image

    # Show the plot
    plt.show()


def actionToLayer(splitDecision: list[float]) -> tuple[int, int]:
    """ It returns the offloading points for the given action ( op1 , op2 )"""

    totalWorkLoad = sum(config.COMP_WORK_LOAD[1:])

    op1: int
    op2: int  # Offloading points op1, op2

    op1_workload = splitDecision[0] * totalWorkLoad
    for i in range(1, config.LAYER_NUM):
        difference = abs(sum(config.COMP_WORK_LOAD[:i + 1]) - op1_workload)
        temp2 = abs(sum(config.COMP_WORK_LOAD[:i + 2]) - op1_workload)
        if temp2 >= difference:
            op1 = i
            break

    remindedWorkLoad = sum(config.COMP_WORK_LOAD[op1 + 1:]) * splitDecision[1]

    for i in range(op1 + 1, len(config.COMP_WORK_LOAD)):
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
