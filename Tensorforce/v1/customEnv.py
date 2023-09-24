import logging

import numpy as np
from tensorforce import Environment

import Tensorforce.config as config
from System.Device import Device
from Tensorforce import utils

logger = logging.getLogger()


class CustomEnvironment(Environment):

    def __init__(self, iotDevices: list[Device], edgeDevices: list[Device], cloud: Device):

        super().__init__()

        self.iotDeviceNum: int = len(iotDevices)
        self.edgeDeviceNum: int = len(edgeDevices)

        self.iotDevices: list[Device] = iotDevices
        self.edgeDevices: list[Device] = edgeDevices
        self.cloud: Device = cloud

        self.trainingTime = list()

    def states(self):
        # State = [AvgEnergy, TrainingTime, edgeCapacity, cloudCapacity, prevAction ]
        return dict(type="float", shape=(1 + 1 + self.edgeDeviceNum + 1 + self.iotDeviceNum * 2))

    def actions(self):
        return dict(type="float", shape=(self.iotDeviceNum * 2,), min_value=0.0, max_value=1.0)

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def close(self):
        super().close()

    def reset(self):
        randActions = np.random.uniform(low=0.0, high=1.0, size=(self.iotDeviceNum * 2))
        reward, newState = self.rewardFun(randActions)
        randEnergy = newState[0]
        randTrainingTime = newState[1]
        randEdgeCapacity = newState[2:len(newState) - len(randActions) - 1]
        randCloudCapacity = newState[len(newState) - len(randActions) - 1]
        state = [randEnergy, randTrainingTime]
        state.extend(randEdgeCapacity)
        state.append(randCloudCapacity)
        state.extend(randActions)
        return state
        # temp = np.append(randEnergy, randTrainingTime)
        # return np.append(temp, randActions)

    def rewardFun(self, actions):
        totalEnergyConsumption = 0
        maxTrainingTime = 0
        offloadingPointsList = []
        edgeCapacity = [edges.capacity for edges in self.edgeDevices]
        cloudCapacity = self.cloud.capacity

        for i in range(0, len(actions), 2):
            op1, op2 = utils.actionToLayer(actions[i:i + 2])
            cloudCapacity -= sum(config.COMP_WORK_LOAD[op2 + 1:])
            edgeCapacity[self.iotDevices[int(i / 2)].edgeIndex] -= sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1])

        for i in range(0, len(actions), 2):
            # Mapping float number to Offloading points
            op1, op2 = utils.actionToLayer(actions[i:i + 2])
            offloadingPointsList.append(op1)
            offloadingPointsList.append(op2)

            # computing training time of this action
            iotTrainingTime = self.iotDevices[int(i / 2)].trainingTime([op1, op2])
            edgeTrainingTime = self.edgeDevices[self.iotDevices[int(i / 2)].edgeIndex].trainingTime([op1, op2])
            cloudTrainingTime = self.cloud.trainingTime([op1, op2])

            if edgeCapacity[self.iotDevices[int(i / 2)].edgeIndex] < 0 and (actions[i] != actions[i + 1]):
                edgeTrainingTime *= (1 + abs(edgeCapacity[self.iotDevices[int(i / 2)].edgeIndex]) / 10)
            if cloudCapacity < 0 and actions[i + 1] < config.LAYER_NUM - 1:
                cloudTrainingTime *= (1 + abs(cloudCapacity) / 10)

            totalTrainingTime = iotTrainingTime + edgeTrainingTime + cloudTrainingTime
            if totalTrainingTime > maxTrainingTime:
                maxTrainingTime = totalTrainingTime

            # computing energy consumption of iot devices
            iotEnergy = self.iotDevices[int(i / 2)].energyConsumption([op1, op2])
            totalEnergyConsumption += iotEnergy

        averageEnergyConsumption = totalEnergyConsumption / self.iotDeviceNum
        rewardOfEnergy = utils.tanhActivation((-averageEnergyConsumption + 43.5) / 30) + 1
        # rewardOfTrainingTime = utils.tanhActivation((-maxTrainingTime + 120) / 200) + 1
        rewardOfTrainingTime = (-1/300) * maxTrainingTime + (120/300) + 1
        reward = (0.5 * rewardOfEnergy) + (0.5 * rewardOfTrainingTime)

        logger.info("-------------------------------------------")
        logger.info(f"Offloading layer : {offloadingPointsList} \n")
        logger.info(f"Avg Energy : {averageEnergyConsumption} \n")
        logger.info(f"Training time : {maxTrainingTime} \n")
        logger.info(f"Reward of this action : {reward} \n")
        logger.info(f"Reward of energy : {0.5 * rewardOfEnergy} \n")
        logger.info(f"Reward of training time : {0.5 * rewardOfTrainingTime} \n")
        logger.info(f"Edges Capacities : {edgeCapacity} \n")
        logger.info(f"Cloud Capacities : {cloudCapacity} \n")

        newState = [averageEnergyConsumption, maxTrainingTime]
        newState.extend(edgeCapacity)
        newState.append(cloudCapacity)
        newState.extend(actions)
        # temp = np.append(averageEnergyConsumption, maxTrainingTime)
        # newState = np.append(temp, actions)
        return reward, newState

    def execute(self, actions: list):
        terminal = False
        reward, newState = self.rewardFun(actions)
        return newState, terminal, reward

    def getTrainingTime(self):
        return self.trainingTime

    def deleteTrainingTime(self):
        self.trainingTime = []

    def appendToTrainingTime(self, data):
        self.trainingTime.append(data)
