import logging
import random

import numpy as np
from tensorforce import Environment

import Tensorforce.config as config
from System.Device import Device
from Tensorforce import utils

logger = logging.getLogger()


class CustomEnvironment(Environment):

    def __init__(self, rewardTuningParams, iotDevices: list[Device], edgeDevices: list[Device], cloud: Device,
                 fraction=0.8):
        super().__init__()

        self.iotDeviceNum: int = len(iotDevices)
        self.edgeDeviceNum: int = len(edgeDevices)

        self.iotDevices: list[Device] = iotDevices
        self.edgeDevices: list[Device] = edgeDevices
        self.cloud: Device = cloud

        self.minEnergy = rewardTuningParams[0]
        self.maxEnergy = rewardTuningParams[1]
        self.minTrainingTime = rewardTuningParams[2]
        self.maxTrainingTime = rewardTuningParams[3]

        self.fraction = fraction

    def states(self):
        # State = [AvgEnergy, TrainingTime, edge capacity, cloud capacity, prevAction ]
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

    def rewardFun(self, actions):
        totalEnergyConsumption = 0
        maxTrainingTime = 0
        offloadingPointsList = []

        iotDeviceCapacity = [iotdevice.capacity for iotdevice in self.iotDevices]
        edgeCapacity = [edges.capacity for edges in self.edgeDevices]
        cloudCapacity = self.cloud.capacity

        for i in range(0, len(actions), 2):
            op1, op2 = utils.actionToLayer(actions[i:i + 2])
            cloudCapacity -= sum(config.COMP_WORK_LOAD[op2 + 1:])
            edgeCapacity[self.iotDevices[int(i / 2)].edgeIndex] -= sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1])
            iotDeviceCapacity[int(i / 2)] -= sum(config.COMP_WORK_LOAD[0:op1 + 1])

        for i in range(0, len(actions), 2):
            # Mapping float number to Offloading points
            op1, op2 = utils.actionToLayer(actions[i:i + 2])
            offloadingPointsList.append(op1)
            offloadingPointsList.append(op2)

            # computing training time of this action
            iotTrainingTime = self.iotDevices[int(i / 2)].trainingTime([op1, op2])
            edgeTrainingTime = self.edgeDevices[self.iotDevices[int(i / 2)].edgeIndex].trainingTime([op1, op2])
            cloudTrainingTime = self.cloud.trainingTime([op1, op2])

            if iotDeviceCapacity[int(i / 2)] < 0:
                iotTrainingTime += abs(iotDeviceCapacity[int(i / 2)])
            if edgeCapacity[self.iotDevices[int(i / 2)].edgeIndex] < 0 and (actions[i] != actions[i + 1]):
                edgeTrainingTime += 50 * abs(edgeCapacity[self.iotDevices[int(i / 2)].edgeIndex])
            if cloudCapacity < 0 and actions[i + 1] < config.LAYER_NUM - 1:
                cloudTrainingTime += 50 * abs(cloudCapacity)

            totalTrainingTime = iotTrainingTime + edgeTrainingTime + cloudTrainingTime
            if totalTrainingTime > maxTrainingTime:
                maxTrainingTime = totalTrainingTime

            # computing energy consumption of iot devices
            iotEnergy = self.iotDevices[int(i / 2)].energyConsumption([op1, op2])
            totalEnergyConsumption += iotEnergy

        averageEnergyConsumption = totalEnergyConsumption / self.iotDeviceNum

        rewardOfEnergy = utils.normalizeReward(maxAmount=self.maxEnergy, minAmount=self.minEnergy,
                                               x=averageEnergyConsumption)
        rewardOfTrainingTime = utils.normalizeReward(maxAmount=self.maxTrainingTime, minAmount=self.minTrainingTime,
                                                     x=maxTrainingTime)

        if self.fraction <= 1:
            reward = (self.fraction * rewardOfEnergy) + ((1 - self.fraction) * rewardOfTrainingTime)

        else:
            raise Exception("Fraction must be less than 1")

        logger.info("-------------------------------------------")
        logger.info(f"Offloading layer : {offloadingPointsList} \n")
        logger.info(f"Avg Energy : {averageEnergyConsumption} \n")
        logger.info(f"Training time : {maxTrainingTime} \n")
        logger.info(f"Reward of this action : {reward} \n")
        logger.info(f"Reward of energy : {self.fraction * rewardOfEnergy} \n")
        logger.info(f"Reward of training time : {(1 - self.fraction) * rewardOfTrainingTime} \n")
        logger.info(f"IOTs Capacities : {iotDeviceCapacity} \n")
        logger.info(f"Edges Capacities : {edgeCapacity} \n")
        logger.info(f"Cloud Capacities : {cloudCapacity} \n")

        newState = [averageEnergyConsumption, maxTrainingTime]
        newState.extend(edgeCapacity)
        newState.append(cloudCapacity)
        newState.extend(actions)
        return reward, newState

    def execute(self, actions: list):
        terminal = False
        reward, newState = self.rewardFun(actions)
        return newState, terminal, reward
