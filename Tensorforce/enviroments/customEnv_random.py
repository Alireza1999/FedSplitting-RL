import logging

from tensorforce import Environment

from System.Device import Device
from Tensorforce import config
from Tensorforce import utils
import numpy as np

logger = logging.getLogger()


class CustomEnvironment(Environment):

    def __init__(self, iotDevices: list[Device], edgeDevices: list[Device], cloud: Device):

        super().__init__()

        self.iotDeviceNum: int = len(iotDevices)
        self.edgeDeviceNum: int = len(edgeDevices)

        self.iotDevices: list[Device] = iotDevices
        self.edgeDevices: list[Device] = edgeDevices
        self.cloud: Device = cloud

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
        rewardOfEnergy = utils.normalizeReward(maxAmount=self.maxEnergy, minAmount=self.minEnergy,
                                               x=averageEnergyConsumption)
        rewardOfTrainingTime = utils.normalizeReward(self.maxTrainingTime, self.minTrainingTime, maxTrainingTime)
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
        logger.info(f"Edges Capacities : {edgeCapacity} \n")
        logger.info(f"Cloud Capacities : {cloudCapacity} \n")

        newState = [averageEnergyConsumption, maxTrainingTime]
        newState.extend(edgeCapacity)
        newState.append(cloudCapacity)
        newState.extend(actions)
        return reward, newState

    def execute(self, actions):
        terminal = False
        reward, newState = self.rewardFun(actions)
        return newState, terminal, reward