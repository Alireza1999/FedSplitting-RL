import logging

import numpy as np
from tensorforce import Environment

import Tensorforce.config as config
from entities.Device import Device
from Tensorforce import utils

logger = logging.getLogger()


class CustomEnvironmentNoEdge(Environment):

    def __init__(self, rewardTuningParams, iotDevices: list[Device], cloud: Device, fraction=0.0):
        super().__init__()

        self.iotDeviceNum: int = len(iotDevices)

        self.iotDevices: list[Device] = iotDevices
        self.cloud: Device = cloud

        self.ClassicFLTrainingTime = rewardTuningParams[2]
        self.rewardOfEnergy = 0
        self.rewardOfTrainingTime = 0
        self.effectiveBandwidth = [[self.iotDevices[i].bandwidth] for i in range(self.iotDeviceNum)]

        self.fraction = fraction

    def states(self):

        # State = [AvgEnergy, TrainingTime, number of connected device to each edge, number of devices connected to
        # cloud , prevAction ]
        # return dict(type="float", shape=(1 + 1 + self.edgeDeviceNum + 1 + self.iotDeviceNum * 2))

        # State = [maxTrainingTime , TrainingTime of Each device, number of devices connected to cloud ,
        # prevAction]
        return dict(type="float", shape=(1 + self.iotDeviceNum + 1 + self.iotDeviceNum))

    def actions(self):
        return dict(type="float", shape=(self.iotDeviceNum,), min_value=0.0, max_value=1.0)

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def close(self):
        super().close()

    def reset(self):
        randActions = np.random.uniform(low=0.0, high=1.0, size=(self.iotDeviceNum))
        # randActions = [config.LAYER_NUM-1] * 2 * self.iotDeviceNum
        reward, newState = self.rewardFun(randActions)
        return newState

    def rewardFun(self, action):
        allTrainingTimes = []
        self.cloud.connectedDevice = 0
        totalEnergyConsumption = 0
        maxTrainingTime = 0
        offloadingPointsList = []

        iotRemainingFLOP = [iot.FLOPS for iot in self.iotDevices]
        cloudRemainingFLOP = self.cloud.FLOPS

        for i in range(0, len(action)):
            op, _ = utils.actionToLayer([action[i], -1])
            cloudRemainingFLOP -= sum(config.COMP_WORK_LOAD[op + 1:])
            iotRemainingFLOP[i] -= sum(config.COMP_WORK_LOAD[0:op + 1])

            if sum(config.COMP_WORK_LOAD[op + 1:]) != 0:
                self.cloud.connectedDevice += 1

        for i in range(0, len(action)):
            # Mapping float number to Offloading points
            op, _ = utils.actionToLayer([action[i], -1])
            offloadingPointsList.append(op)

            # computing training time of this action
            iotTrainingTime = self.iotDevices[int(i / 2)].trainingTime([op, op], remainingFlops=iotRemainingFLOP[i])
            cloudTrainingTime = self.cloud.trainingTime([op, op], remainingFlops=cloudRemainingFLOP)

            self.effectiveBandwidth[i].append(self.iotDevices[i].effectiveBandwidth)

            totalTrainingTime = iotTrainingTime + cloudTrainingTime
            allTrainingTimes.append(totalTrainingTime)
            if totalTrainingTime > maxTrainingTime:
                maxTrainingTime = totalTrainingTime

            # computing energy consumption of iot devices
        rewardOfTrainingTime = maxTrainingTime
        rewardOfTrainingTime -= 600
        rewardOfTrainingTime /= 100
        rewardOfTrainingTime *= -1

        rewardOfTrainingTime = min(max(rewardOfTrainingTime, -1), 1)

        self.rewardOfTrainingTime = (1 - self.fraction) * rewardOfTrainingTime

        if self.fraction <= 1:
            reward = (1 - self.fraction) * rewardOfTrainingTime
        else:
            raise Exception("Fraction must be less than 1")

        logger.info("-------------------------------------------")
        logger.info(f"Offloading layer : {offloadingPointsList} \n")
        logger.info(f"Training time : {maxTrainingTime} \n")
        logger.info(f"Reward of this action : {reward} \n")
        logger.info(f"Reward of training time : {(1 - self.fraction) * rewardOfTrainingTime} \n")
        logger.info(f"IOTs Capacities : {iotRemainingFLOP} \n")
        logger.info(f"Cloud Capacities : {cloudRemainingFLOP} \n")

        newState = [maxTrainingTime]
        newState.extend(allTrainingTimes)
        newState.append(self.cloud.connectedDevice)
        newState.extend(action)
        logger.info(f"New State : {newState} \n")

        return reward, newState

    def execute(self, actions: list):
        terminal = False
        reward, newState = self.rewardFun(actions)
        return newState, terminal, reward
