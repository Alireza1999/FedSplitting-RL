import logging

import numpy as np
from tensorforce import Environment

import Tensorforce.config as config
from entities.Device import Device
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

        # self.maxEnergy = rewardTuningParams[0]
        # self.minEnergy = rewardTuningParams[1]
        self.ClassicFLEnergy = rewardTuningParams[0]
        self.ClassicFLTrainingTime = rewardTuningParams[1]
        self.rewardOfEnergy = 0
        self.rewardOfTrainingTime = 0
        self.effectiveBandwidth = [[self.iotDevices[i].bandwidth] for i in range(self.iotDeviceNum)]

        self.fraction = fraction

    def states(self):

        # State = [AvgEnergy, TrainingTime, number of connected device to each edge, number of devices connected to
        # cloud , prevAction ]
        # return dict(type="float", shape=(1 + 1 + self.edgeDeviceNum + 1 + self.iotDeviceNum * 2))

        # State = [AvgEnergy ,maxTrainingTime , TrainingTime of Each device,  number of connected device to each edge,
        # number of devices connected to cloud , prevAction]
        return dict(type="float", shape=(1 + 1 + self.iotDeviceNum + self.edgeDeviceNum + 1 + self.iotDeviceNum * 2))

    def actions(self):
        return dict(type="float", shape=(self.iotDeviceNum * 2,), min_value=0.0, max_value=1.0)

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def close(self):
        super().close()

    def reset(self):
        randActions = np.random.uniform(low=0.0, high=1.0, size=(self.iotDeviceNum * 2))
        # randActions = [config.LAYER_NUM-1] * 2 * self.iotDeviceNum
        reward, newState = self.rewardFun(randActions)
        return newState

    def rewardFun(self, action):
        total_comp_e = 0
        total_comm_e = 0
        allTrainingTimes = []
        edgesConnectedDeviceNum = [0] * self.edgeDeviceNum

        for i in range(0, len(action), 2):
            self.edgeDevices[self.iotDevices[int(i / 2)].edgeIndex].connectedDevice = 0
            self.cloud.connectedDevice = 0

        totalEnergyConsumption = 0
        maxTrainingTime = 0
        offloadingPointsList = []

        iotRemainingFLOP = [iot.FLOPS for iot in self.iotDevices]
        edgeRemainingFLOP = [edge.FLOPS for edge in self.edgeDevices]
        cloudRemainingFLOP = self.cloud.FLOPS

        for i in range(0, len(action), 2):
            op1, op2 = utils.actionToLayer(action[i:i + 2])
            cloudRemainingFLOP -= sum(config.COMP_WORK_LOAD[op2 + 1:])
            edgeRemainingFLOP[self.iotDevices[int(i / 2)].edgeIndex] -= sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1])
            iotRemainingFLOP[int(i / 2)] -= sum(config.COMP_WORK_LOAD[0:op1 + 1])

            if sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1]):
                self.edgeDevices[self.iotDevices[int(i / 2)].edgeIndex].connectedDevice += 1
                edgesConnectedDeviceNum[self.iotDevices[int(i / 2)].edgeIndex] += 1
            if sum(config.COMP_WORK_LOAD[op2 + 1:]) != 0:
                self.cloud.connectedDevice += 1

        for i in range(0, len(action), 2):
            # Mapping float number to Offloading points
            op1, op2 = utils.actionToLayer(action[i:i + 2])
            offloadingPointsList.append(op1)
            offloadingPointsList.append(op2)

            # computing training time of this action
            iot_comp_e, iot_comm_e, iot_comp_tt, iot_comm_tt = self.iotDevices[int(i/2)].energy_tt(splitPoints=[op1, op2],
                                                                                                   remainingFlops=iotRemainingFLOP[int(i/2)])
            _, _, edge_comp_tt, edge_comm_tt = self.edgeDevices[self.iotDevices[int(i/2)].edgeIndex] \
                .energy_tt(splitPoints=[op1, op2],
                           remainingFlops=edgeRemainingFLOP[self.iotDevices[int(i/2)].edgeIndex])
            _, _, cloud_comp_tt, cloud_comm_tt = self.cloud.energy_tt([op1, op2], remainingFlops=cloudRemainingFLOP)

            totalTrainingTime = (iot_comm_tt + iot_comp_tt) + (edge_comm_tt + edge_comp_tt) + (
                    cloud_comm_tt + cloud_comp_tt)
            allTrainingTimes.append(totalTrainingTime)

            if totalTrainingTime > maxTrainingTime:
                maxTrainingTime = totalTrainingTime

            # computing energy consumption of iot devices
            total_comp_e += iot_comp_e
            total_comm_e += iot_comm_e

        totalEnergyConsumption = (total_comm_e + total_comp_e)
        averageEnergyConsumption = totalEnergyConsumption / self.iotDeviceNum

        rewardOfTrainingTime = maxTrainingTime
        rewardOfTrainingTime -= (self.ClassicFLTrainingTime)
        rewardOfTrainingTime /= 4
        rewardOfTrainingTime *= -1

        rewardOfTrainingTime = min(max(rewardOfTrainingTime, -1), 1)

        rewardOfEnergy = averageEnergyConsumption
        rewardOfEnergy -= self.ClassicFLEnergy
        rewardOfEnergy /= 100
        rewardOfEnergy *= -1

        rewardOfEnergy = min(max(rewardOfEnergy, -1), 1)

        # rewardOfEnergy = utils.normalizeReward(maxAmount=self.maxEnergy, minAmount=self.minEnergy,
        #                                        x=averageEnergyConsumption, minNormalized=-1, maxNormalized=1)

        self.rewardOfEnergy = (self.fraction * rewardOfEnergy)
        self.rewardOfTrainingTime = (1 - self.fraction) * rewardOfTrainingTime

        if self.fraction <= 1:
            reward = (self.fraction * rewardOfEnergy) + ((1 - self.fraction) * rewardOfTrainingTime)
        else:
            raise Exception("Fraction must be less than 1")

        logger.info("-------------------------------------------")
        logger.info(f"Offloading layer: {offloadingPointsList} \n")
        logger.info(f"Avg Energy: {averageEnergyConsumption} \n")
        logger.info(f"Training time: {maxTrainingTime} \n")
        logger.info(f"Reward of this action: {reward} \n")
        logger.info(f"Reward of energy: {self.fraction * rewardOfEnergy} \n")
        logger.info(f"Reward of training time: {(1 - self.fraction) * rewardOfTrainingTime} \n")
        logger.info(f"IOTs Capacities: {iotRemainingFLOP} \n")
        logger.info(f"Edges Capacities: {edgeRemainingFLOP} \n")
        logger.info(f"Cloud Capacities: {cloudRemainingFLOP} \n")

        # newState = [1 - utils.normalizeReward(self.maxEnergy, self.minEnergy, averageEnergyConsumption),
        #             1 - utils.normalizeReward(800, 49, maxTrainingTime)]
        #
        # allTrainingTimes = [1 - utils.normalizeReward(650, 49, trainingTime) for trainingTime in allTrainingTimes]

        newState = [averageEnergyConsumption, maxTrainingTime]
        newState.extend(allTrainingTimes)
        newState.extend(edgesConnectedDeviceNum)
        newState.append(self.cloud.connectedDevice)
        # newState.extend(edgeRemainingFLOP)
        # newState.append(cloudRemainingFLOP)
        newState.extend(action)
        logger.info(f"New State : {newState} \n")

        return reward, newState

    def execute(self, actions: list):
        terminal = False
        reward, newState = self.rewardFun(actions)
        return newState, terminal, reward
