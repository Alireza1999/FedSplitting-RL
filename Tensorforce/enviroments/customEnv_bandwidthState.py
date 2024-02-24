import logging

import numpy as np
from tensorforce import Environment

import Tensorforce.config as config
from Tensorforce import utils
from entities.Device_bandwidthState import Device

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

        self.ClassicFLEnergy = rewardTuningParams[0]
        self.ClassicFLTrainingTime = rewardTuningParams[1]

        self.avgEnergy = 0
        self.tt = 0
        self.rewardOfEnergy = 0
        self.rewardOfTrainingTime = 0
        self.energyOfComputation = 0
        self.energyOfCommunication = 0
        self.trainingTimeOfComputation = 0
        self.trainingTimeOfCommunication = 0

        self.effectiveBandwidth = []

        self.fraction = fraction

    def states(self):
        # State = [Bandwidth of each client, bandwidth of each edge, MaxTrainingTime, EnergyConsumption]
        return dict(type="float", shape=(self.iotDeviceNum + self.edgeDeviceNum,))

    def actions(self):
        return dict(type="float", shape=(self.iotDeviceNum * 2,), min_value=0.0, max_value=1.0)

    def setBandwidth(self, bandwidth: list[float]):
        self.effectiveBandwidth = bandwidth

    def getBandwidth(self):
        return self.effectiveBandwidth

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def close(self):
        super().close()

    def reset(self):
        iotBandwidths = []
        for iotDevice in self.iotDevices:
            iotBandwidths.append(np.random.uniform(low=iotDevice.bandwidth * 0.5, high=iotDevice.bandwidth))

        edgeBandwidths = []
        for edgeDevice in self.edgeDevices:
            edgeBandwidths.append(np.random.uniform(low=edgeDevice.bandwidth * 0.5, high=edgeDevice.bandwidth))

        self.setBandwidth(bandwidth=np.concatenate((iotBandwidths,edgeBandwidths),axis=0))
        return self.getBandwidth()

    def rewardFun(self, action):
        allTrainingTimes = []
        total_comp_e = 0
        total_comm_e = 0
        edgesConnectedDeviceNum = [0] * self.edgeDeviceNum

        for i in range(0, len(action), 2):
            self.iotDevices[int(i / 2)].setEffectiveBW(self.effectiveBandwidth[int(i / 2)])
            self.edgeDevices[self.iotDevices[int(i / 2)].edgeIndex].connectedDevice = 0
            self.cloud.connectedDevice = 0

        for i in range(self.edgeDeviceNum):
            self.edgeDevices[i].setEffectiveBW(self.effectiveBandwidth[i + self.iotDeviceNum])

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
            iot_comp_e, iot_comm_e, iot_comp_tt, iot_comm_tt = self.iotDevices[int(i / 2)].energy_tt(
                splitPoints=[op1, op2],
                remainingFlops=iotRemainingFLOP[int(i / 2)])
            _, _, edge_comp_tt, edge_comm_tt = self.edgeDevices[self.iotDevices[int(i / 2)].edgeIndex] \
                .energy_tt(splitPoints=[op1, op2],
                           remainingFlops=edgeRemainingFLOP[self.iotDevices[int(i / 2)].edgeIndex])
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
        rewardOfTrainingTime -= self.ClassicFLTrainingTime
        rewardOfTrainingTime /= 5
        rewardOfTrainingTime *= -1

        rewardOfTrainingTime = min(max(rewardOfTrainingTime, -1), 1)

        rewardOfEnergy = averageEnergyConsumption
        rewardOfEnergy -= self.ClassicFLEnergy
        rewardOfEnergy /= 100
        rewardOfEnergy *= -1

        rewardOfEnergy = min(max(rewardOfEnergy, -1), 1)

        self.avgEnergy = averageEnergyConsumption
        self.tt = maxTrainingTime
        self.rewardOfEnergy = (self.fraction * rewardOfEnergy)
        self.rewardOfTrainingTime = (1 - self.fraction) * rewardOfTrainingTime

        if self.fraction <= 1:
            reward = (self.fraction * rewardOfEnergy) + ((1 - self.fraction) * rewardOfTrainingTime)
        else:
            raise Exception("Fraction must be less than 1")

        logger.info("-------------------------------------------")
        logger.info(f"Offloading layer : {offloadingPointsList} \n")
        logger.info(f"Avg Energy : {averageEnergyConsumption} \n")
        logger.info(f"Training time : {maxTrainingTime} \n")
        logger.info(f"Bandwidth : {maxTrainingTime} \n")
        logger.info(f"Reward of this action : {reward} \n")
        logger.info(f"Reward of energy : {self.fraction * rewardOfEnergy} \n")
        logger.info(f"Reward of training time : {(1 - self.fraction) * rewardOfTrainingTime} \n")
        logger.info(f"IOTs Capacities : {iotRemainingFLOP} \n")
        logger.info(f"Edges Capacities : {edgeRemainingFLOP} \n")
        logger.info(f"Cloud Capacities : {cloudRemainingFLOP} \n")

        iotBandwidths = []
        for iotDevice in self.iotDevices:
            iotBandwidths.append(np.random.uniform(low=iotDevice.bandwidth * 0.5, high=iotDevice.bandwidth))

        edgeBandwidths = []
        for edgeDevice in self.edgeDevices:
            edgeBandwidths.append(np.random.uniform(low=edgeDevice.bandwidth * 0.5, high=edgeDevice.bandwidth))
        newState = np.concatenate((iotBandwidths,edgeBandwidths),axis=0)
        self.setBandwidth(newState)
        return reward, newState

    def execute(self, actions: list):
        terminal = False
        reward, newState = self.rewardFun(actions)
        return newState, terminal, reward
