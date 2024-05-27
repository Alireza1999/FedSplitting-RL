import logging
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config as config
import utils
from entities.Device_bandwidthState import Device

logger = logging.getLogger()


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, rewardTuningParams, iotDevices: list, edgeDevices: list, cloud: Device,
                 fraction=0.8, ep_length: int = 100):
        super().__init__()

        self.iotDeviceNum: int = len(iotDevices)
        self.edgeDeviceNum: int = len(edgeDevices)

        self.iotDevices: list = iotDevices
        self.edgeDevices: list = edgeDevices
        self.cloud: Device = cloud

        self.ClassicFLEnergy = rewardTuningParams[0]
        self.ClassicFLTrainingTime = rewardTuningParams[1]
        self.currentClassicFLEnergy = 0
        self.currentClassicFLTrainingTime = 0

        self.currentAction = [0] * self.iotDeviceNum * 2
        self.currentOffloadingPoint = [config.LAYER_NUM - 1] * self.iotDeviceNum * 2
        self.currentState = dict()

        self.currentTimestep = 0
        self.currentEpisode = 0
        self.cumulativeEnergy = 0
        self.cumulativeTT = 0

        self.avgEnergy = 0
        self.tt = 0
        self.rewardOfEnergy = 0
        self.rewardOfTrainingTime = 0
        self.energyOfComputation = 0
        self.energyOfCommunication = 0
        self.trainingTimeOfComputation = 0
        self.trainingTimeOfCommunication = 0

        self.timestep_energy = []
        self.timestep_tt = []
        self.timestep_reward = []

        self.episode_energy = []
        self.episode_tt = []
        self.episode_reward = []

        self.effectiveBandwidth = []

        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.

        self.fraction = fraction

        # discrete action : 0% offload, 25% offload, 50% offload, 100% offload
        # [6,6] , [0, 2] , [5, 6], [3, 4]
        action_spec = [4] * self.iotDeviceNum
        print(f"action spec: {action_spec}")
        self.action_space = spaces.MultiDiscrete(action_spec)

        # bandwidths : 0% fluctuation, 10% fluctuation, 20% fluctuation,..., 90% fluctuation.
        observation_spec = [10] * (self.iotDeviceNum + self.edgeDeviceNum)
        self.observation_space = spaces.MultiDiscrete(observation_spec)

    def rewardFun(self, action):
        offloadingPoint = self.actionToOffloadingPoint(action)
        self.currentOffloadingPoint = offloadingPoint
        # print("-------------------------------")
        # print(f"Action: {action}")
        # print(f"current action len: {len(self.currentAction)}")
        # print(f"current offloading points: {self.currentOffloadingPoint}")

        allTrainingTimes = []
        total_comp_e = 0
        total_comm_e = 0
        edgesConnectedDeviceNum = [0] * self.edgeDeviceNum

        isValid = self.isActionValid(self.currentAction)

        if not isValid:
            logger.info("-------------------------------------------")
            logger.info(f"Ops! INVALID ACTION")
            logger.info(f"Action: {action}")
            return -1, self.currentState
        else:
            logger.info("-------------------------------------------")
            logger.info(f"Current Offloading: {self.currentOffloadingPoint}\n")
            logger.info(f"Current Action: {action}\n")

            for i in range(self.iotDeviceNum):
                self.iotDevices[i].setEffectiveBW(self.effectiveBandwidth[i])
                self.edgeDevices[self.iotDevices[i].edgeIndex].connectedDevice = 0
                self.cloud.connectedDevice = 0

            for i in range(self.edgeDeviceNum):
                self.edgeDevices[i].setEffectiveBW(self.effectiveBandwidth[i + self.iotDeviceNum])

            totalEnergyConsumption = 0
            maxTrainingTime = 0

            iotRemainingFLOP = [iot.FLOPS for iot in self.iotDevices]
            edgeRemainingFLOP = [edge.FLOPS for edge in self.edgeDevices]
            cloudRemainingFLOP = self.cloud.FLOPS

            for i in range(0, len(offloadingPoint), 2):
                op1 = offloadingPoint[i]
                op2 = offloadingPoint[i + 1]
                cloudRemainingFLOP -= sum(config.COMP_WORK_LOAD[op2 + 1:])
                edgeRemainingFLOP[self.iotDevices[int(i / 2)].edgeIndex] -= sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1])
                iotRemainingFLOP[int(i / 2)] -= sum(config.COMP_WORK_LOAD[0:op1 + 1])

                if sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1]):
                    self.edgeDevices[self.iotDevices[int(i / 2)].edgeIndex].connectedDevice += 1
                    edgesConnectedDeviceNum[self.iotDevices[int(i / 2)].edgeIndex] += 1
                if sum(config.COMP_WORK_LOAD[op2 + 1:]) != 0:
                    self.cloud.connectedDevice += 1

            for i in range(0, len(offloadingPoint), 2):
                op1 = offloadingPoint[i]
                op2 = offloadingPoint[i + 1]

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

            normalizedAvgEnergy = averageEnergyConsumption / self.ClassicFLEnergy
            normalizedTT = maxTrainingTime / self.ClassicFLTrainingTime

            self.currentClassicFLEnergy, self.currentClassicFLTrainingTime = self.calculateClassicFLEnergyTT()
            rewardOfTrainingTime = maxTrainingTime
            rewardOfTrainingTime -= self.currentClassicFLTrainingTime
            rewardOfTrainingTime /= 1200
            rewardOfTrainingTime *= -1

            rewardOfTrainingTime = min(max(rewardOfTrainingTime, -1), 1)

            rewardOfEnergy = averageEnergyConsumption
            rewardOfEnergy -= self.currentClassicFLEnergy
            rewardOfEnergy /= 10
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

            self.timestep_energy.append(averageEnergyConsumption)
            self.timestep_tt.append(maxTrainingTime)
            self.timestep_reward.append(reward)
            logger.info(f"Current ClassicFL Energy: {self.currentClassicFLEnergy}\n")
            logger.info(f"Current ClassicFL TrainingTime: {self.currentClassicFLTrainingTime}\n")
            logger.info(f"Average Energy : {averageEnergyConsumption} \n")
            logger.info(f"Training Time : {maxTrainingTime} \n")
            logger.info(f"Bandwidth : {self.getBandwidth()} \n")
            logger.info(f"Reward of this action : {reward} \n")
            logger.info(f"Reward of energy : {self.rewardOfEnergy} \n")
            logger.info(f"Reward of training time : {self.rewardOfTrainingTime} \n")

            iotBandwidths = []
            edgeBandwidths = []

            for iotDevice in self.iotDevices:
                randomBW = random.randint(1, 11)
                iotBandwidths.append(iotDevice.bandwidth * randomBW * 0.1)

            for edgeDevice in self.edgeDevices:
                randomBW = random.randint(1, 11)
                edgeBandwidths.append(edgeDevice.bandwidth * randomBW * 0.1)

            newBW = np.concatenate((iotBandwidths, edgeBandwidths), axis=0)
            self.setBandwidth(newBW)

            self.currentState = self.getBandwidth()
            logger.info(f"New State: {self.currentState}")
            return reward, self.currentState

    def step(self, action):
        terminated = False
        reward, observation = self.rewardFun(action)
        truncated = self.getCurrentTimestep() >= self.ep_length - 1
        # if truncated == True:
        # print(f"current Time stamp: {self.getCurrentTimestep()}")
        # print(f"trancated: {truncated}")
        # print(f"Current EP: {self.currentEpisode}")
        if truncated == False:
            self.setCurrentTimestep(self.getCurrentTimestep() + 1)
        return observation, reward, terminated, truncated, {}

    def reset(self, seed=1, options=None):
        super().reset(seed=seed)
        if (self.getCurrentTimestep() != self.ep_length - 1) and self.currentEpisode != 0:
            print(f"BE GA RAFTIM. timestep:{self.getCurrentTimestep()} ============================>>>>>")
        if self.currentEpisode != 0:
            self.setCurrentTimestep(0)
            self.episode_energy.append(sum(self.timestep_energy) / self.ep_length)
            self.episode_tt.append(sum(self.timestep_tt) / self.ep_length)
            self.episode_reward.append(sum(self.timestep_reward) / self.ep_length)

        self.currentEpisode += 1
        self.num_resets += 1
        self.timestep_tt = []
        self.timestep_energy = []
        self.timestep_reward = []

        self.setCumulativeEnergy(0)
        self.setCumulativeTT(0)

        iotBandwidths = []
        edgeBandwidths = []

        for iotDevice in self.iotDevices:
            randomBW = random.randint(1, 11)
            iotBandwidths.append(iotDevice.bandwidth * randomBW * 0.1)

        for edgeDevice in self.edgeDevices:
            randomBW = random.randint(1, 11)
            edgeBandwidths.append(edgeDevice.bandwidth * randomBW * 0.1)

        self.setBandwidth(bandwidth=np.concatenate((iotBandwidths, edgeBandwidths), axis=0))
        ClFLEnergy, ClFlTT = self.calculateClassicFLEnergyTT()

        self.currentState = self.getBandwidth()
        return self.currentState, {}

    def render(self):
        pass

    # this function checks the action, it's ok or not.
    def isActionValid(self, currentAction: list) -> [bool]:
        isValid = True
        allowed_digits = {'0', '1', '2', '3'}

        for digit in str(currentAction[0]):
            if digit not in allowed_digits:
                return False

        return True

    def calculateClassicFLEnergyTT(self):
        allTrainingTimes = []
        total_comp_e = 0
        total_comm_e = 0
        edgesConnectedDeviceNum = [0] * self.edgeDeviceNum

        for i in range(self.iotDeviceNum):
            self.iotDevices[i].setEffectiveBW(self.effectiveBandwidth[i])
            self.edgeDevices[self.iotDevices[i].edgeIndex].connectedDevice = 0
            self.cloud.connectedDevice = 0

        for i in range(self.edgeDeviceNum):
            self.edgeDevices[i].setEffectiveBW(self.effectiveBandwidth[i + self.iotDeviceNum])

        totalEnergyConsumption = 0
        maxTrainingTime = 0

        iotRemainingFLOP = [iot.FLOPS for iot in self.iotDevices]
        edgeRemainingFLOP = [edge.FLOPS for edge in self.edgeDevices]
        cloudRemainingFLOP = self.cloud.FLOPS

        offloadingPointsList = [config.LAYER_NUM - 1] * self.iotDeviceNum * 2
        for i in range(0, len(offloadingPointsList), 2):
            op1 = offloadingPointsList[i]
            op2 = offloadingPointsList[i + 1]
            cloudRemainingFLOP -= sum(config.COMP_WORK_LOAD[op2 + 1:])
            edgeRemainingFLOP[self.iotDevices[int(i / 2)].edgeIndex] -= sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1])
            iotRemainingFLOP[int(i / 2)] -= sum(config.COMP_WORK_LOAD[0:op1 + 1])

        if sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1]):
            edgesConnectedDeviceNum[self.iotDevices[int(i / 2)].edgeIndex] += 1

        for i in range(0, len(offloadingPointsList), 2):
            # Mapping float number to Offloading points
            op1 = offloadingPointsList[i]
            op2 = offloadingPointsList[i + 1]

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
        return averageEnergyConsumption, maxTrainingTime

    def actionToOffloadingPoint(self, action):
        offloadingPoint = []
        for i in range(self.iotDeviceNum):
            if action[i] == 0:
                offloadingPoint.append(6)
                offloadingPoint.append(6)
            elif action[i] == 1:
                offloadingPoint.append(5)
                offloadingPoint.append(6)
            elif action[i] == 2:
                offloadingPoint.append(3)
                offloadingPoint.append(4)
            elif action[i] == 3:
                offloadingPoint.append(0)
                offloadingPoint.append(2)
        return offloadingPoint

    def close(self):
        super().close()

    def setBandwidth(self, bandwidth):
        self.effectiveBandwidth = bandwidth

    def getBandwidth(self):
        return self.effectiveBandwidth

    def setCumulativeEnergy(self, energy):
        self.cumulativeEnergy = energy

    def getCumulativeEnergy(self):
        return self.cumulativeEnergy

    def setCumulativeTT(self, TT):
        self.cumulativeTT = TT

    def getCumulativeTT(self):
        return self.cumulativeTT

    def setCurrentTimestep(self, ts):
        self.currentTimestep = ts

    def getCurrentTimestep(self):
        return self.currentTimestep
