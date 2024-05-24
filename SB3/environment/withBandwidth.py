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
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.iotDeviceNum * 2,),
                                       dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-50.0, high=100,
                                            shape=(2 + self.iotDeviceNum + self.edgeDeviceNum + 2 * self.iotDeviceNum,),
                                            dtype=np.float32)

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

        normalizedAvgEnergy = averageEnergyConsumption / self.ClassicFLEnergy
        normalizedTT = maxTrainingTime / self.ClassicFLTrainingTime

        rewardOfTrainingTime = maxTrainingTime
        rewardOfTrainingTime -= self.ClassicFLTrainingTime
        rewardOfTrainingTime /= 1200
        rewardOfTrainingTime *= -1

        rewardOfTrainingTime = min(max(rewardOfTrainingTime, -1), 1)

        rewardOfEnergy = averageEnergyConsumption
        rewardOfEnergy -= self.ClassicFLEnergy
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

        logger.info("-------------------------------------------")
        logger.info(f"Offloading layer : {offloadingPointsList} \n")
        logger.info(f"Avg Energy : {averageEnergyConsumption} \n")
        logger.info(f"Training time : {maxTrainingTime} \n")
        logger.info(f"Bandwidth : {self.getBandwidth()} \n")
        logger.info(f"Reward of this action : {reward} \n")
        logger.info(f"Reward of energy : {self.fraction * rewardOfEnergy} \n")
        logger.info(f"Reward of training time : {(1 - self.fraction) * rewardOfTrainingTime} \n")

        iotBandwidths = []
        edgeBandwidths = []

        if self.getCurrentTimestep() < 50:
            for iotDevice in self.iotDevices:
                iotBandwidths.append(iotDevice.bandwidth * 1.0)

            for edgeDevice in self.edgeDevices:
                edgeBandwidths.append(edgeDevice.bandwidth * 1.0)
        elif 50 <= self.getCurrentTimestep() <= 100:
            for iotDevice in self.iotDevices:
                iotBandwidths.append(iotDevice.bandwidth * 1.0)

            for edgeDevice in self.edgeDevices:
                edgeBandwidths.append(edgeDevice.bandwidth * 1.0)
        # elif 100 < self.getCurrentTimestep() < 150:
        #     for iotDevice in self.iotDevices:
        #         iotBandwidths.append(iotDevice.bandwidth * 3.0)
        #
        #     for edgeDevice in self.edgeDevices:
        #         edgeBandwidths.append(edgeDevice.bandwidth * 3.0)
        # else:
        #     for iotDevice in self.iotDevices:
        #         iotBandwidths.append(iotDevice.bandwidth * 4.0)
        #
        #     for edgeDevice in self.edgeDevices:
        #         edgeBandwidths.append(edgeDevice.bandwidth * 4.0)

        newBW = np.concatenate((iotBandwidths, edgeBandwidths), axis=0)
        self.setBandwidth(newBW)
        newState = [normalizedAvgEnergy, normalizedTT]
        newState = np.concatenate((newState, iotBandwidths, edgeBandwidths, action), axis=0)
        logger.info(f"New State: {newState}")
        return reward, newState

    def step(self, action):
        terminated = False
        reward, observation = self.rewardFun(action)
        truncated = self.getCurrentTimestep() >= self.ep_length-1
        #if truncated == True:       
            # print(f"current Time stamp: {self.getCurrentTimestep()}")
            # print(f"trancated: {truncated}")
            # print(f"Current EP: {self.currentEpisode}")
        if truncated == False:
            self.setCurrentTimestep(self.getCurrentTimestep() + 1)
        return observation, reward, terminated, truncated, {}

    def reset(self, seed=1, options=None):
        super().reset(seed=seed)
        if (self.getCurrentTimestep() != self.ep_length-1) and self.currentEpisode !=0:
            print(f"BE GA RAFTIM. timestep:{self.getCurrentTimestep()} ============================>>>>>")
        if self.currentEpisode !=0:
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

        # randActions = np.random.uniform(low=0.0, high=1.0, size=(self.iotDeviceNum * 2))
        randActions = [1.0] * 2 * self.iotDeviceNum
        iotBandwidths = []
        for iotDevice in self.iotDevices:
            iotBandwidths.append(np.random.uniform(low=iotDevice.bandwidth * 1.0, high=iotDevice.bandwidth))

        edgeBandwidths = []
        for edgeDevice in self.edgeDevices:
            edgeBandwidths.append(np.random.uniform(low=edgeDevice.bandwidth * 1.0, high=edgeDevice.bandwidth))

        self.setBandwidth(bandwidth=np.concatenate((iotBandwidths, edgeBandwidths), axis=0))

        reward, observation = self.rewardFun(randActions)

        return observation, {}

    def render(self):
        pass

    def close(self):
        super().close()

    def setBandwidth(self, bandwidth: list):
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
