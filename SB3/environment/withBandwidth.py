import collections
import copy
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

    def __init__(self, iotDevices: list, edgeDevices: list, cloud: Device,
                 fraction=0.8, ep_length: int = 100):
        super().__init__()

        self.iotDeviceNum: int = len(iotDevices)
        self.edgeDeviceNum: int = len(edgeDevices)
        self.iotDevices: list = iotDevices
        self.edgeDevices: list = edgeDevices
        self.cloud: Device = cloud

        self.isEvaluation: bool = False
        self.maxEnergyOfIotDevice = [iot.remainingEnergy for iot in self.iotDevices]

        self.currentClassicFLEnergy = 0
        self.currentClassicFLTrainingTime = 0
        self.currentRemainingEnergy = [iot.remainingEnergy for iot in self.iotDevices]
        self.currentConsumedEnergy = [0 for _ in range(self.iotDeviceNum)]
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

        # np.zeros((Episode Number, Timestep Number, Number Of Clients))
        self.remainingEnergy = np.zeros((10, ep_length, self.iotDeviceNum))

        self.timestep_effectiveBW = []
        self.timestep_consumedEnergy = []
        self.timestep_remainingEnergy = []
        self.timestep_classicFL_energy = []
        self.timestep_classicFL_TT = []
        self.timestep_energy = []
        self.timestep_tt = []
        self.timestep_reward = []
        self.timestep_energy_reward = []
        self.timestep_tt_reward = []

        self.episode_effectiveBW = []
        self.episode_consumedEnergy = []
        self.episode_remainingEnergy = []
        self.episode_classicFL_energy = []
        self.episode_classicFL_TT = []
        self.episode_energy = []
        self.episode_tt = []
        self.episode_reward = []
        self.episode_energy_reward = []
        self.episode_tt_reward = []

        self.effectiveBandwidth = []

        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.

        self.fraction = fraction

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2 * self.iotDeviceNum,), dtype=np.float32, seed=None)

        # bandwidths : 0% fluctuation, 10% fluctuation, 20% fluctuation,..., 90% fluctuation.
        # observation_spec = [10] * (self.iotDeviceNum + self.edgeDeviceNum)
        # self.observation_space = spaces.MultiDiscrete(observation_spec)
        self.observation_space = spaces.Dict()
        self.observation_space = spaces.Box(low=0.0, high=20,
                                            shape=(self.iotDeviceNum + self.edgeDeviceNum + self.iotDeviceNum,),
                                            dtype=np.float32, seed=None)

    def rewardFun(self, action):
        total_comp_e = 0
        total_comm_e = 0
        edgesConnectedDeviceNum = [0] * self.edgeDeviceNum
        offloadingPoints = []

        logger.info("-------------------------------------------")
        logger.info(f"Current Episode: {self.currentEpisode}")
        logger.info(f"Current Timestep: {self.currentTimestep}")
        logger.info(f"Current Action: {action}\n")

        for i in range(self.iotDeviceNum):
            self.iotDevices[i].setEffectiveBW(self.effectiveBandwidth[i])
            self.edgeDevices[self.iotDevices[i].edgeIndex].connectedDevice = 0
            self.cloud.connectedDevice = 0

        for i in range(self.edgeDeviceNum):
            self.edgeDevices[i].setEffectiveBW(self.effectiveBandwidth[i + self.iotDeviceNum])

        iotRemainingFLOP = [iot.FLOPS for iot in self.iotDevices]
        edgeRemainingFLOP = [edge.FLOPS for edge in self.edgeDevices]
        cloudRemainingFLOP = self.cloud.FLOPS

        for i in range(0, len(action), 2):
            op1, op2 = utils.actionToLayer(action[i:i + 2])
            offloadingPoints.append(op1)
            offloadingPoints.append(op2)

            cloudRemainingFLOP -= sum(config.COMP_WORK_LOAD[op2 + 1:])
            edgeRemainingFLOP[self.iotDevices[int(i / 2)].edgeIndex] -= sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1])
            iotRemainingFLOP[int(i / 2)] -= sum(config.COMP_WORK_LOAD[0:op1 + 1])

            if sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1]):
                self.edgeDevices[self.iotDevices[int(i / 2)].edgeIndex].connectedDevice += 1
                edgesConnectedDeviceNum[self.iotDevices[int(i / 2)].edgeIndex] += 1
            if sum(config.COMP_WORK_LOAD[op2 + 1:]) != 0:
                self.cloud.connectedDevice += 1
        self.currentOffloadingPoint = offloadingPoints
        logger.info(f"Current Offloading: {self.currentOffloadingPoint}\n")

        remainingEnergyBefore = copy.deepcopy(self.currentRemainingEnergy)

        clientsWithMinEnergy = sorted(range(len(remainingEnergyBefore)), key=lambda k: remainingEnergyBefore[k])

        clientInfo = dict()
        for i in range(self.edgeDeviceNum):
            clientInfo[f"edge{i}"] = list()

        # Phase 1: We calculate the computation and communication time of each iot device
        for i in range(0, len(offloadingPoints), 2):
            op1 = offloadingPoints[i]
            op2 = offloadingPoints[i + 1]

            # Calculating computation time, computation energy, communication time, communication energy of this action
            # on i th iot device
            iot_comp_e, iot_comm_e, iot_comp_tt, iot_comm_tt = self.iotDevices[int(i / 2)].energy_tt(
                splitPoints=[op1, op2],
                remainingFlops=iotRemainingFLOP[int(i / 2)])
            _, _, edge_comp_tt, edge_comm_tt = self.edgeDevices[self.iotDevices[int(i / 2)].edgeIndex] \
                .energy_tt(splitPoints=[op1, op2],
                           remainingFlops=edgeRemainingFLOP[self.iotDevices[int(i / 2)].edgeIndex])
            client = dict(name=f"{int(i / 2)}", start_time=float(round(iot_comp_tt + iot_comm_tt, 2)),
                          duration=float(round(edge_comp_tt, 2)),
                          training_time=float(iot_comp_tt + iot_comm_tt + edge_comm_tt))
            clientInfo[f"edge{self.iotDevices[int(i / 2)].edgeIndex}"].append(client)

            totalEnergyConsumptionOfClient = iot_comp_e + iot_comm_e
            if (self.currentRemainingEnergy[int(i / 2)] - totalEnergyConsumptionOfClient) <= 0:
                self.currentRemainingEnergy[int(i / 2)] = 0
            else:
                self.currentRemainingEnergy[int(i / 2)] -= totalEnergyConsumptionOfClient

            # computing energy consumption of iot devices
            total_comp_e += iot_comp_e
            total_comm_e += iot_comm_e

        # Average energy consumption calculation
        totalEnergyConsumption = (total_comm_e + total_comp_e)
        averageEnergyConsumption = totalEnergyConsumption / self.iotDeviceNum

        # Phase 2: Calculate computation time in each edge considering Round-Robin scheduling in each edge
        allClient = []
        for i in range(self.edgeDeviceNum):
            total_execution_time, conflict_time, waiting_times, turnaround_times, start_times, end_times = \
                utils.round_robin_scheduling(clientInfo[f"edge{i}"], time_slice=0.2)
            clientOfEdge = sorted(clientInfo[f"edge{i}"], key=lambda x: x[f"name"])
            turnaround_times = collections.OrderedDict(sorted(turnaround_times.items()))
            turnaround_times = list(turnaround_times.values())
            for j in range(len(clientInfo[f"edge{i}"])):
                clientOfEdge[j]["training_time"] += turnaround_times[j]
                clientInfo[f"edge{i}"][j]["start_time"] = clientOfEdge[j]["training_time"]
            allClient = np.concatenate((allClient, clientInfo[f"edge{i}"]), axis=0)

        # phase 3: Calculate computation time in central server considering Round-Robin scheduling in it
        total_execution_time, conflict_time, waiting_times, turnaround_times, start_times, end_times = \
            utils.round_robin_scheduling(allClient, time_slice=0.2)

        allClient = sorted(allClient, key=lambda x: x[f"name"])
        turnaround_times = collections.OrderedDict(sorted(turnaround_times.items()))
        turnaround_times = list(turnaround_times.values())
        for i in range(len(allClient)):
            allClient[i]["training_time"] += turnaround_times[i]

        maxTrainingTime = max(allClient, key=lambda x: x['training_time'])['training_time']

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

        rewardOfRemainingEnergy = 0
        consumedEnergy = [a - b for a, b in zip(remainingEnergyBefore, self.currentRemainingEnergy)]
        self.currentConsumedEnergy = consumedEnergy
        sortedConsumedEnergyIndex = sorted(range(len(consumedEnergy)), key=lambda k: consumedEnergy[k])

        for i in range(len(consumedEnergy)):
            if ((sortedConsumedEnergyIndex[i] == clientsWithMinEnergy[i]) and
                    consumedEnergy[sortedConsumedEnergyIndex[i]] != 0):
                rewardOfRemainingEnergy += 5
            else:
                rewardOfRemainingEnergy += -3

        # rewardOfRemainingEnergy = min(max(rewardOfRemainingEnergy, -2), 5)
        remainingEnergyVariance = np.std(self.currentRemainingEnergy)

        self.avgEnergy = averageEnergyConsumption
        self.tt = maxTrainingTime
        self.rewardOfEnergy = (self.fraction * rewardOfEnergy)
        self.rewardOfTrainingTime = (1 - self.fraction) * rewardOfTrainingTime

        # if remainingEnergyVariance != 0:
        #     reward = (1 / remainingEnergyVariance) * 100
        # elif all(x <= 0 for x in self.currentRemainingEnergy):
        #     reward = 0
        # else:
        #     reward = 10
        reward = rewYardOfRemainingEnergy
        # print(reward)
        # if self.fraction <= 1:
        #     reward = self.rewardOfEnergy + self.rewardOfTrainingTime + rewardOfRemainingEnergy
        # else:
        #     raise Exception("Fraction must be less than 1")

        self.timestep_classicFL_energy.append(self.currentClassicFLEnergy)
        self.timestep_classicFL_TT.append(self.currentClassicFLTrainingTime)
        self.timestep_energy.append(averageEnergyConsumption)
        self.timestep_tt.append(maxTrainingTime)
        self.timestep_reward.append(reward)
        self.timestep_energy_reward.append(self.rewardOfEnergy)
        self.timestep_tt_reward.append(self.rewardOfTrainingTime)

        logger.info(f"Current ClassicFL Energy: {self.currentClassicFLEnergy}\n")
        logger.info(f"Current ClassicFL TrainingTime: {self.currentClassicFLTrainingTime}\n")
        logger.info(f"Average Energy : {averageEnergyConsumption} \n")
        logger.info(f"Training Time : {maxTrainingTime} \n")
        logger.info(f"Bandwidth : {self.getBandwidth()} \n")
        logger.info(f"Reward of this action : {reward} \n")
        logger.info(f"Reward of energy : {self.rewardOfEnergy} \n")
        logger.info(f"Reward of training time : {self.rewardOfTrainingTime} \n")

        if self.isEvaluation:
            self.timestep_remainingEnergy.append(copy.deepcopy(self.currentRemainingEnergy))
            self.timestep_consumedEnergy.append(copy.deepcopy(self.currentConsumedEnergy))
            self.timestep_effectiveBW.append(copy.deepcopy(self.effectiveBandwidth))

        iotBandwidths = []
        edgeBandwidths = []

        # for iotDevice in self.iotDevices:
        #     iotBandwidths.append((iotDevice.bandwidth * 1.0))
        # for edgeDevice in self.edgeDevices:
        #     edgeBandwidths.append((edgeDevice.bandwidth * 1.0))

        for iotDevice in self.iotDevices:
            probability = round(random.uniform(0.0, 1.0), 2)
            if probability <= 0.5:
                iotBandwidths.append((iotDevice.bandwidth * round(random.uniform(0.1, 0.2), 10)))
            elif 0.5 < probability <= 0.6:
                iotBandwidths.append((iotDevice.bandwidth * round(random.uniform(0.2, 0.4), 10)))
            elif 0.6 < probability <= 0.7:
                iotBandwidths.append((iotDevice.bandwidth * round(random.uniform(0.4, 0.6), 10)))
            elif 0.7 < probability <= 0.8:
                iotBandwidths.append((iotDevice.bandwidth * round(random.uniform(0.6, 0.8), 10)))
            else:
                iotBandwidths.append((iotDevice.bandwidth * round(random.uniform(0.8, 1.0), 10)))

        for edgeDevice in self.edgeDevices:
            probability = round(random.uniform(0.0, 1.0), 2)
            if probability <= 0.5:
                edgeBandwidths.append((edgeDevice.bandwidth * round(random.uniform(0.1, 0.2), 10)))
            elif 0.5 < probability <= 0.6:
                edgeBandwidths.append((edgeDevice.bandwidth * round(random.uniform(0.2, 0.4), 10)))
            elif 0.6 < probability <= 0.7:
                edgeBandwidths.append((edgeDevice.bandwidth * round(random.uniform(0.4, 0.6), 10)))
            elif 0.7 < probability <= 0.8:
                edgeBandwidths.append((edgeDevice.bandwidth * round(random.uniform(0.6, 0.8), 10)))
            else:
                edgeBandwidths.append((edgeDevice.bandwidth * round(random.uniform(0.8, 1.0), 10)))

        newBW = np.concatenate((iotBandwidths, edgeBandwidths), axis=0)
        self.setBandwidth(newBW)

        self.currentState = []
        for i in range(len(self.getBandwidth())):
            self.currentState.append(self.getBandwidth()[i])

        for j in range(len(self.currentRemainingEnergy)):
            self.currentState.append(self.currentRemainingEnergy[j] / 100)

        logger.info(f"New State: {self.currentState}")
        return reward, self.currentState

    def step(self, action):
        self.currentAction = action
        terminated = False
        reward, observation = self.rewardFun(action)
        truncated = self.getCurrentTimestep() >= self.ep_length - 1
        # if truncated == True:
        # print(f"current Time stamp: {self.getCurrentTimestep()}")
        # print(f"trancated: {truncated}")
        # print(f"Current EP: {self.currentEpisode}")
        if truncated == False:
            self.setCurrentTimestep(self.getCurrentTimestep() + 1)

        if (all(x <= 0 for x in observation[self.iotDeviceNum + self.edgeDeviceNum:])
                or (self.currentTimestep >= self.ep_length)):
            terminated = True
        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.isEvaluation:
            self.episode_remainingEnergy.append(copy.deepcopy(self.timestep_remainingEnergy))
            self.episode_consumedEnergy.append(copy.deepcopy(self.timestep_consumedEnergy))
            self.episode_effectiveBW.append(copy.deepcopy(self.timestep_effectiveBW))

        if self.currentEpisode != 0:
            self.episode_energy.append(sum(self.timestep_energy) / self.getCurrentTimestep())
            self.episode_tt.append(sum(self.timestep_tt) / self.getCurrentTimestep())
            self.episode_classicFL_energy.append(sum(self.timestep_classicFL_energy) / self.getCurrentTimestep())
            self.episode_classicFL_TT.append(sum(self.timestep_classicFL_TT) / self.getCurrentTimestep())
            self.episode_reward.append(sum(self.timestep_reward) / self.getCurrentTimestep())
            self.episode_energy_reward.append(sum(self.timestep_energy_reward) / self.getCurrentTimestep())
            self.episode_tt_reward.append(sum(self.timestep_tt_reward) / self.getCurrentTimestep())

        self.setCurrentTimestep(0)
        self.currentEpisode += 1
        self.num_resets += 1
        self.timestep_tt = []
        self.timestep_energy = []
        self.timestep_reward = []
        self.timestep_tt_reward = []
        self.timestep_energy_reward = []
        self.timestep_classicFL_energy = []
        self.timestep_classicFL_TT = []
        self.timestep_remainingEnergy = []
        self.timestep_consumedEnergy = []
        self.timestep_effectiveBW = []

        self.setCumulativeEnergy(0)
        self.setCumulativeTT(0)

        iotBandwidths = []
        edgeBandwidths = []

        # for iotDevice in self.iotDevices:
        #     iotBandwidths.append((iotDevice.bandwidth * 1.0))
        # for edgeDevice in self.edgeDevices:
        #     edgeBandwidths.append((edgeDevice.bandwidth * 1.0))

        for iotDevice in self.iotDevices:
            probability = round(random.uniform(0.0, 1.0), 2)
            if probability <= 0.5:
                iotBandwidths.append((iotDevice.bandwidth * round(random.uniform(0.1, 0.2), 10)))
            elif 0.5 < probability <= 0.6:
                iotBandwidths.append((iotDevice.bandwidth * round(random.uniform(0.2, 0.4), 10)))
            elif 0.6 < probability <= 0.7:
                iotBandwidths.append((iotDevice.bandwidth * round(random.uniform(0.4, 0.6), 10)))
            elif 0.7 < probability <= 0.8:
                iotBandwidths.append((iotDevice.bandwidth * round(random.uniform(0.6, 0.8), 10)))
            else:
                iotBandwidths.append((iotDevice.bandwidth * round(random.uniform(0.8, 1.0), 10)))

        for edgeDevice in self.edgeDevices:
            probability = round(random.uniform(0.0, 1.0), 2)
            if probability <= 0.5:
                edgeBandwidths.append((edgeDevice.bandwidth * round(random.uniform(0.1, 0.2), 10)))
            elif 0.5 < probability <= 0.6:
                edgeBandwidths.append((edgeDevice.bandwidth * round(random.uniform(0.2, 0.4), 10)))
            elif 0.6 < probability <= 0.7:
                edgeBandwidths.append((edgeDevice.bandwidth * round(random.uniform(0.4, 0.6), 10)))
            elif 0.7 < probability <= 0.8:
                edgeBandwidths.append((edgeDevice.bandwidth * round(random.uniform(0.6, 0.8), 10)))
            else:
                edgeBandwidths.append((edgeDevice.bandwidth * round(random.uniform(0.8, 1.0), 10)))

        self.setBandwidth(bandwidth=np.concatenate((iotBandwidths, edgeBandwidths), axis=0))

        self.currentState = []
        for i in range(len(self.getBandwidth())):
            self.currentState.append(self.getBandwidth()[i])

        self.currentRemainingEnergy = [iot.remainingEnergy for iot in self.iotDevices]
        for j in range(len(self.currentRemainingEnergy)):
            self.currentState.append(self.currentRemainingEnergy[j] / 100)

        return self.currentState, {}

    def render(self):
        pass

    def calculateClassicFLEnergyTT(self):
        allTrainingTimes = []
        offloadingPointsList = [config.LAYER_NUM - 1] * self.iotDeviceNum * 2
        total_comp_e = 0
        total_comm_e = 0
        edgesConnectedDeviceNum = [0] * self.edgeDeviceNum

        iotRemainingFLOP = [iot.FLOPS for iot in self.iotDevices]
        edgeRemainingFLOP = [edge.FLOPS for edge in self.edgeDevices]
        cloudRemainingFLOP = self.cloud.FLOPS

        for i in range(0, len(offloadingPointsList), 2):
            op1 = offloadingPointsList[i]
            op2 = offloadingPointsList[i + 1]

            cloudRemainingFLOP -= sum(config.COMP_WORK_LOAD[op2 + 1:])
            edgeRemainingFLOP[self.iotDevices[int(i / 2)].edgeIndex] -= sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1])
            iotRemainingFLOP[int(i / 2)] -= sum(config.COMP_WORK_LOAD[0:op1 + 1])

            if sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1]):
                self.edgeDevices[self.iotDevices[int(i / 2)].edgeIndex].connectedDevice += 1
                edgesConnectedDeviceNum[self.iotDevices[int(i / 2)].edgeIndex] += 1
            if sum(config.COMP_WORK_LOAD[op2 + 1:]) != 0:
                self.cloud.connectedDevice += 1

        clientInfo = dict()
        for i in range(self.edgeDeviceNum):
            clientInfo[f"edge{i}"] = list()

        # Phase 1: We calculate the computation and communication time of each iot device
        for i in range(0, len(offloadingPointsList), 2):
            op1 = offloadingPointsList[i]
            op2 = offloadingPointsList[i + 1]

            # Calculating computation time, computation energy, communication time, communication energy of this action
            # on i th iot device
            iot_comp_e, iot_comm_e, iot_comp_tt, iot_comm_tt = self.iotDevices[int(i / 2)].energy_tt(
                splitPoints=[op1, op2],
                remainingFlops=iotRemainingFLOP[int(i / 2)])
            _, _, edge_comp_tt, edge_comm_tt = self.edgeDevices[self.iotDevices[int(i / 2)].edgeIndex] \
                .energy_tt(splitPoints=[op1, op2],
                           remainingFlops=edgeRemainingFLOP[self.iotDevices[int(i / 2)].edgeIndex])
            client = dict(name=f"{int(i / 2)}", start_time=float(round(iot_comp_tt + iot_comm_tt, 2)),
                          duration=float(round(edge_comp_tt, 2)),
                          training_time=float(iot_comp_tt + iot_comm_tt + edge_comm_tt))
            clientInfo[f"edge{self.iotDevices[int(i / 2)].edgeIndex}"].append(client)

            # computing energy consumption of iot devices
            total_comp_e += iot_comp_e
            total_comm_e += iot_comm_e

        # Average energy consumption calculation
        totalEnergyConsumption = (total_comm_e + total_comp_e)
        averageEnergyConsumption = totalEnergyConsumption / self.iotDeviceNum

        # Phase 2: Calculate computation time in each edge considering Round-Robin scheduling in each edge
        allClient = []
        for i in range(self.edgeDeviceNum):
            total_execution_time, conflict_time, waiting_times, turnaround_times, start_times, end_times = \
                utils.round_robin_scheduling(clientInfo[f"edge{i}"], time_slice=0.2)
            clientOfEdge = sorted(clientInfo[f"edge{i}"], key=lambda x: x[f"name"])
            turnaround_times = collections.OrderedDict(sorted(turnaround_times.items()))
            turnaround_times = list(turnaround_times.values())
            for j in range(len(clientInfo[f"edge{i}"])):
                clientOfEdge[j]["training_time"] += turnaround_times[j]
                clientInfo[f"edge{i}"][j]["start_time"] = clientOfEdge[j]["training_time"]
            allClient = np.concatenate((allClient, clientInfo[f"edge{i}"]), axis=0)

        # phase 3: Calculate computation time in central server considering Round-Robin scheduling in it
        total_execution_time, conflict_time, waiting_times, turnaround_times, start_times, end_times = \
            utils.round_robin_scheduling(allClient, time_slice=0.2)

        allClient = sorted(allClient, key=lambda x: x[f"name"])
        turnaround_times = collections.OrderedDict(sorted(turnaround_times.items()))
        turnaround_times = list(turnaround_times.values())
        for i in range(len(allClient)):
            allClient[i]["training_time"] += turnaround_times[i]

        maxTrainingTime = max(allClient, key=lambda x: x['training_time'])['training_time']
        return averageEnergyConsumption, maxTrainingTime

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
