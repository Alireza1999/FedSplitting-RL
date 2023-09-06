from tensorforce import Environment
from Tensorforce import utils
from System.Device import Device
import numpy as np
import logging

logger = logging.getLogger()


class CustomEnvironment(Environment):

    def __init__(self, iotDevices: list[Device], edgeDevices: list[Device], cloud: Device,
                 energyWithoutSplitting: float):

        super().__init__()

        self.iotDeviceNum: int = len(iotDevices)
        self.edgeDeviceNum: int = len(edgeDevices)

        self.iotDevices: list[Device] = iotDevices
        self.edgeDevices: list[Device] = edgeDevices
        self.cloud: Device = cloud

        self.avgEnergyWithoutSplitting = energyWithoutSplitting

    def states(self):
        """Returns the state space specification. energy of each iot device, capacity of each edge device,
        and actions previously taken by agent."""

        # return dict(type="float", shape=(self.iotDeviceNum * 3,))
        return dict(type="float", shape=(self.iotDeviceNum * 3 + self.edgeDeviceNum,))

    def actions(self):
        return dict(type="float", shape=(self.iotDeviceNum * 2,), min_value=0.0, max_value=1.0)

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def close(self):
        super().close()

    def reset(self):
        randEnergy = np.random.uniform(low=0.0, high=50.0, size=(self.iotDeviceNum,))
        randActions = np.random.uniform(low=0.0, high=1.0, size=(self.iotDeviceNum * 2))
        edgeCapacity = [np.random.randint(0, edges.capacity) for edges in self.edgeDevices]
        temp = np.append(randEnergy, edgeCapacity)
        return np.append(temp, randActions)

    def rewardFun(self, actions):
        reward = 0
        totalTrainingTime = 0
        totalEnergyConsumption = 0
        maxTrainingTime = 0
        iotEnergyList = []
        offloadingPointsList = []

        edgeCapacity = [edges.capacity for edges in self.edgeDevices]

        for i in range(0, len(actions), 2):
            # Mapping float number to Offloading points
            op1, op2 = utils.actionToLayer(actions[i:i + 2])
            offloadingPointsList.append(op1)
            offloadingPointsList.append(op2)

            edgeCapacity[self.iotDevices[int(i / 2)].edgeIndex] -= (op2 - op1)

            # computing training time of this action
            iotTrainingTime = self.iotDevices[int(i / 2)].trainingTime([op1, op2])
            edgeTrainingTime = self.edgeDevices[self.iotDevices[int(i / 2)].edgeIndex].trainingTime([op1, op2])
            cloudTrainingTime = self.cloud.trainingTime([op1, op2])
            totalTrainingTime += iotTrainingTime + edgeTrainingTime + cloudTrainingTime
            if totalTrainingTime > maxTrainingTime:
                maxTrainingTime = totalTrainingTime

            # calculating energy consumption of iot devices
            iotEnergy = self.iotDevices[int(i / 2)].energyConsumption([op1, op2])
            totalEnergyConsumption += iotEnergy

            # add this iot device energy consumption for new state
            iotEnergyList.append(iotEnergy)

        averageEnergyConsumption = totalEnergyConsumption / self.iotDeviceNum
        # add a float number to reward using tanh activation function
        if averageEnergyConsumption > self.avgEnergyWithoutSplitting:
            reward += (self.avgEnergyWithoutSplitting / averageEnergyConsumption) - 1
        else:
            reward += 2 - (averageEnergyConsumption / self.avgEnergyWithoutSplitting)

        for i in range(self.edgeDeviceNum):
            if edgeCapacity[i] < 0:
                reward += edgeCapacity[i] / self.edgeDevices[i].capacity

        reward += utils.tanhActivation(maxTrainingTime)

        logger.info("Offloading layer :\n{} \n".format(offloadingPointsList))
        logger.info("Edges Capacities :\n{} \n".format(edgeCapacity))
        logger.info("Total TrainingTime :\n{} \n".format(maxTrainingTime))
        logger.info("Avg Energy consumption :\n{} \n".format(averageEnergyConsumption))
        logger.info("==================================================================")


        temp = np.append(iotEnergyList, edgeCapacity)
        newState = np.append(temp, actions)
        return reward, newState

    def execute(self, actions: list):
        terminal = False
        reward, newState = self.rewardFun(actions)
        return newState, terminal, reward
