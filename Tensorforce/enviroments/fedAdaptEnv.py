import logging

from tensorforce import Environment

import Tensorforce.config as config
from System.Device import Device
from Tensorforce import utils

logger = logging.getLogger()


class FedAdaptEnv(Environment):

    def __init__(self, allTrainingTime, iotDevices: list[Device], cloud: Device, groupNum: int = 1):
        super().__init__()

        self.groupNum = groupNum
        self.iotDeviceNum: int = len(iotDevices)
        self.iotDevices: list[Device] = iotDevices
        self.cloud: Device = cloud

        self.rewardTuningParams = allTrainingTime
        self.rewardOfTrainingTime = 0
        self.effectiveBandwidth = [[self.iotDevices[i].bandwidth] for i in range(self.iotDeviceNum)]

    def states(self):
        # State = [Training Time of the slowest device in each group ,previous action for each group]
        return dict(type="float", shape=(2 * self.groupNum,))

    def actions(self):
        return dict(type="float", shape=(self.groupNum,), min_value=0.0, max_value=1.0)

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def close(self):
        super().close()

    def reset(self):
        # randActions = np.random.uniform(low=0.0, high=1.0, size=(self.groupNum,))
        randActions = [1.0] * self.groupNum
        reward, newState = self.rewardFun(randActions)
        return newState

    def rewardFun(self, action):
        allTrainingTimes = []
        self.cloud.connectedDevice = 0
        maxTrainingTime = 0
        offloadingPointsList = []

        iotRemainingFLOP = [iot.FLOPS for iot in self.iotDevices]
        cloudRemainingFLOP = self.cloud.FLOPS

        for i in range(0, self.iotDeviceNum):
            op, _ = utils.actionToLayer([action[0], -1])
            cloudRemainingFLOP -= sum(config.COMP_WORK_LOAD[op + 1:])
            iotRemainingFLOP[int(i)] -= sum(config.COMP_WORK_LOAD[0:op + 1])

            if sum(config.COMP_WORK_LOAD[op + 1:]) != 0:
                self.cloud.connectedDevice += 1

        for i in range(0, self.iotDeviceNum):
            # Mapping float number to Offloading points
            op1, op2 = utils.actionToLayer([action[0], -1])
            offloadingPointsList.append(op)

            # computing training time of this action
            iotTrainingTime = self.iotDevices[int(i)].trainingTime([op, op],
                                                                   remainingFlops=iotRemainingFLOP[int(i)])

            cloudTrainingTime = self.cloud.trainingTime([op, op], remainingFlops=cloudRemainingFLOP)

            self.effectiveBandwidth[int(i)].append(self.iotDevices[int(i)].effectiveBandwidth)

            totalTrainingTime = iotTrainingTime + cloudTrainingTime
            allTrainingTimes.append(totalTrainingTime)
            if totalTrainingTime > maxTrainingTime:
                maxTrainingTime = totalTrainingTime

        reward = 0
        for i in range(self.iotDeviceNum):
            if allTrainingTimes[i] > self.rewardTuningParams[i]:
                reward += ((self.rewardTuningParams[i] / allTrainingTimes[i]) - 1)
            else:
                reward += (1 - (allTrainingTimes[i] / self.rewardTuningParams[i]))

        logger.info("-------------------------------------------")
        logger.info(f"Offloading layer : {offloadingPointsList} \n")
        logger.info(f"Training time : {maxTrainingTime} \n")
        logger.info(f"Reward of this action : {reward} \n")
        logger.info(f"IOTs Capacities : {iotRemainingFLOP} \n")
        logger.info(f"Cloud Capacities : {cloudRemainingFLOP} \n")

        newState = [maxTrainingTime]
        newState.extend(action)
        logger.info(f"New State : {newState} \n")
        return reward, newState

    def execute(self, actions: list):
        terminal = False
        reward, newState = self.rewardFun(actions)
        return newState, terminal, reward
