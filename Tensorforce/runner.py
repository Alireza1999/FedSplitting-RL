import logging
from pathlib import Path

import numpy as np
from tensorforce import Environment

from System.Device import Device
from Tensorforce import config as conf
from Tensorforce import utils
from Tensorforce.enviroments import customEnv
from Tensorforce.splittingMethods import RandomAgent, NoSplitting, AC, PPO, TensorforceAgent, TRPO


class Runner:

    def __init__(self, agentType='tensorforce', episodeNum=501, timestepNum=200, fraction=0.8, summaries=False,
                 log=False):
        self.agentType = agentType
        self.episodeNum = episodeNum
        self.timestepNum = timestepNum
        self.fraction = fraction
        self.summaries = summaries
        self.log = log

        self.saveGraphPath = f"Graphs/{self.agentType}/{self.fraction}"
        self.saveSummariesPath = f"{Path(__file__).parent}"

    def run(self):
        iotDevices = utils.createDeviceFromCSV(csvFilePath="../System/iotDevicesSmallScale.csv",
                                               deviceType='iotDevice')
        edgeDevices = utils.createDeviceFromCSV(csvFilePath="../System/edgesSmallScale.csv")
        cloud = utils.createDeviceFromCSV(csvFilePath="../System/cloud.csv")[0]

        maxEnergy, minEnergy = minMaxAvgEnergy(iotDevices, edgeDevices, cloud)
        FLTrainingTime = ClassicFLTrainingTime(iotDevices, edgeDevices, cloud)

        # rewardTuningParams = preTrain(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud)

        rewardTuningParams = [maxEnergy, minEnergy, FLTrainingTime]

        print(f"------------------------------------------------")
        print(f"Max Energy : \n{rewardTuningParams[0]}")
        print(f"Min Energy : \n{rewardTuningParams[1]}")
        print(f"------------------------------------------------")
        print(f"Classic FL Training Time : \n{rewardTuningParams[2]}")
        print(f"------------------------------------------------")

        # print(f"------------------------------------------------")
        # print(f"MIN TrainingTime : {rewardTuningParams[2]}")
        #
        # print(f"------------------------------------------------")
        # print(f"MAX TrainingTime : {rewardTuningParams[3]}")
        # print(f"------------------------------------------------")

        env = createEnv(rewardTuningParams=rewardTuningParams,
                        iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                        timestepNum=self.timestepNum,
                        fraction=self.fraction)

        agent = createAgent(agentType=self.agentType,
                            fraction=self.fraction,
                            environment=env,
                            timestepNum=self.timestepNum,
                            saveSummariesPath=self.saveSummariesPath)

        if self.log:
            logger = createLog(fileName=f"{self.agentType}_{self.fraction}")

        sumRewardOfEpisodes = list()
        energyConsumption = list()
        trainingTimeOfEpisode = list()
        trainingTimeOfAllTimesteps = list()

        x = list()
        AvgEnergyOfIotDevices = list()
        timestepCounter = 0
        for i in range(self.episodeNum):
            if self.log:
                logger.info("===========================================")
                logger.info("Episode {} started ...\n".format(i))

            episode_energy = list()
            episode_trainingTime = list()
            episode_reward = list()

            states = env.reset()
            internals = agent.initial_internals()
            terminal = False
            while not terminal:
                if self.log:
                    logger.info("-------------------------------------------")
                    logger.info(f"Timestep {timestepCounter} \n")

                actions = agent.act(states=states)
                states, terminal, reward = env.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)

                episode_energy.append(states[0])
                episode_trainingTime.append(states[1])
                episode_reward.append(reward)

                timestepCounter += 1
                # x.append(timestepCounter)
                AvgEnergyOfIotDevices.append(states[0])

            # sumRewardOfEpisodes = np.append(sumRewardOfEpisodes, episode_reward)
            sumRewardOfEpisodes.append(sum(episode_reward) / self.timestepNum)
            energyConsumption.append(sum(episode_energy) / self.timestepNum)
            trainingTimeOfEpisode.append(sum(episode_trainingTime) / self.timestepNum)
            trainingTimeOfAllTimesteps = np.append(trainingTimeOfAllTimesteps, episode_trainingTime)

            x.append(i)
            if i != 0 and i % int(self.episodeNum / 2) == 0:
                utils.draw_graph(title="Reward vs Episode",
                                 xlabel="Episode",
                                 ylabel="Reward",
                                 figSizeX=10,
                                 figSizeY=5,
                                 x=x,
                                 y=sumRewardOfEpisodes,
                                 savePath=self.saveGraphPath,
                                 pictureName=f"Reward_episode{i}")

                utils.draw_graph(title="Avg Energy vs Episode",
                                 xlabel="Episode",
                                 ylabel="Average Energy",
                                 figSizeX=10,
                                 figSizeY=5,
                                 x=x,
                                 y=energyConsumption,
                                 savePath=self.saveGraphPath,
                                 pictureName=f"Energy_episode{i}")

                utils.draw_graph(title="Avg TrainingTime vs Episode",
                                 xlabel="Episode",
                                 ylabel="TrainingTime",
                                 figSizeX=10,
                                 figSizeY=5,
                                 x=x,
                                 y=trainingTimeOfEpisode,
                                 savePath=self.saveGraphPath,
                                 pictureName=f"TrainingTime_episode{i}")

                utils.draw_scatter(title="Energy vs TrainingTime",
                                   xlabel="Energy",
                                   ylabel="TrainingTime",
                                   x=energyConsumption,
                                   y=trainingTimeOfEpisode,
                                   savePath=self.saveGraphPath,
                                   pictureName=f"Scatter{i}")

        utils.draw_hist(title='Avg Energy of IoT Devices',
                        x=AvgEnergyOfIotDevices,
                        xlabel="Average Energy",
                        savePath=self.saveGraphPath,
                        pictureName='AvgEnergy_hist')

        utils.draw_hist(title='TrainingTime of IoT Devices',
                        x=trainingTimeOfAllTimesteps,
                        xlabel="TrainingTime",
                        savePath=self.saveGraphPath,
                        pictureName='TrainingTime_hist')

        utils.draw_3dGraph(
            x=energyConsumption,
            y=trainingTimeOfEpisode,
            z=sumRewardOfEpisodes,
            xlabel=f"Energy {self.saveGraphPath}",
            ylabel="Training Time",
            zlabel="reward"
        )

        if self.agentType != "random":
            # Evaluate for 100 episodes
            rewardEvalEpisode = []
            sum_rewards = 0.0
            x = []

            for i in range(100):
                rewardEval = []
                states = env.reset()
                internals = agent.initial_internals()
                terminal = False
                while not terminal:
                    actions, internals = agent.act(states=states, internals=internals, evaluation=True)
                    states, terminal, reward = env.execute(actions=actions)
                    rewardEval.append(reward)
                    sum_rewards += reward
                rewardEvalEpisode.append(sum(rewardEval) / self.timestepNum)
                x.append(i)
            utils.draw_graph(title="Reward vs Episode Eval",
                             xlabel="Episode",
                             ylabel="Reward",
                             figSizeX=10,
                             figSizeY=5,
                             x=x,
                             y=rewardEvalEpisode,
                             savePath=self.saveGraphPath,
                             pictureName=f"Reward_episode_evaluation")
            print('Mean episode reward:', sum_rewards / 100)
        agent.close()
        env.close()


def createAgent(agentType, fraction, timestepNum, environment, saveSummariesPath):
    if agentType == 'ppo':
        return PPO.create(fraction=fraction, environment=environment, timestepNum=timestepNum,
                          saveSummariesPath=saveSummariesPath)
    elif agentType == 'ac':
        return AC.create(fraction=fraction, environment=environment, timestepNum=timestepNum,
                         saveSummariesPath=saveSummariesPath)
    elif agentType == 'tensorforce':
        return TensorforceAgent.create(fraction=fraction, environment=environment,
                                       timestepNum=timestepNum, saveSummariesPath=saveSummariesPath)
    elif agentType == 'trpo':
        return TRPO.create(fraction=fraction, environment=environment,
                           timestepNum=timestepNum, saveSummariesPath=saveSummariesPath)
    elif agentType == 'random':
        return RandomAgent.RandomAgent(environment=environment)
    elif agentType == 'noSplitting':
        return NoSplitting.NoSplitting(environment=environment)
    else:
        raise Exception('Invalid config select from [ppo, ac, tensorforce, random]')


def createEnv(timestepNum, iotDevices, edgeDevices, cloud, fraction, rewardTuningParams) -> Environment:
    return Environment.create(
        environment=customEnv.CustomEnvironment(rewardTuningParams=rewardTuningParams, iotDevices=iotDevices,
                                                edgeDevices=edgeDevices, cloud=cloud, fraction=fraction),
        max_episode_timesteps=timestepNum)


def createLog(fileName):
    logging.basicConfig(filename=f"./Logs/{fileName}.log",
                        format='%(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


def preTrainEnv(iotDevices: list[Device], edgeDevices: list[Device], cloud: Device, action):
    edgesConnectedDeviceNum = [0] * len(edgeDevices)

    for i in range(0, len(action), 2):
        edgeDevices[iotDevices[int(i / 2)].edgeIndex].connectedDevice = 0
        cloud.connectedDevice = 0

    totalEnergyConsumption = 0
    maxTrainingTime = 0
    offloadingPointsList = []

    iotRemainingFLOP = [iot.FLOPS for iot in iotDevices]
    edgeRemainingFLOP = [edge.FLOPS for edge in edgeDevices]
    cloudRemainingFLOP = cloud.FLOPS

    for i in range(0, len(action), 2):
        op1, op2 = utils.actionToLayer(action[i:i + 2])
        cloudRemainingFLOP -= sum(conf.COMP_WORK_LOAD[op2 + 1:])
        edgeRemainingFLOP[iotDevices[int(i / 2)].edgeIndex] -= sum(conf.COMP_WORK_LOAD[op1 + 1:op2 + 1])
        iotRemainingFLOP[int(i / 2)] -= sum(conf.COMP_WORK_LOAD[0:op1 + 1])

        if sum(conf.COMP_WORK_LOAD[op1 + 1:op2 + 1]):
            edgeDevices[iotDevices[int(i / 2)].edgeIndex].connectedDevice += 1
            edgesConnectedDeviceNum[iotDevices[int(i / 2)].edgeIndex] += 1
        if sum(conf.COMP_WORK_LOAD[op2 + 1:]) != 0:
            cloud.connectedDevice += 1

    for i in range(0, len(action), 2):
        # Mapping float number to Offloading points
        op1, op2 = utils.actionToLayer(action[i:i + 2])
        offloadingPointsList.append(op1)
        offloadingPointsList.append(op2)

        # computing training time of this action
        iotTrainingTime = iotDevices[int(i / 2)].trainingTime(splitPoints=[op1, op2],
                                                              remainingFlops=iotRemainingFLOP[int(i / 2)],
                                                              preTrain=True)
        edgeTrainingTime = edgeDevices[iotDevices[int(i / 2)].edgeIndex] \
            .trainingTime(splitPoints=[op1, op2],
                          remainingFlops=edgeRemainingFLOP[iotDevices[int(i / 2)].edgeIndex],
                          preTrain=True)
        cloudTrainingTime = cloud.trainingTime([op1, op2], remainingFlops=cloudRemainingFLOP, preTrain=True)

        totalTrainingTime = iotTrainingTime + edgeTrainingTime + cloudTrainingTime
        if totalTrainingTime > maxTrainingTime:
            maxTrainingTime = totalTrainingTime

        # computing energy consumption of iot devices
        iotEnergy = iotDevices[int(i / 2)].energyConsumption([op1, op2])
        totalEnergyConsumption += iotEnergy
    averageEnergyConsumption = totalEnergyConsumption / len(iotDevices)

    return averageEnergyConsumption, maxTrainingTime


def preTrain(iotDevices, edgeDevices, cloud):
    rewardTuningParams = [0, 0, 0, 0]
    min_Energy = 1.0e7
    max_Energy = 0

    min_trainingTime = 1.0e7
    max_trainingTime = 0

    splittingLayer = utils.allPossibleSplitting(modelLen=conf.LAYER_NUM - 1, deviceNumber=len(iotDevices))

    for splitting in splittingLayer:
        splittingArray = list()
        for char in splitting:
            splittingArray.append(int(char))

        avgEnergy, trainingTime = preTrainEnv(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                                              action=splittingArray)
        if avgEnergy < min_Energy:
            min_Energy = avgEnergy
            rewardTuningParams[0] = min_Energy
            min_energy_splitting = splittingArray
            min_Energy_TrainingTime = trainingTime
        if avgEnergy > max_Energy:
            max_Energy = avgEnergy
            rewardTuningParams[1] = max_Energy
            max_Energy_splitting = splittingArray
            max_Energy_TrainingTime = trainingTime

        if trainingTime < min_trainingTime:
            min_trainingTime = trainingTime
            rewardTuningParams[2] = min_trainingTime
            min_trainingtime_splitting = splittingArray
            min_trainingTime_energy = avgEnergy
        if trainingTime > max_trainingTime:
            max_trainingTime = trainingTime
            rewardTuningParams[3] = max_trainingTime
            max_trainingtime_splitting = splittingArray
            max_trainingTime_energy = avgEnergy
    return rewardTuningParams


def minMaxAvgEnergy(iotDevices, edgeDevices, cloud):
    splittingLayer = utils.allPossibleSplitting(modelLen=conf.LAYER_NUM - 1, deviceNumber=1)
    maxAvgEnergyOfOneDevice = 0
    minAvgEnergyOfOneDevice = 1.0e7
    maxEnergySplitting = []
    minEnergySplitting = []

    for splitting in splittingLayer:
        splittingArray = list()
        for char in splitting:
            splittingArray.append(int(char))

        avgEnergyOfOneDevice, trainingTimeOfOneDevice = preTrainEnv(iotDevices=iotDevices, edgeDevices=edgeDevices,
                                                                    cloud=cloud,
                                                                    action=splittingArray)
        if avgEnergyOfOneDevice > maxAvgEnergyOfOneDevice:
            maxAvgEnergyOfOneDevice = avgEnergyOfOneDevice
            maxEnergySplitting = splittingArray * len(iotDevices)
        if avgEnergyOfOneDevice < minAvgEnergyOfOneDevice:
            minAvgEnergyOfOneDevice = avgEnergyOfOneDevice
            minEnergySplitting = splittingArray * len(iotDevices)

    maxAvgEnergy, trainingTime = preTrainEnv(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                                             action=maxEnergySplitting)
    minAvgEnergy, trainingTime = preTrainEnv(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                                             action=minEnergySplitting)
    return maxAvgEnergy, minAvgEnergy


def ClassicFLTrainingTime(iotDevices, edgeDevices, cloud):
    maxTrainingTime = 0

    action = [conf.LAYER_NUM - 1, conf.LAYER_NUM - 1] * len(iotDevices)

    for i in range(0, len(action), 2):
        edgeDevices[iotDevices[int(i / 2)].edgeIndex].connectedDevice = 0
        cloud.connectedDevice = 0

    iotRemainingFLOP = [iot.FLOPS for iot in iotDevices]
    edgeRemainingFLOP = [edge.FLOPS for edge in edgeDevices]
    cloudRemainingFLOP = cloud.FLOPS

    for i in range(0, len(action), 2):
        op1 = action[i]
        op2 = action[i + 1]
        cloudRemainingFLOP -= sum(conf.COMP_WORK_LOAD[op2 + 1:])
        edgeRemainingFLOP[iotDevices[int(i / 2)].edgeIndex] -= sum(conf.COMP_WORK_LOAD[op1 + 1:op2 + 1])
        iotRemainingFLOP[int(i / 2)] -= sum(conf.COMP_WORK_LOAD[0:op1 + 1])

        if sum(conf.COMP_WORK_LOAD[op1 + 1:op2 + 1]):
            edgeDevices[iotDevices[int(i / 2)].edgeIndex].connectedDevice += 1
        if sum(conf.COMP_WORK_LOAD[op2 + 1:]) != 0:
            cloud.connectedDevice += 1

    for i in range(0, len(action), 2):
        # Mapping float number to Offloading points
        op1 = action[i]
        op2 = action[i + 1]

        # computing training time of this action
        iotTrainingTime = iotDevices[int(i / 2)].trainingTime(splitPoints=[op1, op2],
                                                              remainingFlops=iotRemainingFLOP[int(i / 2)],
                                                              preTrain=True)
        edgeTrainingTime = edgeDevices[iotDevices[int(i / 2)].edgeIndex] \
            .trainingTime(splitPoints=[op1, op2],
                          remainingFlops=edgeRemainingFLOP[iotDevices[int(i / 2)].edgeIndex],
                          preTrain=True)
        cloudTrainingTime = cloud.trainingTime([op1, op2], remainingFlops=cloudRemainingFLOP, preTrain=True)

        totalTrainingTime = iotTrainingTime + edgeTrainingTime + cloudTrainingTime
        if totalTrainingTime > maxTrainingTime:
            maxTrainingTime = totalTrainingTime
    return maxTrainingTime
