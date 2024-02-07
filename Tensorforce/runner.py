import logging
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorforce import Environment

from System.Device import Device
from Tensorforce import config as conf
from Tensorforce import utils
from Tensorforce.enviroments import customEnv, fedAdaptEnv, customEnvNoEdge
from Tensorforce.splittingMethods import RandomAgent, NoSplitting, AC, PPO, TensorforceAgent, TRPO, FirstFit


class Runner:

    def __init__(self, envType="default", agentType='tensorforce', episodeNum=501, timestepNum=200, fraction=0.8,
                 summaries=False,
                 log=False):

        self.envType = envType
        if self.envType == "fedAdapt":
            self.agentType = 'ppo'
            self.fraction = 0.0
        else:
            self.agentType = agentType
            self.fraction = fraction

        self.episodeNum = episodeNum
        self.timestepNum = timestepNum

        self.summaries = summaries
        self.log = log

        self.saveGraphPath = f"Graphs/{self.envType}/{self.agentType}/{self.fraction}/ScalabilityTest"
        self.saveSummariesPath = f"{Path(__file__).parent}"

    def run(self):
        iotDevices = utils.createDeviceFromCSV(csvFilePath="../System/iotDevicesScalabilityTest50Device.csv",
                                               deviceType='iotDevice')
        edgeDevices = utils.createDeviceFromCSV(csvFilePath="../System/edgesScalabilityTest50Device.csv")
        cloud = utils.createDeviceFromCSV(csvFilePath="../System/cloud50Device.csv")[0]

        if self.envType == "fedAdapt":
            allTrainingTime = ClassicFLTrainingTimeWithoutEdge(iotDevices, cloud)
            rewardTuningParams = allTrainingTime
            FedAdaptRunner(self.timestepNum, self.episodeNum, rewardTuningParams=rewardTuningParams)
        else:
            if self.envType == "defaultNoEdge":
                FLTrainingTime = max(ClassicFLTrainingTimeWithoutEdge(iotDevices, cloud))
                rewardTuningParams = [0, 0, FLTrainingTime]
            else:
                FLTrainingTime, FLEnergy = ClassicFLTrainingTime(iotDevices, edgeDevices, cloud)
                maxEnergy, minEnergy = minMaxAvgEnergy(iotDevices, edgeDevices, cloud)
                rewardTuningParams = [maxEnergy, minEnergy, FLTrainingTime]

            print(f"------------------------------------------------")
            print(f"Max Energy : \n{rewardTuningParams[0]}")
            print(f"Min Energy : \n{rewardTuningParams[1]}")
            print(f"------------------------------------------------")
            print(f"Classic FL Training Time : \n{rewardTuningParams[2]}")
            if self.envType == 'default':
                print(f"Classic FL Energy : \n{FLEnergy}")
            print(f"------------------------------------------------")

            envObject = createEnv(rewardTuningParams=rewardTuningParams,
                                  iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                                  fraction=self.fraction,
                                  envType=self.envType)

            env = Environment.create(environment=envObject,
                                     max_episode_timesteps=self.timestepNum)

            agent = createAgent(agentType=self.agentType,
                                fraction=self.fraction,
                                environment=env,
                                timestepNum=self.timestepNum,
                                saveSummariesPath=self.saveSummariesPath,
                                iotDevices=iotDevices,
                                edgeDevices=edgeDevices,
                                cloud=cloud)

            if self.log:
                logger = createLog(fileName=f"{self.envType}_{self.agentType}_{self.fraction}")

            sumRewardOfEpisodes = list()
            rewardOfEnergy = list()
            rewardOfTrainingTime = list()
            energyConsumption = list()
            trainingTimeOfEpisode = list()
            trainingTimeOfAllTimesteps = list()

            y = list()
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
                episode_rewardOfEnergy = list()
                episode_rewardOfTrainingTime = list()

                states = env.reset()

                y.append(timestepCounter)
                timestepCounter += 1

                internals = agent.initial_internals()
                terminal = False
                while not terminal:
                    if self.log:
                        logger.info("-------------------------------------------")
                        logger.info(f"Timestep {timestepCounter} \n")

                    actions = agent.act(states=states)
                    states, terminal, reward = env.execute(actions=actions)
                    agent.observe(terminal=terminal, reward=reward)

                    if self.envType == "default":
                        episode_energy.append(states[0])
                        episode_trainingTime.append(states[1])
                    elif self.envType == "defaultNoEdge":
                        episode_trainingTime.append(states[0])
                    episode_reward.append(reward)
                    episode_rewardOfEnergy.append(envObject.rewardOfEnergy)
                    episode_rewardOfTrainingTime.append(envObject.rewardOfTrainingTime)

                    y.append(timestepCounter)
                    timestepCounter += 1

                    AvgEnergyOfIotDevices.append(states[0])

                rewardOfEnergy.append(sum(episode_rewardOfEnergy) / self.timestepNum)
                rewardOfTrainingTime.append(sum(episode_rewardOfTrainingTime) / self.timestepNum)
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
            y.append(timestepCounter)
            timestepCounter += 1

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

            trainingTimes = np.array(trainingTimeOfEpisode)
            np.save(f'{self.envType}_{self.agentType}_trainingTimes.npy', trainingTimes)

            rewards = np.array(sumRewardOfEpisodes)
            np.save(f'{self.envType}_{self.agentType}_rewards.npy', rewards)

            # Create a plot
            plt.figure(figsize=(int(10), int(5)))  # Set the figure size
            plt.plot(x, rewardOfEnergy, color='red', label='Energy reward')
            plt.plot(x, rewardOfTrainingTime, color='green', label='TrainingTime reward')
            plt.plot(x, sumRewardOfEpisodes, color='blue', label='Total Reward')
            plt.legend()
            plt.title("All Reward Graphs")
            plt.xlabel("episode")
            plt.ylabel("reward")
            plt.savefig(os.path.join(self.saveGraphPath, f"Reward_energy_trainingTime"))
            plt.close()

            hexadecimal_alphabets = '0123456789ABCDEF'
            color = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in range(6)]) for i in
                     range(len(iotDevices))]
            plt.figure(figsize=(int(10), int(5)))  # Set the figure size
            for i in range(len(iotDevices)):
                plt.plot(y, envObject.effectiveBandwidth[i], color=color[i], label=f'client-{i}')
            plt.legend()
            plt.title("All Reward Graphs")
            plt.xlabel("timesteps")
            plt.ylabel("Bandwidth")
            plt.savefig(os.path.join(self.saveGraphPath, f"Bandwidth"))
            plt.close()

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
                evaluationTrainingTimes = []
                sum_rewards = 0.0
                x = []

                for i in range(100):
                    # if i >= 50:
                    #     for k in range(len(iotDevices)):
                    #         iotDevices[k].bandwidth = iotDevices[k].bandwidth * 0.7
                    rewardEval = []
                    states = env.reset()
                    internals = agent.initial_internals()
                    terminal = False
                    while not terminal:
                        actions, internals = agent.act(states=states, internals=internals, evaluation=True)
                        states, terminal, reward = env.execute(actions=actions)
                        if self.envType == "default":
                            evaluationTrainingTimes.append(states[1])
                        elif self.envType == "defaultNoEdge":
                            evaluationTrainingTimes.append(states[0])
                        rewardEval.append(reward)
                        sum_rewards += reward
                    rewardEvalEpisode.append(sum(rewardEval) / self.timestepNum)
                    x.append(i)

                trainingTimes = np.array(evaluationTrainingTimes)
                np.save(f'{self.envType}_{self.agentType}_trainingTimesHistEvaluation.npy', trainingTimes)

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


def createAgent(agentType, fraction, timestepNum, environment, saveSummariesPath, iotDevices=None, edgeDevices=None,
                cloud=None):
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
    elif agentType == 'firstFit':
        return FirstFit.FirstFit(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud)
    else:
        raise Exception('Invalid config select from [ppo, ac, tensorforce, random]')


def createEnv(iotDevices, edgeDevices, cloud, fraction, rewardTuningParams,
              envType=None, groupNum=1):
    if envType == 'default':
        return customEnv.CustomEnvironment(rewardTuningParams=rewardTuningParams, iotDevices=iotDevices,
                                           edgeDevices=edgeDevices, cloud=cloud, fraction=fraction)
    elif envType == "fedAdapt":
        return fedAdaptEnv.FedAdaptEnv(allTrainingTime=rewardTuningParams,
                                       iotDevices=iotDevices,
                                       cloud=cloud,
                                       groupNum=groupNum)
    elif envType == "defaultNoEdge":
        return customEnvNoEdge.CustomEnvironmentNoEdge(rewardTuningParams=rewardTuningParams,
                                                       iotDevices=iotDevices,
                                                       cloud=cloud)
    else:
        raise "Invalid Environment Parameter. Valid option : default, fedAdapt, defaultNoEdge"


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
    totalEnergyConsumption = 0

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
        # computing energy consumption of iot devices
        iotEnergy = iotDevices[int(i / 2)].energyConsumption([op1, op2])
        totalEnergyConsumption += iotEnergy

        if totalTrainingTime > maxTrainingTime:
            maxTrainingTime = totalTrainingTime
    averageEnergy = totalEnergyConsumption / len(iotDevices)

    return maxTrainingTime, averageEnergy


def ClassicFLTrainingTimeWithoutEdge(iotDevices, cloud):
    allTrainingTime = []
    maxTrainingTime = 0
    action = [conf.LAYER_NUM - 1] * 1
    cloud.connectedDevice = 0

    iotRemainingFLOP = [iot.FLOPS for iot in iotDevices]
    cloudRemainingFLOP = cloud.FLOPS

    for i in range(0, len(iotDevices)):
        op = action[0]
        cloudRemainingFLOP -= sum(conf.COMP_WORK_LOAD[op + 1:])
        iotRemainingFLOP[int(i)] -= sum(conf.COMP_WORK_LOAD[0:op + 1])
        if sum(conf.COMP_WORK_LOAD[op + 1:]) != 0:
            cloud.connectedDevice += 1

    for i in range(0, len(iotDevices)):
        # Mapping float number to Offloading points
        op = action[0]
        # computing training time of this action
        iotTrainingTime = iotDevices[int(i)].trainingTime(splitPoints=[op, op],
                                                          remainingFlops=iotRemainingFLOP[int(i)],
                                                          preTrain=True)
        cloudTrainingTime = cloud.trainingTime([op, op],
                                               remainingFlops=cloudRemainingFLOP,
                                               preTrain=True)

        totalTrainingTime = iotTrainingTime + cloudTrainingTime
        allTrainingTime.append(totalTrainingTime)

        if totalTrainingTime > maxTrainingTime:
            maxTrainingTime = totalTrainingTime
    return allTrainingTime


def FedAdaptRunner(timestepNum, episodeNum, rewardTuningParams):
    iotDevices = utils.createDeviceFromCSV(csvFilePath="../System/iotDevicesScalabilityTest50Device.csv",
                                           deviceType='iotDevice')
    edgeDevices = utils.createDeviceFromCSV(csvFilePath="../System/edgesScalabilityTest50Device.csv")
    cloud = utils.createDeviceFromCSV(csvFilePath="../System/cloud50Device.csv")[0]

    saveGraphPath = f"Graphs/FedAdapt/ScalabilityTest"

    print(f"------------------------------------------------")
    print(f"All Training Time: \n{rewardTuningParams}")
    print(f"------------------------------------------------")

    envObject = createEnv(rewardTuningParams=rewardTuningParams,
                          iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                          fraction=0.0,
                          envType="fedAdapt")

    env = Environment.create(environment=envObject,
                             max_episode_timesteps=timestepNum)

    agent = createAgent(agentType='ppo',
                        fraction=0.0,
                        environment=env,
                        timestepNum=timestepNum,
                        saveSummariesPath=None)

    logger = createLog(fileName=f"fedAdapt")

    sumRewardOfEpisodes = list()
    trainingTimeOfEpisode = list()
    trainingTimeOfAllTimesteps = list()

    y = list()
    x = list()
    timestepCounter = 0
    for i in range(episodeNum):
        logger.info("===========================================")
        logger.info("Episode {} started ...\n".format(i))

        episode_trainingTime = list()
        episode_reward = list()

        y.append(timestepCounter)
        timestepCounter += 1
        states = env.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            logger.info("-------------------------------------------")
            logger.info(f"Timestep {timestepCounter} \n")
            actions = agent.act(states=states)
            states, terminal, reward = env.execute(actions=actions)
            episode_trainingTime.append(states[0])
            episode_reward.append(reward)
            agent.observe(terminal=terminal, reward=reward)

            y.append(timestepCounter)
            timestepCounter += 1

        sumRewardOfEpisodes.append(sum(episode_reward) / timestepNum)
        trainingTimeOfEpisode.append(sum(episode_trainingTime) / timestepNum)
        trainingTimeOfAllTimesteps = np.append(trainingTimeOfAllTimesteps, episode_trainingTime)

        x.append(i)
        if i != 0 and i % int(episodeNum / 2) == 0:
            utils.draw_graph(title="Reward vs Episode",
                             xlabel="Episode",
                             ylabel="Reward",
                             figSizeX=10,
                             figSizeY=5,
                             x=x,
                             y=sumRewardOfEpisodes,
                             savePath=saveGraphPath,
                             pictureName=f"Reward_episode{i}")

            utils.draw_graph(title="Avg TrainingTime vs Episode",
                             xlabel="Episode",
                             ylabel="TrainingTime",
                             figSizeX=10,
                             figSizeY=5,
                             x=x,
                             y=trainingTimeOfEpisode,
                             savePath=saveGraphPath,
                             pictureName=f"TrainingTime_episode{i}")

    y.append(timestepCounter)
    timestepCounter += 1

    trainingTimes = np.array(trainingTimeOfEpisode)
    np.save(f'fedAdapt_ppo_trainingTimes.npy', trainingTimes)

    rewards = np.array(sumRewardOfEpisodes)
    np.save(f'fedAdapt_ppo_rewards.npy', rewards)

    utils.draw_hist(title='TrainingTime of IoT Devices',
                    x=trainingTimeOfAllTimesteps,
                    xlabel="TrainingTime",
                    savePath=saveGraphPath,
                    pictureName='TrainingTime_hist')

    hexadecimal_alphabets = '0123456789ABCDEF'
    color = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in range(6)]) for i in
             range(len(iotDevices))]
    plt.figure(figsize=(int(10), int(5)))  # Set the figure size
    for i in range(len(iotDevices)):
        plt.plot(y, envObject.effectiveBandwidth[i], color=color[i], label=f'client-{i}')
    plt.legend()
    plt.title("All Reward Graphs")
    plt.xlabel("timesteps")
    plt.ylabel("Bandwidth")
    plt.savefig(os.path.join(saveGraphPath, f"Bandwidth"))
    plt.close()

   # Evaluate for 100 episodes
    rewardEvalEpisode = []
    sum_rewards = 0.0
    x = []
    evaluationTrainingTimes = []

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
            evaluationTrainingTimes.append(states[0])
        rewardEvalEpisode.append(sum(rewardEval) / timestepNum)
        x.append(i)

    trainingTimes = np.array(evaluationTrainingTimes)
    np.save(f'fedAdapt_ppo_trainingTimesHistEvaluation.npy', trainingTimes)

    utils.draw_graph(title="Reward vs Episode Eval",
                     xlabel="Episode",
                     ylabel="Reward",
                     figSizeX=10,
                     figSizeY=5,
                     x=x,
                     y=rewardEvalEpisode,
                     savePath=saveGraphPath,
                     pictureName=f"Reward_episode_evaluation")

    print('Mean episode reward:', sum_rewards / 100)
    agent.close()
    env.close()
