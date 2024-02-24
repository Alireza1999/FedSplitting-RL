import logging
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorforce import Environment

from Tensorforce import utils

logger = logging.getLogger()


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
        iotDevices = utils.createDeviceFromCSV(csvFilePath="../System/iotDevices.csv",
                                               deviceType='iotDevice')
        edgeDevices = utils.createDeviceFromCSV(csvFilePath="../System/edges.csv")
        cloud = utils.createDeviceFromCSV(csvFilePath="../System/cloud.csv")[0]

        if self.envType == "fedAdapt":
            allTrainingTime = utils.ClassicFLTrainingTimeWithoutEdge(iotDevices, cloud)
            rewardTuningParams = allTrainingTime
            FedAdaptRunner(self.timestepNum, self.episodeNum, rewardTuningParams=rewardTuningParams)
        else:
            if self.envType == "defaultNoEdge":
                FLTrainingTime = max(utils.ClassicFLTrainingTimeWithoutEdge(iotDevices, cloud))
                rewardTuningParams = [0, 0, FLTrainingTime]
            else:
                FLEnergy, FLTrainingTime = utils.ClassicFLTrainingTime(iotDevices, edgeDevices, cloud)
                # maxEnergy, minEnergy = minMaxAvgEnergy(iotDevices, edgeDevices, cloud)
                rewardTuningParams = [FLEnergy, FLTrainingTime]

            print(f"------------------------------------------------")
            # print(f"Max Energy : \n{rewardTuningParams[0]}")
            # print(f"Min Energy : \n{rewardTuningParams[1]}")
            # print(f"------------------------------------------------")
            print(f"Classic FL Training Time : {FLTrainingTime}")
            print(f"Classic FL Energy: {FLEnergy}\n")

            envObject = utils.createEnv(rewardTuningParams=rewardTuningParams,
                                        iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                                        fraction=self.fraction,
                                        envType=self.envType)

            env = Environment.create(environment=envObject,
                                     max_episode_timesteps=self.timestepNum)

            agent = utils.createAgent(agentType=self.agentType,
                                      fraction=self.fraction,
                                      environment=env,
                                      timestepNum=self.timestepNum,
                                      saveSummariesPath=self.saveSummariesPath,
                                      iotDevices=iotDevices,
                                      edgeDevices=edgeDevices,
                                      cloud=cloud)

            if self.envType == "defaultWithBandwidth":
                bandwidthInStateRunner(envObject=envObject,
                                       env=env,
                                       agent=agent,
                                       timestepNum=self.timestepNum,
                                       episodeNum=self.episodeNum,
                                       saveGraphPath=self.saveGraphPath,
                                       saveLog=self.log)
            else:
                if self.log:
                    logger = utils.createLog(fileName=f"{self.envType}_{self.agentType}_{self.fraction}")

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

                # hexadecimal_alphabets = '0123456789ABCDEF'
                # color = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in range(6)]) for i in
                #          range(len(iotDevices))]
                # plt.figure(figsize=(int(10), int(5)))  # Set the figure size
                # for i in range(len(iotDevices)):
                #     plt.plot(y, envObject.effectiveBandwidth[i], color=color[i], label=f'client-{i}')
                # plt.legend()
                # plt.title("All Reward Graphs")
                # plt.xlabel("timesteps")
                # plt.ylabel("Bandwidth")
                # plt.savefig(os.path.join(self.saveGraphPath, f"Bandwidth"))
                # plt.close()

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


def FedAdaptRunner(timestepNum, episodeNum, rewardTuningParams):
    iotDevices = utils.createDeviceFromCSV(csvFilePath="../System/iotDevicesScalabilityTest50Device.csv",
                                           deviceType='iotDevice')
    edgeDevices = utils.createDeviceFromCSV(csvFilePath="../System/edgesScalabilityTest50Device.csv")
    cloud = utils.createDeviceFromCSV(csvFilePath="../System/cloud50Device.csv")[0]

    saveGraphPath = f"Graphs/FedAdapt/ScalabilityTest"

    print(f"------------------------------------------------")
    print(f"All Training Time: \n{rewardTuningParams}")
    print(f"------------------------------------------------")

    envObject = utils.createEnv(rewardTuningParams=rewardTuningParams,
                                iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                                fraction=0.0,
                                envType="fedAdapt")

    env = Environment.create(environment=envObject,
                             max_episode_timesteps=timestepNum)

    agent = utils.createAgent(agentType='ppo',
                              fraction=0.0,
                              environment=env,
                              timestepNum=timestepNum,
                              saveSummariesPath=None)

    logger = utils.createLog(fileName=f"fedAdapt")

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


def bandwidthInStateRunner(envObject, env, agent, timestepNum, episodeNum, saveGraphPath,
                           saveLog: bool = True):
    sumRewardOfEpisodes = list()
    rewardOfEnergy = list()
    rewardOfTrainingTime = list()
    energyConsumption = list()
    trainingTimeOfEpisode = list()
    trainingTimeOfAllTimesteps = list()
    logger = utils.createLog(fileName=f"defaultBandwidthInState_tensorforce_1.0")

    y = list()
    x = list()
    AvgEnergyOfIotDevices = list()
    timestepCounter = 0
    for i in range(episodeNum):
        if saveLog:
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
        for j in range(timestepNum):
            if saveLog:
                logger.info("-------------------------------------------")
                logger.info(f"Timestep {timestepCounter} \n")

            actions = agent.act(states=states)
            states, terminal, reward = env.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)

            episode_reward.append(reward)
            episode_rewardOfEnergy.append(envObject.rewardOfEnergy)
            episode_rewardOfTrainingTime.append(envObject.rewardOfTrainingTime)
            episode_energy.append(envObject.avgEnergy)
            episode_trainingTime.append(envObject.tt)
            AvgEnergyOfIotDevices.append(envObject.avgEnergy)

            y.append(timestepCounter)
            timestepCounter += 1

        rewardOfEnergy.append(sum(episode_rewardOfEnergy) / timestepNum)
        rewardOfTrainingTime.append(sum(episode_rewardOfTrainingTime) / timestepNum)
        sumRewardOfEpisodes.append(sum(episode_reward) / timestepNum)
        energyConsumption.append(sum(episode_energy) / timestepNum)
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

            utils.draw_graph(title="Avg Energy vs Episode",
                             xlabel="Episode",
                             ylabel="Average Energy",
                             figSizeX=10,
                             figSizeY=5,
                             x=x,
                             y=energyConsumption,
                             savePath=saveGraphPath,
                             pictureName=f"Energy_episode{i}")

            utils.draw_graph(title="Avg TrainingTime vs Episode",
                             xlabel="Episode",
                             ylabel="TrainingTime",
                             figSizeX=10,
                             figSizeY=5,
                             x=x,
                             y=trainingTimeOfEpisode,
                             savePath=saveGraphPath,
                             pictureName=f"TrainingTime_episode{i}")

            utils.draw_scatter(title="Energy vs TrainingTime",
                               xlabel="Energy",
                               ylabel="TrainingTime",
                               x=energyConsumption,
                               y=trainingTimeOfEpisode,
                               savePath=saveGraphPath,
                               pictureName=f"Scatter{i}")
    y.append(timestepCounter)
    timestepCounter += 1

    utils.draw_hist(title='Avg Energy of IoT Devices',
                    x=AvgEnergyOfIotDevices,
                    xlabel="Average Energy",
                    savePath=saveGraphPath,
                    pictureName='AvgEnergy_hist')

    utils.draw_hist(title='TrainingTime of IoT Devices',
                    x=trainingTimeOfAllTimesteps,
                    xlabel="TrainingTime",
                    savePath=saveGraphPath,
                    pictureName='TrainingTime_hist')

    # trainingTimes = np.array(trainingTimeOfEpisode)
    # np.save(f'{self.envType}_{self.agentType}_trainingTimes.npy', trainingTimes)
    #
    # rewards = np.array(sumRewardOfEpisodes)
    # np.save(f'{self.envType}_{self.agentType}_rewards.npy', rewards)

    # Create a plot
    plt.figure(figsize=(int(10), int(5)))  # Set the figure size
    plt.plot(x, rewardOfEnergy, color='red', label='Energy reward')
    plt.plot(x, rewardOfTrainingTime, color='green', label='TrainingTime reward')
    plt.plot(x, sumRewardOfEpisodes, color='blue', label='Total Reward')
    plt.legend()
    plt.title("All Reward Graphs")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig(os.path.join(saveGraphPath, f"Reward_energy_trainingTime"))
    plt.close()

    # hexadecimal_alphabets = '0123456789ABCDEF'
    # color = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in range(6)]) for i in
    #          range(len(iotDevices))]
    # plt.figure(figsize=(int(10), int(5)))  # Set the figure size
    # for i in range(len(iotDevices)):
    #     plt.plot(y, envObject.effectiveBandwidth[i], color=color[i], label=f'client-{i}')
    # plt.legend()
    # plt.title("All Reward Graphs")
    # plt.xlabel("timesteps")
    # plt.ylabel("Bandwidth")
    # plt.savefig(os.path.join(self.saveGraphPath, f"Bandwidth"))
    # plt.close()

    utils.draw_3dGraph(
        x=energyConsumption,
        y=trainingTimeOfEpisode,
        z=sumRewardOfEpisodes,
        xlabel=f"Energy {saveGraphPath}",
        ylabel="Training Time",
        zlabel="reward"
    )

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
            rewardEval.append(reward)
            sum_rewards += reward
        rewardEvalEpisode.append(sum(rewardEval) / timestepNum)
        x.append(i)

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
