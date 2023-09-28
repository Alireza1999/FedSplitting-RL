import logging
import numpy as np
from tensorforce import Agent, Environment
from Tensorforce import utils
from Tensorforce.enviroments import customEnv
from Tensorforce.splittingMethods import randomAgent
from Tensorforce import config as conf
from pathlib import Path


class Runner:

    def __init__(self, agentType='ppo', episodeNum=501, timestepNum=200, fraction=0.8, summaries=False, log=False):
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

        rewardTuningParams = preTrain(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud)

        print(f"------------------------------------------------")
        print(f"MIN Energy : \n{rewardTuningParams[0]}")

        print(f"------------------------------------------------")
        print(f"MAX Energy : {rewardTuningParams[1]}")

        print(f"------------------------------------------------")
        print(f"MIN TrainingTime : {rewardTuningParams[2]}")

        print(f"------------------------------------------------")
        print(f"MAX TrainingTime : {rewardTuningParams[3]}")
        print(f"------------------------------------------------")

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
            sumRewardOfEpisodes.append(sum(episode_reward))
            energyConsumption.append(sum(episode_energy) / self.timestepNum)
            trainingTimeOfEpisode.append(sum(episode_trainingTime) / self.timestepNum)
            trainingTimeOfAllTimesteps = np.append(trainingTimeOfAllTimesteps, episode_trainingTime)

            x.append(i)
            if i != 0 and i % int(self.episodeNum / 4) == 0:
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
            xlabel="Energy",
            ylabel="Training Time",
            zlabel="reward"
        )

        agent.close()
        env.close()


def createTensorforceAgent(agentType, fraction, environment, timestepNum, saveSummariesPath) -> Agent:
    return Agent.create(
        agent='tensorforce',
        environment=environment,
        max_episode_timesteps=timestepNum,

        # Reward estimation
        reward_estimation=dict(
            horizon=1,
            discount=0.96),

        # Optimizer
        optimizer=dict(
            optimizer='adam', learning_rate=0.001, clipping_threshold=0.01,
            multi_step=10, subsampling_fraction=0.99
        ),

        # update network every 2 timestep
        update=dict(
            unit='timesteps',
            batch_size=2,
        ),

        policy=dict(
            type='parametrized_distributions',
            network='auto',
            distributions=dict(
                float=dict(type='gaussian'),
                bounded_action=dict(type='beta')
            ),
            temperature=dict(
                type='decaying', decay='exponential', unit='timesteps',
                decay_steps=5, initial_value=0.01, decay_rate=0.5
            )
        ),

        objective='policy_gradient',
        # Preprocessing
        preprocessing=None,

        # Exploration
        exploration=0.1, variable_noise=0.0,

        # Regularization
        l2_regularization=0.5, entropy_regularization=0.1,
        memory=200,
        # TensorFlow etc
        # saver=dict(directory='model', filename='model'),
        summarizer=dict(directory=f"{saveSummariesPath}/summaries/{agentType}_{fraction}",
                        frequency=50,
                        labels='all',
                        ),
        recorder=None,

        # Config
        config=dict(name='agent',
                    device="GPU",
                    parallel_interactions=1,
                    seed=None,
                    execution=None,
                    )
    )


def createACAgents(agentType, fraction, environment, timestepNum, saveSummariesPath) -> Agent:
    return Agent.create(
        agent='ac',
        environment=environment,
        max_episode_timesteps=timestepNum,
        # Automatically configured network
        network='auto',
        # Optimization
        batch_size=10, update_frequency=2, learning_rate=0.0001,
        # Reward estimation
        discount=0.99, estimate_terminal=False,
        # Critic
        horizon=1,
        critic_network='auto',
        critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=0.0001),
        # Preprocessing
        preprocessing=None,
        # Exploration
        exploration=0.1, variable_noise=0.0,
        # Regularization
        l2_regularization=0.1, entropy_regularization=0.01,
        # TensorFlow etc
        name='agent',
        device=None,
        parallel_interactions=1,
        seed=None,
        execution=None,
        saver=None,
        summarizer=dict(directory=f"{saveSummariesPath}/summaries/{agentType}_{fraction}",
                        frequency=50,
                        labels='all',
                        ),
        recorder=None
    )


def createPPOAgent(agentType, fraction, environment, timestepNum, saveSummariesPath) -> Agent:
    return Agent.create(
        agent='ppo',
        environment=environment,
        max_episode_timesteps=timestepNum,
        # Automatically configured network
        network='auto',

        # Optimization
        batch_size=5, update_frequency=5, learning_rate=1e-5, subsampling_fraction=0.2,
        optimization_steps=5,

        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=0.96, estimate_terminal=False,

        # Critic
        critic_network='auto',
        critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-5),

        # Preprocessing
        preprocessing=None,

        # Exploration
        exploration=0.1, variable_noise=0.0,

        # Regularization
        l2_regularization=0.0, entropy_regularization=0.2,

        # TensorFlow etc
        name='agent',
        device=None,
        parallel_interactions=1,
        seed=None,
        execution=None,
        saver=None,
        summarizer=dict(directory=f"{saveSummariesPath}/summaries/{agentType}_{fraction}",
                        frequency=50,
                        labels='all',
                        ),
        recorder=None
    )


def createRandomAgent(environment):
    return randomAgent.RandomAgent.create(agent='random', environment=environment)


def createAgent(agentType, fraction, timestepNum, environment, saveSummariesPath):
    if agentType == 'ppo':
        return createPPOAgent(agentType=agentType, fraction=fraction, environment=environment, timestepNum=timestepNum,
                              saveSummariesPath=saveSummariesPath)
    elif agentType == 'ac':
        return createACAgents(agentType=agentType, fraction=fraction, environment=environment, timestepNum=timestepNum,
                              saveSummariesPath=saveSummariesPath)
    elif agentType == 'tensorforce':
        return createTensorforceAgent(agentType=agentType, fraction=fraction, environment=environment,
                                      timestepNum=timestepNum, saveSummariesPath=saveSummariesPath)
    elif agentType == 'random':
        return createRandomAgent(environment=environment)
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


def preTrainEnv(iotDevices, edgeDevices, cloud, actions):
    totalEnergyConsumption = 0
    maxTrainingTime = 0
    offloadingPointsList = []

    iotDeviceCapacity = [iotDevice.capacity for iotDevice in iotDevices]
    edgeCapacity = [edges.capacity for edges in edgeDevices]
    cloudCapacity = cloud.capacity

    for i in range(0, len(actions), 2):
        op1 = actions[i]
        op2 = actions[i + 1]
        cloudCapacity -= sum(conf.COMP_WORK_LOAD[op2 + 1:])
        edgeCapacity[iotDevices[int(i / 2)].edgeIndex] -= sum(conf.COMP_WORK_LOAD[op1 + 1:op2 + 1])
        iotDeviceCapacity[int(i / 2)] -= sum(conf.COMP_WORK_LOAD[0:op1 + 1])

    for i in range(0, len(actions), 2):
        # Mapping float number to Offloading points
        op1 = actions[i]
        op2 = actions[i + 1]
        offloadingPointsList.append(op1)
        offloadingPointsList.append(op2)

        # computing training time of this action
        iotTrainingTime = iotDevices[int(i / 2)].trainingTime([op1, op2], preTrain=True)
        edgeTrainingTime = edgeDevices[iotDevices[int(i / 2)].edgeIndex].trainingTime([op1, op2], preTrain=True)
        cloudTrainingTime = cloud.trainingTime([op1, op2], preTrain=True)

        if iotDeviceCapacity[int(i / 2)] < 0:
            iotTrainingTime *= (1 + abs(iotDeviceCapacity[int(i / 2)]) / 10)
        if edgeCapacity[iotDevices[int(i / 2)].edgeIndex] < 0 and (actions[i] != actions[i + 1]):
            edgeTrainingTime *= (1 + abs(edgeCapacity[iotDevices[int(i / 2)].edgeIndex]) / 10)
        if cloudCapacity < 0 and actions[i + 1] < conf.LAYER_NUM - 1:
            cloudTrainingTime *= (1 + abs(cloudCapacity) / 10)

        totalTrainingTime = iotTrainingTime + edgeTrainingTime + cloudTrainingTime
        if totalTrainingTime > maxTrainingTime:
            maxTrainingTime = totalTrainingTime

        # computing energy consumption of iot devices
        iotEnergy = iotDevices[int(i / 2)].energyConsumption([op1, op2])
        totalEnergyConsumption += iotEnergy

    averageEnergyConsumption = totalEnergyConsumption / len(iotDevices)

    return averageEnergyConsumption, maxTrainingTime


def preTrain(iotDevices, edgeDevices, cloud):
    rewardTuningParams = [10000, 0, 10000, 0]
    min_Energy = 10000
    max_Energy = 0

    min_trainingTime = 10000
    max_trainingTime = 0

    splittingLayer = utils.allPossibleSplitting(modelLen=conf.LAYER_NUM - 1, deviceNumber=len(iotDevices))

    for splitting in splittingLayer:
        splittingArray = list()
        for char in splitting:
            splittingArray.append(int(char))

        avgEnergy, trainingTime = preTrainEnv(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                                              actions=splittingArray)
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