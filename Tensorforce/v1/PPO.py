import logging

import numpy as np
from tensorforce import Agent, Environment

from Tensorforce import utils
from Tensorforce.v1.customEnv import CustomEnvironment

logging.basicConfig(filename="./Logs/PPO/info.log",
                    format='%(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

iotDevices = utils.createDeviceFromCSV(csvFilePath="../../System/iotDevicesSmallScale.csv", deviceType='iotDevice')
edgeDevices = utils.createDeviceFromCSV(csvFilePath="../../System/edgesSmallScale.csv", deviceType='edge')
cloud = utils.createDeviceFromCSV(csvFilePath="../../System/cloud.csv")[0]

environment = Environment.create(
    environment=CustomEnvironment(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud),
    max_episode_timesteps=200
)

agent = Agent.create(
    agent='ppo', environment=environment,
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
    name='agent', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
    summarizer=None, recorder=None
)
sumRewardOfEpisodes = list()
energyConsumption = list()

trainingTimeOfEpisode = list()
trainingTimeOfAllTimesteps = list()

x = list()
AvgEnergyOfIotDevices = list()
timestepCounter = 0
for i in range(501):
    logger.info("===========================================")
    logger.info("Episode {} started ...\n".format(i))

    episode_energy = list()
    episode_trainingTime = list()
    episode_reward = list()

    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        logger.info("-------------------------------------------")
        logger.info(f"Timestep {timestepCounter} \n")

        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        episode_energy.append(states[0])
        episode_trainingTime.append(states[1])
        episode_reward.append(reward)

        timestepCounter += 1
        # x.append(timestepCounter)
        AvgEnergyOfIotDevices.append(states[0])

    # sumRewardOfEpisodes = np.append(sumRewardOfEpisodes, episode_reward)
    sumRewardOfEpisodes.append(sum(episode_reward) / 200)
    energyConsumption.append(sum(episode_energy) / 200)
    trainingTimeOfEpisode.append(sum(episode_trainingTime) / 200)
    trainingTimeOfAllTimesteps = np.append(trainingTimeOfAllTimesteps, episode_trainingTime)

    x.append(i)
    if i != 0 and i % 250 == 0:
        utils.draw_graph(title="Reward vs Episode",
                         xlabel="Episode",
                         ylabel="Reward",
                         figSizeX=10,
                         figSizeY=5,
                         x=x,
                         y=sumRewardOfEpisodes,
                         savePath="Graphs/PPO_0.7_0.3",
                         pictureName=f"PPO_Reward_episode{i}")

        utils.draw_graph(title="Avg Energy vs Episode",
                         xlabel="Episode",
                         ylabel="Average Energy",
                         figSizeX=10,
                         figSizeY=5,
                         x=x,
                         y=energyConsumption,
                         savePath="Graphs/PPO_0.7_0.3",
                         pictureName=f"PPO_Energy_episode{i}")

        utils.draw_graph(title="Avg TrainingTime vs Episode",
                         xlabel="Episode",
                         ylabel="TrainingTime",
                         figSizeX=10,
                         figSizeY=5,
                         x=x,
                         y=trainingTimeOfEpisode,
                         savePath="Graphs/PPO_0.7_0.3",
                         pictureName=f"PPO_TrainingTime_episode{i}")

        utils.draw_scatter(title="Energy vs TrainingTime",
                           xlabel="Energy",
                           ylabel="TrainingTime",
                           x=energyConsumption,
                           y=trainingTimeOfEpisode,
                           savePath="Graphs/PPO",
                           pictureName=f"PPO_Scatter{i}")

utils.draw_hist(title='Avg Energy of IoT Devices',
                x=AvgEnergyOfIotDevices,
                xlabel="Average Energy",
                savePath='Graphs/PPO_0.7_0.3',
                pictureName='PPO_AvgEnergy_hist')

utils.draw_hist(title='TrainingTime of IoT Devices',
                x=trainingTimeOfAllTimesteps,
                xlabel="TrainingTime",
                savePath='Graphs/PPO_0.7_0.3',
                pictureName='TF_TrainingTime_hist')

utils.draw_3dGraph(x=energyConsumption,
                   y=trainingTimeOfEpisode,
                   z=sumRewardOfEpisodes,
                   xlabel="Energy 0.7",
                   ylabel="Training Time 0.3",
                   zlabel="reward")

# Evaluate for 100 episodes
# sum_rewards = 0.0
# for i in range(1000):
#     states = environment.reset()
#     internals = agent.initial_internals()
#     terminal = False
#     episode_energy = list()
#     episode_reward = list()
#
#     while not terminal:
#         actions, internals = agent.act(
#             states=states, internals=internals,
#             independent=True, deterministic=True
#         )
#         states, terminal, reward = environment.execute(actions=actions)
#         sum_rewards += reward
#
#         episode_reward.append(reward)
#         episode_energy.append(sum(states[:len(iotDevices)]))
#
#     sumRewardOfEpisodes.append(sum(episode_reward) / 200)
#     energyConsumption.append(sum(episode_energy))
#
#     if i % 150 == 0:
#         plt.figure(figsize=(30, 10))
#         plt.plot(sumRewardOfEpisodes)
#         plt.title("Reward vs Episodes")
#         plt.show()
#
#         plt.figure(figsize=(30, 10))
#         plt.plot(energyConsumption)
#         plt.title("Energy vs Episodes")
#         plt.show()

# Close agent and environment
agent.close()
environment.close()
