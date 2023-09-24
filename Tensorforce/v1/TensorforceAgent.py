import logging

import numpy as np
from tensorforce import Agent, Environment

from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append("/home/alireza_soleymani/UniversityWorks/Thesis/FedSplitting-RL/Tensorforce/v1")
sys.path.append("/home/alireza_soleymani/UniversityWorks/Thesis/FedSplitting-RL")
print(Path(__file__).parents[1])

from Tensorforce import config
from Tensorforce import utils
from Tensorforce.v1.customEnv import CustomEnvironment

logging.basicConfig(filename=f"{Path(__file__).parents[0]}/Logs/TF_agent/info.log",
                    format='%(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

iotDevices = utils.createDeviceFromCSV(csvFilePath=f"{Path(__file__).parents[0]}/../../System/iotDevicesSmallScale.csv", deviceType='iotDevice')
edgeDevices = utils.createDeviceFromCSV(csvFilePath=f"{Path(__file__).parents[0]}/../../System/edgesSmallScale.csv", deviceType='edge')
cloud = utils.createDeviceFromCSV(csvFilePath=f"{Path(__file__).parents[0]}/../../System/cloud.csv")[0]

env = CustomEnvironment(iotDevices=iotDevices,
                        edgeDevices=edgeDevices,
                        cloud=cloud)

environment = Environment.create(
    environment=env,
    max_episode_timesteps=200)

agent = Agent.create(
    agent='tensorforce', environment=environment,
    max_episode_timesteps=200,
    # Reward estimation
    reward_estimation=dict(
        horizon=1,
        discount=0.96),

    # Optimizer
    optimizer=dict(
        optimizer='adam', learning_rate=0.001, clipping_threshold=0.01,
        multi_step=10, subsampling_fraction=0.99
    ),

    # update network every 5 timestep
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
    summarizer=dict(directory="summaries/TF_Agent_newState/",
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
                         savePath="Graphs/TF_agent_newState_0.5_0.5",
                         pictureName=f"TF_Reward_episode{i}")

        utils.draw_graph(title="Avg Energy vs Episode",
                         xlabel="Episode",
                         ylabel="Average Energy",
                         figSizeX=10,
                         figSizeY=5,
                         x=x,
                         y=energyConsumption,
                         savePath="Graphs/TF_agent_newState_0.5_0.5",
                         pictureName=f"TF_Energy_episode{i}")

        utils.draw_graph(title="Avg TrainingTime vs Episode",
                         xlabel="Episode",
                         ylabel="TrainingTime",
                         figSizeX=10,
                         figSizeY=5,
                         x=x,
                         y=trainingTimeOfEpisode,
                         savePath="Graphs/TF_agent_newState_0.5_0.5",
                         pictureName=f"TF_TrainingTime_episode{i}")

        utils.draw_scatter(title="Energy vs TrainingTime",
                           xlabel="Energy",
                           ylabel="TrainingTime",
                           x=energyConsumption,
                           y=trainingTimeOfEpisode,
                           savePath="Graphs/TF_agent_newState_0.5_0.5",
                           pictureName=f"TF_Scatter{i}")

utils.draw_hist(title='Avg Energy of IoT Devices',
                x=AvgEnergyOfIotDevices,
                xlabel="Average Energy",
                savePath='Graphs/TF_agent_newState_0.5_0.5',
                pictureName='TF_AvgEnergy_hist')

utils.draw_hist(title='TrainingTime of IoT Devices',
                x=trainingTimeOfAllTimesteps,
                xlabel="TrainingTime",
                savePath='Graphs/TF_agent_newState_0.5_0.5',
                pictureName='TF_TrainingTime_hist')

utils.draw_3dGraph(x=energyConsumption,
                   y=trainingTimeOfEpisode,
                   z=sumRewardOfEpisodes,
                   xlabel="Energy 0.5",
                   ylabel="Training Time 0.5",
                   zlabel="reward")

# print("Evaluation started ...")
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
#         episode_energy.append(states[0])
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

# Initialize the runner
# runner = Runner(agent=agent, environment=environment, evaluation=False, max_episode_timesteps=200)
# runner.run(num_episodes=700)
# runner.close()

agent.close()
environment.close()
