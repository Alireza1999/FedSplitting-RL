from tensorforce import Agent, Environment
from Tensorforce import utils
from Tensorforce import config
from Tensorforce.simpleEnergy.customEnv import CustomEnvironment
import logging
from matplotlib import pyplot as plt
import numpy as np

logging.basicConfig(filename="./Logs/PPO/info.log",
                    format='%(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

iotDevices = utils.createDeviceFromCSV(csvFilePath="../../System/iotDevicesSmallScale.csv", deviceType='iotDevice')
edgeDevices = utils.createDeviceFromCSV(csvFilePath="../../System/edgesSmallScale.csv", deviceType='edge')
cloud = utils.createDeviceFromCSV(csvFilePath="../../System/cloud.csv")[0]

environment = Environment.create(
    environment=CustomEnvironment(iotDevices=iotDevices,
                                  edgeDevices=edgeDevices,
                                  cloud=cloud),
    max_episode_timesteps=200
)

agent = Agent.create(
    # Agent and Env
    agent='ppo', environment=environment,

    # Automatically configured network
    network='auto',
    use_beta_distribution=True,

    # Optimization

    # Number of episodes per update batch
    batch_size=1,
    # Frequency of updates (default: batch_size).
    update_frequency=2,
    # Optimizer learning rate
    learning_rate=3.0e-3,
    # Fraction of batch timesteps to subsample (default: 0.33).
    subsampling_fraction=0.99,
    # Number of optimization steps
    optimization_steps=2,

    # Reward estimation
    likelihood_ratio_clipping=0.2, discount=0.96, estimate_terminal=False,

    # Critic
    critic_network='auto',
    critic_optimizer=dict(optimizer='adam', multi_step=2, learning_rate=1.0e-3),

    # Preprocessing
    preprocessing=None,

    # Exploration
    exploration=0.1, variable_noise=0.0,

    # Regularization
    l2_regularization=0.1, entropy_regularization=0.2,

    # TensorFlow etc
    name='agent',
    device=None,
    parallel_interactions=1,
    seed=None,
    execution=None,
    saver=None,
    summarizer=dict(directory="summaries/PPO/",
                    labels='all'),
    recorder=None
)
sumRewardOfEpisodes = list()
energyConsumption = list()

x = list()
j = 0
# Train for 100 episodes
for i in range(2501):
    episode_states = list()
    episode_internals = list()
    episode_actions = list()
    episode_terminal = list()
    episode_reward = list()
    episode_energy = list()

    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        j += 1
        episode_states.append(states)
        episode_internals.append(internals)
        actions, internals = agent.act(
            states=states, internals=internals, independent=True
        )
        episode_actions.append(actions)
        states, terminal, reward = environment.execute(actions=actions)
        episode_terminal.append(terminal)
        episode_reward.append(reward)
        episode_energy.append(states[0])
        x.append(j)

    sumRewardOfEpisodes = np.append(sumRewardOfEpisodes, episode_reward)
    energyConsumption = np.append(energyConsumption, episode_energy)

    # x.append(j)
    if i % 500 == 0:
        utils.draw_graph(title="Reward vs Episode",
                         xlabel="Episode",
                         ylabel="Reward",
                         figSizeX=15,
                         figSizeY=5,
                         x=x,
                         y=sumRewardOfEpisodes,
                         savePath="Graphs/PPO",
                         pictureName=f"PPO_Reward_episode{i}")

        utils.draw_graph(title="Avg Energy vs Episode",
                         xlabel="Episode",
                         ylabel="Average Energy",
                         figSizeX=15,
                         figSizeY=5,
                         x=x,
                         y=energyConsumption,
                         savePath="Graphs/PPO",
                         pictureName=f"PPO_Energy_episode{i}")

    agent.experience(
        states=episode_states, internals=internals,
        actions=episode_actions, terminal=episode_terminal,
        reward=episode_reward
    )
    agent.update()

plt.hist(energyConsumption, 10)
plt.show()

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
