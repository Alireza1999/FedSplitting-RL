from tensorforce import Agent, Environment
from Tensorforce import utils
from Tensorforce import config
from Tensorforce.v0.customEnv_v0 import CustomEnvironment
import logging
from matplotlib import pyplot as plt

logging.basicConfig(filename="./Logs/info.log",
                    format='%(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

iotDevices = utils.createDeviceFromCSV(csvFilePath="../../System/iotDevices.csv", deviceType='iotDevice')
edgeDevices = utils.createDeviceFromCSV(csvFilePath="../../System/edges.csv", deviceType='edge')
cloud = utils.createDeviceFromCSV(csvFilePath="../../System/cloud.csv")[0]

avgEnergyWithoutSplitting = 0
for device in iotDevices:
    avgEnergyWithoutSplitting += device.energyConsumption([config.LAYER_NUM - 1, config.LAYER_NUM - 1])
avgEnergyWithoutSplitting /= len(iotDevices)

environment = Environment.create(
    environment=CustomEnvironment(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                                  energyWithoutSplitting=avgEnergyWithoutSplitting),
    max_episode_timesteps=200
)

agent = Agent.create(
    agent='ppo', environment=environment,
    # Automatically configured network
    network='auto',
    use_beta_distribution=True,
    # Optimization
    batch_size=10, update_frequency=2, learning_rate=1e-5, subsampling_fraction=0.2,
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
AvgEnergyOfIotDevices = list()

x = list()

# Train for 100 episodes
for i in range(2001):
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
        episode_states.append(states)
        episode_internals.append(internals)
        actions, internals = agent.act(
            states=states, internals=internals, independent=True
        )
        episode_actions.append(actions)
        states, terminal, reward = environment.execute(actions=actions)
        episode_terminal.append(terminal)
        episode_reward.append(reward)
        episode_energy.append(sum(states[:len(iotDevices)]) / len(iotDevices))
        AvgEnergyOfIotDevices.append(sum(states[:len(iotDevices)]) / len(iotDevices))

    sumRewardOfEpisodes.append(sum(episode_reward) / 200)
    energyConsumption.append(sum(episode_energy) / 200)

    x.append(i)
    if i % 500 == 0:
        utils.draw_graph(title="Reward vs Episode",
                         xlabel="Episode",
                         ylabel="Reward",
                         figSizeX=15,
                         figSizeY=5,
                         x=x,
                         y=sumRewardOfEpisodes,
                         savePath="Graphs/PPO_v0",
                         pictureName=f"PPO_Reward_episode{i}")

        utils.draw_graph(title="Avg Energy vs Episode",
                         xlabel="Episode",
                         ylabel="Average Energy",
                         figSizeX=15,
                         figSizeY=5,
                         x=x,
                         y=energyConsumption,
                         savePath="Graphs/PPO_v0",
                         pictureName=f"PPO_Energy_episode{i}")

    agent.experience(
        states=episode_states, internals=internals,
        actions=episode_actions, terminal=episode_terminal,
        reward=episode_reward
    )
    agent.update()

plt.hist(AvgEnergyOfIotDevices, 10)
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
print(agent.get_available_summaries())

# Close agent and environment
agent.close()
environment.close()