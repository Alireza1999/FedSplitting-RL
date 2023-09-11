from tensorforce import Agent, Environment
from Tensorforce import utils
from Tensorforce import config
from Tensorforce.v0.customEnv_v0 import CustomEnvironment
import logging
from matplotlib import pyplot as plt
from tensorforce import Runner

logging.basicConfig(filename="./Logs/TF_agent/info.log",
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
print(avgEnergyWithoutSplitting)
environment = Environment.create(
    environment=CustomEnvironment(iotDevices=iotDevices,
                                  edgeDevices=edgeDevices, cloud=cloud,
                                  energyWithoutSplitting=avgEnergyWithoutSplitting),
)

agent = Agent.create(
    agent='tensorforce', environment=environment,
    max_episode_timesteps=10,
    # Reward estimation
    reward_estimation=dict(
        horizon=1,
        discount=0.99),

    # Optimizer
    optimizer=dict(
        optimizer='adam', learning_rate=0.001, clipping_threshold=0.01,
        multi_step=10, subsampling_fraction=1.0
    ),

    # update network every 5 timestep
    update=dict(
        unit='timesteps',
        batch_size=5,
    ),

    objective='policy_gradient',
    # Preprocessing
    preprocessing=None,

    # Exploration
    exploration=0.1, variable_noise=0.0,

    # Regularization
    l2_regularization=0.1, entropy_regularization=0.01,
    memory=200,
    # TensorFlow etc
    saver=dict(directory='model', filename='model'),
    summarizer=dict(directory='summaries'),
    recorder=None,

    config=dict(name='agent',
                device="CPU",
                parallel_interactions=1,
                seed=None,
                execution=None,
                )
)

# sumRewardOfEpisodes = list()
# energyConsumption = list()
# x = list()
# AvgEnergyOfIotDevices = list()

# Train for 100 episodes
# for i in range(50):
#     episode_states = list()
#     episode_internals = list()
#     episode_actions = list()
#     episode_terminal = list()
#     episode_reward = list()
#     episode_energy = list()
#     logger.info("Episode {} started ...\n".format(i))
#     logger.info("=========================================")
#
#     states = environment.reset()
#     internals = agent.initial_internals()
#     terminal = False
#     while not terminal:
#         logger.info("new time step ...\n".format(i))
#         logger.info("------------------------------------")
#         episode_states.append(states)
#         episode_internals.append(internals)
#         actions, internals = agent.act(
#             states=states, internals=internals, independent=True
#         )
#         episode_actions.append(actions)
#         states, terminal, reward = environment.execute(actions=actions)
#         episode_terminal.append(terminal)
#         episode_reward.append(reward)
#         episode_energy.append(sum(states[:len(iotDevices)]) / len(iotDevices))
#         AvgEnergyOfIotDevices.append(sum(states[:len(iotDevices)]) / len(iotDevices))
#
#     sumRewardOfEpisodes.append(sum(episode_reward) / 200)
#     energyConsumption.append(sum(episode_energy) / 200)
#
#     x.append(i)
#     if i % 500 == 0:
#         utils.draw_graph(title="Reward vs Episode",
#                          xlabel="Episode",
#                          ylabel="Reward",
#                          figSizeX=15,
#                          figSizeY=5,
#                          x=x,
#                          y=sumRewardOfEpisodes,
#                          savePath="Graphs/TF_agent",
#                          pictureName=f"TF_Reward_episode{i}")
#
#         utils.draw_graph(title="Avg Energy vs Episode",
#                          xlabel="Episode",
#                          ylabel="Average Energy",
#                          figSizeX=15,
#                          figSizeY=5,
#                          x=x,
#                          y=energyConsumption,
#                          savePath="Graphs/TF_agent",
#                          pictureName=f"TF_Energy_episode{i}")
#
#     agent.experience(
#         states=episode_states, internals=internals,
#         actions=episode_actions, terminal=episode_terminal,
#         reward=episode_reward
#     )
#     agent.update()

# plt.hist(AvgEnergyOfIotDevices, 10)
# plt.show()

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
# Initialize the runner
runner = Runner(agent=agent, environment=environment, max_episode_timesteps=10)

# Train for 200 episodes
runner.run(num_episodes=10)
runner.close()
agent.close()
environment.close()
