from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
from Tensorforce.v1.customEnv import CustomEnvironment
import utils
import logging

logging.basicConfig(filename="./Logs/info.log",
                    format='%(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

iotDevices = utils.createDeviceFromCSV(csvFilePath="../System/iotDevices.csv", deviceType='iotDevice')
edgeDevices = utils.createDeviceFromCSV(csvFilePath="../System/edges.csv", deviceType='edge')
cloud = utils.createDeviceFromCSV(csvFilePath="../System/cloud.csv")[0]

environment = Environment.create(
    environment=CustomEnvironment(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud),
    max_episode_timesteps=200)

agent = Agent.create(
    agent='ppo', environment=environment,
    # Automatically configured network
    network='auto',
    # Optimization
    batch_size=5, update_frequency=5, learning_rate=0.0001, subsampling_fraction=0.2,
    optimization_steps=5,
    # Reward estimation
    likelihood_ratio_clipping=0.3, discount=0.96, estimate_terminal=False,
    # Critic
    critic_network='auto',
    critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=0.0001),
    # Preprocessing
    preprocessing=None,
    # Exploration
    exploration=0.05, variable_noise=0.0,
    # Regularization
    l2_regularization=0.1, entropy_regularization=0.01,
    # TensorFlow etc
    name='agent', device=None, parallel_interactions=1, seed=1, execution=None, saver=None,
    summarizer=None, recorder=None
)

# agent = Agent.create(
#     agent='ac', environment=environment,
#     # Automatically configured network
#     network='auto',
#     # Optimization
#     batch_size=5, update_frequency=5, learning_rate=0.0001,
#     # Reward estimation
#     discount=0.99, estimate_terminal=False,
#     # Critic
#     horizon=1,
#     critic_network='auto',
#     critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=0.0001),
#     # Preprocessing
#     preprocessing=None,
#     # Exploration
#     exploration=0.05, variable_noise=0.0,
#     # Regularization
#     l2_regularization=0.1, entropy_regularization=0.01,
#     # TensorFlow etc
#     name='agent', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
#     summarizer=None, recorder=None
# )

rewards = []
energyConsumption = []


def run(environment, agent, n_episodes, max_step_per_episode, test=False):
    environment.max_step_per_episode = max_step_per_episode
    # Loop over episodes
    for i in range(n_episodes):
        # Initialize episode
        logger.info("\n Episode {} started ... \n ====================================".format(i))

        sum_reward = 0
        energyOfOneEpisode = 0
        episode_length = 0
        states = environment.reset()

        logger.info("Initial state : \n {}".format(states))

        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            # Run episode

            logger.info("time step {} in Episode {} started : \n".format(episode_length, i))
            actions = agent.act(states=states, independent=False)
            logger.info("Action in step {} in episode {} : \n {} \n".format(episode_length, i, actions))

            episode_length += 1
            states, terminal, reward = environment.execute(actions=actions)
            logger.info("Reward in step {} in episode {} :\n {} \n".format(episode_length, i, reward))
            agent.observe(terminal=terminal, reward=reward)
            sum_reward += reward
            print(states)
            energyOfOneEpisode += sum(states[:len(iotDevices)])

        logger.info("Average Reward of episode {} : \n {}\n".format(i, sum_reward / episode_length))
        rewards.append(sum_reward / episode_length)
        energyConsumption.append(energyOfOneEpisode / episode_length)

    plt.figure(figsize=(30, 10))
    plt.plot(energyConsumption)
    plt.title("Energy Consumption")
    plt.show()

    plt.figure(figsize=(30, 10))
    plt.plot(rewards)
    plt.title("Episodes rewards")
    plt.show()


def runner(environment, agent, max_step_per_episode, n_episodes, n_episodes_test=1, combination="reward"):
    # Train agent
    result_vec = []  # initialize the result list
    for i in range(round(n_episodes / 100)):  # Divide the number of episodes into batches of 100 episodes
        if result_vec:
            print("batch", i, "Best result", result_vec[-1])  # Show the results for the current batch
        # Train Agent for 100 episode
        run(environment, agent, 100, max_step_per_episode)
        # Test Agent for this batch
        # test_results = run(
        #     environment,
        #     agent,
        #     n_episodes_test,
        #     max_step_per_episode,
        #     test=True
        # )
        # Append the results for this batch
    # result_vec.append(test_results)
    # Plot the evolution of the agent over the batches
    utils.draw_graph(
        Series=[rewards],
        labels=["Reward"],
        xlabel="episodes",
        ylabel="Reward",
        title="Reward vs episodes",
        save_fig=True,
        path="Graphs",
        folder=str("reward"),
        time=False,
    )
    # Terminate the agent and the environment
    agent.close()
    environment.close()


# Call runner
runner(environment, agent, max_step_per_episode=200, n_episodes=1000, combination="reward")
