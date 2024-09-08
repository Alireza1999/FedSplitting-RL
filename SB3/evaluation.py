import argparse
import sys

import utils
from SB3.environment.withBandwidth import CustomEnv
from config import ROOT_DIR

sys.path.append(f"{ROOT_DIR}")

arguments = {
    '-a': ['--agent', 'ddpg', '[String] Name of agent model for evaluation]'],
    '-e': ['--episode_num', 50, '[Integer] number of episode for evaluation'],
    '-l': ['--log', True, '[Boolean] save log or not']
}


def parse_argument(parser: argparse.ArgumentParser(), arg: dict):
    for op in arguments.keys():
        parser.add_argument(op, arguments.get(op)[0], help=arguments.get(op)[2], type=str,
                            default=arguments.get(op)[1])
    args = parser.parse_args()
    option = vars(args)
    return option


def evaluation(logger, env, agentType, modelPath, modelName):
    # logger = utils.createLog(fileName=f"SB3/Logs/{modelName}")

    modelPath = f"{ROOT_DIR}/{modelPath}/{modelName}"
    saveGraphPath = f"{ROOT_DIR}/SB3/Graphs/{modelName}"

    model = utils.loadAgent(env=env, agentType=agentType, agent_index=modelName)
    # logger.info("Evaluation Started")

    env.isEvaluation = True
    timestepReward = []
    episodeReward = []
    for i in range(200):
        observation = env.reset()[0]
        terminated = False
        while not terminated:
            # logger.info(f"---------------------------------")
            # logger.info(f"State: {observation}")
            actions, states = model.predict(observation=observation, deterministic=True)
            new_observations, rewards, terminated, infos, _ = env.step(actions)
            timestepReward.append(rewards)
            # logger.info(f"Action: {actions}")
            # logger.info(f"Reward: {rewards}")
        episodeReward.append(sum(timestepReward) / len(timestepReward))
        timestepReward = []

    # x = [i for i in range(len(episodeReward))]
    # plt.figure(figsize=(10, 5))  # Set the figure size
    # plt.plot(x, episodeReward)
    # plt.title("Evaluation Reward")
    # plt.xlabel("episode")
    # plt.ylabel("mean reward")
    #
    # plt.savefig(os.path.join(saveGraphPath, "Evaluation Reward"))
    # plt.close()
    #
    # saveInterval = 1
    # x = [i for i in range(int((self.total_time_step) / self.timestepNum))]

    # for i in range(len(env.episode_reward)):
    # if i % saveInterval == 0:0
    # meanReward = sum(env.episode_reward[i - saveInterval:i]) / saveInterval
    # meanRewardOfEnergy = sum(env.episode_energy_reward[i - saveInterval:i]) / saveInterval
    # meanRewardOfTT = sum(env.episode_tt_reward[i - saveInterval:i]) / saveInterval
    # meanTT = sum(env.episode_tt[i - saveInterval:i]) / saveInterval
    # meanEnergy = sum(env.episode_energy[i - saveInterval:i]) / saveInterval
    # meanClassicFLEnergy = sum(env.episode_classicFL_energy[i - saveInterval:i]) / saveInterval
    # meanClassicFLTT = sum(env.episode_classicFL_TT[i - saveInterval:i]) / saveInterval

    x = [i for i in range(len(env.episode_reward))]
    reward = (env.episode_reward)
    rewardOfEnergy = (env.episode_energy_reward)
    rewardOfTT = (env.episode_tt_reward)
    tt = (env.episode_tt)
    energy = (env.episode_energy)
    classicFLEnergy = (env.episode_classicFL_energy)
    classicFLTT = (env.episode_classicFL_TT)

    utils.saveGraphs(savePath=saveGraphPath, energy=energy, rewardOfEnergy=rewardOfEnergy,
                     rewardOfTrainingTime=rewardOfTT,
                     trainingTime=tt, reward=reward, x=x, allEnergy=env.episode_energy,
                     allTrainingTime=env.episode_tt, classicFL_trainingTime=classicFLTT,
                     classicFL_energy=classicFLEnergy, )

    import matplotlib.pyplot as plt
    import random
    import os

    remainingEnergy = env.episode_remainingEnergy
    for i in range(len(env.episode_remainingEnergy)):
        plt.figure(figsize=(int(15), int(5)))
        x = [i for i in range(len(remainingEnergy[i]) - 1)]
        for k in range(env.iotDeviceNum):
            iotDevice_K = []
            for j in range(1, len(remainingEnergy[i])):
                iotDevice_K.append(remainingEnergy[i][j][k])
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.plot(x, iotDevice_K, color=color, label=f"Device {k}")
        plt.legend()
        plt.title(f"Remaining Energy of iot device - episode {i}")
        plt.xlabel("timestep")
        plt.ylabel("remaining energy")
        plt.savefig(os.path.join(saveGraphPath, f"Remaining Energy - episode {i}"))
        plt.close()

    consumedEnergy = env.episode_consumedEnergy
    for i in range(len(env.episode_consumedEnergy)):
        plt.figure(figsize=(int(15), int(5)))
        x = [i for i in range(len(consumedEnergy[i]) - 1)]
        for k in range(env.iotDeviceNum):
            iotDevice_K = []
            for j in range(1, len(consumedEnergy[i])):
                iotDevice_K.append(consumedEnergy[i][j][k])
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.plot(x, iotDevice_K, color=color, label=f"Device {k}")
        plt.legend()
        plt.title(f"Consumed Energy of iot device - episode {i}")
        plt.xlabel("timestep")
        plt.ylabel("consumed energy")
        plt.savefig(os.path.join(saveGraphPath, f"Consumed Energy - episode {i}"))
        plt.close()


if __name__ == '__main__':
    iotDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/iotDevices.csv",
                                           deviceType='iotDevice')
    edgeDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/edges.csv", deviceType='edgeDevice')
    cloud = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/cloud.csv")[0]

    env = CustomEnv(iotDevices, edgeDevices, cloud, fraction=self.fraction, ep_length=self.episode_len)

    parser = argparse.ArgumentParser()
    options = parse_argument(parser=parser, arg=arguments)

    evaluation(logger=None, env=env, agentType=options['agent'], modelName=options[''])
