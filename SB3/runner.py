from stable_baselines3 import PPO, A2C, DDPG, DQN
from SB3.environment.withBandwidth import CustomEnv
from config import ROOT_DIR

import sys
import utils

sys.path.append(f"{ROOT_DIR}")


def createAgent(env, fraction, agentType='ppo'):
    if agentType == 'ppo':
        return PPO("MlpPolicy", env, learning_rate=utils.linear_schedule(0.0007), verbose=2, clip_range=0.5,
                   gamma=1.0, batch_size=100, n_steps=1000,
                   tensorboard_log=f'{ROOT_DIR}/SB3/TensorboardLog/{agentType}/{fraction}/',
                   device="auto")
    elif agentType == 'ac':
        return A2C("MlpPolicy", env, learning_rate=utils.linear_schedule(0.0007), verbose=2, gamma=1.0,
                   n_steps=1000, tensorboard_log=f'{ROOT_DIR}/SB3/TensorboardLog/{agentType}/{fraction}/',
                   device="auto")
    else:
        raise Exception('Invalid config select from [ppo, ac, tensorforce, random]')


def loadAgent(env, agentType='ppo', fraction=None):
    if fraction is None:
        return Exception("fraction required for loading agent!")
    if agentType == 'ppo':
        return PPO.load(f"{ROOT_DIR}/SB3/models/{agentType}_{fraction}", env=env)
    elif agentType == 'ac':
        return A2C.load(f"{ROOT_DIR}/SB3/models/{agentType}_{fraction}", env=env)
    else:
        raise Exception('Invalid config select from [ppo, ac, tensorforce, random]')


class Runner:
    def __init__(self, agentType='ppo', episodeNum=10000, timestepNum=1, fraction=1.0, summaries=True, log=True):
        self.agentType = agentType
        self.episodeNum = episodeNum
        self.timestepNum = timestepNum
        self.fraction = fraction
        self.summaries = summaries
        self.log = log
        self.total_time_step = self.episodeNum * self.timestepNum
        self.episode_len = timestepNum
        self.env = None

    def run(self):
        if self.log:
            logger = utils.createLog(fileName=f"SB3/Logs/SB3_{self.agentType}_{self.fraction}")

        iotDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/iotDevices.csv",
                                               deviceType='iotDevice')
        edgeDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/edges.csv")
        cloud = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/cloud.csv")[0]

        env = CustomEnv(iotDevices, edgeDevices, cloud, fraction=self.fraction, ep_length=self.episode_len)
        self.env = env
        model = createAgent(agentType=self.agentType, env=env, fraction=self.fraction)

        model.learn(total_timesteps=self.total_time_step)
        model.save(f"{ROOT_DIR}/SB3/models/{self.agentType}_{self.fraction}")

        saveGraphPath = f"{ROOT_DIR}/SB3/Graphs/bandwidth/{self.agentType}/{self.fraction}/"
        x = [i for i in range(int(self.total_time_step / 100))]
        reward = []
        tt = []
        energy = []
        print(len(env.episode_reward))
        for i in range(len(env.episode_reward)):
            if i % 100 == 0:
                meanReward = sum(env.episode_reward[i - 100:i]) / 100
                meanTT = sum(env.episode_tt[i - 100:i]) / 100
                meanEnergy = sum(env.episode_energy[i - 100:i]) / 100
                reward.append(meanReward)
                tt.append(meanTT)
                energy.append(meanEnergy)

        utils.draw_graph(title="Reward vs Episode",
                         xlabel="Episode",
                         ylabel="Reward",
                         figSizeX=10,
                         figSizeY=5,
                         x=x,
                         y=reward,
                         savePath=saveGraphPath,
                         pictureName=f"Reward_episode")

        utils.draw_graph(title="Avg Energy vs Episode",
                         xlabel="Episode",
                         ylabel="Average Energy",
                         figSizeX=10,
                         figSizeY=5,
                         x=x,
                         y=energy,
                         savePath=saveGraphPath,
                         pictureName=f"Energy_episode")

        utils.draw_graph(title="Avg TrainingTime vs Episode",
                         xlabel="Episode",
                         ylabel="TrainingTime",
                         figSizeX=10,
                         figSizeY=5,
                         x=x,
                         y=tt,
                         savePath=saveGraphPath,
                         pictureName=f"TrainingTime_episode")

        utils.draw_scatter(title="Energy vs TrainingTime",
                           xlabel="Energy",
                           ylabel="TrainingTime",
                           x=env.episode_energy,
                           y=env.episode_tt,
                           savePath=saveGraphPath,
                           pictureName=f"Scatter")

    def evaluation(self):
        from stable_baselines3.common.evaluation import evaluate_policy

        model = loadAgent(env=self.env, agentType=self.agentType, fraction=self.fraction)
        mean_reward, std_reward = evaluate_policy(model, self.env, n_eval_episodes=100, deterministic=True)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
