import copy

from stable_baselines3 import PPO, A2C, DDPG, DQN
from SB3.environment.withBandwidth import CustomEnv
from config import ROOT_DIR
import json

import sys
import utils

sys.path.append(f"{ROOT_DIR}")


def createSummaryFromModel(model, lr, fraction, agentType, clip) -> dict:
    model_dict = model.__dict__
    summary = dict()
    summary["policy_kwargs"] = model_dict["policy_kwargs"]
    summary["num_timesteps"] = model_dict["num_timesteps"]
    summary["_total_timesteps"] = model_dict["_total_timesteps"]
    summary["seed"] = model_dict["seed"]
    summary["learning_rate"] = lr
    summary["fraction"] = fraction
    summary["agentType"] = agentType
    summary["tensorboard_log"] = model_dict["tensorboard_log"]
    summary["use_sde"] = model_dict["use_sde"]
    summary["sde_sample_freq"] = model_dict["sde_sample_freq"]
    summary["_stats_window_size"] = model_dict["_stats_window_size"]
    summary["_n_updates"] = model_dict["_n_updates"]
    summary["n_steps"] = model_dict["n_steps"]
    summary["gamma"] = model_dict["gamma"]
    summary["gae_lambda"] = model_dict["gae_lambda"]
    summary["ent_coef"] = model_dict["ent_coef"]
    summary["vf_coef"] = model_dict["vf_coef"]
    summary["max_grad_norm"] = model_dict["max_grad_norm"]
    summary["batch_size"] = model_dict["batch_size"]
    summary["n_epochs"] = model_dict["n_epochs"]
    summary["clip_range"] = clip
    summary["normalize_advantage"] = model_dict["normalize_advantage"]
    summary["target_kl"] = model_dict["target_kl"]
    return summary


def checkSummaryAndSaveConfig(configPath: str, summary: dict):
    """ First, this function check that new summary is saved before or not
    If we ran model with this config before, so it does not save new picture
    but if it was new config we'll create new config record and we wll save picture """

    isDuplicate = False
    lastIndex = -1
    f = open(f"{configPath}.json")

    # returns JSON object as
    # a dictionary
    configList = json.load(f)

    # Iterate through each dictionary in the list
    for item in configList:
        isDuplicate = False
        # Iterate through each key in the dictionary
        for key in item:
            lastIndex += 1
            print(summary)
            print(item[key])
            if summary == item[key]:
                isDuplicate = True

    temp = dict()
    temp[f"{lastIndex}"] = summary
    if not isDuplicate:
        configList.append(temp)

    with open(f"{configPath}.json", 'w', encoding='utf-8') as f:
        json.dump(configList, f, ensure_ascii=False, indent=4)

    # Closing file
    f.close()


def createAgent(env, fraction, lr, clip, batch_size, n_step, agentType='ppo'):
    if agentType == 'ppo':
        return PPO("MlpPolicy", env, learning_rate=lr, verbose=2, clip_range=clip,
                   gamma=1.0, batch_size=batch_size, n_steps=n_step,
                   tensorboard_log=f'{ROOT_DIR}/SB3/TensorboardLog/{agentType}/{fraction}/',
                   device="auto")
    elif agentType == 'ac':
        return A2C("MlpPolicy", env, learning_rate=utils.linear_schedule(lr), verbose=2, gamma=1.0,
                   n_steps=n_step, tensorboard_log=f'{ROOT_DIR}/SB3/TensorboardLog/{agentType}/{fraction}/',
                   device="auto")
    else:
        raise Exception('Invalid config select from [ppo, ac, tensorforce, random]')


def loadAgent(env, fraction, lr, clip, batch_size, episodeNum, n_step, agentType='ppo'):
    loadPath = f"{ROOT_DIR}/SB3/models/agent_{agentType}_fraction_{fraction}_episode_{episodeNum}_batchSize_{batch_size}_lr_{lr}_clip_{clip}"
    if fraction is None:
        return Exception("fraction required for loading agent!")
    if agentType == 'ppo':
        return PPO.load(loadPath, env=env)
    elif agentType == 'ac':
        return A2C.load(loadPath, env=env)
    else:
        raise Exception('Invalid config select from [ppo, ac]')


class Runner:
    def __init__(self, agentType='ppo', episodeNum=10000, timestepNum=1, fraction=1.0, batch_size=100, lr=0.003,
                 n_step=1000, clip=0.3, summaries=True, log=True):
        self.agentType = agentType
        self.episodeNum = episodeNum
        self.timestepNum = timestepNum
        self.fraction = fraction

        self.batch_size = batch_size
        self.lr = lr
        self.n_step = n_step
        self.clip = clip

        self.summaries = summaries
        self.log = log
        self.total_time_step = self.episodeNum * self.timestepNum
        self.episode_len = timestepNum
        self.env = None

    def run(self):
        if self.log:
            logger = utils.createLog(
                fileName=f"SB3/Logs/SB3_agent_{self.agentType}_fraction_{self.fraction}_episode_{self.episodeNum}_batchSize_{self.batch_size}_lr_{self.lr}_clip_{self.clip}")

        iotDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/iotDevices.csv",
                                               deviceType='iotDevice')
        edgeDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/edges.csv")
        cloud = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/cloud.csv")[0]

        env = CustomEnv(iotDevices, edgeDevices, cloud, fraction=self.fraction, ep_length=self.episode_len)
        self.env = env
        model = createAgent(agentType=self.agentType, env=env, fraction=self.fraction, lr=self.lr, clip=self.clip,
                            n_step=self.n_step, batch_size=self.batch_size)

        model.learn(total_timesteps=self.total_time_step)

        model.save(
            f"{ROOT_DIR}/SB3/models/agent_{self.agentType}_fraction_{self.fraction}_episode_{self.episodeNum}_batchSize_{self.batch_size}_lr_{self.lr}_clip_{self.clip}")

        modelSummary = createSummaryFromModel(model, lr=self.lr, fraction=self.fraction, agentType=self.agentType,
                                              clip=self.clip)
        config = dict()
        config["1"] = modelSummary
        array = [{i: config[i]} for i in config]
        checkSummaryAndSaveConfig(f"{ROOT_DIR}/SB3/Graphs/configList", modelSummary)


        saveGraphPath = f"{ROOT_DIR}/SB3/Graphs/agent_{self.agentType}_fraction_{self.fraction}_episode_{self.episodeNum}_batchSize_{self.batch_size}_lr_{self.lr}_clip_{self.clip}/"
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
