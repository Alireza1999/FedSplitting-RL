import sys

import numpy as np

import utils
from SB3.environment.withBandwidth import CustomEnv
from config import ROOT_DIR

sys.path.append(f"{ROOT_DIR}")


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
        iotDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/iotDevices.csv",
                                               deviceType='iotDevice')
        edgeDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/edges.csv")
        cloud = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/cloud.csv")[0]

        env = CustomEnv(iotDevices, edgeDevices, cloud, fraction=self.fraction, ep_length=self.episode_len)
        self.env = env

        model = utils.createAgent(agentType=self.agentType, env=env, lr=self.lr, clip=self.clip, n_step=self.n_step,
                                  batch_size=self.batch_size)
        modelSummary = utils.createSummaryFromModel(model, lr=self.lr, fraction=self.fraction, agentType=self.agentType,
                                                    clip=self.clip, episodeNum=self.episodeNum,
                                                    timestep=self.timestepNum, batchSize=self.batch_size)
        isDuplicate, folderName = utils.checkSummaryAndSaveConfig(configPath=f"{ROOT_DIR}/SB3/Graphs/configList",
                                                                  summary=modelSummary)
        model.__setattr__('tensorboard_log', f'{ROOT_DIR}/SB3/TensorboardLog/{folderName}')

        if self.log:
            logger = utils.createLog(fileName=f"SB3/Logs/{folderName}")

        model.learn(total_timesteps=self.total_time_step)
        model.save(f"{ROOT_DIR}/SB3/models/{folderName}")

        self.evaluation(folderName=folderName, logger=logger, env=self.env)
        if isDuplicate:
            print("You have run an agent with this configuration before.")
            print(f"Pictures of new train version saved in folder: {folderName}")
        else:
            print(f"New Configuration added to configList.json with ID: {folderName}")
            print(f"Graphs was saved in folder: {folderName}")

    def evaluation(self, logger, env, folderName):
        from stable_baselines3.common.evaluation import evaluate_policy
        model = utils.loadAgent(env=self.env, agentType=self.agentType, agent_index=folderName)
        logger.info("Evaluation Started")

        for i in range(2000):
            logger.info(f"---------------------------------")
            observation = env.reset()[0]
            logger.info(f"State: {observation}")
            actions, states = model.predict(observation=observation, deterministic=True)
            new_observations, rewards, dones, infos, _ = env.step(actions)
            logger.info(f"Action: {actions}")
            logger.info(f"Reward: {rewards}")

        saveGraphPath = f"{ROOT_DIR}/SB3/Graphs/{folderName}"

        saveInterval = 200
        x = [i for i in range(int((self.total_time_step+2000) / saveInterval))]
        reward = []
        rewardOfEnergy = []
        classicFLEnergy = []
        classicFLTT = []
        rewardOfTT = []
        tt = []
        energy = []
        for i in range(len(env.episode_reward)):
            if i % saveInterval == 0:
                meanReward = sum(env.episode_reward[i - saveInterval:i]) / saveInterval
                meanRewardOfEnergy = sum(env.episode_energy_reward[i - saveInterval:i]) / saveInterval
                meanRewardOfTT = sum(env.episode_tt_reward[i - saveInterval:i]) / saveInterval
                meanTT = sum(env.episode_tt[i - saveInterval:i]) / saveInterval
                meanEnergy = sum(env.episode_energy[i - saveInterval:i]) / saveInterval
                meanClassicFLEnergy = sum(env.episode_classicFL_energy[i - saveInterval:i]) / saveInterval
                meanClassicFLTT = sum(env.episode_classicFL_TT[i - saveInterval:i]) / saveInterval

                reward.append(meanReward)
                rewardOfEnergy.append(meanRewardOfEnergy)
                rewardOfTT.append(meanRewardOfTT)
                tt.append(meanTT)
                energy.append(meanEnergy)
                classicFLEnergy.append(meanClassicFLEnergy)
                classicFLTT.append(meanClassicFLTT)

        utils.saveGraphs(savePath=saveGraphPath, energy=energy, rewardOfEnergy=rewardOfEnergy,
                         rewardOfTrainingTime=rewardOfTT,
                         trainingTime=tt, reward=reward, x=x, allEnergy=env.episode_energy,
                         allTrainingTime=env.episode_tt, classicFL_trainingTime=classicFLTT,
                         classicFL_energy=classicFLEnergy, )


        # mean_reward, std_reward = evaluate_policy(model, self.env, n_eval_episodes=100, deterministic=True)
        # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
