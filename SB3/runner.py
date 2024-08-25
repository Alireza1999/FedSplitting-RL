import sys

import utils
from SB3.environment.withBandwidth import CustomEnv
from config import ROOT_DIR

sys.path.append(f"{ROOT_DIR}")


class Runner:
    def __init__(self, agentType='ddpg', episodeNum=10000, timestepNum=1, fraction=1.0, batch_size=100, lr=0.003,
                 n_step=1000, clip=0.3, summaries=True, log=True, justEval=False, modelName=None):
        self.agentType = agentType
        self.episodeNum = episodeNum
        self.timestepNum = timestepNum
        self.fraction = fraction
        self.justEval = justEval
        self.modelName = modelName
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
        edgeDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/edges.csv", deviceType='edgeDevice')
        cloud = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/cloud.csv")[0]

        env = CustomEnv(iotDevices, edgeDevices, cloud, fraction=self.fraction, ep_length=self.episode_len)
        self.env = env

        if self.justEval:
            if self.modelName is not None:
                folderName = self.modelName
                logger = utils.createLog(fileName=f"SB3/Logs/{folderName}_eval")
                self.evaluation(folderName=folderName, logger=logger, env=self.env, isEvaluation=True)
            else:
                print("You should specify a model name")
        else:
            model = utils.createAgent(agentType=self.agentType, env=env, lr=self.lr, clip=self.clip, n_step=self.n_step,
                                      batch_size=self.batch_size)
            modelSummary = utils.createSummaryFromModel(model, lr=self.lr, fraction=self.fraction,
                                                        agentType=self.agentType,
                                                        clip=self.clip, episodeNum=self.episodeNum,
                                                        timestep=self.timestepNum, batchSize=self.batch_size)
            isDuplicate, folderName = utils.checkSummaryAndSaveConfig(configPath=f"{ROOT_DIR}/SB3/Graphs/configList",
                                                                      summary=modelSummary)
            model.__setattr__('tensorboard_log', f'{ROOT_DIR}/SB3/TensorboardLog/{folderName}')

            if self.log:
                logger = utils.createLog(fileName=f"SB3/Logs/{folderName}")

            model.learn(total_timesteps=self.total_time_step, progress_bar=True)
            model.save(f"{ROOT_DIR}/SB3/models/{folderName}")

            self.evaluation(folderName=folderName, logger=logger, env=self.env)
            if isDuplicate:
                print("You have run an agent with this configuration before.")
                print(f"Pictures of new train version saved in folder: {folderName}")
            else:
                print(f"New Configuration added to configList.json with ID: {folderName}")
                print(f"Graphs was saved in folder: {folderName}")

    def evaluation(self, logger, env, folderName, isEvaluation=False):

        saveGraphPath = f"{ROOT_DIR}/SB3/Graphs/{folderName}/"
        saveGraphPathEval = f"{ROOT_DIR}/SB3/Graphs/{folderName}/evaluations/"

        model = utils.loadAgent(env=self.env, agentType=self.agentType, agent_index=folderName)
        logger.info("Evaluation Started")
        print(f"Evaluation Started...")
        env.isEvaluation = True
        timestepReward = []
        episodeReward = []
        episodeAction = []
        timestepAction = []
        for i in range(50):
            observation, _ = env.reset()
            terminated = False
            while not terminated:
                logger.info(f"---------------------------------")
                logger.info(f"State: {observation}")
                actions, states = model.predict(observation=observation, deterministic=True)
                timestepAction.append(actions)
                new_observations, rewards, terminated, infos, _ = env.step(actions)
                observation = new_observations
                timestepReward.append(rewards)
                logger.info(f"Action: {actions}")
                logger.info(f"Reward: {rewards}")
            episodeAction.append(timestepAction)
            timestepAction = []
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

        if not isEvaluation:
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

        remainingEnergy = env.episode_remainingEnergy[1:]
        remainingEnergyVariance = env.episode_remainingEnergyVariance[1:]
        iotBW = env.episode_effectiveBW[1:]
        assert len(iotBW) == len(remainingEnergy)

        for i in range(len(remainingEnergy)):
            x = [i for i in range(len(remainingEnergy[i]) - 1)]
            plt.figure(figsize=(int(25), int(5)))
            for k in range(env.iotDeviceNum):
                iotDevice_K = []
                for j in range(1, len(remainingEnergy[i])):
                    iotDevice_K.append(remainingEnergy[i][j][k])
                r = random.random()
                b = random.random()
                g = random.random()
                color = (r, g, b)
                plt.subplot(2, 1, 1)
                plt.title(f"Remaining Energy of iot device - episode {i}")
                plt.xlabel("timestep")
                plt.ylabel("remaining energy")
                plt.plot(x, iotDevice_K, color=color, linewidth='3', label=f"Device {k}")
                plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(x, remainingEnergyVariance[i][:-1], color='black', linewidth='3', label=f"Variance")
            plt.xlabel("timestep")
            plt.ylabel("remaining energy variance")
            if not os.path.exists(saveGraphPathEval):
                os.makedirs(saveGraphPathEval)
            plt.savefig(os.path.join(saveGraphPathEval, f"Remaining Energy - episode {i}"))
            plt.close()

        consumedEnergy = env.episode_consumedEnergy[1:]
        for i in range(len(consumedEnergy)):
            color = []
            plt.figure(figsize=(int(40), int(10)))
            x = [i for i in range(len(consumedEnergy[i]) - 1)]
            for k in range(env.iotDeviceNum):
                actionIndex = k * 2
                iotDevice_K = []
                for j in range(1, len(consumedEnergy[i])):
                    iotDevice_K.append(consumedEnergy[i][j][k])
                r = random.random()
                b = random.random()
                g = random.random()
                color.append((r, g, b))
                plt.subplot(3, 1, 1)
                plt.plot(x, iotDevice_K, color=color[k], marker='o', label=f"Device {k}")
                plt.legend()
                plt.ylabel("consumed energy")

                iotDevice_K_BW = []
                for j in range(1, len(iotBW[i])):
                    iotDevice_K_BW.append(iotBW[i][j][k])
                plt.subplot(3, 1, 2)
                plt.plot(x, iotDevice_K_BW, color=color[k], linewidth='2', marker='o', label=f"BW of Device {k}")
                plt.legend()
                plt.ylabel("Bandwidth")

                iotDevice_K_Action = []
                for j in range(1, len(episodeAction[i])):
                    op1, op2 = utils.actionToLayer(episodeAction[i][j][actionIndex:actionIndex + 2])
                    iotDevice_K_Action.append(op1)
                plt.subplot(3, 1, 3)
                plt.plot(x, iotDevice_K_Action, color=color[k], linewidth='2', marker='o',
                         label=f"Action of Device {k}")
                plt.legend()
                plt.ylabel("Action")
            plt.title(f"Consumed Energy of iot device - episode {i}")
            plt.xlabel("timestep")
            if not os.path.exists(saveGraphPathEval):
                os.makedirs(saveGraphPathEval)
            plt.savefig(os.path.join(saveGraphPathEval, f"Consumed Energy - episode {i}"))
            plt.close()

    # mean_reward, std_reward = evaluate_policy(model, self.env, n_eval_episodes=100, deterministic=True)
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
