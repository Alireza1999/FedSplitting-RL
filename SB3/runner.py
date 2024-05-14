import sys
import os
from pathlib import Path
ROOT_DIR = Path.cwd().parent
sys.path.append(f"{ROOT_DIR}")

from stable_baselines3 import A2C
import utils


agent = 'AC'
fractions = 1.0
total_time_step = 100000
episode_len = 100

logger = utils.createLog(fileName=f"SB3/Logs/SB3_{agent}_{fractions}")

from SB3.environment.withBandwidth import CustomEnv

iotDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/iotDevices.csv",
                                       deviceType='iotDevice')
edgeDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/edges.csv")
cloud = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/cloud.csv")[0]

FLEnergy, FLTrainingTime = utils.ClassicFLTrainingTime(iotDevices, edgeDevices, cloud)

rewardTuningParams = [FLEnergy, FLTrainingTime]
print(f"Energy of ClssicFL: {FLEnergy}")
print(f"TrainingIme of Clasic FL: {FLTrainingTime}")
env = CustomEnv(rewardTuningParams, iotDevices, edgeDevices, cloud, fraction=1.0, ep_length=episode_len)

model = A2C("MlpPolicy", env, verbose=1, device="auto", learning_rate=0.00007)
model.learn(total_timesteps=total_time_step)
model.save(f"{ROOT_DIR}/SB3/models/{agent}_{fractions}")

# model = A2C.load(f"{ROOT_DIR}/SB3/models/{agent}_{fractions}", env=env)
# print(model.get_parameters())

# Evaluate the trained agent
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
# print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
# x = [i for i in range(100)]


saveGraphPath = f"{ROOT_DIR}/SB3/Graphs/bandwidth/{agent}/{fractions}/"
x = [i for i in range(int(total_time_step / episode_len))]
utils.draw_graph(title="Reward vs Episode",
                 xlabel="Episode",
                 ylabel="Reward",
                 figSizeX=10,
                 figSizeY=5,
                 x=x,
                 y=env.episode_reward[1:],
                 savePath=saveGraphPath,
                 pictureName=f"Reward_episode")

utils.draw_graph(title="Avg Energy vs Episode",
                 xlabel="Episode",
                 ylabel="Average Energy",
                 figSizeX=10,
                 figSizeY=5,
                 x=x,
                 y=env.episode_energy[1:],
                 savePath=saveGraphPath,
                 pictureName=f"Energy_episode")

utils.draw_graph(title="Avg TrainingTime vs Episode",
                 xlabel="Episode",
                 ylabel="TrainingTime",
                 figSizeX=10,
                 figSizeY=5,
                 x=x,
                 y=env.episode_tt[1:],
                 savePath=saveGraphPath,
                 pictureName=f"TrainingTime_episode")

utils.draw_scatter(title="Energy vs TrainingTime",
                   xlabel="Energy",
                   ylabel="TrainingTime",
                   x=env.episode_energy[1:],
                   y=env.episode_tt[1:],
                   savePath=saveGraphPath,
                   pictureName=f"Scatter")
