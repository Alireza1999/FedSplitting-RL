from stable_baselines3 import PPO
from SB3.environment.withBandwidth import CustomEnv
from config import ROOT_DIR

import sys
import utils

sys.path.append(f"{ROOT_DIR}")

agent = 'PPO'
fractions = 0.5
total_time_step = 100000
episode_len = 1


def main():
    logger = utils.createLog(fileName=f"SB3/Logs/SB3_{agent}_{fractions}")

    iotDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/iotDevices.csv",
                                           deviceType='iotDevice')
    edgeDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/edges.csv")
    cloud = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/cloud.csv")[0]

    FLEnergy, FLTrainingTime = utils.ClassicFLTrainingTime(iotDevices, edgeDevices, cloud)

    rewardTuningParams = [FLEnergy, FLTrainingTime]
    print(f"Energy of ClssicFL: {FLEnergy}")
    print(f"TrainingIme of Clasic FL: {FLTrainingTime}")

    env = CustomEnv(rewardTuningParams, iotDevices, edgeDevices, cloud, fraction=fractions, ep_length=episode_len)

    model = PPO("MlpPolicy", env,
                learning_rate=utils.linear_schedule(0.003), verbose=2, clip_range=0.5, gamma=1.0,
                batch_size=2000, n_steps=1000, tensorboard_log=f'{ROOT_DIR}/SB3/TensorboardLog/{agent}/{fractions}/',
                device="auto")

    model.learn(total_timesteps=total_time_step)
    model.save(f"{ROOT_DIR}/SB3/models/{agent}_{fractions}")

    saveGraphPath = f"{ROOT_DIR}/SB3/Graphs/bandwidth/{agent}/{fractions}/"
    x = [i for i in range(int(total_time_step / 100))]
    reward = []
    tt = []
    energy = []
    for i in range(len(env.episode_reward)):
        if i % 100==0:
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


def evaluation():
    iotDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/iotDevices.csv",
                                           deviceType='iotDevice')
    edgeDevices = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/edges.csv")
    cloud = utils.createDeviceFromCSV(csvFilePath=f"{ROOT_DIR}/envs_stats/cloud.csv")[0]

    FLEnergy, FLTrainingTime = utils.ClassicFLTrainingTime(iotDevices, edgeDevices, cloud)

    rewardTuningParams = [FLEnergy, FLTrainingTime]
    print(f"Energy of ClssicFL: {FLEnergy}")
    print(f"TrainingIme of Clasic FL: {FLTrainingTime}")

    env = CustomEnv(rewardTuningParams, iotDevices, edgeDevices, cloud, fraction=fractions, ep_length=episode_len)

    model = PPO.load(f"{ROOT_DIR}/SB3/models/{agent}_{fractions}", env=env)
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


if __name__ == '__main__':
    main()
    evaluation()
