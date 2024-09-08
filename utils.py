import csv
import json
import logging
import os
import random
from collections import deque
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from stable_baselines3 import PPO, A2C, DDPG, SAC

import config
from entities.Device_bandwidthState import Device as Device


# from entities.Device import Device

def round_robin_scheduling(clientInfo, time_slice: float = 0.01):
    clientInfo = sorted(clientInfo, key=lambda x: x['start_time'])
    # print("Client info at RR function util", clientInfo)

    remaining_durations = dict()
    waiting_times = dict()
    turnaround_times = dict()
    start_times = dict()
    end_times = dict()
    arrival_times = dict()

    isOffloaded = list()
    for i in range(len(clientInfo)):
        if clientInfo[i]["duration"] != 0:
            isOffloaded.append(clientInfo[i])
            remaining_durations[f"{clientInfo[i]['name']}"] = clientInfo[i]["duration"]
            waiting_times[f"{clientInfo[i]['name']}"] = 0
            turnaround_times[f"{clientInfo[i]['name']}"] = 0
            start_times[f"{clientInfo[i]['name']}"] = None
            end_times[f"{clientInfo[i]['name']}"] = 0
            arrival_times[f"{clientInfo[i]['name']}"] = clientInfo[i]["start_time"]
        else:
            turnaround_times[f"{clientInfo[i]['name']}"] = 0

    clientInfo = isOffloaded
    # print(isOffloaded)
    # Queue to manage the round-robin scheduling
    rr_queue = deque()

    current_time = 0.00
    total_execution_time = 0.00
    conflict_time = 0.00

    # Keep track of active processes
    active_processes = set()

    while remaining_durations or rr_queue or active_processes:
        # print(current_time)
        # print(remaining_durations)

        # Add processes to the round-robin queue as they start
        for process in clientInfo:
            if process["start_time"] == current_time and process["name"] not in rr_queue:
                rr_queue.append(process["name"])

        # Process the round-robin queue
        if rr_queue:
            name = rr_queue.popleft()
            if start_times[name] is None:
                start_times[name] = current_time
            actual_time_slice = round(min(time_slice, remaining_durations[name]), 2)
            # actual_time_slice = time_slice

            # Update waiting times for all processes in the queue
            for process_name in rr_queue:
                waiting_times[process_name] = round(waiting_times[process_name] + actual_time_slice, 2)

                # Check for conflicts and update conflict time
            if active_processes:
                conflict_time = round(conflict_time + actual_time_slice * len(active_processes), 2)

            # Update remaining duration
            remaining_durations[name] -= actual_time_slice
            remaining_durations[name] = round(remaining_durations[name], 2)

            current_time = round(current_time + actual_time_slice, 2)

            # If the process finishes, calculate its turnaround time and end time
            if remaining_durations[name] <= 0.00:
                turnaround_times[name] = round(current_time - arrival_times[name], 2)
                end_times[name] = current_time
                del remaining_durations[name]
                active_processes.remove(name)
            else:
                active_processes.add(name)
                rr_queue.append(name)

        else:
            # If the round-robin queue is empty, jump to the next start time
            if remaining_durations:
                next_start_time = min(arrival_times[name] for name in remaining_durations.keys())
                current_time = next_start_time
                continue

        total_execution_time = current_time

    return total_execution_time, conflict_time, waiting_times, turnaround_times, start_times, end_times


def saveGraphs(savePath, energy, trainingTime, reward, rewardOfEnergy, rewardOfTrainingTime, rewardOfRemainingEnergy, x,
               allEnergy,
               allTrainingTime, classicFL_energy, classicFL_trainingTime):
    draw_graph(title="Energy vs Episode",
               xlabel="Episode",
               ylabel="Energy",
               figSizeX=25,
               figSizeY=5,
               x=x,
               y=energy,
               y_2=classicFL_energy,
               y_1_label="Our Method",
               y_2_label="Classic FL",
               savePath=savePath,
               pictureName=f"energy_episode")

    draw_graph(title="Training Time vs Episode",
               xlabel="Episode",
               ylabel="Training Time",
               figSizeX=25,
               figSizeY=5,
               x=x,
               y=trainingTime,
               y_2=classicFL_trainingTime,
               y_1_label="Our Method",
               y_2_label="Classic FL",
               savePath=savePath,
               pictureName=f"trainingTime_episode")

    # draw_graph(title="Training Time vs Episode",
    #            xlabel="Episode",
    #            ylabel="Training Time",
    #            figSizeX=10,
    #            figSizeY=5,
    #            x=x,
    #            y=trainingTime,
    #            y_2=classicFL_trainingTime,
    #            y_1_label="Our Method",
    #            y_2_lable="Classic FL",
    #            savePath=savePath,
    #            pictureName=f"trainingTime_episode")

    draw_graph(title="Reward vs Episode",
               xlabel="Episode",
               ylabel="Reward",
               figSizeX=25,
               figSizeY=5,
               x=x,
               y=reward,
               savePath=savePath,
               pictureName=f"reward_episode")

    # draw_graph(title="Avg Energy vs Episode",
    #            xlabel="Episode",
    #            ylabel="Average Energy",
    #            figSizeX=10,
    #            figSizeY=5,
    #            x=x,
    #            y=energy,
    #            savePath=savePath,
    #            pictureName=f"energy_episode")

    # draw_graph(title="Avg TrainingTime vs Episode",
    #            xlabel="Episode",
    #            ylabel="TrainingTime",
    #            figSizeX=10,
    #            figSizeY=5,
    #            x=x,
    #            y=trainingTime,
    #            savePath=savePath,
    #            pictureName=f"training_time_episode")

    draw_scatter(title="Energy vs TrainingTime",
                 xlabel="Energy",
                 ylabel="TrainingTime",
                 x=allEnergy,
                 y=allTrainingTime,
                 savePath=savePath,
                 pictureName=f"energy_training_time_Scatter")

    plt.figure(figsize=(int(25), int(5)))
    plt.plot(x, rewardOfEnergy, color='red', label='Energy reward')
    plt.plot(x, rewardOfTrainingTime, color='green', label='Training time reward')
    plt.plot(x, rewardOfRemainingEnergy, color='yellow', label='Remaining energy reward')
    plt.plot(x, reward, color='blue', label='Total Reward')
    plt.legend()
    plt.title("All Reward Graphs")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig(os.path.join(savePath, f"rewards_fraction"))
    plt.close()


def createSummaryFromModel(model, lr, fraction, agentType, clip, episodeNum, timestep, batchSize) -> dict:
    model_dict = model.__dict__
    summary = dict()
    print(model_dict)
    summary["learning_rate"] = lr
    summary["fraction"] = fraction
    summary["agentType"] = agentType
    summary["episode_num"] = episodeNum
    summary["num_timesteps"] = timestep
    summary["total_timesteps"] = timestep * episodeNum
    summary["clip_range"] = clip
    summary["batch_size"] = batchSize

    summary["policy_kwargs"] = model_dict["policy_kwargs"]
    summary["seed"] = model_dict["seed"]
    summary["tensorboard_log"] = model_dict["tensorboard_log"]
    summary["use_sde"] = model_dict["use_sde"]
    summary["sde_sample_freq"] = model_dict["sde_sample_freq"]
    summary["_stats_window_size"] = model_dict["_stats_window_size"]
    summary["_n_updates"] = model_dict["_n_updates"]
    summary["n_steps"] = model_dict["n_steps"] if agentType == "ppo" else None
    summary["gamma"] = model_dict["gamma"]
    summary["gae_lambda"] = model_dict["gae_lambda"] if agentType == "ppo" else None
    summary["ent_coef"] = model_dict["ent_coef"] if agentType == "ppo" else None
    summary["vf_coef"] = model_dict["vf_coef"] if agentType == "ppo" else None
    summary["max_grad_norm"] = model_dict["max_grad_norm"] if agentType == "ppo" else None
    summary["n_epochs"] = model_dict["n_epochs"] if agentType == "ppo" else None
    summary["normalize_advantage"] = model_dict["normalize_advantage"] if agentType == "ppo" else None
    summary["target_kl"] = model_dict["target_kl"] if agentType == "ppo" else None
    return summary


def checkSummaryAndSaveConfig(configPath: str, summary: dict):
    """ First, this function check that new summary has been saved before or not
    If we ran model with this config before, so it does not save new config
    but if it was new config we'll create new config record and we wll save picture """

    isDuplicate = False
    duplicateIndex = 0
    lastIndex = 0
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
            if summary == item[key]:
                isDuplicate = True
                duplicateIndex = lastIndex

    temp = dict()
    lastIndex += 1
    temp[f"{lastIndex}"] = summary

    if not isDuplicate:
        configList.append(temp)
        folderName = lastIndex
    else:
        folderName = duplicateIndex

    with open(f"{configPath}.json", 'w', encoding='utf-8') as f:
        json.dump(configList, f, ensure_ascii=False, indent=4)

    # Closing file
    f.close()
    return isDuplicate, folderName


def createAgent(env, lr, clip, batch_size, n_step, agentType='ppo'):
    if agentType == 'ppo':
        return PPO("MlpPolicy", env, learning_rate=lr, verbose=1, clip_range=clip,
                   gamma=1.0, batch_size=batch_size, n_steps=n_step, device="mps", ent_coef=0.1)
    elif agentType == 'ddpg':
        mean = np.array([0.3])
        sigma = np.array([0.2])
        return DDPG("MlpPolicy", env, learning_rate=lr, verbose=2, batch_size=batch_size, device="mps",
                    buffer_size=10_000_000, gamma=1.0, seed=1234)
    elif agentType == 'sac':
        return SAC("MlpPolicy", env, learning_rate=linear_schedule(lr), verbose=2, batch_size=batch_size, device="mps",
                   buffer_size=1_000_000, gamma=1.0, ent_coef='auto_0.1', use_sde=True)
    elif agentType == 'ac':
        return A2C("MlpPolicy", env, learning_rate=linear_schedule(lr), verbose=2, gamma=1.0,
                   n_steps=n_step, device="auto")
    else:
        raise Exception('Invalid config select from [ppo, ac, tensorforce, random]')


def loadAgent(agentType, agent_index, env=None):
    loadPath = f"{config.ROOT_DIR}/SB3/models/{agent_index}"
    if agentType == 'ppo':
        return PPO.load(loadPath, env=env)
    elif agentType == 'ddpg':
        return DDPG.load(loadPath, env=env, print_system_info=True)
    elif agentType == 'sac':
        return SAC.load(loadPath, env=env, print_system_info=True)
    elif agentType == 'ac':
        return A2C.load(loadPath, env=env)
    else:
        raise Exception('Invalid config select from [ppo, ac, ddpg]')


def createDeviceFromCSV(csvFilePath: str, deviceType: str = 'cloud') -> list:
    devices = list()
    with open(csvFilePath, 'r') as device:
        csvreader = csv.reader(device)
        for row in csvreader:
            if row[0] == 'FLOPS':
                continue
            if deviceType == 'iotDevice':
                device = Device(deviceType=deviceType, FLOPS=int(row[0]), bandwidth=float(row[1]),
                                edgeIndex=int(row[2]), maxPower=float(row[3]), remainingEnergy=float(row[4]))
            elif deviceType == 'edgeDevice':
                device = Device(deviceType=deviceType, FLOPS=int(row[0]), bandwidth=float(row[1]),
                                maxPower=float(row[2]))
            else:
                device = Device(deviceType=deviceType, FLOPS=int(row[0]), bandwidth=float(row[1]),
                                maxPower=float(row[2]))
            devices.append(device)
    return devices


def draw_graph(figSizeX, figSizeY, x, y, title, xlabel, ylabel, savePath, pictureName, y_1_label=None, y_2_label=None,
               y_2=None,
               saveFig=True):
    # Create a plot

    if y_2 is not None:
        plt.figure(figsize=(int(figSizeX), int(figSizeY)))
        plt.plot(x, y, color='red', label=y_1_label)
        plt.plot(x, y_2, color='blue', label=y_2_label)
        plt.legend()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if saveFig:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig(os.path.join(savePath, pictureName))
        plt.close()
    else:
        plt.figure(figsize=(int(figSizeX), int(figSizeY)))  # Set the figure size
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if saveFig:
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig(os.path.join(savePath, pictureName))
        plt.close()


def draw_hist(x, title, xlabel, savePath, pictureName, saveFig=True):
    # Create a plot
    plt.hist(x, 10)
    plt.title(title)
    plt.xlabel(xlabel)
    if saveFig:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, pictureName))
    plt.close()
    # plt.show()


def draw_scatter(x, y, title, xlabel, ylabel, savePath, pictureName, saveFig=True):
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if saveFig:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, pictureName))
    plt.close()
    # plt.show()


def draw_3dGraph(x, y, z, xlabel, ylabel, zlabel):
    fig = go.Figure(data=[go.Mesh3d(x=x,
                                    y=y,
                                    z=z,
                                    opacity=0.7, )])

    fig.update_layout(scene=dict(xaxis_title=xlabel,
                                 yaxis_title=ylabel,
                                 zaxis_title=zlabel,
                                 xaxis_showspikes=False,
                                 yaxis_showspikes=False))

    fig.show()


def actionToLayer(splitDecision: list) -> tuple:
    """ It returns the offloading points for the given action ( op1 , op2 ), split decision can be between -1 to 1"""
    if splitDecision[0] >= 0.96:
        return 6, 6
    else:
        op1: float
        op2: float  # Offloading points op1, op2
        workLoad = []
        model_state_flops = []

        for l in config.COMP_WORK_LOAD:
            workLoad.append(l)
            model_state_flops.append(sum(workLoad))

        totalWorkLoad = sum(workLoad)
        model_flops_list = np.array(model_state_flops)
        model_flops_list = (model_flops_list / totalWorkLoad)

        idx = np.where(np.abs(model_flops_list - splitDecision[0]) == np.abs(model_flops_list - splitDecision[0]).min())
        op1 = int(idx[0][-1])

        op2_totalWorkload = sum(workLoad[op1:])
        model_state_flops = []
        for l in range(op1, config.LAYER_NUM):
            model_state_flops.append(sum(workLoad[op1:l + 1]))
        model_flops_list = np.array(model_state_flops)
        model_flops_list = model_flops_list / op2_totalWorkload

        idx = np.where(np.abs(model_flops_list - splitDecision[1]) == np.abs(model_flops_list - splitDecision[1]).min())
        op2 = int(idx[0][-1]) + op1

        return op1, op2


print(actionToLayer([0.0, 0.0]))


# def actionToLayer(splitDecision: list[float]) -> tuple[int, int]:
#     """ It returns the offloading points for the given action ( op1 , op2 )"""
#
#     totalWorkLoad = sum(config.COMP_WORK_LOAD[1:])
#
#     op1: int
#     op2: int = 0  # Offloading points op1, op2
#
#     op1_workload = splitDecision[0] * totalWorkLoad
#     print(f"op1 WL: {op1_workload}")
#     for i in range(0, config.LAYER_NUM):
#         difference = abs(sum(config.COMP_WORK_LOAD[:i + 1]) - op1_workload)
#         if i < 6:
#             temp2 = abs(sum(config.COMP_WORK_LOAD[:i + 2]) - op1_workload)
#         else:
#             temp2 = abs(sum(config.COMP_WORK_LOAD) - op1_workload)
#         print()
#         print(f"i: {i}")
#         print(f"def : {difference}")
#         print(f"temp: {temp2}")
#         if temp2 > difference:
#             op1 = i
#             break
#
#     if splitDecision[1] != -1:
#         remindedWorkLoad = sum(config.COMP_WORK_LOAD[op1 + 1:]) * splitDecision[1]
#
#         for i in range(op1, len(config.COMP_WORK_LOAD)):
#             difference = abs(sum(config.COMP_WORK_LOAD[op1 + 1:i + 1]) - remindedWorkLoad)
#             temp2 = abs(sum(config.COMP_WORK_LOAD[op1 + 1:i + 2]) - remindedWorkLoad)
#             if temp2 >= difference:
#                 op2 = i
#                 break
#         if op2 == 0:
#             op2 = op2 + 1
#         if op1 == config.LAYER_NUM - 1:
#             op2 = config.LAYER_NUM - 1
#
#     return op1, op2


def sigmoidActivation(x: float) -> float:
    """ It returns 1/(1+exp(-x)). where the values lies between zero and one """

    return 1 / (1 + np.exp(-x))


def tanhActivation(x: float) -> float:
    """ It returns the value (1-exp(-2x))/(1+exp(-2x)) and the value returned will be lies in between -1 to 1."""

    return np.tanh(x)


def normalizeReward(maxAmount, minAmount, x, minNormalized, maxNormalized):
    P = [maxAmount, minNormalized]
    Q = [minAmount, maxNormalized]
    lineGradient = (P[1] - Q[1]) / (P[0] - Q[0])
    y = lineGradient * (x - Q[0]) + Q[1]
    return y


def normalizeReward_tan(x, turning_point):
    y = max(min(-pow(x - turning_point, 3) / pow(turning_point, 3), 1), -1)
    return y


def convert_To_Len_th_base(n, arr, modelLen, deviceNumber, allPossible):
    a: str = ""
    for i in range(deviceNumber * 2):
        a += str(arr[n % modelLen])
        n //= modelLen
    allPossible.append(a)


def randomSelectionSplitting(modelLen, deviceNumber) -> list:
    splittingForOneDevice = []
    for i in range(0, modelLen):
        for j in range(0, i + 1):
            splittingForOneDevice.appendq([j, i])

    result = []
    for i in range(deviceNumber):
        rand = random.randint(0, len(splittingForOneDevice) - 1)
        result.append(splittingForOneDevice[rand])
    return result


def allPossibleSplitting(modelLen, deviceNumber):
    arr = [i for i in range(0, modelLen + 1)]
    allPossible = list()
    for i in range(pow(modelLen, deviceNumber * 2)):
        # Convert i to Len th base
        convert_To_Len_th_base(i, arr, modelLen, deviceNumber, allPossible)
    result = list()
    for item in allPossible:
        isOk = True
        for j in range(0, len(item) - 1, 2):
            if int(item[j]) > int(item[j + 1]):
                isOk = False
        if isOk:
            result.append(item)
    return result


def ClassicFLTrainingTimeWithoutEdge(iotDevices, cloud):
    allTrainingTime = []
    maxTrainingTime = 0
    action = [config.LAYER_NUM - 1] * 1
    cloud.connectedDevice = 0

    iotRemainingFLOP = [iot.FLOPS for iot in iotDevices]
    cloudRemainingFLOP = cloud.FLOPS

    for i in range(0, len(iotDevices)):
        op = action[0]
        cloudRemainingFLOP -= sum(config.COMP_WORK_LOAD[op + 1:])
        iotRemainingFLOP[int(i)] -= sum(config.COMP_WORK_LOAD[0:op + 1])
        if sum(config.COMP_WORK_LOAD[op + 1:]) != 0:
            cloud.connectedDevice += 1

    for i in range(0, len(iotDevices)):
        # Mapping float number to Offloading points
        op = action[0]
        # computing training time of this action
        iotTrainingTime = iotDevices[int(i)].trainingTime(splitPoints=[op, op],
                                                          remainingFlops=iotRemainingFLOP[int(i)],
                                                          preTrain=True)
        cloudTrainingTime = cloud.trainingTime([op, op],
                                               remainingFlops=cloudRemainingFLOP,
                                               preTrain=True)

        totalTrainingTime = iotTrainingTime + cloudTrainingTime
        allTrainingTime.append(totalTrainingTime)

        if totalTrainingTime > maxTrainingTime:
            maxTrainingTime = totalTrainingTime
    return allTrainingTime


def ClassicFLTrainingTime(iotDevices, edgeDevices, cloud):
    offloadingPointsList = []
    allTrainingTime = []
    maxTrainingTime = 0
    totalEnergyConsumption = 0
    total_comp_e = 0
    total_comm_e = 0

    action = [[config.LAYER_NUM - 1, config.LAYER_NUM - 1]] * len(iotDevices)
    for i in range(len(action)):
        edgeDevices[iotDevices[i].edgeIndex].connectedDevice = 0
        cloud.connectedDevice = 0

    iotRemainingFLOP = [iot.FLOPS for iot in iotDevices]
    edgeRemainingFLOP = [edge.FLOPS for edge in edgeDevices]
    cloudRemainingFLOP = cloud.FLOPS

    for i in range(len(action)):
        op1 = action[i][0]
        op2 = action[i][1]
        cloudRemainingFLOP -= sum(config.COMP_WORK_LOAD[op2 + 1:])
        edgeRemainingFLOP[iotDevices[i].edgeIndex] -= sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1])
        iotRemainingFLOP[i] -= sum(config.COMP_WORK_LOAD[0:op1 + 1])

        if sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1]):
            edgeDevices[iotDevices[i].edgeIndex].connectedDevice += 1
        if sum(config.COMP_WORK_LOAD[op2 + 1:]) != 0:
            cloud.connectedDevice += 1

    for i in range(len(action)):
        # Mapping float number to Offloading points
        op1 = action[i][0]
        op2 = action[i][1]
        offloadingPointsList.append(op1)
        offloadingPointsList.append(op2)

        # computing training time of this action
        iot_comp_e, iot_comm_e, iot_comp_tt, iot_comm_tt = iotDevices[i].energy_tt(splitPoints=[op1, op2],
                                                                                   remainingFlops=iotRemainingFLOP[i])
        _, _, edge_comp_tt, edge_comm_tt = edgeDevices[iotDevices[i].edgeIndex] \
            .energy_tt(splitPoints=[op1, op2],
                       remainingFlops=edgeRemainingFLOP[iotDevices[i].edgeIndex])
        _, _, cloud_comp_tt, cloud_comm_tt = cloud.energy_tt([op1, op2], remainingFlops=cloudRemainingFLOP)

        totalTrainingTime = (iot_comm_tt + iot_comp_tt) + (edge_comm_tt + edge_comp_tt) + (
                cloud_comm_tt + cloud_comp_tt)
        allTrainingTime.append(totalTrainingTime)

        if totalTrainingTime > maxTrainingTime:
            maxTrainingTime = totalTrainingTime

        # computing energy consumption of iot devices
        total_comp_e += iot_comp_e
        total_comm_e += iot_comm_e

    totalEnergyConsumption = (total_comm_e + total_comp_e)
    avgCommE = total_comm_e / len(iotDevices)
    avgCompE = total_comp_e / len(iotDevices)
    print(f"Computation E: {total_comp_e}")
    print(f"Communication E: {total_comm_e}")
    averageEnergyConsumption = totalEnergyConsumption / len(iotDevices)
    return averageEnergyConsumption, maxTrainingTime


def minMaxAvgEnergy(iotDevices, edgeDevices, cloud):
    splittingLayer = allPossibleSplitting(modelLen=config.LAYER_NUM, deviceNumber=1)
    maxAvgEnergyOfOneDevice = 0
    minAvgEnergyOfOneDevice = 1.0e7
    maxEnergySplitting = []
    minEnergySplitting = []

    for splitting in splittingLayer:
        splittingArray = list()
        for char in splitting:
            splittingArray.append(int(char))

        avgEnergyOfOneDevice, trainingTimeOfOneDevice = preTrainEnv(iotDevices=iotDevices, edgeDevices=edgeDevices,
                                                                    cloud=cloud,
                                                                    action=splittingArray * len(iotDevices))
        if avgEnergyOfOneDevice > maxAvgEnergyOfOneDevice:
            maxAvgEnergyOfOneDevice = avgEnergyOfOneDevice
            maxEnergySplitting = splittingArray * len(iotDevices)
        if avgEnergyOfOneDevice < minAvgEnergyOfOneDevice:
            minAvgEnergyOfOneDevice = avgEnergyOfOneDevice
            minEnergySplitting = splittingArray * len(iotDevices)

    maxAvgEnergy, maxEnergyTrainingTime = preTrainEnv(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                                                      action=maxEnergySplitting)
    minAvgEnergy, minEnergyTrainingTime = preTrainEnv(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                                                      action=minEnergySplitting)
    print(f"Max Energy Splitting : {maxEnergySplitting}\nMin Energy Splitting : {minEnergySplitting}")
    print(f"Max Energy Training Time : {maxEnergyTrainingTime}\nMin Energy Training Time : {minEnergyTrainingTime}")
    return maxAvgEnergy, minAvgEnergy


def createLog(fileName):
    from SB3.runner import ROOT_DIR

    logging.basicConfig(filename=f"{ROOT_DIR}/{fileName}.log",
                        format='%(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


# this function changes Learning Rate of agent depends on the remaining round, decrease the LR with passing time
# USED IN STABLE-BASELINE 3
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def preTrainEnv(iotDevices: list, edgeDevices: list, cloud: Device, action) -> tuple:
    edgesConnectedDeviceNum = [0] * len(edgeDevices)

    for i in range(0, len(action), 2):
        edgeDevices[iotDevices[int(i / 2)].edgeIndex].connectedDevice = 0
        cloud.connectedDevice = 0

    totalEnergyConsumption = 0
    maxTrainingTime = 0
    offloadingPointsList = []

    iotRemainingFLOP = [iot.FLOPS for iot in iotDevices]
    edgeRemainingFLOP = [edge.FLOPS for edge in edgeDevices]
    cloudRemainingFLOP = cloud.FLOPS

    for i in range(0, len(action), 2):
        op1 = action[0]
        op2 = action[1]
        cloudRemainingFLOP -= sum(config.COMP_WORK_LOAD[op2 + 1:])
        edgeRemainingFLOP[iotDevices[int(i / 2)].edgeIndex] -= sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1])
        iotRemainingFLOP[int(i / 2)] -= sum(config.COMP_WORK_LOAD[0:op1 + 1])

        if sum(config.COMP_WORK_LOAD[op1 + 1:op2 + 1]):
            edgeDevices[iotDevices[int(i / 2)].edgeIndex].connectedDevice += 1
            edgesConnectedDeviceNum[iotDevices[int(i / 2)].edgeIndex] += 1
        if sum(config.COMP_WORK_LOAD[op2 + 1:]) != 0:
            cloud.connectedDevice += 1
    print(f"Action : {action}\nRemaining FLOP : {iotRemainingFLOP}\n{edgeRemainingFLOP}\n{cloudRemainingFLOP}")
    for i in range(0, len(action), 2):
        # Mapping float number to Offloading points
        op1 = action[0]
        op2 = action[1]
        offloadingPointsList.append(op1)
        offloadingPointsList.append(op2)

        # computing training time of this action
        iotEnergy, iotTrainingTime = iotDevices[int(i / 2)] \
            .energy_tt(splitPoints=[op1, op2], remainingFlops=iotRemainingFLOP[int(i / 2)], preTrain=True)

        _, edgeTrainingTime = edgeDevices[iotDevices[int(i / 2)].edgeIndex] \
            .energy_tt(splitPoints=[op1, op2], remainingFlops=edgeRemainingFLOP[iotDevices[int(i / 2)].edgeIndex],
                       preTrain=True)

        _, cloudTrainingTime = cloud.energy_tt([op1, op2], remainingFlops=cloudRemainingFLOP, preTrain=True)

        totalTrainingTime = iotTrainingTime + edgeTrainingTime + cloudTrainingTime
        if totalTrainingTime > maxTrainingTime:
            maxTrainingTime = totalTrainingTime

        # computing energy consumption of iot devices
        totalEnergyConsumption += iotEnergy
    averageEnergyConsumption = totalEnergyConsumption / len(iotDevices)

    return averageEnergyConsumption, maxTrainingTime


def preTrain(iotDevices, edgeDevices, cloud):
    rewardTuningParams = [0, 0, 0, 0]
    min_Energy = 1.0e7
    max_Energy = 0

    min_trainingTime = 1.0e7
    max_trainingTime = 0

    splittingLayer = allPossibleSplitting(modelLen=config.LAYER_NUM - 1, deviceNumber=len(iotDevices))

    for splitting in splittingLayer:
        splittingArray = list()
        for char in splitting:
            splittingArray.append(int(char))

        avgEnergy, trainingTime = preTrainEnv(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud,
                                              action=splittingArray)
        if avgEnergy < min_Energy:
            min_Energy = avgEnergy
            rewardTuningParams[0] = min_Energy
            min_energy_splitting = splittingArray
            min_Energy_TrainingTime = trainingTime
        if avgEnergy > max_Energy:
            max_Energy = avgEnergy
            rewardTuningParams[1] = max_Energy
            max_Energy_splitting = splittingArray
            max_Energy_TrainingTime = trainingTime

        if trainingTime < min_trainingTime:
            min_trainingTime = trainingTime
            rewardTuningParams[2] = min_trainingTime
            min_trainingtime_splitting = splittingArray
            min_trainingTime_energy = avgEnergy
        if trainingTime > max_trainingTime:
            max_trainingTime = trainingTime
            rewardTuningParams[3] = max_trainingTime
            max_trainingtime_splitting = splittingArray
            max_trainingTime_energy = avgEnergy
    return rewardTuningParams
