import logging

from tensorforce import Environment

from System.Device import Device
from Tensorforce import config as conf
from Tensorforce import utils
from Tensorforce.enviroments import customEnv


class Runner:

    def __init__(self):
        self.log = True

    def run(self):
        saveGraphPath = f"Graphs/allPossibleSplitting/"
        energy = []
        trainingTime = []

        minEnergy = 1.0e7
        maxEnergy = 0
        minTT = 1.0e7
        maxTT = 0

        iotDevices = utils.createDeviceFromCSV(csvFilePath="../System/iotDevicesSmallScale.csv",
                                               deviceType='iotDevice')
        edgeDevices = utils.createDeviceFromCSV(csvFilePath="../System/edgesSmallScale.csv")
        cloud = utils.createDeviceFromCSV(csvFilePath="../System/cloud.csv")[0]

        logger = createLog(fileName=f"AllPossibleSplitting")

        splittingLayer = utils.allPossibleSplitting(modelLen=conf.LAYER_NUM, deviceNumber=len(iotDevices))
        print(len(splittingLayer))

        for splitting in splittingLayer:
            splittingArray = list()
            for char in splitting:
                splittingArray.append(int(char))

            avgEnergy, maxTrainingTime, iotFLOPS, edgeFLOPS, cloudFLOPS, allTrainingTime = preTrainEnv(
                iotDevices=iotDevices,
                edgeDevices=edgeDevices,
                cloud=cloud,
                action=splittingArray)

            logger.info(f'Action : {splittingArray}')
            logger.info(f"Energy : {avgEnergy}")
            logger.info(f"Training Time : {maxTrainingTime}")
            logger.info(f"IOT FLOPS : {iotFLOPS}")
            logger.info(f"Edge FLOPS : {edgeFLOPS}")
            logger.info(f"Cloud FLOPS : {cloudFLOPS}")
            logger.info(f"All Training Time : {allTrainingTime}")
            logger.info(f"--------------------------------------------------")

            energy.append(avgEnergy)
            trainingTime.append(maxTrainingTime)

            if avgEnergy < minEnergy:
                minEnergy = avgEnergy
                min_energy_splitting = splittingArray
                minEnergyTrainingTime = trainingTime
            if avgEnergy > maxEnergy:
                maxEnergy = avgEnergy
                max_Energy_splitting = splittingArray
                maxEnergyTrainingTime = trainingTime

            if maxTrainingTime < minTT:
                minTT = maxTrainingTime
                min_trainingtime_splitting = splittingArray
                minTrainingTimeEnergy = avgEnergy
            if maxTrainingTime > maxTT:
                maxTT = maxTrainingTime
                max_trainingtime_splitting = splittingArray
                maxTrainingTimeEnergy = avgEnergy
        print(f"Max Energy : {maxEnergy}\n")
        print(f"Min Energy : {minEnergy}\n")
        print(f"Max Training Time : {maxTT}\n")
        print(f"Min Training Time : {minTT}\n")
        # utils.draw_graph(title="Reward vs Episode",
        #                  xlabel="Episode",
        #                  ylabel="Reward",
        #                  figSizeX=10,
        #                  figSizeY=5,
        #                  x=x,
        #                  y=sumRewardOfEpisodes,
        #                  savePath=self.saveGraphPath,
        #                  pictureName=f"Reward_episode{i}")
        #
        # utils.draw_graph(title="Avg Energy vs Episode",
        #                  xlabel="Episode",
        #                  ylabel="Average Energy",
        #                  figSizeX=10,
        #                  figSizeY=5,
        #                  x=x,
        #                  y=energyConsumption,
        #                  savePath=self.saveGraphPath,
        #                  pictureName=f"Energy_episode{i}")
        #
        # utils.draw_graph(title="Avg TrainingTime vs Episode",
        #                  xlabel="Episode",
        #                  ylabel="TrainingTime",
        #                  figSizeX=10,
        #                  figSizeY=5,
        #                  x=x,
        #                  y=trainingTimeOfEpisode,
        #                  savePath=self.saveGraphPath,
        #                  pictureName=f"TrainingTime_episode{i}")

        utils.draw_scatter(title="Energy vs TrainingTime",
                           xlabel="Energy",
                           ylabel="TrainingTime",
                           x=energy,
                           y=trainingTime,
                           savePath=saveGraphPath,
                           pictureName=f"Scatter")

        utils.draw_hist(title='Avg Energy of IoT Devices',
                        x=energy,
                        xlabel="Average Energy",
                        savePath=saveGraphPath,
                        pictureName='AvgEnergy_hist')

        utils.draw_hist(title='TrainingTime of IoT Devices',
                        x=trainingTime,
                        xlabel="TrainingTime",
                        savePath=saveGraphPath,
                        pictureName='TrainingTime_hist')

        # utils.draw_3dGraph(
        #     x=energyConsumption,
        #     y=trainingTimeOfEpisode,
        #     z=sumRewardOfEpisodes,
        #     xlabel=f"Energy {self.saveGraphPath}",
        #     ylabel="Training Time",
        #     zlabel="reward"
        # )


def createEnv(timestepNum, iotDevices, edgeDevices, cloud, fraction, rewardTuningParams) -> Environment:
    return Environment.create(
        environment=customEnv.CustomEnvironment(rewardTuningParams=rewardTuningParams, iotDevices=iotDevices,
                                                edgeDevices=edgeDevices, cloud=cloud, fraction=fraction),
        max_episode_timesteps=timestepNum)


def createLog(fileName):
    logging.basicConfig(filename=f"./Logs/{fileName}.log",
                        format='%(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


def preTrainEnv(iotDevices: list[Device], edgeDevices: list[Device], cloud: Device, action):
    totalEnergyConsumption = 0
    maxTrainingTime = 0
    offloadingPointsList = []
    allTrainingTime = []

    for i in range(0, len(action), 2):
        edgeDevices[iotDevices[int(i / 2)].edgeIndex].connectedDevice = 0
        cloud.connectedDevice = 0

    iotRemainingFLOP = [iot.FLOPS for iot in iotDevices]
    edgeRemainingFLOP = [edge.FLOPS for edge in edgeDevices]
    cloudRemainingFLOP = cloud.FLOPS

    for i in range(0, len(action), 2):
        op1 = action[i]
        op2 = action[i + 1]

        cloudRemainingFLOP -= sum(conf.COMP_WORK_LOAD[op2 + 1:])
        edgeRemainingFLOP[iotDevices[int(i / 2)].edgeIndex] -= sum(conf.COMP_WORK_LOAD[op1 + 1:op2 + 1])
        iotRemainingFLOP[int(i / 2)] -= sum(conf.COMP_WORK_LOAD[0:op1 + 1])

        if sum(conf.COMP_WORK_LOAD[op1 + 1:op2 + 1]):
            edgeDevices[iotDevices[int(i / 2)].edgeIndex].connectedDevice += 1
        if sum(conf.COMP_WORK_LOAD[op2 + 1:]) != 0:
            cloud.connectedDevice += 1

    for i in range(0, len(action), 2):
        # Mapping float number to Offloading points
        op1 = action[i]
        op2 = action[i + 1]
        offloadingPointsList.append(op1)
        offloadingPointsList.append(op2)

        # computing training time of this action
        iotTrainingTime = iotDevices[int(i / 2)].trainingTime(splitPoints=[op1, op2],
                                                              remainingFlops=iotRemainingFLOP[int(i / 2)])
        edgeTrainingTime = edgeDevices[iotDevices[int(i / 2)].edgeIndex] \
            .trainingTime(splitPoints=[op1, op2],
                          remainingFlops=edgeRemainingFLOP[iotDevices[int(i / 2)].edgeIndex],
                          preTrain=True)
        cloudTrainingTime = cloud.trainingTime([op1, op2], remainingFlops=cloudRemainingFLOP)

        totalTrainingTime = iotTrainingTime + edgeTrainingTime + cloudTrainingTime
        allTrainingTime.append(totalTrainingTime)

        if totalTrainingTime > maxTrainingTime:
            maxTrainingTime = totalTrainingTime

        # computing energy consumption of iot devices
        iotEnergy = iotDevices[int(i / 2)].energyConsumption([op1, op2])
        totalEnergyConsumption += iotEnergy

    averageEnergyConsumption = totalEnergyConsumption / len(iotDevices)
    # averageEnergyConsumption = 1 - utils.normalizeReward(13.98, 1.68, averageEnergyConsumption),
    # maxTrainingTime = 1 - utils.normalizeReward(140, 0, maxTrainingTime)

    return averageEnergyConsumption, maxTrainingTime, iotRemainingFLOP, edgeRemainingFLOP, cloudRemainingFLOP, allTrainingTime


runer = Runner()
runer.run()
