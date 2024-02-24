import logging

from tensorforce import Environment

from entities.Device import Device
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

        iotDevices = utils.createDeviceFromCSV(csvFilePath="../envs_stats/iotDevices.csv",
                                               deviceType='iotDevice')
        edgeDevices = utils.createDeviceFromCSV(csvFilePath="../envs_stats/edges.csv")
        cloud = utils.createDeviceFromCSV(csvFilePath="../envs_stats/cloud.csv")[0]

        logger = createLog(fileName=f"AllPossibleSplitting")

        for i in range(10000):
            splitting = utils.randomSelectionSplitting(modelLen=conf.LAYER_NUM, deviceNumber=len(iotDevices))
            avgEnergy, maxTrainingTime, iotFLOPS, edgeFLOPS, cloudFLOPS, allTrainingTime, avgCommE, avgCompE = preTrainEnv(
                iotDevices=iotDevices,
                edgeDevices=edgeDevices,
                cloud=cloud,
                action=splitting)

            logger.info(f'Action: {splitting}')
            logger.info(f"Energy: {avgEnergy}")
            logger.info(f"Avg Communication E: {avgCommE}")
            logger.info(f"Avg Computation E: {avgCompE}")
            logger.info(f"Training Time: {maxTrainingTime}")
            logger.info(f"IOT FLOPS: {iotFLOPS}")
            logger.info(f"Edge FLOPS: {edgeFLOPS}")
            logger.info(f"Cloud FLOPS: {cloudFLOPS}")
            logger.info(f"All Training Time: {allTrainingTime}")
            logger.info(f"--------------------------------------------------")

            energy.append(avgEnergy)
            trainingTime.append(maxTrainingTime)

            if avgEnergy < minEnergy:
                minEnergy = avgEnergy
                min_energy_splitting = splitting
                minEnergyTrainingTime = maxTrainingTime
                minEnergyEdgeFLOPS = edgeFLOPS
                minEnergyIOTFLOPS = iotFLOPS
                minEnergyCloudFLOPS = cloudFLOPS

            if avgEnergy > maxEnergy:
                maxEnergy = avgEnergy
                max_Energy_splitting = splitting
                maxEnergyTrainingTime = maxTrainingTime
                maxEnergyEdgeFLOPS = edgeFLOPS
                maxEnergyIOTFLOPS = iotFLOPS
                maxEnergyCloudFLOPS = cloudFLOPS

            if maxTrainingTime < minTT:
                minTT = maxTrainingTime
                min_trainingtime_splitting = splitting
                minTrainingTimeEnergy = avgEnergy
                minTTEdgeFLOPS = edgeFLOPS
                minTTIOTFLOPS = iotFLOPS
                minTTCloudFLOPS = cloudFLOPS

            if maxTrainingTime > maxTT:
                maxTT = maxTrainingTime
                max_trainingtime_splitting = splitting
                maxTrainingTimeEnergy = avgEnergy
                maxTTEdgeFLOPS = edgeFLOPS
                maxTTIOTFLOPS = iotFLOPS
                maxTTCloudFLOPS = cloudFLOPS

        print("--------------------------------------------------")
        print(f"Max Energy : {maxEnergy}")
        print(f"Max Energy Splitting : {max_Energy_splitting}")
        print(f"Max Energy Splitting Training Time : {maxEnergyTrainingTime}")
        print(f"Max Energy Edge FLOPS : {maxEnergyEdgeFLOPS}")
        print(f"Max Energy IOT FLOPS : {maxEnergyIOTFLOPS}")
        print(f"Max Energy Cloud FLOPS : {maxEnergyCloudFLOPS}")
        print("--------------------------------------------------")
        print(f"Min Energy : {minEnergy}")
        print(f"Min Energy Splitting : {min_energy_splitting}")
        print(f"Min Energy Splitting Training Time : {minEnergyTrainingTime}")
        print(f"Min Energy Edge FLOPS : {minEnergyEdgeFLOPS}")
        print(f"Min Energy IOT FLOPS : {minEnergyIOTFLOPS}")
        print(f"Min Energy Cloud FLOPS : {minEnergyCloudFLOPS}")
        print("--------------------------------------------------")
        print(f"Max Training Time : {maxTT}")
        print(f"Max Training Time Splitting : {max_trainingtime_splitting}")
        print(f"Max Training Time Splitting Energy : {maxTrainingTimeEnergy}")
        print(f"Max Training Time Edge FLOPS : {maxTTEdgeFLOPS}")
        print(f"Max Training Time IOT FLOPS : {maxTTIOTFLOPS}")
        print(f"Max Training Time Cloud FLOPS : {maxTTCloudFLOPS}")
        print("--------------------------------------------------")
        print(f"Min Training Time : {minTT}")
        print(f"Min Training Time Splitting : {min_trainingtime_splitting}")
        print(f"Min Training Time Splitting Energy : {minTrainingTimeEnergy}")
        print(f"Min Training Time Edge FLOPS : {minTTEdgeFLOPS}")
        print(f"Min Training Time IOT FLOPS : {minTTIOTFLOPS}")
        print(f"Min Training Time Cloud FLOPS : {minTTCloudFLOPS}")
        print("--------------------------------------------------")

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
    total_comp_e = 0
    total_comm_e = 0

    for i in range(len(action)):
        edgeDevices[iotDevices[i].edgeIndex].connectedDevice = 0
        cloud.connectedDevice = 0

    iotRemainingFLOP = [iot.FLOPS for iot in iotDevices]
    edgeRemainingFLOP = [edge.FLOPS for edge in edgeDevices]
    cloudRemainingFLOP = cloud.FLOPS

    for i in range(len(action)):
        op1 = action[i][0]
        op2 = action[i][1]

        cloudRemainingFLOP -= sum(conf.COMP_WORK_LOAD[op2 + 1:])
        edgeRemainingFLOP[iotDevices[i].edgeIndex] -= sum(conf.COMP_WORK_LOAD[op1 + 1:op2 + 1])
        iotRemainingFLOP[i] -= sum(conf.COMP_WORK_LOAD[0:op1 + 1])

        if sum(conf.COMP_WORK_LOAD[op1 + 1:op2 + 1]):
            edgeDevices[iotDevices[i].edgeIndex].connectedDevice += 1
        if sum(conf.COMP_WORK_LOAD[op2 + 1:]) != 0:
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
    averageEnergyConsumption = totalEnergyConsumption / len(iotDevices)
    # averageEnergyConsumption = 1 - utils.normalizeReward(13.98, 1.68, averageEnergyConsumption),
    # maxTrainingTime = 1 - utils.normalizeReward(140, 0, maxTrainingTime)

    return averageEnergyConsumption, maxTrainingTime, iotRemainingFLOP, edgeRemainingFLOP, cloudRemainingFLOP, allTrainingTime, avgCommE, avgCompE


runer = Runner()
runer.run()
