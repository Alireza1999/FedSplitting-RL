import logging
import random

import Tensorforce.config as config

logger = logging.getLogger()


class Device:
    def __init__(self, connectedDevice: int = 1, deviceType: str = 'cloud', edgeIndex: int = 0, FLOPS: int = 200,
                 bandwidth: float = 2.0):
        # for Iot Device we use Edge index to find out each iot connected to which edge
        self.edgeIndex = int(edgeIndex)
        self.FLOPS = int(FLOPS)
        self.bandwidth = float(bandwidth)
        self.deviceType = str(deviceType)
        self.connectedDevice = int(connectedDevice)

    def trainingTime(self, splitPoints: list, remainingFlops: int, numOfBatch: int = 1,
                     preTrain: bool = False) -> float:

        if not preTrain:
            # 80% the bandwidth has not changed and 20% the bandwidth has decreased by 30%.
            if random.uniform(a=0.0, b=1.0) > 0.8:
                effectiveBandwidth = self.bandwidth * 0.7
            else:
                effectiveBandwidth = self.bandwidth
        else:
            effectiveBandwidth = self.bandwidth * 0.7

        if splitPoints[0] < config.LAYER_NUM and config.LAYER_NUM > splitPoints[1] >= splitPoints[0]:
            computationTime = 0
            communicationTime = 0

            for _ in range(numOfBatch):
                if self.deviceType == 'iotDevice':
                    compWorkLoad = sum(config.COMP_WORK_LOAD[:splitPoints[0] + 1])
                    computationTime = compWorkLoad / self.FLOPS
                    if remainingFlops < 0:
                        computationTime *= (1 + (abs(remainingFlops)/100))

                    if splitPoints[0] < config.LAYER_NUM - 1:
                        sizeOfDataTransferred = config.SIZE_OF_PARAM[splitPoints[0]]
                    else:
                        sizeOfDataTransferred = 0
                    communicationTime += sizeOfDataTransferred / effectiveBandwidth

                elif self.deviceType == 'edge':
                    compWorkLoad = sum(config.COMP_WORK_LOAD[splitPoints[0] + 1:splitPoints[1] + 1])
                    if self.connectedDevice != 0:
                        computationTime = compWorkLoad / (self.FLOPS / self.connectedDevice)
                    if remainingFlops < 0 and (splitPoints[1] != splitPoints[2]):
                        computationTime *= (1 + abs(remainingFlops)/100)

                    if splitPoints[1] < config.LAYER_NUM - 1:
                        sizeOfDataTransferred = config.SIZE_OF_PARAM[splitPoints[1]]
                    else:
                        sizeOfDataTransferred = 0
                    communicationTime += sizeOfDataTransferred / effectiveBandwidth
                else:
                    compWorkLoad = sum(config.COMP_WORK_LOAD[splitPoints[1] + 1:])
                    if self.connectedDevice != 0:
                        computationTime += compWorkLoad / (self.FLOPS / self.connectedDevice)
                    if remainingFlops < 0 and splitPoints[1] < config.LAYER_NUM - 1:
                        computationTime *= (1 + abs(remainingFlops) / 100)

            # End of epoch and sending model to cloud
            if self.deviceType == 'iotDevice':
                sizeOfDataTransferred = sum(config.SIZE_OF_PARAM[:splitPoints[0]])
                communicationTime += sizeOfDataTransferred / effectiveBandwidth
            elif self.deviceType == 'edge':
                sizeOfDataTransferred = sum(config.SIZE_OF_PARAM[splitPoints[0] + 1:splitPoints[1]])
                communicationTime += sizeOfDataTransferred / effectiveBandwidth

            return communicationTime + computationTime
        else:
            raise Exception("out of range split point!!")

    def energyConsumption(self, splitPoints, batchSize: int = 1) -> float:
        compWorkLoad = 0
        sizeOfDataTransferred = 0
        if splitPoints[0] <= config.LAYER_NUM and splitPoints[1] <= config.LAYER_NUM:

            for _ in range(batchSize):
                compWorkLoad += sum(config.COMP_WORK_LOAD[:splitPoints[0] + 1])
                if splitPoints[0] < config.LAYER_NUM - 1:
                    sizeOfDataTransferred += config.SIZE_OF_PARAM[splitPoints[0]]
                else:
                    sizeOfDataTransferred = 0

            # End of epoch and sending model to cloud
            if self.deviceType == 'iotDevice':
                sizeOfDataTransferred += sum(config.SIZE_OF_PARAM[:splitPoints[0]])

            computationTime = compWorkLoad / self.FLOPS
            computationEnergy = computationTime * 1

            communicationEnergy = sizeOfDataTransferred * 0.01

            return computationEnergy + communicationEnergy
        else:
            raise Exception("out of range split point!!")
