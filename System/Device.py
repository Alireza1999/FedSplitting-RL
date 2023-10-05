import Tensorforce.config as config
import random


class Device:
    def __init__(self, deviceType: str = 'cloud', edgeIndex: int = 0, CPU: int = 2, bandwidth: float = 2.0,
                 capacity: int = 5):
        # for Iot Device we use Edge index to find out each iot connected to which edge
        self.edgeIndex = int(edgeIndex)
        self.CPU = int(CPU)
        self.bandwidth = float(bandwidth)
        self.deviceType = str(deviceType)
        self.capacity = int(capacity)

    def trainingTime(self, splitPoints: list, preTrain: bool = False, batchSize: int = 50) -> float:
        communicationTime = 0
        computationTime = 0
        if splitPoints[0] < config.LAYER_NUM and config.LAYER_NUM > splitPoints[1] >= splitPoints[0]:
            sizeOfDataTransferred = 0.0
            compWorkLoad = 0.0
            for _ in range(batchSize):
                if self.deviceType == 'iotDevice':
                    compWorkLoad += sum(config.COMP_WORK_LOAD[:splitPoints[0] + 1])
                    if splitPoints[0] < config.LAYER_NUM - 1:
                        sizeOfDataTransferred += config.SIZE_OF_PARAM[splitPoints[0]]
                    else:
                        sizeOfDataTransferred = 0

                elif self.deviceType == 'edge':
                    compWorkLoad += sum(config.COMP_WORK_LOAD[splitPoints[0] + 1:splitPoints[1] + 1])
                    if splitPoints[1] < config.LAYER_NUM - 1:
                        sizeOfDataTransferred += config.SIZE_OF_PARAM[splitPoints[1]]
                    else:
                        sizeOfDataTransferred = 0
                else:
                    compWorkLoad += sum(config.COMP_WORK_LOAD[splitPoints[1] + 1:])

                computationTime = compWorkLoad / self.CPU

            # End of epoch and sending model to cloud
            if self.deviceType == 'iotDevice':
                sizeOfDataTransferred += sum(config.SIZE_OF_PARAM[:splitPoints[0]])
            elif self.deviceType == 'edge':
                sizeOfDataTransferred += sum(config.SIZE_OF_PARAM[splitPoints[0] + 1:splitPoints[1]])

            communicationTime = sizeOfDataTransferred / self.bandwidth

            # if not preTrain:
            #     effectiveBandwidth = 0
            #     # 80% the bandwidth has not changed and 20% the bandwidth has decreased by 30%.
            #     if random.uniform(a=0.0, b=1.0) < 0.8:
            #         communicationTime = sizeOfDataTransferred / self.bandwidth
            #         effectiveBandwidth = self.bandwidth
            #     else:
            #         communicationTime = sizeOfDataTransferred / (self.bandwidth * 0.7)
            #         effectiveBandwidth = self.bandwidth * 0.7
            # else:
            #     communicationTime = sizeOfDataTransferred / 0.7 * self.bandwidth

            return communicationTime + computationTime
        else:
            raise Exception("out of range split point!!")

    def energyConsumption(self, splitPoints, batchSize: int = 50) -> float:
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

            computationTime = compWorkLoad / self.CPU
            computationEnergy = computationTime * 1

            communicationEnergy = sizeOfDataTransferred * 1

            return computationEnergy + communicationEnergy
        else:
            raise Exception("out of range split point!!")
