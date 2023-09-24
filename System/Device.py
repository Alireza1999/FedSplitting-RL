import sys
# sys.path.insert(0, '/home/alireza_soleymani/UniversityWorks/Thesis/RL/Tensorforce')
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

    def trainingTime(self, splitPoints: list) -> float:
        if splitPoints[0] < config.LAYER_NUM and config.LAYER_NUM > splitPoints[1] >= splitPoints[0]:
            sizeOfDataTransferred = 0

            if self.deviceType == 'iotDevice':
                compWorkLoad = sum(config.COMP_WORK_LOAD[:splitPoints[0] + 1])
                if splitPoints[0] < config.LAYER_NUM - 1:
                    sizeOfDataTransferred = config.SIZE_OF_PARAM[splitPoints[0]]
                else:
                    sizeOfDataTransferred = 0

            elif self.deviceType == 'edge':
                compWorkLoad = sum(config.COMP_WORK_LOAD[splitPoints[0] + 1:splitPoints[1] + 1])
                if splitPoints[1] < config.LAYER_NUM - 1:
                    sizeOfDataTransferred = config.SIZE_OF_PARAM[splitPoints[1]]
                else:
                    sizeOfDataTransferred = 0
            else:
                compWorkLoad = sum(config.COMP_WORK_LOAD[splitPoints[1] + 1:])

            computationTime = compWorkLoad / self.CPU

            effectiveBandwidth = []
            # 80% the bandwidth has not changed and 20% the bandwidth has decreased by 30%.
            if random.random() < 0.8:
                communicationTime = sizeOfDataTransferred / self.bandwidth
                effectiveBandwidth.append(self.bandwidth)
            else:
                communicationTime = sizeOfDataTransferred / (self.bandwidth * 0.7)
                effectiveBandwidth.append(self.bandwidth * 0.7)

            return communicationTime + computationTime
        else:
            raise Exception("out of range split point!!")

    def energyConsumption(self, splitPoints) -> float:
        if splitPoints[0] <= config.LAYER_NUM and splitPoints[1] <= config.LAYER_NUM:
            compWorkLoad = sum(config.COMP_WORK_LOAD[:splitPoints[0] + 1])
            if splitPoints[0] < config.LAYER_NUM - 1:
                sizeOfDataTransferred = config.SIZE_OF_PARAM[splitPoints[0]]
            else:
                sizeOfDataTransferred = 0

            computationTime = compWorkLoad / self.CPU
            computationEnergy = computationTime * 1

            communicationEnergy = sizeOfDataTransferred * 1

            return computationEnergy + communicationEnergy
        else:
            raise Exception("out of range split point!!")
