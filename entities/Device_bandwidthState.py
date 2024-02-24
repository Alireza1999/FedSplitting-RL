import logging

import Tensorforce.config as config

logger = logging.getLogger()


class Device:
    def __init__(self, connectedDevice: int = 1, deviceType: str = 'cloud', edgeIndex: int = 0, FLOPS: int = 200,
                 bandwidth: float = 2.0, maxPower: float = 15):
        # for Iot Device we use Edge index to find out each iot connected to which edge
        self.edgeIndex = int(edgeIndex)
        self.FLOPS = int(FLOPS)
        self.bandwidth = float(bandwidth)
        self.maxPower = float(maxPower)
        self.deviceType = str(deviceType)
        self.connectedDevice = int(connectedDevice)
        self.effectiveBandwidth = self.bandwidth

    def setEffectiveBW(self, bw: float):
        self.effectiveBandwidth = bw

    def energy_tt(self, splitPoints: list, remainingFlops: int, numOfBatch: int = 1, preTrain: bool = False):

        # if not preTrain:
        #     # 80% the bandwidth has not changed and 20% the bandwidth has decreased by 30%.
        #     if random.uniform(a=0.0, b=1.0) > 0.8:
        #         self.effectiveBandwidth = np.random.uniform(low=self.bandwidth * 0.7, high=self.bandwidth)
        #     else:
        #         self.effectiveBandwidth = self.bandwidth
        # else:
        #     self.effectiveBandwidth = self.bandwidth * 0.7
        # self.effectiveBandwidth = self.bandwidth

        if splitPoints[0] < config.LAYER_NUM and config.LAYER_NUM > splitPoints[1] >= splitPoints[0]:
            computationTime = 0
            communicationTime = 0
            computationEnergy = 0
            communicationEnergy = 0

            for _ in range(numOfBatch):
                if self.deviceType == 'iotDevice':
                    compWorkLoad = sum(config.COMP_WORK_LOAD[:splitPoints[0] + 1])
                    computationTime = compWorkLoad / self.FLOPS
                    # if remainingFlops < 0:
                    #     computationTime *= (1 + (abs(remainingFlops) / 100))

                    if splitPoints[0] < config.LAYER_NUM - 1:
                        sizeOfDataTransferred = config.SIZE_OF_PARAM[splitPoints[0]]
                    else:
                        sizeOfDataTransferred = 0
                    communicationTime += sizeOfDataTransferred / self.effectiveBandwidth

                elif self.deviceType == 'edge':
                    compWorkLoad = sum(config.COMP_WORK_LOAD[splitPoints[0] + 1:splitPoints[1] + 1])
                    if self.connectedDevice != 0:
                        computationTime = compWorkLoad / (self.FLOPS / self.connectedDevice)
                    # if remainingFlops < 0 and (splitPoints[1] != splitPoints[2]):
                    #     computationTime *= (1 + abs(remainingFlops) / 100)

                    if splitPoints[1] < config.LAYER_NUM - 1:
                        sizeOfDataTransferred = config.SIZE_OF_PARAM[splitPoints[1]]
                    else:
                        sizeOfDataTransferred = 0
                    communicationTime += sizeOfDataTransferred / self.effectiveBandwidth
                else:
                    compWorkLoad = sum(config.COMP_WORK_LOAD[splitPoints[1] + 1:])
                    if self.connectedDevice != 0:
                        computationTime += compWorkLoad / (self.FLOPS / self.connectedDevice)
                    # if remainingFlops < 0 and splitPoints[1] < config.LAYER_NUM - 1:
                    #     computationTime *= (1 + abs(remainingFlops) / 100)

            # End of epoch and sending model to cloud
            if self.deviceType == 'iotDevice':
                sizeOfDataTransferred = sum(config.SIZE_OF_PARAM[:splitPoints[0]])
                communicationTime += sizeOfDataTransferred / self.effectiveBandwidth
                computationEnergy = computationTime * self.maxPower
                communicationEnergy = communicationTime * 0.15
            elif self.deviceType == 'edge':
                sizeOfDataTransferred = sum(config.SIZE_OF_PARAM[splitPoints[0] + 1:splitPoints[1]])
                communicationTime += sizeOfDataTransferred / self.effectiveBandwidth

            return computationEnergy, communicationEnergy, computationTime, communicationTime
        else:
            raise Exception("out of range split point!!")
