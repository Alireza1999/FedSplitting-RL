import logging

import config as config

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

    def energy_tt(self, splitPoints: list, remainingFlops: int, numOfBatch: int = 100, preTrain: bool = False):

        if splitPoints[0] < config.LAYER_NUM and config.LAYER_NUM > splitPoints[1] >= splitPoints[0]:
            computationTime = 0
            communicationTime = 0
            computationEnergy = 0
            communicationEnergy = 0
            sizeOfDataTransferred = 0

            if self.deviceType == 'iotDevice':
                compWorkLoad = sum(config.COMP_WORK_LOAD[:splitPoints[0] + 1])
                computationTime = numOfBatch * sum(config.COMP_TIME_OF_LAYERS_clients[:splitPoints[0] + 1])
                computationEnergy = sum(config.COMP_ENERGY_OF_LAYERS_clients[:splitPoints[0] + 1])
                # if remainingFlops < 0:
                #     computationTime *= (1 + (abs(remainingFlops) / 100))

                if splitPoints[0] < config.LAYER_NUM - 1:
                    sizeOfDataTransferred = numOfBatch * config.SIZE_OF_PARAM[splitPoints[0]]
                communicationTime = sizeOfDataTransferred / self.effectiveBandwidth

            elif self.deviceType == 'edge':
                compWorkLoad = sum(config.COMP_WORK_LOAD[splitPoints[0] + 1:splitPoints[1] + 1])
                if self.connectedDevice != 0:
                    computationTime = numOfBatch * sum(
                        config.COMP_TIME_OF_LAYERS_edges[splitPoints[0] + 1:splitPoints[1] + 1])

                # if remainingFlops < 0 and (splitPoints[1] != splitPoints[2]):
                #     computationTime *= (1 + abs(remainingFlops) / 100)

                if splitPoints[1] < config.LAYER_NUM - 1:
                    sizeOfDataTransferred = numOfBatch * config.SIZE_OF_PARAM[splitPoints[1]]
                communicationTime = sizeOfDataTransferred / self.effectiveBandwidth
            else:
                compWorkLoad = sum(config.COMP_WORK_LOAD[splitPoints[1] + 1:])
                if self.connectedDevice != 0:
                    computationTime = numOfBatch * sum(config.COMP_TIME_OF_LAYERS_cloud[splitPoints[1] + 1:])
                # if remainingFlops < 0 and splitPoints[1] < config.LAYER_NUM - 1:
                #     computationTime *= (1 + abs(remainingFlops) / 100)

            # End of epoch and sending models to cloud
            if self.deviceType == 'iotDevice':
                sizeOfDataTransferred = sum(config.SIZE_OF_PARAM[:splitPoints[0]])
                communicationTime += sizeOfDataTransferred / self.effectiveBandwidth
                communicationEnergy = communicationTime * 0.2
            elif self.deviceType == 'edge':
                sizeOfDataTransferred = sum(config.SIZE_OF_PARAM[splitPoints[0] + 1:splitPoints[1]])
                communicationTime += sizeOfDataTransferred / self.effectiveBandwidth

            return computationEnergy, communicationEnergy, computationTime, communicationTime
        else:
            raise Exception("out of range split point!!")
