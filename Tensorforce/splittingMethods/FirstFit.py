import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import entities.Device as Device
import Tensorforce.config as conf


class FirstFit:

    def __init__(self, iotDevices: list[Device.Device], edgeDevices: list[Device], cloud: Device):
        self.iotDevices: list[Device.Device] = iotDevices
        self.edgeDevices: list[Device.Device] = edgeDevices
        self.cloud: Device.Device = cloud

    def initial_internals(self):
        pass

    def observe(self, reward, terminal=False, parallel=0, query=None, **kwargs):
        pass

    def act(self, **kwargs):

        edgesRemainingFlops = [edge.FLOPS for edge in self.edgeDevices]
        cloudRemainingFLOPS = self.cloud.FLOPS
        action = []
        print(edgesRemainingFlops)
        print(cloudRemainingFLOPS)
        for iotDevice in self.iotDevices:
            if edgesRemainingFlops[iotDevice.edgeIndex] > 0 or cloudRemainingFLOPS > 0:
                if edgesRemainingFlops[iotDevice.edgeIndex] > 0:
                    op1 = find_index_with_sum(conf.COMP_WORK_LOAD[1:], edgesRemainingFlops[iotDevice.edgeIndex])
                    if op1 == conf.LAYER_NUM - 2:
                        edgesRemainingFlops[iotDevice.edgeIndex] -= sum(conf.COMP_WORK_LOAD[:op1])
                        action.extend([0, op1 + 1])
                    elif op1 < conf.LAYER_NUM - 1:
                        action.extend([0, op1 + 1])
                        edgesRemainingFlops[iotDevice.edgeIndex] -= sum(conf.COMP_WORK_LOAD[1:op1 + 2])
                        cloudRemainingFLOPS -= sum(conf.COMP_WORK_LOAD[op1 + 2:])
                elif cloudRemainingFLOPS > 0:
                    action.extend([0, 0])
                    cloudRemainingFLOPS -= sum(conf.COMP_WORK_LOAD[1:])
            else:
                action.extend([conf.LAYER_NUM - 1, conf.LAYER_NUM - 1])
            print(edgesRemainingFlops)
            print(cloudRemainingFLOPS)
            print(action)
            print("------------------------------")
        return action

    def close(self):
        pass


def find_index_with_sum(arr, target_sum):
    current_sum = 0
    index = 0
    min_difference = float('inf')

    for i, num in enumerate(arr):
        current_sum += num
        difference = abs(current_sum - target_sum)

        if difference < min_difference:
            min_difference = difference
            index = i

    return index


# iotDevices = utils.createDeviceFromCSV(csvFilePath="../envs_stats/iotDevicesSmallScale.csv",
#                                        deviceType='iotDevice')
# edgeDevices = utils.createDeviceFromCSV(csvFilePath="../envs_stats/edgesSmallScale.csv")
# cloud = utils.createDeviceFromCSV(csvFilePath="../envs_stats/cloud.csv")[0]
# a = FirstFit(iotDevices=iotDevices, edgeDevices=edgeDevices, cloud=cloud)
# action = a.act()
# print(action)
