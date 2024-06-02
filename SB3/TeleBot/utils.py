import json


def readConfigList(configPath):
    f = open(f"{configPath}.json")
    configNum = 0
    # returns JSON object as
    # a dictionary
    configList = json.load(f)
    configs = ""
    for item in configList:
        configs += f"------------------\n"
        configNum = len(configList)
        for key in item:
            configs += f"Config ID: {key}\n\t"
            for attribute in item[key]:
                configs += f"{attribute} : {item[key][attribute]}\n\t"
    f.close()
    return configNum, configs
