COMP_WORK_LOAD = [884736, 0, 4718592, 0, 2359296, 524288, 1280]  # Computation workload for each layer (FLOP) VGG5 FLOPS
LAYER_NUM = len(COMP_WORK_LOAD)  # Number of model Layer
SIZE_OF_PARAM = [0.0008325, 0.0008325, 0.010492, 0.010492, 0.029, 0.291]  # Size of parameter send from each layer to next layer (MB)
