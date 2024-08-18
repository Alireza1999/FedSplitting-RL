import sys
from pathlib import Path

ROOT_DIR = Path.cwd().parent
sys.path.append(f"{ROOT_DIR}")

import argparse
from SB3.runner import Runner

arguments = {
    '-a': ['--agent', 'ppo',
           '[String] name of the RL agent[ppo, firstFit, ac, trpo]'],
    '-f': ['--fraction', 0.8, '[float] The fraction of energy and training time that is used for training RL'],
    '-lr': ['--learningRate', 0.003, '[float] The learning rate for RL'],
    '-eval': ['--evaluation', False, '[Boolean] It indicates whether or not to just evaluate RL'],
    '-model': ['--modelName', None, '[String] The name of the RL model to use'],
    '-b': ['--batchSize', 1000, '[int] The batch size for RL'],
    '-ns': ['--numSteps', 1000, '[int] The number of training steps for RL'],
    '-c': ['--clip', 0.3, '[float] The clipping value for RL'],
    '-e': ['--episode', 5000, '[int] Number of episodes'],
    '-t': ['--timestep', 1, '[int] Number of timestep of each episode'],
    '-s': ['--summaries', True, '[boolean] Save the summaries or not'],
    '-l': ['--log', True, '[boolean] save log or not']
}


def parse_argument(parser: argparse.ArgumentParser(), arg: dict):
    for op in arguments.keys():
        parser.add_argument(op, arguments.get(op)[0], help=arguments.get(op)[2], type=str,
                            default=arguments.get(op)[1])
    args = parser.parse_args()
    option = vars(args)
    return option


def mainRunner():
    parser = argparse.ArgumentParser()
    options = parse_argument(parser=parser, arg=arguments)

    runner = Runner(agentType=options['agent'], episodeNum=int(options['episode']),
                    timestepNum=int(options['timestep']), fraction=float(options['fraction']),
                    summaries=options['summaries'], log=options['log'], batch_size=int(options['batchSize']),
                    lr=float(options['learningRate']), n_step=int(options['numSteps']), clip=float(options['clip']),
                    justEval=options['evaluation'], modelName=options['modelName'])
    runner.run()


if __name__ == '__main__':
    mainRunner()
