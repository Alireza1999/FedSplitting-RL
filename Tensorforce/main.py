import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import argparse
from Tensorforce import runner

arguments = {
    '-a': ['--agent', 'tensorforce',
           '[String] name of the RL agent[ppo, firstFit, ac, trpo, tensorforce, random, noSplitting]'],
    '-env': ['--env', 'default', '[String] name of the environment [fedAdapt, default, defaultNoEdge, bandwidthInState]'],
    '-f': ['--fraction', 0.8, '[float] The fraction of energy and training time that is used for training RL'],
    '-e': ['--episode', 501, '[int] Number of episodes'],
    '-t': ['--timestep', 200, '[int] Number of timestep of each episode'],
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


parser = argparse.ArgumentParser()
options = parse_argument(parser=parser, arg=arguments)

runner = runner.Runner(agentType=options['agent'], envType=options["env"], episodeNum=int(options['episode']),
                       timestepNum=int(options['timestep']), fraction=float(options['fraction']),
                       summaries=options['summaries'], log=options['log'])
runner.run()
