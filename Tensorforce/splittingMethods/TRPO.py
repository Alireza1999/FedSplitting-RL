from tensorforce import Agent


def create(fraction, environment, timestepNum, saveSummariesPath):
    return Agent.create(
        agent='trpo',
        environment=environment,
        max_episode_timesteps=timestepNum,

        # Reward estimation
        discount=0.96,
        batch_size=10,
        network='auto',
        # Preprocessing
        preprocessing=None,

        # Exploration
        exploration=0.1, variable_noise=0.0,

        # Regularization
        l2_regularization=0.1, entropy_regularization=0.1,
        memory=2200,
        # TensorFlow etc
        # saver=dict(directory='model', filename='model'),
        summarizer=dict(directory=f"{saveSummariesPath}/summaries/tensorforce_{fraction}",
                        frequency=50,
                        labels='all',
                        ),
        recorder=None,

        # Config
        config=dict(name='agent',
                    device="GPU",
                    parallel_interactions=1,
                    seed=None,
                    execution=None,
                    )
    )
