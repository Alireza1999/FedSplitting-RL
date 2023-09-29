from tensorforce import Agent


def create(fraction, environment, timestepNum, saveSummariesPath):
    return Agent.create(

        agent='ac',
        environment=environment,
        max_episode_timesteps=timestepNum,

        # Automatically configured network
        network='auto',

        # Optimization
        batch_size=5,
        update_frequency=5,
        learning_rate=1e-5,

        # Reward estimation
        discount=0.96,
        estimate_terminal=False,

        # Critic
        critic_network='auto',
        critic_optimizer=dict(
            optimizer='adam',
            multi_step=10,
            learning_rate=1e-5),

        # Preprocessing
        preprocessing=None,

        # Exploration
        exploration=0.1,
        variable_noise=0.0,

        # Regularization
        l2_regularization=0.0,
        entropy_regularization=0.2,

        # TensorFlow etc
        name='agent',
        device=None,
        parallel_interactions=1,
        seed=None,
        execution=None,
        saver=None,
        summarizer=dict(directory=f"{saveSummariesPath}/summaries/ac_{fraction}",
                        frequency=50,
                        labels='all',
                        ),
        recorder=None
    )
