from tensorforce import Agent


def create(fraction, environment, timestepNum, saveSummariesPath):
    return Agent.create(
        agent='ppo',
        environment=environment,
        max_episode_timesteps=timestepNum,
        # Automatically configured network
        network="auto",

        # Optimization
        batch_size=10,
        update_frequency=1,
        learning_rate=0.003,
        subsampling_fraction=0.99,
        optimization_steps=5,

        # Reward estimation
        likelihood_ratio_clipping=0.6,
        discount=0.96,
        estimate_terminal=False,
        # preprocessing=dict(type='linear_normalization', min_value=0.0, max_value=1.0),
        # Critic
        critic_network="auto",

        critic_optimizer=dict(optimizer='adam', multi_step=5, learning_rate=0.003),

        # Exploration
        exploration=0.1, variable_noise=0.0,

        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,

        # TensorFlow etc
        name='agent',
        device=None,
        parallel_interactions=1,
        seed=None,
        execution=None,
        saver=None,
        summarizer=dict(directory=f"{saveSummariesPath}/summaries/ppo_{fraction}",
                        frequency=50,
                        labels='all',
                        ),
        recorder=None
    )
