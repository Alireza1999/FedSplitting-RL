from tensorforce import Agent


def create(fraction, environment, timestepNum, saveSummariesPath):
    return Agent.create(
        agent='ppo',
        environment=environment,
        max_episode_timesteps=timestepNum,
        # Automatically configured network
        network=[dict(type='dense', size=64, activation='tanh'),
                 dict(type='dense', size=64, activation='tanh'),
                 dict(type='dense', size=32, activation='sigmoid')],

        # Optimization
        batch_size=10, update_frequency=10, learning_rate=1.0e-3, subsampling_fraction=0.33,
        optimization_steps=5,

        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=0.90, estimate_terminal=False,
        preprocessing=dict(type='linear_normalization', min_value=0.0, max_value=1.0),
        # Critic
        critic_network=[dict(type='dense', size=64, activation='tanh'),
                        dict(type='dense', size=64, activation='tanh'),
                        dict(type='dense', size=32, activation='sigmoid')],

        critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1.0e-3),

        # Exploration
        exploration=0.2, variable_noise=0.0,

        # Regularization
        l2_regularization=0.0, entropy_regularization=0.2,

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
