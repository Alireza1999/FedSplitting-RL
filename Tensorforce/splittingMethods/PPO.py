from tensorforce import Agent


def create(fraction, environment, timestepNum, saveSummariesPath):
    return Agent.create(
        agent='ppo',
        environment=environment,
        max_episode_timesteps=timestepNum,
        # Automatically configured network
        network=[dict(type='dense', size=64, activation='tanh'),dict(type='dense', size=64, activation='tanh')],

        # Optimization
        batch_size=3,
        update_frequency=1,
        learning_rate=0.003,
        subsampling_fraction=0.33,
        multi_step=1,

        # Reward estimation
        likelihood_ratio_clipping=0.7,
        discount=0.96,
        predict_terminal_values=False,
        # preprocessing=dict(type='linear_normalization', min_value=0.0, max_value=1.0),
        # Critic
        baseline="auto",

        baseline_optimizer=dict(optimizer='adam', multi_step=5, learning_rate=0.003),

        # Exploration
        exploration=0.1, variable_noise=0.0,

        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,

        # TensorFlow etc
        config=dict(name='agent', device=None, seed=None),
        parallel_interactions=1,
        # saver=dict(directory=f"/home/alireza_soleymani/UniversityWorks/Thesis/FedSplitting-RL/Tensorforce/agent/ppo_1_1_1/{fraction}/",
        #            frequency=30),
        # summarizer=dict(directory=f"{saveSummariesPath}/summaries/ppo_{fraction}",
        #                 frequency=50,
        #                 labels='all',
        #                 ),
        recorder=None
    )
