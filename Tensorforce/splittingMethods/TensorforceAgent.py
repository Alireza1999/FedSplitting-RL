from tensorforce import Agent


def create(fraction, environment, timestepNum, saveSummariesPath):
    return Agent.create(
        agent='tensorforce',
        environment=environment,
        max_episode_timesteps=timestepNum,

        # Reward estimation
        reward_estimation=dict(
            horizon=1,
            discount=0.96,
        ),

        # Preprocessing
        preprocessing=dict(type='linear_normalization', min_value=0.0, max_value=1.0),

        # Optimizer
        optimizer=dict(
            optimizer='adam',
            learning_rate=0.003,
            clipping_threshold=0.1,
            multi_step=5,
            subsampling_fraction=0.99
        ),

        update=dict(
            unit='timesteps',
            batch_size=5,
        ),

        policy=dict(
            type='parametrized_distributions',
            network='auto',
            distributions=dict(
                float=dict(type='beta'),
                bounded_action=dict(type='beta')
            ),
            # temperature=dict(
            #     type='decaying', decay='exponential', unit='timesteps',
            #     decay_steps=5, initial_value=0.01, decay_rate=0.5
            # ),
        ),

        objective='policy_gradient',

        # Exploration
        exploration=0.1,
        variable_noise=0.0,

        # Regularization
        l2_regularization=0.0, entropy_regularization=0.1,
        memory=200,
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
