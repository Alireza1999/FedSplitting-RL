from tensorforce import Agent


def create(fraction, environment, timestepNum, saveSummariesPath):
    return Agent.create(
        agent='tensorforce',
        environment=environment,
        max_episode_timesteps=timestepNum,

        # Reward estimation
        reward_estimation=dict(
            horizon=1,
            discount=0.96),

        # Optimizer
        optimizer=dict(
            optimizer='adam', learning_rate=0.001, clipping_threshold=0.01,
            multi_step=10, subsampling_fraction=0.99
        ),

        # update network every 2 timestep
        update=dict(
            unit='timesteps',
            batch_size=10,
        ),

        policy=dict(
            type='parametrized_distributions',
            network='auto',
            distributions=dict(
                float=dict(type='gaussian'),
                bounded_action=dict(type='beta')
            ),
            temperature=dict(
                type='decaying', decay='exponential', unit='timesteps',
                decay_steps=5, initial_value=0.01, decay_rate=0.5
            )
        ),

        objective='policy_gradient',
        # Preprocessing
        preprocessing=None,

        # Exploration
        exploration=0.1, variable_noise=0.0,

        # Regularization
        l2_regularization=0.1, entropy_regularization=0.1,
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
