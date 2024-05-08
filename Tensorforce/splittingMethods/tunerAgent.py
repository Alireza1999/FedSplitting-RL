from tensorforce import Agent


def create(fraction, environment, timestepNum, saveSummariesPath):
    return Agent.create(
        environment=environment,
        max_episode_timesteps=timestepNum,
        update=1,
        optimizer=dict(),
        objective='policy_gradient',
        reward_estimation=dict(horizon=27),
        baseline="same",
        batch_size=2,
        clipping_value=0.154506736683916,
        discount=0.8292602888292065,
        entropy_regularization=0.00014980729001386585,
        importance_sampling="yes",
        learning_rate=0.0016468839654683637,
        multi_step=3,
        baseline_weight=71.44664337402382,
        estimate_advantage="no",
    )
