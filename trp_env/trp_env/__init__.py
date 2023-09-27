from gym.envs.registration import register

register(
    id='SmallLowGearAntTRP-v0',
    entry_point='trp_env.envs:LowGearAntSmallTwoResourceEnv',
)
