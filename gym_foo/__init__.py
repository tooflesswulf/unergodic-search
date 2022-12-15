from gym.envs.registration import register
register(
    id='701-simple-env-v0',
    entry_point='gym_foo.envs:SimpleMap',
)

register(
    id='701-expectation-env-v0',
    entry_point='gym_foo.envs:ExpectationMap',
)

register(
    id='701-simple-env-v1',
    entry_point='gym_foo.envs:SimpleMapAuto',
)

register(
    id='701-expectation-env-v1',
    entry_point='gym_foo.envs:ExpectationMapAuto',
)
