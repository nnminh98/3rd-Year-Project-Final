from gym.envs.registration import register, registry, make, spec

register(
    id='SinglePacketRouting-v0',
    entry_point='env.Minh:SinglePacketRouting',
    max_episode_steps=1000,
    reward_threshold=50
)

register(
    id='MultiplePacketRouting-v0',
    entry_point='env.Minh:MultiplePacketRouting',
    max_episode_steps=1000,
    reward_threshold=25
)

