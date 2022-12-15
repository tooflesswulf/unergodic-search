import gym
import gym_foo
from gym_foo import dijkstra, util
# from ray.rllib.algorithms.ppo import PPO

# from ray.tune.registry import register_env
# from gym_foo.envs import ExpectationMapAuto
import numpy as np

# register_env('701-expectation-env-v1', lambda cfg: ExpectationMapAuto(cfg))

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "701-expectation-env-v1",
    'env_config': {'num_agent': 5, 'num_target': 20},
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 2,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },

    'horizon': 2000,

    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    },
}


if __name__ == '__main__':
    i = 8
    start_locs = [
        [(13,13), (13,37), (37,37), (37,13), (25,25)],
        [(13,10), (20,15), (37,37), (37,13), (30,25)],
        [(13,13), (13,37), (37,37), (37,13), (25,25)],
        [(13,13), (13,37), (37,37), (37,13), (30,10)],
        [(13,13), (13,37), (37,37), (37,13), (30,10)],
        [(13,13), (13,37), (37,37), (37,13), (30,10)],
        [(13,13), (13,37), (40,37), (37,13), (25,25)],
        [(17,18), (13,37), (33,37), (45,13), (25,25)],
        [(13,13), (13,37), (37,48), (37,1), (25,25)],
    ]

    data = np.load(f'gym_foo/envs/maps/map{i}.npy', allow_pickle=True).item()
    gt_map = data['map']
    pellet_locs = np.where(gt_map == 2)
    num_targ = np.count_nonzero(gt_map == 2)
    gt_map[gt_map == 2] = 0

    robot_starts = start_locs[i]

    gt_prior = np.zeros_like(gt_map)
    sd = 5
    for i, (tx, ty) in enumerate(zip(*pellet_locs)):
        d = dijkstra.dijkstra(gt_map, (tx, ty)) / sd
        p = np.exp(-d*d/2)
        p[d < 0] = 0
        p = p / np.sum(p)

        gt_prior = gt_prior + p / num_targ

    env = gym.make('701-simple-env-v0', num_agent=5, num_target=len(pellet_locs[0]), map=gt_map, prior=gt_prior, render_mode='human')

    agents = [util.Agent('robot', s, gt_map, env.metadata['px_scale']) for s in robot_starts]
    env.set_agents(agents)
    env.reset()

    env.set_targets(pellet_locs)

    while True:
        env.render(mode='human')

    # import matplotlib.pyplot as plt
    # plt.imshow(env.map)
    # plt.show()
    # plt.imshow(env.prior)
    # plt.show()

    # algo = PPO(config=config)
    # algo.restore('model_checkpoints/epoch300/checkpoint_000301')

    # env = gym.make('701-expectation-env-v1', config={'num_agent': 5, 'num_target': 20})
    # obs = env.reset()

    # act = algo.compute_single_action(obs)
    # obs, rew, done, info = env.step(act)

    # print(act)
    # print(rew)

