import gym
import gym_foo
from gym_foo import dijkstra, util
from ray.rllib.algorithms.ppo import PPO

from ray.tune.registry import register_env
from gym_foo.envs import ExpectationMapAuto
import numpy as np

from tqdm import tqdm

register_env('701-expectation-env-v1', lambda cfg: ExpectationMapAuto(cfg))

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

def get_actions_list(algo, env, obs0):
    act = algo.compute_single_action(obs0)

    actions = [act]
    while True:
        obs, rew, done, info = env.step(act)
        if done:
            break

        act = algo.compute_single_action(obs)
        actions.append(act)

    return actions


def load_env(i):
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

    env = gym.make('701-simple-env-v0', num_agent=5, num_target=len(pellet_locs[0]), map=gt_map, prior=gt_prior)
    agents = [util.Agent('robot', s, gt_map, env.metadata['px_scale']) for s in robot_starts]
    env.set_agents(agents)
    obs = env.reset()
    env.set_targets(np.array(pellet_locs).T)
    return env, obs


if __name__ == '__main__':
    all_checkpoints = []
    for i in range(100):
        all_checkpoints.append(f'model_checkpoints/exp_env_v1/checkpoint_{10*i+1:06d}')

    algo = PPO(config=config)
    for check_path in tqdm(all_checkpoints):
        check_name = check_path.split('/')[-1]
        algo.restore(check_path)

        for i in range(9):
            env, obs0 = load_env(i)
            actions = get_actions_list(algo, env, obs0)

            actions = np.array(actions)
            np.save(f'cached_actions/exp_env_v1/{check_name}_map{i}.npy', actions)

    exit(0)
    # i = 1
    # start_locs = [
    #     [(13,13), (13,37), (37,37), (37,13), (25,25)],
    #     [(13,10), (20,15), (37,37), (37,13), (30,25)],
    #     [(13,13), (13,37), (37,37), (37,13), (25,25)],
    #     [(13,13), (13,37), (37,37), (37,13), (30,10)],
    #     [(13,13), (13,37), (37,37), (37,13), (30,10)],
    #     [(13,13), (13,37), (37,37), (37,13), (30,10)],
    #     [(13,13), (13,37), (40,37), (37,13), (25,25)],
    #     [(17,18), (13,37), (33,37), (45,13), (25,25)],
    #     [(13,13), (13,37), (37,48), (37,1), (25,25)],
    # ]

    # data = np.load(f'gym_foo/envs/maps/map{i}.npy', allow_pickle=True).item()
    # gt_map = data['map']
    # pellet_locs = np.where(gt_map == 2)
    # num_targ = np.count_nonzero(gt_map == 2)
    # gt_map[gt_map == 2] = 0

    # robot_starts = start_locs[i]

    # gt_prior = np.zeros_like(gt_map)
    # sd = 5
    # for i, (tx, ty) in enumerate(zip(*pellet_locs)):
    #     d = dijkstra.dijkstra(gt_map, (tx, ty)) / sd
    #     p = np.exp(-d*d/2)
    #     p[d < 0] = 0
    #     p = p / np.sum(p)

    #     gt_prior = gt_prior + p / num_targ

    # env = gym.make('701-simple-env-v0', num_agent=5, num_target=len(pellet_locs[0]), map=gt_map, prior=gt_prior, render_mode='human')

    # agents = [util.Agent('robot', s, gt_map, env.metadata['px_scale']) for s in robot_starts]
    # env.set_agents(agents)
    # obs = env.reset()
    # env.set_targets(np.array(pellet_locs).T)

    # algo = PPO(config=config)
    # algo.restore('model_checkpoints/exp_env_v1/checkpoint_000201')
    # act = algo.compute_single_action(obs)

    # while True:
    #     obs, rew, done, info = env.step(act)
    #     if done:
    #         break
    #     act = algo.compute_single_action(obs)
    #     env.render(mode='human')

    # while True:
    #     env.render(mode='human')
