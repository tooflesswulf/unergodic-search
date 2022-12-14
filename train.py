import gym
import gym_foo
from ray.rllib.algorithms.ppo import PPO

from ray.tune.registry import register_env
from gym_foo.envs import ExpectationMapAuto

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


if __name__ == '__main__':
    algo = PPO(config=config)

    for i in range(1000):
        algo.train()

        if i % 10 == 0:
            vv = algo.evaluate()
            print(f'Iteration {i+1} done! Saving model.')
            print(vv['episode_reward_mean'])
            algo.save(f'model_checkpoints/epoch{i}')
