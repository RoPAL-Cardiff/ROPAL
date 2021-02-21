# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
example.py
==================================================
This is an example of how to use the habitat environment class.
"""

import habitat
from habitat_baselines.utils.env_utils import (
    make_env_fn,
    construct_envs,
    construct_threaded_envs,
)
from habitat_baselines.config.default import get_config


def example():
    """
    Initialises a `habitat.Env` environment, then takes random steps.
    """

    baseline_config = get_config("configs/tasks/pointnav.yaml")
    config = habitat.get_config("configs/tasks/pointnav.yaml")

    config.defrost()
    config.TASK_CONFIG = baseline_config.TASK_CONFIG
    config.freeze()

    env_class = habitat.Env

    env = make_env_fn(config, env_class)

    n_steps = 500
    n_episodes = env.number_of_episodes

    for episode in range(n_steps):
        count_steps = 0
        _ = env.reset()

        scene_name = env.current_episode.scene_id.split(sep="/")[-1].strip(".glb")
        print(f"\nStarting episode {episode+1}/{n_episodes} in {scene_name}...")

        print("Stepping around...")
        for step in range(n_steps):
            action = env.action_space.sample()

            while action["action"] == "STOP":
                action = env.action_space.sample()

            _ = env.step(action)
            count_steps += 1

        print(
            f"Episode {episode+1} in scene {scene_name} finished after {count_steps} steps."
        )


def vector_example():
    """
    Initialises a `habitat.VectorEnv` environment, then takes random steps.
    """

    baseline_config = get_config("configs/tasks/pointnav.yaml")
    config = habitat.get_config("configs/tasks/pointnav.yaml")

    config.defrost()
    config.TASK_CONFIG = baseline_config.TASK_CONFIG
    config.NUM_ENVIRONMENTS = 14
    config.SENSORS = baseline_config.SENSORS
    config.SIMULATOR_GPU_ID = 0
    config.freeze()

    env_class = habitat.Env
    envs = construct_envs(config, env_class)

    _ = envs.reset()

    n_steps = 500
    n_envs = envs.num_envs
    n_episodes = envs.number_of_episodes[0]

    for episode in range(n_episodes):
        count_steps = 0
        print(
            f"Starting episode {episode}/{n_episodes} in parallel processes across {n_envs} environments."
        )
        scene_names = [
            ep.scene_id.split("/")[-1].strip(".glb") for ep in envs.current_episodes()
        ]

        for i, scene in enumerate(scene_names):
            print(f"Stepping around in Environment {i+1}: {scene}..")

        for step in range(n_steps):
            random_actions = [
                action_space.sample() for action_space in envs.action_spaces
            ]
            _ = envs.step(random_actions)
            count_steps += 1

        print("Episodes finished after {} steps.".format(count_steps))


def threaded_vector_example():
    """
    Initialises a `habitat.ThreadedVectorEnv` environment, then takes random steps.
    """

    baseline_config = get_config("configs/tasks/pointnav.yaml")
    config = habitat.get_config("configs/tasks/pointnav.yaml")

    config.defrost()
    config.TASK_CONFIG = baseline_config.TASK_CONFIG
    config.NUM_ENVIRONMENTS = 14
    config.SENSORS = baseline_config.SENSORS
    config.SIMULATOR_GPU_ID = 0
    config.freeze()

    env_class = habitat.Env

    envs = construct_threaded_envs(config, env_class)

    _ = envs.reset()

    n_steps = 500
    n_envs = envs.num_envs
    n_episodes = envs.number_of_episodes[0]

    for episode in range(n_episodes):
        count_steps = 0
        print(
            f"Starting episode {episode}/{n_episodes} in parallel threads across {n_envs} environments."
        )
        scene_names = [
            ep.scene_id.split("/")[-1].strip(".glb") for ep in envs.current_episodes()
        ]

        for i, scene in enumerate(scene_names):
            print(f"Stepping around in Environment {i+1}: {scene}..")

        for step in range(n_steps):
            random_actions = [
                action_space.sample() for action_space in envs.action_spaces
            ]
            _ = envs.step(random_actions)
            count_steps += 1

        print("Episodes finished after {} steps.".format(count_steps))


if __name__ == "__main__":
    import time

    start = time.time()
    example()
    end = time.time()
    example_time = end - start

    start = time.time()
    vector_example()
    end = time.time()
    vector_example_time = end - start

    start = time.time()
    threaded_vector_example()
    end = time.time()
    threaded_example_time = end - start

    print(
        f"\n\n\nNormal Env took {round(example_time, 1)}s to take 500 steps in 2 Environments"
    )
    print(
        f"VectorEnv took {round(vector_example_time, 1)}s to take 500 steps in 2 Environments"
    )
    print(
        f"ThreadedVectorEnv took {round(threaded_example_time, 1)}s to take 500 steps in 2 Environments"
    )
