# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat


def example():
    """TODO: Docstring for example.

    :function: TODO
    :returns: TODO

    """

    config = habitat.get_config("configs/pointnav.yaml")

    with habitat.Env(config=config) as env:

        print("Environment creation successful")
        _ = env.reset()

        print("Agent stepping around inside environment.")

        count_steps = 0
        while not env.episode_over:
            _ = env.step(env.action_space.sample())
            count_steps += 1

        print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
    example()
