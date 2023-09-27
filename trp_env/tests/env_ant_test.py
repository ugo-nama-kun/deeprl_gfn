import pytest
import numpy as np

from pytest import approx

from trp_env.envs.two_resource_env import FoodClass
from trp_env.envs.ant_trp_env import AntTwoResourceEnv, AntSmallTwoResourceEnv


class TestEnv:

    def test_instance(self):
        env = AntTwoResourceEnv(
            ego_obs=True,
            no_contact=False,
            sparse=False
        )

    def test_reset_env(self):
        env = AntTwoResourceEnv()
        env.reset()

    def test_run_env(self):
        env = AntTwoResourceEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

    def test_dim(self):
        env = AntTwoResourceEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 27 + 10 + 10 + 2
        assert len(env.action_space.high) == 8
        assert len(obs) == 27 + 10 + 10 + 2
        assert len(env.action_space.sample()) == 8

    def test_dim_small(self):
        env = AntSmallTwoResourceEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 27 + 20 + 20 + 2
        assert len(env.action_space.high) == 8
        assert len(obs) == 27 + 20 + 20 + 2
        assert len(env.action_space.sample()) == 8