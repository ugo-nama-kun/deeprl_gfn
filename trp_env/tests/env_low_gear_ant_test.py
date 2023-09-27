import numpy as np
from pytest import approx

from trp_env.envs.low_gear_ant_trp_env import LowGearAntTwoResourceEnv, LowGearAntSmallTwoResourceEnv


class TestEnv:

    def test_instance(self):
        env = LowGearAntTwoResourceEnv(
            ego_obs=True,
            no_contact=False,
            sparse=False
        )

    def test_reset_env(self):
        env = LowGearAntTwoResourceEnv()
        env.reset()

    def test_run_env(self):
        env = LowGearAntTwoResourceEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

    def test_run_env_render(self):
        env = LowGearAntSmallTwoResourceEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())
            env.render()
        env.close()

    def test_dim(self):
        env = LowGearAntTwoResourceEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 27 + 10 + 10 + 2
        assert len(env.action_space.high) == 8
        assert len(obs) == 27 + 10 + 10 + 2
        assert len(env.action_space.sample()) == 8

    def test_dim_small(self):
        env = LowGearAntSmallTwoResourceEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 27 + 20 + 20 + 2
        assert len(env.action_space.high) == 8
        assert len(obs) == 27 + 20 + 20 + 2
        assert len(env.action_space.sample()) == 8

    def test_render_env(self):
        env = LowGearAntSmallTwoResourceEnv(show_sensor_range=True, n_bins=20, sensor_range=16.)
        for n in range(5):
            env.reset()
            for i in range(10):
                env.step(0.1 * env.action_space.sample())
                env.render()
        env.close()
