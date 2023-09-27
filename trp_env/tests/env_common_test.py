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

    def test_instance_not_ego_obs(self):
        env = AntTwoResourceEnv(
            ego_obs=False,
            no_contact=False,
            sparse=False
        )
        env.reset()

    def test_instance_no_contact(self):
        env = AntTwoResourceEnv(
            ego_obs=True,
            no_contact=True,
            sparse=False
        )
        env.reset()

    def test_reset_internal_state(self):
        env = AntTwoResourceEnv(internal_reset="setpoint")
        env.reset()
        env.internal_state = {
            FoodClass.BLUE: 1.0,
            FoodClass.RED: 1.0,
        }
        for key in FoodClass:
            assert env.internal_state[key] == approx(1.0)

        env.reset()
        initial_internal_state = {
            FoodClass.BLUE: 0.0,
            FoodClass.RED: 0.0,
        }

        for key in FoodClass:
            assert env.internal_state[key] == initial_internal_state[key]

    def test_reset_if_resource_end(self):
        env = AntTwoResourceEnv()
        env.default_metabolic_update = 0.1
        env.reset()
        while True:
            ob, reward, done, info = env.step(0 * env.action_space.sample())
            if done:
                break
            else:
                intero = ob[-2:]
                assert intero[0] > -env.survival_area and intero[1] > -env.survival_area

        intero = ob[-2:]
        assert intero[0] < -env.survival_area or intero[1] < -env.survival_area
        ob = env.reset()
        intero = ob[-2:]
        assert intero[0] == approx(0) and intero[1] == approx(0)

    def test_run_env(self):
        env = AntTwoResourceEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())
        env.close()

    def test_render_env(self):
        env = AntSmallTwoResourceEnv(show_sensor_range=True, n_bins=20, sensor_range=16.,
                                     n_blue=2, n_red=3)
        for n in range(5):
            env.reset()
            for i in range(10):
                env.step(0.1 * env.action_space.sample())
                env.render()
        env.close()

    def test_reset(self):
        env = AntTwoResourceEnv()
        env.reset()
        initial_robot_pos = env.wrapped_env.get_body_com("torso")[:2].copy()
        for i in range(1000):
            env.step(env.action_space.sample())

        env.reset()
        reset_robot_pos = env.wrapped_env.get_body_com("torso")[:2].copy()
        np.testing.assert_allclose(actual=reset_robot_pos,
                                   desired=initial_robot_pos,
                                   atol=0.3)

    @pytest.mark.parametrize("setting,expected_mean, expected_var",
                             [
                                 ("setpoint", np.array([0.0, 0.0]), np.array([0.0, 0.0])),
                                 ("random", np.array([0.0, 0.0]), np.array([0.33, 0.33])),
                                 ("error_case", None, None)
                             ])
    def test_reset_internal(self, setting, expected_mean, expected_var):
        if setting != "error_case":
            env = AntTwoResourceEnv(internal_reset=setting)
        else:
            with pytest.raises(ValueError):
                AntTwoResourceEnv(internal_reset=setting)
            return

        obs_intero_list = []

        for i in range(1000):
            obs = env.reset()

            obs_intero = obs[-2:]  # interoception

            obs_intero_list.append(obs_intero)

        obs_intero_mean = np.array(obs_intero_list).mean(axis=0)
        obs_intero_var = np.array(obs_intero_list).var(axis=0)

        # Test mean
        np.testing.assert_allclose(actual=obs_intero_mean,
                                   desired=expected_mean,
                                   atol=0.06)

        # Test var
        np.testing.assert_allclose(actual=obs_intero_var,
                                   desired=expected_var,
                                   atol=0.06)

    @pytest.mark.parametrize("setting,expected_mean, expected_var",
                             [
                                 ([-1, 1], np.array([0.0, 0.0]), np.array([0.33, 0.33])),
                                 ([-0.5, 0.5], np.array([0.0, 0.0]), np.array([1. / 12, 1. / 12])),
                                 ([0, 1], np.array([0.5, 0.5]), np.array([1. / 12, 1. / 12])),
                             ])
    def test_reset_internal_random_limit(self, setting, expected_mean, expected_var):
        env = AntTwoResourceEnv(internal_reset="random",
                                internal_random_range=setting)

        obs_intero_list = []

        for i in range(3000):
            obs = env.reset()

            obs_intero = obs[-2:]  # interoception

            obs_intero_list.append(obs_intero)

        obs_intero_mean = np.array(obs_intero_list).mean(axis=0)
        obs_intero_var = np.array(obs_intero_list).var(axis=0)

        # Test mean
        np.testing.assert_allclose(actual=obs_intero_mean,
                                   desired=expected_mean,
                                   atol=0.03)

        # Test var
        np.testing.assert_allclose(actual=obs_intero_var,
                                   desired=expected_var,
                                   atol=0.02)

    @pytest.mark.parametrize("reward_setting,expected,param",
                             [
                                 ("homeostatic", -2, None),
                                 ("homeostatic", -2, 0.5),  # Reward bias should be ignored
                                 ("homeostatic_shaped", +2, None),
                                 ("one", 0, None),
                                 ("homeostatic_biased", -1.5, 0.5),
                                 ("something_else", None, None),
                             ])
    def test_reward_definition(self, reward_setting, expected, param):
        env = AntTwoResourceEnv(reward_setting=reward_setting, coef_main_rew=1.0, reward_bias=param)

        action = env.action_space.sample() * 0

        env.prev_interoception = np.array([1, 1])
        env.internal_state = {
            FoodClass.BLUE: 0.0,
            FoodClass.RED: 0.0,
        }

        if reward_setting != "something_else":
            assert env.get_reward(reward_setting, action, False) == approx(expected, abs=0.0001)
        else:
            with pytest.raises(ValueError):
                env.get_reward(reward_setting, action, False)

        if reward_setting == "one":
            env.internal_state = {
                FoodClass.BLUE: -0.90005,
                FoodClass.RED: -0.999999,
            }
            _, reward, done, _ = env.step(action)

            assert done
            assert reward == approx(-1.0, abs=0.001)

    @pytest.mark.parametrize("red_eaten,blue_eaten,expected",
                             [
                                 (0, 0, 0),
                                 (1, 0, 1),
                                 (0, 1, 1),
                                 (1, 1, 2),
                                 (1, 2, 3),
                                 (None, 0, None),
                                 (0, None, None),
                             ])
    def test_greedy_reward(self, red_eaten, blue_eaten, expected):
        env = AntTwoResourceEnv(reward_setting="greedy", coef_main_rew=1.0)

        action = env.action_space.sample() * 0

        if red_eaten is None or blue_eaten is None:
            with pytest.raises(ValueError):
                env.get_reward("greedy", action, False, blue_eaten, red_eaten)
        else:
            reward = env.get_reward("greedy", action, False, blue_eaten, red_eaten)
            assert reward == approx(expected, abs=0.0001)

    @pytest.mark.parametrize("reward_setting,expected,param",
                             [
                                 ("homeostatic", -2, None),
                                 ("homeostatic", -2, 0.5),  # Reward bias should be ignored
                                 ("homeostatic_shaped", +2, None),
                                 ("one", 0, None),
                                 ("homeostatic_biased", -1.5, 0.5),
                                 ("something_else", None, None),
                             ])
    def test_reward_definition_small(self, reward_setting, expected, param):
        env = AntSmallTwoResourceEnv(reward_setting=reward_setting, coef_main_rew=1.0, reward_bias=param)

        action = env.action_space.sample() * 0

        env.prev_interoception = np.array([1, 1])
        env.internal_state = {
            FoodClass.BLUE: 0.0,
            FoodClass.RED: 0.0,
        }

        if reward_setting != "something_else":
            assert env.get_reward(reward_setting, action, False) == approx(expected, abs=0.0001)
        else:
            with pytest.raises(ValueError):
                env.get_reward(reward_setting, action, False)

        if reward_setting == "one":
            env.internal_state = {
                FoodClass.BLUE: -0.90005,
                FoodClass.RED: -0.999999,
            }
            _, reward, done, _ = env.step(action)

            assert done
            assert reward == approx(-1.0, abs=0.0001)

    def test_object_num(self):
        env = AntTwoResourceEnv()
        env.reset()

        n_blue = 0
        n_red = 0
        for obj in env.objects:
            if obj[2] is FoodClass.BLUE:
                n_blue += 1
            elif obj[2] is FoodClass.RED:
                n_red += 1

        assert n_blue == 14
        assert n_red == 8

    def test_object_num_small(self):
        env = AntSmallTwoResourceEnv()
        env.reset()

        n_blue = 0
        n_red = 0
        for obj in env.objects:
            if obj[2] is FoodClass.BLUE:
                n_blue += 1
            elif obj[2] is FoodClass.RED:
                n_red += 1

        assert n_blue == 6
        assert n_red == 4

    def test_dt(self):
        env = AntTwoResourceEnv()
        env.reset()

        assert env.dt == 0.01 * 5

    @pytest.mark.parametrize("env_class,expected",
                             [
                                 (AntTwoResourceEnv, (27, 20, 2)),
                                 (AntSmallTwoResourceEnv, (27, 40, 2)),
                             ])
    def test_multi_modal_dims(self, env_class, expected):
        env = env_class()

        assert env.multi_modal_dims == expected

    @pytest.mark.parametrize("seed,expected,expected_val",
                             [
                                 (0, 0, 0.658194704787239),
                                 (1, 1, 0.6788319225439268),
                                 (2, 2, 0.8128451481188561)
                             ])
    def test_multi_modal_dims(self, seed, expected, expected_val):
        env = AntSmallTwoResourceEnv()
        seed_value = env.seed(seed)

        assert seed_value == [expected]

        obs = env.reset()

        assert obs[0] == approx(expected_val)

    def test_max_time_steps(self):
        env = AntSmallTwoResourceEnv()
        env._max_episode_steps = 10

        num_of_decisions = 0
        env.reset()

        while True:
            a = env.action_space.sample()
            num_of_decisions += 1

            _, _, done, _ = env.step(a)

            if done:
                break

        assert num_of_decisions == 10

    def test_max_time_steps_init(self):
        env = AntSmallTwoResourceEnv(max_episode_steps=42)

        num_of_decisions = 0
        env.reset()

        while True:
            a = env.action_space.sample()
            num_of_decisions += 1

            _, _, done, _ = env.step(a)

            if done:
                break

        assert num_of_decisions == 42

    @pytest.mark.parametrize("setting,expected",
                             [
                                 ((True, False), np.array([0.1, 0])),
                                 ((False, True), np.array([0, 0.1])),
                                 ((True, True), np.array([0.1, 0.1])),
                                 ((False, False), None),
                             ])
    def test_update_by_food(self, setting, expected):
        env = AntTwoResourceEnv(internal_reset="setpoint")
        env.reset()

        if setting[0] or setting[1]:
            env.update_by_food(is_blue=setting[0], is_red=setting[1])

            np.testing.assert_allclose(actual=env.get_interoception(),
                                       desired=expected,
                                       atol=0.03)
        else:
            with pytest.raises(AssertionError):
                env.update_by_food(is_blue=setting[0], is_red=setting[1])

    def test_rgb(self):
        env = AntSmallTwoResourceEnv()
        env.reset()

        im = None
        for i in range(100):
            env.step(0. * env.action_space.sample())
            im = env.render(mode='rgb_array', camera_id=0, width=64, height=64)

        import matplotlib.pyplot as plt
        plt.imshow(im)
        plt.savefig("test_im.png")

        assert im.shape == (64, 64, 3)

        env.close()

    def test_rgbd(self):
        env = AntSmallTwoResourceEnv()
        env.seed()
        env.reset()

        im = None
        for i in range(100):
            env.step(0. * env.action_space.sample())
            im = env.render(mode='rgbd_array', camera_id=0, width=64, height=64)

        import matplotlib.pyplot as plt
        plt.subplot(2, 2, 1)
        plt.imshow(im[:, :, :3] / 255.)
        plt.subplot(2, 2, 2)
        plt.hist(im[:, :, :3].flatten())
        plt.subplot(2, 2, 3)
        plt.imshow(im[:, :, 3], "gray")
        plt.colorbar()
        plt.subplot(2, 2, 4)
        plt.hist(im[:, :, 3].flatten())
        plt.savefig("test_rgbd.png")

        assert im.shape == (64, 64, 4)
        assert im[:, :, :3].shape == (64, 64, 3)

        env.close()

    def test_rgbd_render_and_capture(self):
        env = AntSmallTwoResourceEnv()
        env.seed()
        env.reset()

        im = None
        for i in range(100):
            env.step(0.1 * env.action_space.sample())
            env.render()
            im = env.render(mode='rgbd_array', camera_id=0, width=64, height=64)

        import matplotlib.pyplot as plt
        plt.figure(dpi=300)
        plt.imshow(im[:, :, :3] / 255.)
        plt.axis("off")
        plt.savefig("test_rgbd_render.png")

        assert im.shape == (64, 64, 4)
        assert im[:, :, :3].shape == (64, 64, 3)

        env.close()

    def test_count_eaten(self):
        env = AntSmallTwoResourceEnv()
        env.reset()
        for i in range(200):
            obs, rew, done, info = env.step(env.action_space.sample())
            assert info["food_eaten"] == (env.num_blue_eaten, env.num_red_eaten)

    @pytest.mark.parametrize("setting,expected",
                             [
                                 ("eat_blue", np.array([0.09, 0.02])),
                                 ("eat_red", np.array([0.02, 0.09])),
                                 ("eat_both", np.array([0.11, 0.11])),
                             ])
    def test_update_nutrient_by_dist(self, setting, expected):
        env = AntSmallTwoResourceEnv(red_nutrient=(0.02, 0.09), blue_nutrient=(0.09, 0.02))
        env.reset()

        if setting == "eat_red":
            env.update_by_food(is_red=True, is_blue=False)
        elif setting == "eat_blue":
            env.update_by_food(is_red=False, is_blue=True)
        elif setting == "eat_both":
            env.update_by_food(is_red=True, is_blue=True)

        np.testing.assert_allclose(actual=env.get_interoception(),
                                   desired=expected)

    def test_render_motion_line(self):
        env = AntSmallTwoResourceEnv(show_sensor_range=True, n_bins=20, sensor_range=16., show_move_line=True)
        env.reset()
        for i in range(100):
            env.step(0.1 * env.action_space.sample())
            env.render()
        env.close()

    def test_get_com_from_info(self):
        env = AntSmallTwoResourceEnv()
        env.reset()
        for i in range(10):
            _, _, _, info = env.step(0.1 * env.action_space.sample())

            com = info.get("com")
            assert com is not None
            assert com.shape == (3,)

    def test_space(self):
        env = AntSmallTwoResourceEnv()

        assert env.observation_space.shape == (2 + 27 + 40,)
        assert env.action_space.shape == (8,)

    @pytest.mark.parametrize("n_food",
                             [
                                 (1, 2),
                                 (10, 4),
                                 (20, 28),
                                 (30, 30),
                                 (29, 30)
                             ])
    def test_set_num_food_by_reset(self, n_food):
        n_blue_set, n_red_set = n_food

        env = AntSmallTwoResourceEnv(n_blue=6, n_red=4)

        if n_blue_set + n_red_set > 48:
            with pytest.raises(AssertionError):
                env.reset(n_blue=n_blue_set, n_red=n_red_set)
        else:
            env.reset(n_blue=n_blue_set, n_red=n_red_set)

            n_blue = 0
            n_red = 0
            for obj in env.objects:
                if obj[2] is FoodClass.BLUE:
                    n_blue += 1
                elif obj[2] is FoodClass.RED:
                    n_red += 1

            assert n_blue == n_blue_set
            assert n_red == n_red_set

            for i in range(10):
                _, _, _, info = env.step(0.1 * env.action_space.sample())
                env.render()

    @pytest.mark.parametrize("n_food,a_range",
                             [
                                 ((5**2 - 1, 0), 5),
                                 ((4 ** 2 - 11, 10), 4),
                                 ((10 ** 2 - 21, 20), 10),
                                 ((10 ** 2 - 21, 25), 10),
                                 ((10 ** 2 - 21, 30), 3),
                             ])
    def test_set_num_food_and_range(self, n_food, a_range):
        n_blue_set, n_red_set = n_food

        env = AntSmallTwoResourceEnv(n_blue=1, n_red=1, activity_range=a_range)

        if n_blue_set + n_red_set > (a_range + 1) ** 2:
            with pytest.raises(AssertionError):
                env.reset(n_blue=n_blue_set, n_red=n_red_set)
        else:
            env.reset(n_blue=n_blue_set, n_red=n_red_set)

            n_blue = 0
            n_red = 0
            for obj in env.objects:
                if obj[2] is FoodClass.BLUE:
                    n_blue += 1
                elif obj[2] is FoodClass.RED:
                    n_red += 1

            assert n_blue == n_blue_set
            assert n_red == n_red_set

            for i in range(100):
                _, _, _, info = env.step(0.1 * env.action_space.sample())
                env.render()
