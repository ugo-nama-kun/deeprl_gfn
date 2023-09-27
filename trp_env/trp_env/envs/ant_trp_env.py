import math

import numpy as np
from gym import utils

from trp_env.envs.two_resource_env import TwoResourceEnv
from trp_env.envs.mymujoco import MyMujocoEnv


# Code is mostly based on the original paper and provided code
# https://github.com/alexlioralexli/rllab-finetuning/blob/2dae9141d0fdc284d04f18931907131d66b43023/sandbox/finetuning/envs/mujoco/ant_env.py


def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):  # multiply two quaternion
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]


class MyAntEnv(MyMujocoEnv, utils.EzPickle):
    FILE = "ant.xml"
    ORI_IND = 3
    IS_WALKER = True

    def __init__(self,
                 xml_path,
                 ctrl_cost_coeff=1e-2,  # gym has 1 here!
                 rew_speed=False,  # if True the dot product is taken with the speed instead of the position
                 rew_dir=None,  # (x,y,z) -> Rew=dot product of the CoM SPEED with this dir. Otherwise, DIST to 0
                 ego_obs=True,
                 no_contact=False,
                 sparse=False,
                 *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.reward_dir = rew_dir
        self.rew_speed = rew_speed
        self.ego_obs = ego_obs
        self.no_contct = no_contact
        self.sparse = sparse
        MyMujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.sim.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        if self.rew_speed:
            direction_com = self.get_body_com('torso')
        else:
            direction_com = self.get_body_com('torso')
        if self.reward_dir:
            direction = np.array(self.reward_dir, dtype=float) / np.linalg.norm(self.reward_dir)
            forward_reward = np.dot(direction, direction_com)
        else:
            forward_reward = np.linalg.norm(
                direction_com[0:-1])  # instead of comvel[0] (does this give jumping reward??)
        lb, ub = self.action_space.low, self.action_space.high
        scaling = (ub - lb) * 0.5
        ctrl_cost = .5 * self.ctrl_cost_coeff * np.square(a / scaling).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.05  # Different from Gym Ant

        if self.sparse:  # strip the forward reward, but keep the other costs/rewards!
            if np.linalg.norm(self.get_body_com("torso")[0:2]) > np.inf:  # potentially could specify some distance
                forward_reward = 1.0
            else:
                forward_reward = 0.
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()
        # notdone = np.isfinite(state).all() \
        #     and state[2] >= 0.3 and state[2] <= 1.0  # Different from Gym Ant 0.2 --> 0.3
        notdone = np.isfinite(state).all()  # Agent dies only if the agent broken
        done = not notdone
        ob = self.get_current_obs()
        com = np.concatenate([self.get_body_com("torso").flat]).reshape(-1)
        ori = self.get_ori()
        return ob, reward, done, dict(
            com=com,
            ori=ori,
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def get_current_obs(self):
        if self.ego_obs:
            return np.concatenate([
                self.sim.data.qpos.flat[2:],  # qpos[:1] is (x,y) position of the agent
                self.sim.data.qvel.flat,
            ]).reshape(-1)
        elif self.no_contct:
            return np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                self.get_body_com("torso").flat,
                self.sim.data.get_body_xpos("torso"),  # use xpos instead of com
            ]).reshape(-1)
        else:
            return np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                self.sim.data.get_body_xmat("torso").flat,
                self.sim.data.get_body_xpos("torso"),
            ]).reshape(-1)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self.get_current_obs()

    def viewer_setup(self):
        if self.viewer:
            self.viewer.cam.distance = self.model.stat.extent * 0.5


class AntTwoResourceEnv(TwoResourceEnv):
    """
    Two Resource Problem with approximated scale of Konidaris & Barto paper
    """
    MODEL_CLASS = MyAntEnv
    ORI_IND = 3


class AntSmallTwoResourceEnv(TwoResourceEnv):
    """
    Small-sized Two Resource Problem with the scale of original GatherEnv
    """
    MODEL_CLASS = MyAntEnv
    ORI_IND = 3

    def __init__(self,
                 n_blue=6,
                 n_red=4,
                 activity_range=6.,
                 n_bins=20,
                 sensor_range=16,
                 *args, **kwargs):
        super().__init__(
            n_blue=n_blue,
            n_red=n_red,
            n_bins=n_bins,
            activity_range=activity_range,
            sensor_range=sensor_range,
            *args, **kwargs
        )
