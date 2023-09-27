import logging
import os
import random
import sys

import numpy as np
import seaborn as sns
import gym
import pfrl
import trp_env
import torch

from tqdm import tqdm
from torch import nn
from gym.wrappers import RescaleAction
from util.ppo import PPO_KL
from util.modules import BetaPolicyModel

sns.set()
sns.set_context("talk")

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Configuration:
mode = "ED"  # ED, CD, or NI
seed = 100


# main code from below:)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)


def make_env(extend_settings=dict()):
    env = gym.make(
        "SmallLowGearAntTRP-v0",
        seed=seed,
        # "trp_env:SmallLowGearAntTRP-v1",
        max_episode_steps=np.inf,
        internal_reset="setpoint",
        n_bins=20,
        sensor_range=16,
        **extend_settings,
    )
    env = RescaleAction(env, 0, 1)
    env = pfrl.wrappers.CastObservationToFloat32(env)
    return env


env = make_env()

obs_space = env.observation_space
action_space = env.action_space

obs_size = obs_space.low.size
action_size = action_space.low.size

policy = BetaPolicyModel(obs_size=obs_size,
                         action_size=action_size,
                         hidden1=256,
                         hidden2=64)

value_func = torch.nn.Sequential(
    nn.Linear(obs_size, 256),
    nn.Tanh(),
    nn.Linear(256, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
)

model = pfrl.nn.Branched(policy, value_func)

opt = torch.optim.Adam(model.parameters())

agent = PPO_KL(
    model=model,
    optimizer=opt,
    gpu=-1,
)

if mode == "CD":
    dir_name = "trp-homeostatic_shaped2022-09-19-22-12-55default-10-10"  # CDRule
    agent.load(f"data/cd/{dir_name}/90000000_finish")
    path_extend_settings = f"data/cd/{dir_name}/extend_params.npy"
elif mode == "ED":
    dir_name = "trp-homeostatic_shaped2022-10-18-20-59-40exchange-10-10"  # EDRule
    agent.load(f"data/ed/{dir_name}/90000000_inter")
    path_extend_settings = f"data/ed/{dir_name}/extend_params.npy"
elif mode == "NI":
    dir_name = "trp-homeostatic_shaped2022-09-23-00-16-16default-10-10NI-red"  # NIRule
    agent.load(f"data/ni/{dir_name}/90000000_finish")
    path_extend_settings = f"data/ni/{dir_name}/extend_params.npy"
else:
    raise ValueError("Invalid mode.")

n_sample = 20
MAX_STEPS = 10000000
n_blue_red = (10, 10)
use_extend_settings = True

extend_settings = {}
if os.path.exists(path_extend_settings):
    extend_settings = np.load(path_extend_settings, allow_pickle=True).item()

print("EXTENDED SETTINGS: ", extend_settings)

cum_nutrient_target = np.zeros((n_sample, MAX_STEPS, 2))
cum_foods_target = np.zeros((n_sample, MAX_STEPS, 2))
intero_data_target = np.zeros((n_sample, MAX_STEPS, 2))
is_dead_target = np.zeros(n_sample)
seeds = np.random.randint(0, 100000, n_sample)

target_extend_settings = dict()

if use_extend_settings:
    target_extend_settings["step_IS_rule"] = extend_settings["step_IS_rule"]
    target_extend_settings["n_blue"] = n_blue_red[0]
    target_extend_settings["n_red"] = n_blue_red[1]

for n in tqdm(range(n_sample)):
    env = make_env(target_extend_settings)
    obs = env.reset()
    
    for i in range(MAX_STEPS):
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        
        env.render()
        
        if i > 0:
            # total nutrient consumption
            d_nutrient = info["food_eaten"][0] * np.array((0.1, 0)) + info["food_eaten"][1] * np.array((0, 0.1))
            cum_nutrient_target[n, i] = cum_nutrient_target[n, i - 1] + d_nutrient
            
            # total food consumption
            cum_foods_target[n, i] = cum_foods_target[n, i - 1] + info["food_eaten"]
        
        intero_data_target[n, i] = env.get_interoception()
        
        if done:
            is_dead_target[n] = True
            print("dead... @", info)
            break

print("Finish.")
