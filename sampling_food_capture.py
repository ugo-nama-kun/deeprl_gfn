import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import gym
import pfrl
import trp_env
import torch

from tqdm import tqdm
from torch import nn
from gym.wrappers import RescaleAction
from util.ppo import PPO_KL
from util.modules import BetaPolicyModel

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Configuration
SEED = 100


# main process
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.cuda.manual_seed(SEED)


def make_env(extend_settings=dict()):
    env = gym.make(
        "SmallLowGearAntTRP-v0",
        seed=SEED,
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

MAX_STEPS = 80000
n_blue_red = (10, 10)
use_extend_settings = True

for SEED, mode in enumerate(["cd", "ed", "ni"]):
    
    if mode == "cd":
        dir_name = "trp-homeostatic_shaped2022-09-19-22-12-55default-10-10"  # CDRule
    elif mode == "ed":
        dir_name = "trp-homeostatic_shaped2022-10-18-20-59-40exchange-10-10"  # EDRule
    elif mode == "ni":
        dir_name = "trp-homeostatic_shaped2022-09-23-00-16-16default-10-10NI-red"  # NIRule
    else:
        raise ValueError("Invalid mode.")
    
    agent.load(f"data/{mode}/{dir_name}/90000000")
    path_extend_settings = f"data/{mode}/{dir_name}/extend_params.npy"

    extend_settings = {}
    if os.path.exists(path_extend_settings):
        extend_settings = np.load(path_extend_settings, allow_pickle=True).item()
    
    print(extend_settings)
    
    cum_nutrient = np.zeros((MAX_STEPS, 2))
    cum_foods = np.zeros((MAX_STEPS, 2))
    intero_data = np.zeros((MAX_STEPS, 2))
    food_capture = np.zeros((MAX_STEPS, 2))

    target_extend_settings = dict()
    
    if use_extend_settings:
        target_extend_settings["step_IS_rule"] = extend_settings["step_IS_rule"]
        target_extend_settings["n_blue"] = n_blue_red[0]
        target_extend_settings["n_red"] = n_blue_red[1]
    
    env = make_env(target_extend_settings)
    obs = env.reset(seed=SEED)
    
    for step in tqdm(range(MAX_STEPS), desc=f"{mode} sampling:"):
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        
        # env.render()
        
        # total nutrient consumption
        if step > 0:
            d_nutrient = info["food_eaten"][0] * np.array((0.1, 0)) + info["food_eaten"][1] * np.array((0, 0.1))
            cum_nutrient[step] = cum_nutrient[step - 1] + d_nutrient
            
            # total food consumption
            cum_foods[step] = cum_foods[step - 1] + info["food_eaten"]
        
        intero_data[step] = env.get_interoception()
        food_capture[step] = info["food_eaten"]
        
        if done:
            print(f"{mode} agent dead... @", info)
            break
    
    os.makedirs(f"raw_data_{timestamp}/{mode}", exist_ok=True)
    np.save(f"raw_data_{timestamp}/{mode}/cum_nutrient_target.npy", cum_nutrient)
    np.save(f"raw_data_{timestamp}/{mode}/cum_foods_target.npy", cum_foods)
    np.save(f"raw_data_{timestamp}/{mode}/intero_data.npy", intero_data)
    np.save(f"raw_data_{timestamp}/{mode}/food_capture.npy", food_capture)

print("Finish.")
