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
MODE = "cd"
N_SAMPLE = 20

# data saving directory
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
outdir = f"raw_data_{timestamp}/{MODE}"
os.makedirs(outdir, exist_ok=True)

# main process
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.cuda.manual_seed(SEED)


def make_env(extend_settings={}):
    env = gym.make(
        "SmallLowGearAntTRP-v0",
        max_episode_steps=np.inf,
        internal_reset="setpoint",
        n_bins=20,
        sensor_range=16,
        **extend_settings,
    )
    env = RescaleAction(env, 0, 1)
    return env


# dummy env for initialization
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

if MODE == "cd":
    dir_name = "trp-homeostatic_shaped2022-09-19-22-12-55default-10-10"  # CDRule
elif MODE == "ed":
    dir_name = "trp-homeostatic_shaped2022-10-18-20-59-40exchange-10-10"  # EDRule
elif MODE == "ni":
    dir_name = "trp-homeostatic_shaped2022-09-23-00-16-16default-10-10NI-red"  # NIRule
else:
    raise ValueError("Invalid mode.")

agent.load(f"data/{MODE}/{dir_name}/90000000")
path_extend_settings = f"data/{MODE}/{dir_name}/extend_params.npy"

MAX_STEPS = 10000
n_blue_red = (10, 10)
balance_left, balance_right = 0.02, 0.08
num_balance = 15
use_extend_settings = True

path_extend_settings = f"data/{MODE}/{dir_name}/extend_params.npy"
extend_settings = np.load(path_extend_settings, allow_pickle=True).item() if os.path.exists(path_extend_settings) else {}


def make_biased_env(n_blue_red=(6, 4), blue_nutrient=(0.1, 0), red_nutrient=(0, 0.1), extend_settings=dict()):
    env = gym.make(
        id="SmallLowGearAntTRP-v0",
        max_episode_steps=np.inf,
        internal_reset="setpoint",
        n_bins=20,
        sensor_range=16,
        n_blue=n_blue_red[0],
        n_red=n_blue_red[1],
        blue_nutrient=blue_nutrient,
        red_nutrient=red_nutrient,
        **extend_settings,
    )
    env = RescaleAction(env, 0, 1)
    env = pfrl.wrappers.CastObservationToFloat32(env)
    return env


seeds = np.random.randint(0, 100000, N_SAMPLE).tolist()
print("seeds: ", seeds)

balance_list = np.linspace(balance_left, balance_right, num_balance).tolist()
n_context = len(balance_list)

# data
cum_nutrient = np.zeros((N_SAMPLE, n_context, MAX_STEPS, 2))
cum_foods = np.zeros((N_SAMPLE, n_context, MAX_STEPS, 2))
intero_data = np.zeros((N_SAMPLE, n_context, MAX_STEPS, 2))
is_dead = np.zeros((N_SAMPLE, n_context))
final_step = np.zeros((N_SAMPLE, n_context))
final_intero = np.zeros((N_SAMPLE, n_context, 2))
is_deficit_dead = np.zeros((N_SAMPLE, n_context))
cum_nutrient_target = np.zeros((N_SAMPLE, MAX_STEPS, 2))
cum_foods_target = np.zeros((N_SAMPLE, MAX_STEPS, 2))
intero_data_target = np.zeros((N_SAMPLE, MAX_STEPS, 2))
is_dead_target = np.zeros(N_SAMPLE)

biased_extend_settings = dict()
if use_extend_settings:
    step_IS_rule = extend_settings["step_IS_rule"]
    biased_extend_settings["step_IS_rule"] = step_IS_rule
    if "survival_area" in extend_settings:
        biased_extend_settings["survival_area"] = extend_settings["survival_area"]
    if step_IS_rule == "fast": biased_extend_settings["fast_rate"] = extend_settings["fast_rate"]

for n in tqdm(range(N_SAMPLE)):
    for i, b in enumerate(balance_list):
        
        print(f"balance list @ {b}, {i + 1}/{n_context}")
        
        rate_nutrient = np.array((b, 0.1 - b))
        length_b = n_context
        env = make_biased_env(n_blue_red=n_blue_red, red_nutrient=rate_nutrient, blue_nutrient=rate_nutrient, extend_settings=biased_extend_settings)
        
        env.seed(seeds[n])
        obs = env.reset()
        for j in range(MAX_STEPS):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if j > 0:
                d_nutrient = info["food_eaten"][0] * rate_nutrient + info["food_eaten"][1] * rate_nutrient
                cum_nutrient[n, i, j] = cum_nutrient[n, i, j - 1] + d_nutrient
                cum_foods[n, i, j] = cum_foods[n, i, j - 1] + info["food_eaten"]
            
            intero_data[n, i, j] = env.get_interoception()
            
            if done:
                print("dead... @", b, info)
                is_dead[n, i] = True
                is_deficit_dead[n, i] = np.min(info["interoception"]) < env.survival_area
                cum_foods[n, i, j + 1:] = cum_foods[n, i, j]
                cum_nutrient[n, i, j + 1:] = cum_nutrient[n, i, j]
                final_step[n, i] = j
                final_intero[n, i, :] = info["interoception"]
                break
        if not done:
            final_step[n, i] = MAX_STEPS - 1
            final_intero[n, i, :] = info["interoception"]
        
        env.close()
    
    # save files
    if n == N_SAMPLE - 1:
        np.save(f"{outdir}/cum_nutrient50_{n}.npy", cum_nutrient)
        np.save(f"{outdir}/intero_data_cn50_{n}.npy", intero_data)
        np.save(f"{outdir}/is_dead50_{n}.npy", is_dead)
        np.save(f"{outdir}/is_deficit_dead50_{n}.npy", is_deficit_dead)
        np.save(f"{outdir}/cum_foods50_{n}.npy", cum_foods)
        np.save(f"{outdir}/final_step50_{n}.npy", final_step)
        np.save(f"{outdir}/final_intero50_{n}.npy", final_intero)

print("Finish.")
