# deeprl_gfn
The code for "Modeling long-term nutritional behaviors using deep homeostatic reinforcement learning" by Naoto Yoshida, Etsushi Arikawa, Hoshinori Kanazawa, Yasuo Kuniyoshi.

## Main contents
- **plot_gfn.py**: Plotting code for Fig.5 in the paper.  
- **train.py**:  Training code of the optimization of the agents.
- **visualize_behavior.py**: Visualization of the trained agent.
- **sampling_gfn.py**: Data sampling for the GFN plots.
- **sampling_food_capture.py**: Data sampling for the food capture situations.

## Running Environment
Ubuntu 18.04+
Python 3.9

## Required python packages
```
gym==0.22.0
mujoco-py==2.1.2.14
pfrl==0.3.0
torch==2.0.1
tqdm
```

```bash
# We recommend to make anaconda environment:
conda create -n gfnrl python=3.9

# install the RL environment:
pip install -e trp_env
```

## How to run train.py
```bash
# How to run with GPU. "--rule" takes CD, ED, or NI as argument.  
python train.py --seed 0 --gpu 0 --rule CD

# CPU mode.
python train.py --seed 0 --gpu 0
```

For other code, simply:
```bash
python {name_of_code}.py
```

## Ack
Author: Naoto Yoshida @ The University of Tokyo, Kyoto University.
## License
MIT
