# Testing the AI
# Installing Keras
# conda install -c conda-forge keras

import os
import numpy as np
import random as rn
from keras.models import load_model
from environment import Environment

# Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# SETTING THE PARAMETERS
number_actions = 5
direction_boundary = (number_actions - 1) / 2
temperature_step = 1.5

# BUILDING THE ENVIRONMENT BY SIMPLY CREATING AN OBJECT OF THE ENVIRONMENT CLASS
env = Environment(optimal_temperature=(18.0, 24.0), initial_month=0, initial_number_users=20, initial_rate_data=30)

# LOADING A PRE-TRAINED BRAIN
model = load_model("model.h5")

# CHOOSING THE MODE
train = False

# RUNNING A 1 YEAR SIMULATION IN INFERENCE MODE
env.train = train
current_state, _, _ = env.observe()
for timestep in range(0, 12 * 30 * 24 * 60):
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])
    direction = -1 if action < direction_boundary else 1
    energy_ai = abs(action - direction_boundary) * temperature_step
    next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep / (30 * 24 * 60)))
    current_state = next_state

# PRINTING THE TESTING RESULTS
print("\n")
print(f"Total Energy spent with an AI: {env.total_energy_ai:.0f}")
print(f"Total Energy spent with no AI: {env.total_energy_noai:.0f}")
print(f"ENERGY SAVED: {(env.total_energy_noai - env.total_energy_ai) / env.total_energy_noai * 100:.0f} %")
