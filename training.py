# Training the AI
# Installing Keras
# conda install -c conda-forge keras

import os
import numpy as np
import random as rn
from environment import Environment
from brain import Brain
from dqn import dqn as DQN

# Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# SETTING THE PARAMETERS
epsilon = 0.3
number_actions = 5
direction_boundary = (number_actions - 1) / 2
number_epochs = 10
max_memory = 3000
batch_size = 512
temperature_step = 1.5

# BUILDING THE ENVIRONMENT BY SIMPLY CREATING AN OBJECT OF THE ENVIRONMENT CLASS
env = Environment(optimal_temperature=(18.0, 24.0), initial_month=0, initial_number_users=20, initial_rate_data=30)

# BUILDING THE BRAIN BY SIMPLY CREATING AN OBJECT OF THE BRAIN CLASS
brain = Brain(learning_rate=0.00001, number_actions=number_actions)

# BUILDING THE DQN MODEL BY SIMPLY CREATING AN OBJECT OF THE DQN CLASS
dqn = DQN(max_memory=max_memory, discount=0.9)

# CHOOSING THE MODE
train = True

# TRAINING THE AI
env.train = train
model = brain.model
early_stopping = True
patience = 10
best_total_reward = -np.inf
patience_count = 0

if env.train:
    # STARTING THE LOOP OVER ALL THE EPOCHS (1 Epoch = 5 Months)
    for epoch in range(1, number_epochs + 1):
        # INITIALIZING ALL THE VARIABLES OF BOTH THE ENVIRONMENT AND THE TRAINING LOOP
        total_reward = 0
        loss = 0.0
        new_month = np.random.randint(0, 12)
        env.reset(new_month=new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0

        # STARTING THE LOOP OVER ALL THE TIMESTEPS (1 Timestep = 1 Minute) IN ONE EPOCH
        while not game_over and timestep <= 5 * 30 * 24 * 60:
            # PLAYING THE NEXT ACTION BY EXPLORATION
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                direction = -1 if action < direction_boundary else 1
                energy_ai = abs(action - direction_boundary) * temperature_step
            else:
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
                direction = -1 if action < direction_boundary else 1
                energy_ai = abs(action - direction_boundary) * temperature_step

            # UPDATING THE ENVIRONMENT AND REACHING THE NEXT STATE
            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep / (30 * 24 * 60)))
            total_reward += reward

            # STORING THIS NEW TRANSITION INTO THE MEMORY
            dqn.remember([current_state, action, reward, next_state], game_over)

            # GATHERING IN TWO SEPARATE BATCHES THE INPUTS AND THE TARGETS
            inputs, targets = dqn.get_batch(model, batch_size=batch_size)

            # COMPUTING THE LOSS OVER THE TWO WHOLE BATCHES OF INPUTS AND TARGETS
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state

        # PRINTING THE TRAINING RESULTS FOR EACH EPOCH
        print("\n")
        print(f"Epoch: {epoch:03d}/{number_epochs:03d}")
        print(f"Total Energy spent with an AI: {env.total_energy_ai:.0f}")
        print(f"Total Energy spent with no AI: {env.total_energy_noai:.0f}")

        # EARLY STOPPING
        if early_stopping:
            if total_reward <= best_total_reward:
                patience_count += 1
            else:
                best_total_reward = total_reward
                patience_count = 0
            if patience_count >= patience:
                print("Early Stopping")
                break

    # SAVING THE MODEL
    model.save("model.h5")
