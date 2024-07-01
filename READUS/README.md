# READUS

## Objective

The idea is to create different readme (one for each file in this fork) to understand what this project does, by revers engineering.

## TO DO

- [ ] All

## Files

| File | What |
| --- | --- |
| brain.py | Actor class |
| dqn.py | Critic class |
| environment.py | Environment class |
| training.py | Training algorithm |
| testing.py | Validation algorithm |

### brain.py

Define a class called `Brain`:

- **keras.layers.Input: Used to instantiate a Keras tensor**
https://keras.io/api/layers/core_layers/input/
This class can instantiate an initial layer composed by the inputs.

```python
keras.Input(
    shape=None, # A shape tuple (tuple of integers or None objects), not including the batch size. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors.
    batch_size=None, # Optional static batch size (integer).
    dtype=None, # The data type expected by the input
    sparse=None, # A boolean specifying whether the expected input will be sparse tensors.
    batch_shape=None,
    name=None, # Optional name string for the layer
    tensor=None, # Optional existing tensor to wrap into the Input layer
)
```

- keras.layers.Dense: Just your regular densely-connected NN layer. Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

```python
tf.keras.layers.Dense(
    units, #Positive integer, dimensionality of the output space.
    activation=None, #Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    use_bias=True, #Boolean, whether the layer uses a bias vector.
    kernel_initializer='glorot_uniform', # Initializer for the kernel weights matrix.
    bias_initializer='zeros', # Initializer for the bias vector.
    kernel_regularizer=None, # Regularizer function applied to the kernel weights matrix.
    bias_regularizer=None, # Regularizer function applied to the bias vector.
    activity_regularizer=None, # Regularizer function applied to the output of the layer (its "activation").
    kernel_constraint=None, # Constraint function applied to the kernel weights matrix.
    bias_constraint=None, #  Constraint function applied to the bias vector.
    lora_rank=None, #Optional integer. If set, the layer's forward pass will implement LoRA (Low-Rank Adaptation) with the provided rank. LoRA sets the layer's kernel to non-trainable and replaces it with a delta over the original kernel, obtained via multiplying two lower-rank trainable matrices. This can be useful to reduce the computation cost of fine-tuning large dense layers. You can also enable LoRA on an existing Dense layer by calling layer.enable_lora(rank).
)
```

- keras.layers.Droput: Applies dropout to the input.
https://keras.io/api/layers/regularization_layers/dropout/

```python
keras.layers.Dropout(
    rate, # Float between 0 and 1. Fraction of the input units to drop.
    noise_shape=None, # 1D integer tensor representing the shape of the binary dropout mask that will be multiplied with the input. For instance, if your inputs have shape (batch_size, timesteps, features) and you want the dropout mask to be the same for all timesteps, you can use noise_shape=(batch_size, 1, features).
    seed=None, # A Python integer to use as random seed.
    )
```

```python
keras.layers.Dropout(rate)(
    inputs, # Input tensor (of any rank) (keras.input).
    training #Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (doing nothing).
    )

```

- keras.models.Model: A model grouping layers into an object with training/inference features (at the end of the day connect all the layers).
https://keras.io/api/models/model/

```python
model = keras.Model(
    inputs=inputs, # Input tensor (of any rank) (keras.input)
    outputs=outputs # Last "keras.layers.Dense" (output=keras.layers.Dense(...)(output_layer_n-1))
    )
```

- keras.optimizers.Adam: Optimizer that implements the Adam algorithm
https://keras.io/api/optimizers/adam/

```python
keras.optimizers.Adam(
    learning_rate=0.001,# A float, a keras.optimizers.schedules.LearningRateSchedule instance, or a callable that takes no arguments and returns the actual value to use. The learning rate. Defaults to 0.001.
    beta_1=0.9, # A float value or a constant float tensor, or a callable that takes no arguments and returns the actual value to use. The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2=0.999, # A float value or a constant float tensor, or a callable that takes no arguments and returns the actual value to use. The exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
    epsilon=1e-07, # A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma and Ba paper (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to 1e-7.
    amsgrad=False, # Boolean. Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and beyond". Defaults to False.
    weight_decay=None, # Float. If set, weight decay is applied. 
    clipnorm=None, # Float. If set, the gradient of each weight is individually clipped so that its norm is no higher than this value.
    clipvalue=None, # Float. If set, the gradient of each weight is clipped to be no higher than this value.
    global_clipnorm=None, # Float. If set, the gradient of all weights is clipped so that their global norm is no higher than this value.
    use_ema=False, # Boolean, defaults to False. If True, exponential moving average (EMA) is applied. EMA consists of computing an exponential moving average of the weights of the model (as the weight values change after each training batch), and periodically overwriting the weights with their moving average.
    ema_momentum=0.99, # Float, defaults to 0.99. Only used if use_ema=True. This is the momentum to use when computing the EMA of the model's weights: new_average = ema_momentum * old_average + (1 - ema_momentum) * current_variable_value.
    ema_overwrite_frequency=None, #Int or None, defaults to None. Only used if use_ema=True. Every ema_overwrite_frequency steps of iterations, we overwrite the model variable by its moving average. If None, the optimizer does not overwrite model variables in the middle of training, and you need to explicitly overwrite the variables at the end of training by calling optimizer.finalize_variable_values() (which updates the model variables in-place). When using the built-in fit() training loop, this happens automatically after the last epoch, and you don't need to do anything.
    loss_scale_factor=None, # Float or None. If a float, the scale factor will be multiplied the loss before computing gradients, and the inverse of the scale factor will be multiplied by the gradients before updating variables. Useful for preventing underflow during mixed precision training. Alternately, keras.optimizers.LossScaleOptimizer will automatically set a loss scale factor.
    gradient_accumulation_steps=None, # Int or None. If an int, model & optimizer variables will not be updated at every step; instead they will be updated every gradient_accumulation_steps steps, using the average value of the gradients since the last update. This is known as "gradient accumulation". This can be useful when your batch size is very small, in order to reduce gradient noise at each update step.
    name="adam", # String. The name to use for momentum accumulator weights created by the optimizer.
)
```

- [ ] dropout: For instance, if the hidden layers have 1000 neurons (nodes) and a dropout is applied with drop probability = 0.5, then 500 neurons would be randomly dropped in every iteration (batch). The aim is to reduce the over-fitting. (https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9)

- [ ] regularize: Regularization techniques help improve a neural network’s generalization ability by reducing overfitting. They do this by minimizing needless complexity and exposing the network to more diverse data. (https://www.pinecone.io/learn/regularization-in-neural-networks/)

- [ ] keras.optimizers.schedules.LearningRateSchedule: You can use a learning rate schedule to modulate how the learning rate of your optimizer changes over time. The amount that the weights are updated during training is referred to as the step size or the “learning rate.” (https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/)

- [ ] exponential decay rate for the 1st moment estimates: https://medium.com/the-ml-practitioner/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc

- [ ] Adams alternatives: https://en.wikipedia.org/wiki/Stochastic_gradient_descent

### dqn.py

### environment.py

### testing.py

### training.py
