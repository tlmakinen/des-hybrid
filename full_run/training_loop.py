# embedding loop code
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import optax

from tensorflow_probability.substrates.jax import distributions as tfd


from typing import Sequence, Any, Callable
Array = Any

import numpy as np
import flax.linen as nn
import matplotlib.pyplot as plt
import cloudpickle as pickle



def run_training_loop(model, 
                             key,
                             train_data, # tuple of (d ,theta)
                             test_data,
                             epochs=1000,
                             batch_size=256,
                             learning_rate=1e-4,
                             schedule=False,
                             n_params=3,
                             w=None):

    
    data_train, theta_train = train_data
    data_test, theta_test =  test_data

    data_single_shape = data_train[0].shape
    
    n_train = data_train.shape[0]
    
    remainder = batch_size * (data_train.shape[0] // batch_size)
    
    data_ = data_train.reshape((-1, batch_size,) + data_single_shape)
    theta_ = theta_train.reshape(-1, batch_size, n_params)
    
    # reshape the test data into batches
    data_test = data_test[:remainder].reshape((-1, batch_size,) + data_single_shape)
    theta_test = theta_test[:remainder].reshape(-1, batch_size, n_params)

    @jax.jit
    def logprob_loss(w, x_batched, theta_batched):
    
        def fn(x, theta):
           logp = model.apply(w, x, theta)
           return logp
    
        logp_batched = jax.vmap(fn)(x_batched, theta_batched)
        return -jnp.mean(logp_batched)

    
    # init model again
    if w is None:
        w = model.init(key, data_train[0], jnp.ones(n_params,))

    total_steps = epochs*(data_train.shape[0]) + epochs
    if schedule:
        lr_schedule = optax.exponential_decay(init_value=learning_rate, transition_steps=total_steps,
                                                      decay_rate=0.98, transition_begin=int(total_steps*0.25),
                                                      staircase=False)
    else:
        lr_schedule = learning_rate
    
    # # Clip gradients at max value, and evt. apply weight decay
    transf = [optax.clip(1.0)]
    transf.append(optax.add_decayed_weights(1e-4))
    tx = optax.chain(
        *transf,
        optax.adam(learning_rate=lr_schedule)
    )
    opt_state = tx.init(w)
    loss_grad_fn = jax.value_and_grad(logprob_loss)

    
    # this is a hack to make the for-loop training much faster in jax
    def body_fun(i, inputs):
        w,loss_val, opt_state, _data, _theta, key = inputs
        x_samples = _data[i]
        y_samples = _theta[i]
    
        # apply noise simulator
        keys = jr.split(key, x_samples.shape[0])
        #x_samples = jax.vmap(noise_simulator)(keys, x_samples)
    
    
        loss_val, grads = loss_grad_fn(w, x_samples, y_samples)
        updates, opt_state = tx.update(grads, opt_state, w)
        w = optax.apply_updates(w, updates)
    
        return w, loss_val, opt_state, _data, _theta, key
    
    
    def val_body_fun(i, inputs):
        w,loss_val, _data, _theta, key = inputs
        x_samples = _data[i]
        y_samples = _theta[i]
    
        # apply noise simulator
        keys = jr.split(key, x_samples.shape[0])
        #x_samples = jax.vmap(noise_simulator)(keys, x_samples)
    
        loss_val, grads = loss_grad_fn(w, x_samples, y_samples)
    
        return w, loss_val, _data, _theta, key
    

    
    losses = jnp.zeros(epochs)
    val_losses = jnp.zeros(epochs)
    loss_val = 0.
    val_loss_value = 0.
    best_val_loss = jnp.inf
    lower = 0
    upper = n_train // batch_size

    best_w = w
    
    pbar = tqdm(range(epochs), leave=True, position=0)
    counter = 0
    
    for j in pbar:
          key,rng = jax.random.split(key)
    
          # shuffle data every epoch
          randidx = jr.permutation(key, jnp.arange(theta_.reshape(-1, n_params).shape[0]), independent=True)
          _data = data_.reshape((-1,) + data_single_shape)[randidx].reshape((-1, batch_size,) + data_single_shape)
          _theta = theta_.reshape(-1, n_params)[randidx].reshape(-1, batch_size, n_params)
    
          #print(_data.shape)
    
          inits = (w, loss_val, opt_state, _data, _theta, key)
          w, loss_val, opt_state, _data, _theta, key = jax.lax.fori_loop(lower, upper, body_fun, inits)
          losses = losses.at[j].set(loss_val)
    
    
          # do validation set
          key,rng = jr.split(key)
          inits = (w, loss_val, data_test, theta_test, key)
          w, val_loss_value, data_test, theta_test, key = jax.lax.fori_loop(0, data_test.shape[0], val_body_fun, inits)
          val_losses = val_losses.at[j].set(val_loss_value)

          if val_loss_value < best_val_loss:
              best_val_loss = val_loss_value
              best_w = w
    
          #val_losses.append(val_loss)
          pbar.set_description('epoch %d loss: %.5f  val loss: %.5f'%(j, loss_val, val_loss_value))
    
          counter += 1

    return best_w, (losses, val_losses)
