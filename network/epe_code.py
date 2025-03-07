from functools import partial

import jax
import numpy as np
import optax
from absl import logging
from jax import numpy as jnp
from jax import random as jr
from jax import scipy as jsp
from jax._src.flatten_util import ravel_pytree
from tqdm import tqdm
import sys,os

from sbijax._src._ne_base import NE
from sbijax._src.util.data import as_inference_data
from sbijax._src.util.early_stopping import EarlyStopping
#from sbijax._src.util.dataloader import as_batch_iterators

from collections.abc import Iterable
from typing import Callable, Sequence, Any

import haiku as hk
import haiku.experimental.flax as hkflax
import flax.linen as nn
import jax
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd


from pathlib import Path
import cloudpickle as pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    


#Path("/my/directory").mkdir(parents=True, exist_ok=True)


# custom dataloader class to support additive noise (or simulator)


# ruff: noqa: PLR0913, E501
class myNPE(NE):
    """Neural posterior estimation.

    Implements the method introduced in :cite:t:`greenberg2019automatic`.
    In the literature, the method is usually referred to as APT or NPE-C, but
    here we refer to it simply as NPE.

    Args:
        model_fns: a tuple of calalbles. The first element needs to be a
            function that constructs a tfd.JointDistributionNamed, the second
            element is a simulator function.
        density_estimator: a (neural) conditional density estimator
            to model the posterior distribution
        num_atoms: number of atomic atoms

    Examples:
        >>> from sbijax import NPE
        >>> from sbijax.nn import make_maf
        >>> from tensorflow_probability.substrates.jax import distributions as tfd
        ...
        >>> prior = lambda: tfd.JointDistributionNamed(
        ...     dict(theta=tfd.Normal(0.0, 1.0))
        ... )
        >>> s = lambda seed, theta: tfd.Normal(theta["theta"], 1.0).sample(seed=seed)
        >>> fns = prior, s
        >>> neural_network = make_maf(1)
        >>> model = NPE(fns, neural_network)

    References:
        Greenberg, David, et al. "Automatic posterior transformation for likelihood-free inference." International Conference on Machine Learning, 2019.
    """

    def __init__(self, model_fns, density_estimator, num_atoms=10):
        """Construct an SNP object.

        Args:
            model_fns: a tuple of tuples. The first element is a tuple that
                    consists of functions to sample and evaluate the
                    log-probability of a data point. The second element is a
                    simulator function.
            density_estimator: a (neural) conditional density estimator
                to model the posterior distribution
            num_atoms: number of atomic atoms
        """
        super().__init__(model_fns, density_estimator)
        self.num_atoms = num_atoms
        self.n_round = 0

    # ruff: noqa: D417
    def fit(
        self,
        rng_key,
        data,
        *,
        train_dataset=None,
        val_dataset=None,
        noise_simulator=None,
        optimizer=optax.adam(0.0003),
        n_iter=1000,
        batch_size=128,
        percentage_data_as_validation_set=0.1,
        n_early_stopping_patience=10,
        params = None,
        outdir = "./net-logs/",
        theta_idx = None,
        **kwargs,
    ):
        """Fit an SNP model.

        Args:
            rng_key: a jax random key
            data: data set obtained from calling
                `simulate_data_and_possibly_append`
            noise_simulator: noise function with signature `data_noise = fn(key, data)`
            optimizer: an optax optimizer object
            n_iter: maximal number of training iterations per round
            batch_size:  batch size used for training the model
            percentage_data_as_validation_set: percentage of the simulated
                data that is used for validation and early stopping
            n_early_stopping_patience: number of iterations of no improvement
                of training the flow before stopping optimisation


        Returns:
            a tuple of parameters and a tuple of the training information
        """
        itr_key, rng_key = jr.split(rng_key)

        Path(outdir).mkdir(parents=True, exist_ok=True)
        
        if train_dataset is None:
            train_iter, val_iter = self.as_iterators(
                itr_key, data, batch_size, percentage_data_as_validation_set,
                noise_simulator
            )
        else:
            train_iter = train_dataset
            val_iter = val_dataset

        if params is None:
            params, losses = self._fit_model_single_round(
                seed=rng_key,
                train_iter=train_iter,
                val_iter=val_iter,
                optimizer=optimizer,
                n_iter=n_iter,
                n_early_stopping_patience=n_early_stopping_patience,
                outdir=outdir,
                n_atoms=self.num_atoms,
            )
        else:
            params, losses = self._fit_model_single_round(
                seed=rng_key,
                train_iter=train_iter,
                val_iter=val_iter,
                optimizer=optimizer,
                n_iter=n_iter,
                n_early_stopping_patience=n_early_stopping_patience,
                n_atoms=self.num_atoms,
                params=params,
                outdir=outdir,
                theta_idx=theta_idx
            )

        return params, losses

    # pylint: disable=undefined-loop-variable
    def _fit_model_single_round(
        self,
        seed,
        train_iter,
        val_iter,
        optimizer,
        n_iter,
        n_early_stopping_patience,
        n_atoms,
        params=None,
        outdir="./net-logs/",
        theta_idx=None
    ):
        init_key, seed = jr.split(seed)
        if params is None:
            params = self._init_params(init_key, **next(iter(train_iter)))
        else:
            params = params
        # intialise opt state
        state = optimizer.init(params)

        n_round = self.n_round
        if n_round == 0:

            def loss_fn(params, rng, **batch):
                # upack tuple
                lp = self.model.apply(
                    params,
                    None,
                    method="log_prob",
                    y=jnp.array(batch["theta"]),
                    x=batch["y"],
                )
                return -jnp.mean(lp)

        else:

            def loss_fn(params, rng, **batch):
                lp = self._proposal_posterior_log_prob(
                    params,
                    rng,
                    n_atoms,
                    theta=jnp.array(batch["theta"]),
                    y=jnp.array(batch["y"]),
                )
                return -jnp.mean(lp)

        @jax.jit
        def step(params, rng, state, **batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, rng, **batch)
            updates, new_state = optimizer.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, new_state

        losses = np.zeros([n_iter, 2])
        early_stop = EarlyStopping(1e-3, n_early_stopping_patience)
        best_params, best_loss = None, np.inf
        logging.info("training model")

        pbar = tqdm(range(n_iter), leave=True, position=0) # progress bar
        for i in pbar:
            train_loss = 0.0
            rng_key = jr.fold_in(seed, i)
            for b in range(train_iter.num_batch_per_epoch):

                batch = next(iter(train_iter))
                train_key, rng_key = jr.split(rng_key)
                batch_loss, params, state = step(
                    params, train_key, state, **batch
                )
                #print("batch_loss", batch_loss)
                train_loss += batch_loss * (
                    jax.tree.leaves(batch["y"])[0].shape[0] / train_iter.num_samples
                )
                #print("train_loss", train_loss)
            
            val_key, rng_key = jr.split(rng_key)
            validation_loss = self._validation_loss(
                val_key, params, val_iter, n_atoms
            )
            #print("validation_loss", validation_loss)
            
            losses[i] = jnp.array([train_loss, validation_loss])
            pbar.set_description('epoch %d loss: %.5f  val loss: %.5f'%(i, train_loss, validation_loss))
            
            _, early_stop = early_stop.update(validation_loss)
            if early_stop.should_stop:
                logging.info("early stopping criterion found")
                break
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_params = params.copy()
                save_obj(best_params, outdir + "best_params")
                np.savez(outdir + "history",
                         train_val_losses=jnp.vstack(losses)[: (i + 1), :])

        #self.n_round += 1
        losses = jnp.vstack(losses)[: (i + 1), :]

        return best_params, losses

    def _init_params(self, rng_key, **init_data):
        params = self.model.init(
            rng_key, method="log_prob", y=jnp.array(init_data["theta"]), x=init_data["y"]
        )
        return params

    def _proposal_posterior_log_prob(self, params, rng, n_atoms, theta, y):
        n = theta.shape[0]
        n_atoms = np.maximum(2, np.minimum(n_atoms, n))
        repeated_y = jnp.repeat(y, n_atoms, axis=0)
        probs = jnp.ones((n, n)) * (1 - jnp.eye(n)) / (n - 1)

        choice = partial(
            jr.choice, a=jnp.arange(n), replace=False, shape=(n_atoms - 1,)
        )
        sample_keys = jr.split(rng, probs.shape[0])
        choices = jax.vmap(lambda key, prob: choice(key, p=prob))(
            sample_keys, probs
        )
        contrasting_theta = theta[choices]

        atomic_theta = jnp.concatenate(
            (theta[:, None, :], contrasting_theta), axis=1
        )
        atomic_theta = atomic_theta.reshape(n * n_atoms, -1)

        log_prob_posterior = self.model.apply(
            params, None, method="log_prob", y=atomic_theta, x=repeated_y
        )
        log_prob_posterior = log_prob_posterior.reshape(n, n_atoms)
        log_prob_prior = self.prior_log_density_fn(atomic_theta)
        log_prob_prior = log_prob_prior.reshape(n, n_atoms)

        unnormalized_log_prob = log_prob_posterior - log_prob_prior
        log_prob_proposal_posterior = unnormalized_log_prob[
            :, 0
        ] - jsp.special.logsumexp(unnormalized_log_prob, axis=-1)

        return log_prob_proposal_posterior

    def _validation_loss(self, rng_key, params, val_iter, n_atoms):
        if self.n_round == 0:

            def loss_fn(rng, **batch):
                lp = self.model.apply(
                    params,
                    None,
                    method="log_prob",
                    y=jnp.array(batch["theta"]),
                    x=batch["y"],
                )
                return -jnp.mean(lp)

        else:

            def loss_fn(rng, **batch):
                lp = self._proposal_posterior_log_prob(
                    params, rng, n_atoms, jnp.array(batch["theta"]), batch["y"]
                )
                return -jnp.mean(lp)

        def body_fn(batch, rng_key):
            loss = jax.jit(loss_fn)(rng_key, **batch)
            return loss * (jax.tree.leaves(batch["y"])[0].shape[0] / val_iter.num_samples)

        loss = 0.0
        for b in range(val_iter.num_batch_per_epoch):
            batch = next(iter(val_iter))
            val_key, rng_key = jr.split(rng_key)
            loss += body_fn(batch, val_key)
        return loss

    @staticmethod
    def as_iterators(
        rng_key, data, batch_size, percentage_data_as_validation_set,
        noise_simulator
    ):
        """Convert the data set to an iterable for training.

        Args:
            rng_key: a jax random key
            data: a tuple with 'y' and 'theta' elements
            batch_size: the size of each batch
            percentage_data_as_validation_set: fraction

        Returns:
            two batch iterators
        """
        print("hooray")
        return as_batch_iterators(
            rng_key,
            data,
            batch_size,
            1.0 - percentage_data_as_validation_set,
            True,
            noise_simulator
        )

    def sample_posterior(
        self, rng_key, params, observable, *, n_samples=4_000, **kwargs
    ):
        r"""Sample from the approximate posterior.

        Args:
            rng_key: a jax random key
            params: a pytree of neural network parameters
            observable: observation to condition on
            n_samples: number of samples to draw

        Returns:
            returns an array of samples from the posterior distribution of
            dimension (n_samples \times p)
        """
        observable = jnp.atleast_2d(observable)

        thetas = None
        n_curr = n_samples
        n_total_simulations_round = 0
        _, unravel_fn = ravel_pytree(self.prior_sampler_fn(seed=jr.PRNGKey(1)))
        while n_curr > 0:
            n_sim = jnp.minimum(200, jnp.maximum(200, n_curr))
            n_total_simulations_round += n_sim
            sample_key, rng_key = jr.split(rng_key)
            proposal = self.model.apply(
                params,
                sample_key,
                method="sample",
                sample_shape=(n_sim,),
                x=jnp.tile(observable, [n_sim, 1]),
            )
            proposal_probs = self.prior_log_density_fn(
                jax.vmap(unravel_fn)(proposal)
            )
            proposal_accepted = proposal[jnp.isfinite(proposal_probs)]
            if thetas is None:
                thetas = proposal_accepted
            else:
                thetas = jnp.vstack([thetas, proposal_accepted])
            n_curr -= proposal_accepted.shape[0]
        self.n_total_simulations += n_total_simulations_round

        ess = float(thetas.shape[0] / n_total_simulations_round)
        thetas = jax.tree_map(
            lambda x: x.reshape(1, *x.shape),
            jax.vmap(unravel_fn)(thetas[:n_samples]),
        )
        inference_data = as_inference_data(thetas, jnp.squeeze(observable))
        return inference_data, ess



#TODO: SHOULD WE ADD A LAYERNORM TO THE INITIAL INPUTS OF THE FIRST NETWORK LAYER ??


# pylint: disable=too-many-arguments
def my_make_mdn(
    n_dimension: int,
    n_components: int,
    hidden_sizes: Iterable[int] = (64, 64),
    activation: Callable = jax.nn.swish,
    embedding_net: nn.Module = None,
):
    """Create a mixture density network

    Args:
        n_dimension: dimensionality of theta
        n_components: number of mixture components
        hidden_sizes: sizes of hidden layers for each normalizing flow. E.g.,
            when the hidden sizes are a tuple (64, 64), then each maf layer
            uses a MADE with two layers of size 64 each
        activation: a jax activation function

    Returns:
        a normalizing flow model
    """

    @hk.transform
    def mdn(method, **kwargs):

        #n = kwargs["x"].shape[0]
        #x = jnp.array(kwargs["x"])
        n = jax.tree.leaves(kwargs["x"])[0].shape[0] # all elements of data dict should be batched
        x = kwargs["x"]

        #print("x mdn input", x.shape)

        # optional embedding network
        if embedding_net is not None:
          mod = hkflax.lift(embedding_net, name='flax_embedding_net')
          # add in a vmap here
          x = jax.vmap(mod)(x)

        xembed = x
        # rest of network
        hidden = hk.nets.MLP(
            hidden_sizes, activation=activation, activate_final=True
        )(x)
        #print("hidden", hidden.shape)
        logits = hk.Linear(n_components)(hidden)
        mu_sigma = hk.Linear(n_components * n_dimension * 2)(hidden)
        mu, sigma = jnp.split(mu_sigma, 2, axis=-1)

        mixture = tfd.MixtureSameFamily(
            tfd.Categorical(logits=logits),
            tfd.MultivariateNormalDiag(
                mu.reshape(n, n_components, n_dimension),
                nn.softplus(sigma.reshape(n, n_components, n_dimension)),
            ),
        )
        if method == "sample":
            return mixture.sample(seed=hk.next_rng_key())
        elif method == "embedding":
            # could also return logits, mu_sigma
            return xembed
        else:
            return mixture.log_prob(kwargs["y"])

    return mdn

