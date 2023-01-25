import jax
import jax.numpy as jnp


def loss_vmim(nf, compressor, params, mu, batch, state_resnet):
    y, opt_state_resnet = compressor.apply(params, state_resnet, None,
                                           batch.reshape([-1, 128, 128, 1]))
    log_prob = jax.vmap(lambda theta, x: nf.apply(params, theta.reshape(
        [1, 2]), x.reshape([1, 2])).squeeze())(mu, y)
    return -jnp.mean(log_prob), opt_state_resnet


def loss_fn(nvp_nd, compressor, params, parameters_compressor,
            opt_state_resnet, weight, mu, batch, score):
    y, _ = compressor.apply(parameters_compressor, opt_state_resnet, None,
                            batch.reshape([-1, 128, 128, 1]))
    log_prob, out = jax.vmap(
        jax.value_and_grad(lambda theta, x: nvp_nd.apply(
            params, theta.reshape([1, 2]), x.reshape([1, 2])).squeeze()))(mu,
                                                                          y)
    return -jnp.mean(log_prob) + weight * jnp.mean(
        jnp.sum((out - score)**2, axis=1))
