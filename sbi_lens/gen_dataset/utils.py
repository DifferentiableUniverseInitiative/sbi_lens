from numpyro.handlers import seed, trace, condition
import jax


def get_samples_and_scores(model,
                           key,
                           batch_size=64,
                           score_type=None,
                           thetas=None,
                           with_noise=True):
    """
    Handling function sampling and computing the score from the model.

    model: a numpyro model.
    key: jax random seed.
    batch_size: size of the batch to sample.
    score_type: 'density' for nabla_theta log p(theta | y, z) or 
                'conditional' for nabla_theta log p(y | z, theta), default is 'density'.
    thetas: thetas used to sample simulations or 
            'None' sample thetas from the model, default is 'None'.
    with_noise: add noise in simulations, default is 'True'. 
                note: if no noise the score is only nabla_theta log p(theta, z)
                      and log_prob log p(theta, z)
        
    returns: (log_prob, sample), score
    """

    def log_prob_fn(theta, key):
        cond_model = condition(model, {'theta': theta})
        cond_model = seed(cond_model, key)
        model_trace = trace(cond_model).get_trace()

        sample = {
            'theta': model_trace['theta']['value'],
            'y': model_trace['y']['value']
        }

        if score_type == 'density':
            logp = model_trace['theta']['fn'].log_prob(
                model_trace['theta']['value']).sum()
        elif score_type == 'conditional':
            logp = 0

        if with_noise == True:
            logp += model_trace['y']['fn'].log_prob(
                jax.lax.stop_gradient(model_trace['y']['value'])).sum()

        del model_trace['theta']
        del model_trace['y']

        for i in range(len(model_trace)):
            key, val = list(model_trace.items())[i]
            logp += val['fn'].log_prob(val['value']).sum()

        return logp, sample

    # Split the key by batch
    keys = jax.random.split(key, batch_size)

    # Sample theta from the model
    if thetas is None:
        thetas = jax.vmap(lambda k: trace(seed(model, k)).get_trace()['theta'][
            'value'])(keys)

    return jax.vmap(jax.value_and_grad(log_prob_fn, has_aux=True))(thetas,
                                                                   keys)
