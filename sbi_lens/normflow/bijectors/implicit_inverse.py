# This module is to store our implicit inverse functions
import jax
from jax import lax
import jax.numpy as jnp
from functools import partial
from jaxopt.linear_solve import solve_normal_cg
from jaxopt import Bisection

__all__ = ["make_inverse_fn"]


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def root_bisection(f, params):
  """
  f: optimality fn with input arg (params, x)
  """
  bisec = Bisection(optimality_fun=f, lower=0.0, upper=1., 
                    check_bracket=False, maxiter=100, tol=1e-06)
  return bisec.run(None, params).params

def root_bisection_fwd(f, params):
  z_star = root_bisection(f, params)
  return z_star, (params, z_star)

def root_bwd(f, res, z_star_bar):
  params, z_star = res
  _, vjp_a = jax.vjp(lambda p: f(z_star, p), params)
  _, vjp_z = jax.vjp(lambda z: f(z, params), z_star)
  return vjp_a(solve_normal_cg(lambda u: vjp_z(u)[0], - z_star_bar))

root_bisection.defvjp(root_bisection_fwd, root_bwd)


def make_inverse_fn(f):
  """ Defines the inverse of the input function, and provides implicit gradients
  of the inverse.

  Args:
    f: callable of input shape (params, x)
  Retuns:
    inv_f: callable of with args (params, y)
  """
  def inv_fn(params, y):
    def optimality_fn(x, params):
      p, y = params
      return f(p, x) - y
    return root_bisection(optimality_fn, [params, y])
  return inv_fn