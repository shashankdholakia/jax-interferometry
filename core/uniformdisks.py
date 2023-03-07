import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

from jax.config import config
config.update("jax_enable_x64", True)

mas2rad = jnp.pi / 180.0 / 3600.0/ 1000.0

def j1(x):
    return jax.scipy.special.bessel_jn(x,v=1,n_iter=30)[1]

def j0(x):
    return jax.scipy.special.bessel_jn(x,v=0,n_iter=50)[0]

# @jit
@vmap
def jinc(x):
    dummy = 2*(j1(x)/x)
    return dummy
    # return jax.lax.select(~jnp.isfinite(dummy), 1., dummy)

def vis_gauss(d,u,v):
    d = mas2rad*d
    return jnp.exp(-(jnp.pi*d*jnp.sqrt(u**2+v**2))**2/4./jnp.log(2))

def vis_ud(d,u,v):
    """
    Takes: star diameter in milliarcseconds, u and v in baseline/wavelength
    returns: visibility
    """
    r = jnp.sqrt(u**2+v**2)
    diam = d*mas2rad
    t = jinc(jnp.pi*diam*r)
    return t
