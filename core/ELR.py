import jax.numpy as jnp
from jaxopt import Bisection
import matplotlib.pyplot as plt
from jax.lax import cond
import jax
from jax import grad, jit, vmap
from zodiax import Base
from scipy.spatial import ConvexHull

from . import utils

from jax.config import config

config.update("jax_enable_x64", True)

"""
Provide a vectorized and jaxed version of the ELR model for speed
"""
#constants
G=6.67428e-8
sigma=5.67040e-5
f23=2.0/3.0
Lsun=3.8418e33
Msun=1.989e33
Rsun=6.96e10

#ELR11 equations
#gives the value of phi
def eq24(phi,theta,omega,rtw):
    tau = (jnp.power(omega,2) * jnp.power(rtw*jnp.cos(theta),3) )/3.0 + jnp.cos(theta) + jnp.log(jnp.tan(0.5*theta))
    return jnp.cos(phi) + jnp.log(jnp.tan(0.5*phi)) - tau

#solve for rtw given omega
def eq30(rtw,theta,omega):
    #jaxed and vectorized
    return (1./omega**2)*(1./rtw - 1.0) + 0.5*((rtw*jnp.sin(theta))**2 - 1.0)

#ratio of equatorial to polar Teff
def eq32(omega):
    #jaxed and vectorized
    return jnp.sqrt(2./(2.+omega**2)) * (1. - omega**2)**(1./12.) * jnp.exp(-(4./3.)* omega**2/(2+omega**2)**3)

@jit
def solve_ELR(omega, theta): #eq.26, 27, 28; solve the ELR11 equations
    """

    Takes a float omega where 0<=omega<1
    and a single value theta, or polar angles in radians

    calculates r~, Teff_ratio, and Flux_ratio

    Can be vmapped to solve for an array of thetas (done below)

    """
    #theta is the polar angle.
    #this routine calculates values for 0 <= theta <= pi/2
    #everything else is mapped into this interval by symmetry
    # theta = 0 at the pole(s)
    # theta = pi/2 at the equator
    # -pi/2 < theta < 0: theta -> abs(theta)
    #  pi/2 > theta > pi: theta -> pi - theta
    theta = jnp.where(
            jnp.logical_and(theta>jnp.pi/2, theta<=jnp.pi), #if
            jnp.pi - theta, #then
            theta #else
            )

    theta = jnp.where(
            jnp.logical_and(theta>=-jnp.pi/2, theta<0),
            jnp.abs(theta),
            theta
        )



    #first we solve equation 30 for rtw
    q = Bisection(optimality_fun=eq30, lower=0,upper=1,check_bracket=False, tol=1e-14, maxiter=101)
    rtw = q.run(**{"theta":theta, "omega":omega}).params

    #the following are special solutions for extreme values of theta
    w2r3=omega**2*rtw**3

    q = Bisection(optimality_fun=eq24, lower=0,upper=jnp.pi/2-1e-10,check_bracket=False,tol=1e-14,maxiter=101)

    Fw = jnp.where(
            theta==0, #if
            jnp.exp( f23 * w2r3 ), #then

            jnp.where(
                    theta==0.5*jnp.pi, #elsif
                    (1.0 - w2r3)**(-f23), #then

                    #very cumbersome way of computing the else case without a cond, see the unvectorized version for clarity
                    (jnp.tan(q.run(theta, **{"theta":theta, "omega":omega, "rtw":rtw}).params) / jnp.tan(theta))**2
                                    )
                            )

    #equation 31 and similar for Fw
    term1 = rtw**(-4)
    term2 = omega**4*(rtw*jnp.sin(theta))**2
    term3 = -2*(omega*jnp.sin(theta))**2/rtw
    gterm = jnp.sqrt(term1+term2+term3)
    Flux_ratio = Fw*gterm
    Teff_ratio = Flux_ratio**0.25
    return rtw, Teff_ratio, Flux_ratio



solve_ELR_vec = jax.vmap(solve_ELR, in_axes=[None,0])

@jit
def compute_DFTM1(x,y,uv,wavel):
    '''
    Compute a direct Fourier transform matrix, from coordinates x and y
    (milliarcsec) to uv (metres) at a given wavelength wavel.
    '''

    # Convert to radians
    x = x * jnp.pi / 180.0 / 3600.0/ 1000.0
    y = y * jnp.pi / 180.0 / 3600.0/ 1000.0

    # get uv in nondimensional units
    uv = uv / wavel

    # Compute the matrix
    dftm = jnp.exp(-2j* jnp.pi* (jnp.outer(uv[:,0],x)+jnp.outer(uv[:,1],y)))

    return dftm

@jit
def apply_DFTM1(image,dftm):
    '''Apply a direct Fourier transform matrix to an image.'''
    image /= image.sum()
    return jnp.dot(dftm,image.ravel())


array = jnp.ndarray
class ELR_Model(Base):
    """
    Models a star currently as a monochromatic uniform disk but takes into account
    the Espinosa Lara Rieutord model for an oblate, rapidly rotating star. This
    model calculates oblateness and gravity darkening without an added parameter
    beta
    

    TODO:

    NEED to take into account: 4 parameter/linear limb darkening from Claret

    Refactor to have multiple functions to extract different observables;
    interferometric v^2 but also potentially broadband or color photometry,
    line profiles or (with model atmospheres), even polarization

    POSSIBLY: Interpolated model spectra
    """
    N: int
    uv: array
    wavel: float
    thetas: array #length N
    phi:array

    n: array #length N

    triangulation: array



    def __init__(self,N,uv, wavel):
        """

        Args:
            N (int): Number of latitudes on the stellar grid
            uv (array): UV grid in meters
            wavel (float): wavelength in meters
        """
        self.N =  N
        self.uv = uv
        self.wavel = wavel

        thetas = jnp.linspace(0,jnp.pi,N)
        self.thetas=thetas
        rtws = jnp.ones_like(thetas)
        self.thetas = thetas
        ns = utils.closest_polygon(thetas)
        self.n = ns
        phi = jnp.concatenate([jnp.linspace(0,2*jnp.pi,n, endpoint=False) for n in ns])
        self.phi=phi
        rtw = rtws.repeat(ns)
        theta = thetas.repeat(ns)
        x, y, z = utils.spherical_to_cartesian(rtw,theta,phi)
        points = jnp.stack([x,y,z],axis=1)

        hull = ConvexHull(points)
        self.triangulation = hull.simplices

    def __call__(self,omega, r_eq, inc, obl):
        rtws, Ts, Fs = solve_ELR_vec(omega, self.thetas)

        rtw = rtws.repeat(self.n)
        T = Ts.repeat(self.n)
        F = Fs.repeat(self.n)
        theta = self.thetas.repeat(self.n)

        x, y, z = utils.spherical_to_cartesian(rtw,theta,self.phi)
        points = r_eq*jnp.stack([x,y,z],axis=1)

        points_rotated = utils.rotate_point_cloud(points, -inc, obl)
        # compute the normal vectors
        normals = utils.triangle_normals(points_rotated, self.triangulation)

        barycenter = utils.barycenter(points_rotated,self.triangulation)
        intensity = jnp.mean(F[self.triangulation], axis=1)
        #temperature = jnp.mean(T[self.triangulation],axis=1)
        # compute the areas of the triangles
        areas = utils.triangle_area(points_rotated, self.triangulation)
        cosine = jnp.dot(jnp.array([0,0,1]),normals.T)
        weight = jnp.heaviside(cosine,0)*cosine

        #THIS IS A REALLY EXPENSIVE OPERATION TO PERFORM EVERY TIME
        dftm = compute_DFTM1(barycenter[:,0], barycenter[:,1], self.uv, self.wavel)
        ft = apply_DFTM1(intensity*weight,dftm)
        return ft


if __name__=="__main__":
    thetas = jnp.linspace(0.0, jnp.pi/2,100)
    #theta = 0.01
    omega = 0.8
    rtws, Ts, Fs = solve_ELR_vec(omega, self.theta)

    print(Fs)
    print(rtws)
    print(thetas)
