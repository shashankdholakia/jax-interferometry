import jax.numpy as jnp
from jaxopt import Bisection
import matplotlib.pyplot as plt
from jax.lax import cond
import jax
from jax import grad, jit, vmap
from zodiax import Base
from scipy.spatial import ConvexHull
import numpy as np

if __name__=="__main__":
    import utils
else:
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
        tol = 1e-4
        thetas = jnp.linspace(tol,jnp.pi-tol,N)
        self.thetas=thetas
        rtws = jnp.ones_like(thetas)
        self.thetas = thetas
        ns = utils.closest_polygon(thetas)
        self.n = ns
        phi = jnp.concatenate([jnp.linspace(tol,2*jnp.pi-tol,n, endpoint=False) for n in ns])
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

        #compute the x, y, z coordinates of each barycenter (barycenter vector)
        barycenter = utils.barycenter(points_rotated,self.triangulation)
        #find the intensity of the star at each barycenter (mean of the intensity at the corners of the triangle)
        intensity = jnp.mean(F[self.triangulation], axis=1)
        #temperature = jnp.mean(T[self.triangulation],axis=1)
        # compute the areas of the triangles
        areas = utils.triangle_area(points_rotated, self.triangulation)
        cosine = jnp.dot(jnp.array([0,0,1]),normals.T)
        #apply a step function weight along with the contribution of flux towards the observer
        #should 0 the intensities in the non-visible portion of the star (behind the star)
        weight = jnp.heaviside(cosine,0)*cosine

        #THIS IS A REALLY EXPENSIVE OPERATION TO PERFORM EVERY TIME
        dftm = compute_DFTM1(barycenter[:,0], barycenter[:,1], self.uv, self.wavel)
        ft = apply_DFTM1(intensity*weight,dftm)
        return jnp.abs(ft)**2
    
    def plot(self,omega, r_eq, inc, obl, ax=None):
        
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

        #compute the barycenter vector (x,y,z of each barycenter)
        barycenter = utils.barycenter(points_rotated,self.triangulation)
        #take the intensity at each barycenter (mean of the values at each triangle)
        #same for temperature
        intensity = jnp.mean(F[self.triangulation], axis=1)
        temperature = jnp.mean(T[self.triangulation],axis=1)
        # compute the areas of the triangles
        areas = utils.triangle_area(points_rotated, self.triangulation)
        #take the contribution towards the observer
        cosine = jnp.dot(jnp.array([0,0,1]),normals.T)
        #apply a step function weight such that the behind of the star is 0d in intensity
        mask = jnp.where(normals[:,2] > 0, True, False)
        colors = (intensity[mask]).astype(np.float32)
        barycenters_projected = jnp.delete(barycenter, 2, axis=1)
        if ax is None:
            c = plt.tripcolor(points_rotated[:,0], points_rotated[:,1],triangles=self.triangulation[mask], facecolors=colors,edgecolors='k');
            plt.colorbar(c)
            plt.plot(barycenters_projected[mask,0], barycenters_projected[mask,1], 'ko',ms=0.2);
            plt.gca().set_aspect('equal')
        else:
            c = ax.tripcolor(points_rotated[:,0], points_rotated[:,1],triangles=self.triangulation[mask], facecolors=colors,edgecolors='k', cmap='plasma',lw=0.3);
            ax.set_xlim([-r_eq*1.1,r_eq*1.1])
            ax.set_ylim([-r_eq*1.1,r_eq*1.1])
            
            def colorbar(mappable):
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                import matplotlib.pyplot as plt
                last_axes = plt.gca()
                ax = mappable.axes
                fig = ax.figure
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(mappable, cax=cax)
                plt.sca(last_axes)
                return cbar
            
            colorbar(c)
            #ax.plot(barycenters_projected[mask,0], barycenters_projected[mask,1], 'ko',ms=0.2);
            ax.set_aspect('equal')
    



if __name__=="__main__":
    theta = jnp.pi/4
    #theta = 0.01
    omega = 0.8
    rtws, Ts, Fs = solve_ELR(omega, theta)
    r = y(theta, omega)
    
    print(rtws)
    print(r)