import numpy as np
import matplotlib.pyplot as plt
#this is the way I've found to get jax to use multiple CPU cores on Apple Silicon
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

# add the root directory to path to allow import of the uniform disk model
import sys
from .ELR import ELR_Model

import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev

from numpyro.contrib.nested_sampling import NestedSampler
from jaxns.prior_transforms.prior import UniformBase
import tensorflow_probability
import jaxns

import numpyro
from numpyro import distributions as dist, infer
from numpy import genfromtxt
import pandas as pd
import arviz as az
import corner

try:
    from colorpy import ciexyz
    from colorpy import colormodels
    colorpy_exists=True
except:
    colorpy_exists=False
    from matplotlib import cm
        
class FitELR:
    def __init__(self,l1l2res, hdnum, outputs, cpus=2, method="NUTS"):
        """

        Args:
            l1l2res (str): file path for the reduced data from PAVO (should be called something_l0l1.res_l1l2.res, meaning it's been through both stages of pipeline calibration)
            HDnum (str): HD number of the fil in the format, e.g "HD_84999"
            outputs (str): folder where to put all the output plots and files
            cpus (int): how many cpus to use, defaults to 2
            method (str, optional): Fitting method to use, whether gradient based (NUTS), or nested sampling ("NS"). Defaults to "NUTS".
        """
        
        df = pd.read_csv(l1l2res,delim_whitespace=True,comment='#',header=0)
        #remove bad data (negative errors, above 1 or below 0 visibility)
        df = df[(df.V2CALERR>0.0) & (df.V2CAL<1.0) & (df.V2CAL>0.0)]
        self.u = df[df['STAR']==hdnum]['UCOORDS'].values
        self.v = df[df['STAR']==hdnum]['VCCORDS'].values
        self.wav = df[df['STAR']==hdnum]['LAMBDA'].values
        self.v2 = df[df['STAR']==hdnum]['V2CAL'].values
        self.v2_err = df[df['STAR']==hdnum]['V2CALERR'].values
        
        self.l1l2res = l1l2res
        self.hdnum = hdnum
        if os.path.exists(outputs):
            self.outputs = outputs
        else:
            os.mkdir(outputs)
            self.outputs = outputs
        self.method = method
        self.cpus = cpus
        

        
    def fit(self,max_diam):
        """Fits a given CHARA PAVO reduced file using either NUTS (gradient based) or nested sampling (using jaxns, slower on average)

        Args:
            max_diam (float): maximum diameter to fit for, prevents second or third lobes from being fit to first lobe of visibility
        """
        uvgrid = jnp.vstack([self.u,self.v]).T
        wavels = jnp.vstack([self.wav*1e-6,self.wav*1e-6]).T
        rr = ELR_Model(32,uvgrid,wavels)
        
        #define the projected baselines 
        x = jnp.hypot(self.u/(self.wav*1e-6),self.v/(self.wav*1e-6))
        #baseline angles
        theta = jnp.arctan(self.v/self.u)
        
        #define y and yerr
        y = self.v2
        yerr = self.v2_err

        numpyro.set_host_device_count(self.cpus)

        #helper function to build the model
        def model(yerr, y=None):
            # These are the parameters that we're fitting and we're required to define explicit
            # priors using distributions from the numpyro.distributions module.
            diam = numpyro.sample("diam", dist.Uniform(0.0001, max_diam))
            omega = numpyro.sample("omega", dist.Uniform(0.0, 0.99))
            inc = numpyro.sample("inc", dist.Uniform(0, jnp.pi/2))
            numpyro.factor('isotropy', jnp.log(jnp.cos(inc)))
            jitter = numpyro.sample("logsig", dist.Normal(loc=jnp.log(0.01),scale=5.0))
            obl = numpyro.sample("obl", dist.Uniform(-jnp.pi, jnp.pi))
            numpyro.sample("y", dist.Normal(rr(omega,diam/2,inc,obl), jnp.sqrt(yerr**2 + jnp.exp(jitter)**2)), obs=y)

        if self.method=='NUTS':
            sampler = infer.MCMC(
                infer.NUTS(model),
                num_warmup=2000,
                num_samples=2500,
                num_chains=2,
                progress_bar=True)

            sampler.run(jax.random.PRNGKey(0), yerr, y=y)
            inf_data = az.from_numpyro(sampler)
            inf_data.to_netcdf(os.path.join(self.outputs, (self.hdnum+'_'+self.method+'.h5')))
        elif self.method=='NS':
            ns = NestedSampler(model)
            ns.run(jax.random.PRNGKey(0), yerr, y=y)
            
            ns_samples = ns.get_samples(jax.random.PRNGKey(2), num_samples=10000)
            inf_data = az.from_dict(ns_samples)
            inf_data.to_netcdf(os.path.join(self.outputs, self.hdnum+'_'+self.method+'.h5'))
            
    def create_plots(self):
        
        #first create the baselines plot
        uvgrid = jnp.vstack([self.u,self.v]).T
        wavels = jnp.vstack([self.wav*1e-6,self.wav*1e-6]).T
        rr = ELR_Model(32,uvgrid,wavels)
        
        #define the projected baselines 
        x = jnp.hypot(self.u/(self.wav*1e-6),self.v/(self.wav*1e-6))
        #baseline angles
        theta = jnp.arctan(self.v/self.u)
        
        if colorpy_exists:
            irgb = []
            for i in self.wav:
                xyz = ciexyz.xyz_from_wavelength(i*1000)
                irgb.append(colormodels.irgb_from_xyz(xyz))
            irgb = np.array(irgb)/255
        else:
            irgb = cm.jet(self.wav)
            
        plt.scatter(self.u/(self.wav*1e-6),self.v/(self.wav*1e-6), c=irgb)
        plt.xlabel(r"U (baseline/$\lambda$)")
        plt.ylabel(r"V (baseline/$\lambda$)")
        plt.savefig(os.path.join(self.outputs, self.hdnum+'_baselines.png'), dpi=300)
        
        #get the posterior samples (should be agnostic to method)
        inf_data = az.from_netcdf(os.path.join(self.outputs, self.hdnum+'_'+self.method+'.h5'))
        
        #save the HMC trace if the method is NUTS, otherwise this doesn't make sense
        if self.method=='NUTS':
            az.plot_trace(inf_data, var_names=('diam','omega','inc','obl', 'logsig'));
            plt.subplots_adjust(hspace=0.5,wspace=0.5)
            plt.savefig(os.path.join(self.outputs, self.hdnum+'_trace.png'), dpi=300)
        
        
        #save the plot of the visibility squared overplotted with the final model
        cmap = plt.get_cmap("hsv")
        cmap2 = plt.get_cmap("viridis")

        #get the median posterior samples
        diam = np.median(inf_data.posterior.diam.values)
        inc = np.median(inf_data.posterior.inc.values)
        obl = np.median(inf_data.posterior.obl.values)
        omega = np.median(inf_data.posterior.omega.values)
        jitter = np.exp(np.median(inf_data.posterior.logsig.values))

        x = jnp.hypot(self.u/(self.wav*1e-6),self.v/(self.wav*1e-6))
        theta = jnp.arctan(self.v/self.u)
        y = self.v2
        #add the errors in quadrature
        yerr = np.sqrt(self.v2_err**2+jitter**2)

        #get the median model
        u0 = np.linspace(self.u.min()/(np.mean(self.wav*1e-6)),self.u.max()/(np.mean(self.wav*1e-6)),64)
        v0 = np.linspace(self.v.min()/(np.mean(self.wav*1e-6)),self.v.max()/(np.mean(self.wav*1e-6)),64)
        uu, vv = np.meshgrid(u0,v0)
        uv0 = np.vstack((uu.flatten(),vv.flatten())).T
        x0 = jnp.hypot(uu,vv).flatten()
        theta0 = jnp.arctan2(vv.flatten(),uu.flatten())
        rr0 = ELR_Model(31,uv0,1.0)
        y0 = rr0(omega,diam/2.0,inc,obl)

        y0s = []
        diams = np.random.choice(np.concatenate(inf_data.posterior.diam.values), 10)
        incs = np.random.choice(np.concatenate(inf_data.posterior.inc.values), 10)
        obls = np.random.choice(np.concatenate(inf_data.posterior.obl.values), 10)
        omegas = np.random.choice(np.concatenate(inf_data.posterior.omega.values), 10)

        for diam,inc,obl,omega  in zip(diams,incs,obls,omegas):
            y0s.append(rr0(omega,diam/2.0,inc,obl))

        def plot_data(x,y, theta, x0, y0, yerr, y0s, wavels=self.wav*1e-6, ax=None, alpha=1):
            if ax is None:
                fig, ax = plt.subplots(1,2, figsize=(20,5), sharey=True)
            ax[0].errorbar(x, y, yerr=yerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
            
        #     if wavels is not None:
            ax[0].scatter(x, y, marker="s", s=30, edgecolor="k", zorder=1000, c=cmap(theta))
        #     else:
        #         ax[0].scatter(x, y, marker="s", s=30, edgecolor="k", zorder=1000, c=cmap(x))
            inds = np.argsort(x0)
            ax[0].plot(x0[inds], y0[inds], color="k", lw=1.5, alpha=alpha)
            ax[0].set_xlabel("baseline/$\lambda$", fontsize=20)
            ax[0].set_ylabel("$V^2$", fontsize=20)
            ax[0].set_ylim(0, 1.2)
                
            
            for i in y0s:
                inds = np.argsort(x0)
                ax[0].plot(x0[inds], i[inds], "k", alpha=0.1)
            
            ax[1].errorbar(theta, y, yerr=yerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)        
            ax[1].scatter(theta, y, marker="s", s=30, edgecolor="k", zorder=1000)
            ax[1].set_xlabel(r"$\theta$", fontsize=20)
            ax[1].set_ylabel("$V^2$", fontsize=20)
            ax[1].set_ylim(0, 1.2)
            
            return ax


        ax = plot_data(x,y,theta,x0,y0,yerr, y0s)
        plt.savefig(os.path.join(self.outputs, self.hdnum+'_v2.png'),dpi=300)
        
        corner.corner(inf_data, var_names=('omega','diam','inc','obl','logsig'))
        plt.savefig(os.path.join(self.outputs, self.hdnum+'_corner.png'),dpi=300)


if __name__=="__main__":
    """
    Run from inside the interferometry folder using:
    python -m core.elr_fit
    """
    upsuma = FitELR("/Users/uqsdhola/Projects/Interferometry/tests/upsUMa_l0l1.res_l1l2.res", 'HD_84999', 'upsUma', cpus=2, method="NS")
    #upsuma.fit(2.0)
    upsuma.create_plots()
    