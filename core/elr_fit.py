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
            l1l2res (str): file path for the reduced data from PAVO (should be called something_l1l2.csv, meaning it's been through both stages of pipeline calibration)
            HDnum (str): HD number of the fil in the format, e.g "HD_84999"
            outputs (str): folder where to put all the output plots and files
            cpus (int): how many cpus to use, defaults to 2
            method (str, optional): Fitting method to use, whether gradient based (NUTS), or nested sampling ("NS"). Defaults to "NUTS".
        """
        
        df = pd.read_csv(l1l2res)
        #remove bad data (negative errors, above 1 or below 0 visibility)
        df = df[(df.cal_v2sig>0.0) & (df.cal_v2<1.0) & (df.cal_v2>0.0)]
        self.u = df[df['Star']==hdnum]['u'].values*1e-6
        self.v = df[df['Star']==hdnum]['v'].values*1e-6
        self.wav = df[df['Star']==hdnum]['wl'].values
        self.v2 = df[df['Star']==hdnum]['cal_v2'].values
        self.v2_err = df[df['Star']==hdnum]['cal_v2sig'].values
        
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
        rr = ELR_Model(33,uvgrid,wavels)
        
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
            jitter = numpyro.sample("logsig", dist.Normal(loc=jnp.log(0.001),scale=3.0))
            #obliquities rotated by 180 degrees are analytically degenerate without phases
            obl = numpyro.sample("obl", dist.Uniform(0, jnp.pi))
            numpyro.sample("y", dist.Normal(rr(omega,diam/2,inc,obl), jnp.sqrt(yerr**2 + jnp.exp(jitter)**2)), obs=y)

        if self.method=='NUTS':
            sampler = infer.MCMC(
                infer.NUTS(model),
                num_warmup=2000,
                num_samples=2500,
                num_chains=2,
                progress_bar=True)

            sampler.run(jax.random.PRNGKey(1), yerr, y=y)
            inf_data = az.from_numpyro(sampler)
            inf_data.to_netcdf(os.path.join(self.outputs, (self.hdnum+'_'+self.method+'.h5')))
        elif self.method=='NS':
            ns = NestedSampler(model)
            ns.run(jax.random.PRNGKey(0), yerr, y=y)
            
            ns_samples = ns.get_samples(jax.random.PRNGKey(2), num_samples=10000)
            inf_data = az.from_dict(ns_samples)
            inf_data.to_netcdf(os.path.join(self.outputs, self.hdnum+'_'+self.method+'.h5'))
            
    def create_plots(self, star_name):
        
        #first create the baselines plot
        uvgrid = jnp.vstack([self.u,self.v]).T
        wavels = jnp.vstack([self.wav*1e-6,self.wav*1e-6]).T
        rr = ELR_Model(33,uvgrid,wavels)
        
        #get the posterior samples (should be agnostic to method)
        inf_data = az.from_netcdf(os.path.join(self.outputs, self.hdnum+'_'+self.method+'.h5'))
        
        #get the median posterior samples
        diam = np.median(inf_data.posterior.diam.values)
        inc = np.median(inf_data.posterior.inc.values)
        obl = np.median(inf_data.posterior.obl.values)
        omega = np.median(inf_data.posterior.omega.values)
        jitter = np.exp(np.median(inf_data.posterior.logsig.values))
        
        print("Median diameter is " + str(diam))
        print("Median inclination is " + str(np.degrees(inc)))
        print("Median obliquity is " + str(np.degrees(obl)))
        print("Median omega is " + str(omega))
        print("Median jitter is " + str(jitter))
        
        plt.figure()
        rr.plot(omega, diam/2, inc, obl)
        plt.suptitle(star_name, fontsize=24)
        plt.savefig(os.path.join(self.outputs, self.hdnum+'_star.png'), dpi=300)
        
        #define the projected baselines 
        x = jnp.hypot(self.u/(self.wav*1e-6),self.v/(self.wav*1e-6))
        #baseline angles
        theta = jnp.arctan2(self.v,self.u)
        
        plt.clf()
        median_model = rr(omega, diam/2, inc, obl)
        plt.errorbar(self.v2, median_model, yerr=jnp.sqrt(self.v2_err**2), fmt="ok", ms=1, capsize=0, lw=1)
        plt.plot([0,1.0],[0,1.0])
        plt.suptitle(star_name, fontsize=24)
        plt.savefig(os.path.join(self.outputs, self.hdnum+'_corr.png'), dpi=300)
        
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
        
        #save the HMC trace if the method is NUTS, otherwise this doesn't make sense
        if self.method=='NUTS':
            az.plot_trace(inf_data, var_names=('diam','omega','inc','obl', 'logsig'));
            plt.subplots_adjust(hspace=0.5,wspace=0.5)
            plt.savefig(os.path.join(self.outputs, self.hdnum+'_trace.png'), dpi=300)
        
        
        #save the plot of the visibility squared overplotted with the final model
        cmap = plt.get_cmap("hsv")
        #cmap2 = plt.get_cmap("viridis")


        x = jnp.hypot(self.u/(self.wav*1e-6),self.v/(self.wav*1e-6))
        theta = jnp.arctan2(self.v,self.u)
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
                fig, ax = plt.subplots(1,2, figsize=(15,5),gridspec_kw={'width_ratios':[1,2]})
            ax[1].errorbar(x, y, yerr=yerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
            
        #     if wavels is not None:
            c = ax[1].scatter(x, y, marker="s", s=30, edgecolor="k", zorder=1000, c=theta, cmap='twilight')
        #     else:
        #         ax[0].scatter(x, y, marker="s", s=30, edgecolor="k", zorder=1000, c=cmap(x))
            inds = np.argsort(x0)
            ax[1].plot(x0[inds], y0[inds], color="k", lw=1.5, alpha=alpha)
            ax[1].set_xlabel("baseline/$\lambda$", fontsize=20)
            ax[1].set_ylabel("$V^2$", fontsize=20)
            ax[1].set_ylim(0, 1.2)
            cbar = fig.colorbar(c,ax=ax[1])
            cbar.set_label(r'$\theta$')
            
            for i in y0s:
                inds = np.argsort(x0)
                ax[1].plot(x0[inds], i[inds], "k", alpha=0.1)
            
            #ax[1].errorbar(theta, y, yerr=yerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)        
            #ax[1].scatter(theta, y, marker="s", s=30, edgecolor="k", zorder=1000)
            #ax[1].set_xlabel(r"$\theta$", fontsize=20)
            #ax[1].set_ylabel("$V^2$", fontsize=20)
            #ax[1].set_ylim(0, 1.2)
            
            return fig,ax

        #get the median posterior samples
        diam = np.median(inf_data.posterior.diam.values)
        inc = np.median(inf_data.posterior.inc.values)
        obl = np.median(inf_data.posterior.obl.values)
        omega = np.median(inf_data.posterior.omega.values)
        
        fig,ax = plot_data(x,y,theta,x0,y0,yerr, y0s)
        
        rr.plot(omega, diam/2, inc, obl, ax=ax[0])
        ax[0].set_xlabel("X (mas)")
        ax[0].set_ylabel("Y (mas)")

        plt.suptitle(star_name, fontsize=24)
        plt.savefig(os.path.join(self.outputs, self.hdnum+'_v2_2.pdf'))
        
        corner.corner(inf_data, var_names=('omega','diam','inc','obl','logsig'))
        plt.suptitle(star_name, fontsize=24)
        plt.savefig(os.path.join(self.outputs, self.hdnum+'_corner.png'),dpi=300)
        
        


if __name__=="__main__":
    """
    Run from inside the interferometry folder using:
    python -m core.elr_fit
    """
    #upstau  = FitELR("/Users/uqsdhola/Projects/Interferometry/data/upsTau/pavlist_l1l2.csv", 'HD_28024', 'upsTau', cpus=2, method="NUTS")
    #upstau.fit(1.5)
    #upstau.create_plots(star_name=r'$\upsilon$ Tau')
    
    upsuma = FitELR("/Users/uqsdhola/Projects/Interferometry/data/upsUMa/pavlist_l1l2.csv", 'HD_84999', 'upsUMa', cpus=2, method="NUTS")
    upsuma.fit(1.8)
    upsuma.create_plots(star_name=r'$\upsilon$ UMa')
    
    #epscep = FitELR("/Users/uqsdhola/Projects/Interferometry/data/epsCep/pavlist_l1l2.csv", 'HD_211336', 'epsCep', cpus=2, method="NUTS")
    #epscep.fit(2.0)
    #epscep.create_plots(star_name=r'$\epsilon$ Cep')
    
    #lamboo  = FitELR("/Users/uqsdhola/Projects/Interferometry/data/lamBoo/pavlist_l1l2.csv", 'HD_125162', 'lamBoo', cpus=2, method="NUTS")
    #lamboo.fit(2.0)
    #lamboo.create_plots(star_name=r'$\lambda$ Boo')  