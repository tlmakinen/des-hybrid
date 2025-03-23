import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist

import numpy as np
import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior

from chainconsumer import Chain, ChainConsumer, make_sample, Truth
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import cloudpickle as pickle

import argparse,os,sys

sys.path.append("/home/makinen/repositories/des-hybrid/")


from density_estimation.affine_sample import *



#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print('Device:', device)


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def mask_prior_and_summaries(theta, summs, low, high):
    condition = (~torch.any(theta.lt(low), keepdim=True, dim=-1)) & (~torch.any(theta.gt(high), keepdim=True, dim=-1))
    mask = condition.squeeze()
    return theta.float()[mask], summs.float()[mask], mask


def ret_scaled_param(data,param):
    s = 0.6
    if param == 'AIA':
        hi = 3
        lo = -3

    if param == "s8":
        hi = 1.32
        lo = 0.4

    if param == 'om':
        hi = 0.5
        lo = 0.13

    if param == "h":
        hi = 0.6
        lo = 0.8

    if param == 'w':
        hi = -0.33
        lo = -1.8
    if param == "ns":
        hi = 0.99
        lo = 0.94
    if param == 'ob':
        hi = 0.061
        lo = 0.037

    return (data[param] -lo)*s/(hi - lo) + (1 - s)*0.5


# data["s8"]*tf.math.sqrt(data["om"]/0.3)-0.25

def scale_params(theta):
    hi = np.array([0.5, 1.0, -0.33])
    lo = np.array([0.13, 0.0, -1.8])

    theta_ = theta.copy()
    s = 0.6
    
    theta_[:, 0] = (theta[:, 0] - lo[0])*s / (hi[0] - lo[0]) + (1 - s)*0.5
    theta_[:, 1] -= 0.25
    theta_[:, 2] =  (theta[:, 2] - lo[2])*s / (hi[2] - lo[2]) + (1 - s)*0.5
    
    return theta_

def inv_scale_params(theta):

    hi = np.array([0.5, 1.0, -0.33])
    lo = np.array([0.13, 0.0, -1.8])

    theta_ = theta.copy()
    s = 0.6

    theta_[:, 0] = ((theta[:, 0] - (1 - s)*0.5) * (hi[0] - lo[0]) / s) + lo[0]
    theta_[:, 1] += 0.25
    theta_[:, 2] =  ((theta[:, 2] - (1 - s)*0.5) * (hi[2] - lo[2]) / s) + lo[2]
    
    return theta_


def get_S8(theta):
    return np.array([theta[:, 0], theta[:, 1]*np.sqrt(theta[:, 0]/0.3), theta[:, 2]]).T

def get_sigma8(theta):
    return np.array([theta[:, 0], theta[:, 1]/np.sqrt(theta[:, 0]/0.3), theta[:, 2]]).T



def main():


    parser = argparse.ArgumentParser(description="arguments to pass to Cls compressor")
    parser.add_argument('-c', '--config', default='', type=str, help='config file path')
    parser.add_argument('-f', '--summary_file', type=str, help='summary file path')
    parser.add_argument('-t', '--test_file', type=str, help='test summary file path')
    parser.add_argument('-d', '--nde_file', type=str, help='NDE file path')
    parser.add_argument('-n', '--name', type=str, default="posterior_chains_tarp", help='results file path')
    parser.add_argument('-w', '--w_cut', type=float, default=1.25, help='w cutoff for trainnig')
    parser.add_argument('-nt', '--n_sims_tarp', type=int, default=100, help='number of sims to test')

    args = parser.parse_args()


    def mask_prior_and_summaries(theta, summs, low, high):
        condition = (~torch.any(theta.lt(low), keepdim=True, dim=-1)) & (~torch.any(theta.gt(high), keepdim=True, dim=-1))
        mask = condition.squeeze()
        return theta.float()[mask], summs.float()[mask], mask


    #file = load_obj("/home/makinen/repositories/des-hybrid/final_dry_ABC_6summs.pkl")[-1]

    file = load_obj(args.summary_file)[-1]


    preds_val = file["summs_test"]
    params_val = file["params_test"]

    params_LFI = (file['params_lfi'])
    preds_LFI = file['summs_lfi']

    params_test = (file['params_sys'])
    preds_test = file['summs_sys']

    low_theta=[0.15, 0.5, -1.0]; 
    high_theta=[0.52, 1.0, -0.3333]


    preds_train = file["summs_train"]
    params_train = file["params_train"]




    # MAKE CUTS FOR DENSITY ESTIMATION
    print('cutting w at', -args.w_cut)
    low_theta=[0.15, 0.5, -args.w_cut];  ## CUTTING AT w=-1.25
    high_theta=[0.52, 1.0, -0.3333]


    # step 1: mask the prior from the test set in scaled_params space
    hi_scaled = scale_params(np.array([high_theta]))
    lo_scaled = scale_params(np.array([low_theta]))

    params_test, preds_test, _ = mask_prior_and_summaries(torch.tensor(params_test), 
                                torch.tensor(preds_test),
                                low=torch.tensor(lo_scaled), 
                                high=torch.tensor(hi_scaled))


    params_LFI, preds_LFI, _ = mask_prior_and_summaries(torch.tensor(params_LFI), 
                                torch.tensor(preds_LFI),
                                low=torch.tensor(lo_scaled), 
                                high=torch.tensor(hi_scaled))


    params_val, preds_val, _ = mask_prior_and_summaries(torch.tensor(params_val), 
                                torch.tensor(preds_val),
                                low=torch.tensor(lo_scaled), 
                                high=torch.tensor(hi_scaled))




    # load the ndes
    print("loading NDEs from", args.nde_file)
    like_ensemble_reg = load_obj(args.nde_file + "posterior.pkl")

    # load bijector as well
    bijector = load_obj(args.nde_file + "bijector.pkl")

    phi_LFI = bijector(torch.tensor(params_LFI)).numpy()
    phi_val = bijector(torch.tensor(params_val)).numpy()    


    low=phi_LFI.min(0);
    high=phi_LFI.max(0)

    def mask_prior_and_summaries(theta, summs, low=torch.tensor(low), high=torch.tensor(high)):
        condition = (~torch.any(theta.lt(low), keepdim=True, dim=-1)) & (~torch.any(theta.gt(high), keepdim=True, dim=-1))
        mask = condition.squeeze()
        return theta.float()[mask], summs.float()[mask], mask

    def mask_prior(theta, low=torch.tensor(low), high=torch.tensor(high)):
        condition = (~torch.any(theta.lt(low), keepdim=True, dim=-1)) & (~torch.any(theta.gt(high), keepdim=True, dim=-1))
        mask = condition.squeeze()
        return theta.float()[mask], mask

    def logprob(theta, x, likelihood_estimator):

        theta, mask = mask_prior(theta)
        lgp = likelihood_estimator.potential(theta.float(), 
                                            x=x.float())
        # replace theta not in prior with nan for later removal
        logp = torch.ones_like(mask) * torch.nan
        logp[mask] = lgp
        return logp


    def get_mcmc_chains(target, std=0.5, n_steps=4000, burnin=500, theta0=phi_val.mean(0)):

        logp = lambda t: logprob(t, x=torch.tensor(target), 
                                likelihood_estimator=like_ensemble_reg)
                
        n_walkers = 500
        walkers1 = torch.normal(mean=0, std=std, size=[n_walkers, 3]) + torch.tensor(theta0)
        walkers2 = torch.normal(mean=0, std=std, size=[n_walkers, 3]) + torch.tensor(theta0)
        
        chainreg = affine_sample(logp, n_params=3, n_walkers=n_walkers, 
                n_steps=n_steps, walkers1=walkers1, walkers2=walkers2, progress_bar=True)
        
        chainreg = chainreg[burnin:, ...].reshape(-1, 3)
        
        
        # invert bijector
        chainreg = bijector.inv(torch.tensor(chainreg))
        # convert back to real parameter values
        chainreg = torch.tensor(inv_scale_params(chainreg.numpy()))
        
        chainreg,_ = mask_prior(chainreg, low=torch.tensor([0.15, 0.5, -1.0]), high=torch.tensor([0.52, 1.0, -0.3333]))
        cut = np.all(~np.isnan(chainreg.numpy()), axis=-1)
        chainreg = chainreg.numpy()[cut,:]

        return chainreg


    print("loading resampled test simulations for coverage testing")
    print("testing %d sims"%(args.n_sims_tarp))

    outname = args.name + "_" + str(int(args.w_cut * 100))

    print("saving to ", outname)

    fsims = np.load(args.test_file)


    summs_test_coverage = fsims['summaries'][:args.n_sims_tarp]
    params_test_coverage = fsims['params'][:args.n_sims_tarp]

    chains_test_coverage = [get_mcmc_chains(f, std=0.25, n_steps=1000, burnin=500) for f in summs_test_coverage]

    
    print('saving everything')
    save_obj(
        dict(
        chains=chains_test_coverage,
        params=params_test_coverage,
         summaries=summs_test_coverage
        ),
        outname,
        )

    


if __name__ == "__main__":
    main()



# python run_mcmc_tarp.py --summary_file /home/makinen/repositories/des-hybrid/final_dry_ABC_6summs.pkl --test_file test_sys_sims_prior_resampled_wextend_125_700.npz