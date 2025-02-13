from tqdm import trange
import numpy as np
import torch

from ili.utils import IndependentTruncatedNormal

#tfd = tfp.distributions
#tfb = tfp.bijectors

def mask_prior_and_summaries(theta, summs, low, high):
    condition = (~torch.any(theta.lt(low), keepdim=True, dim=-1)) & (~torch.any(theta.gt(high), keepdim=True, dim=-1))
    mask = condition.squeeze()
    return theta.float()[mask], summs.float()[mask], mask

def mask_prior(theta, low, high):
    condition = (~torch.any(theta.lt(low), keepdim=True, dim=-1)) & (~torch.any(theta.gt(high), keepdim=True, dim=-1))
    mask = condition.squeeze()
    return theta.float()[mask], mask

def affine_sample(log_prob, n_params, n_walkers, n_steps, walkers1, walkers2,
                 progress_bar=True):
    """Run two sets of MCMC walkers in an affine sampler.

    Args:
        log_prob (<callable>): function that returns a log-probability
        n_params (int): dimensionality of target parameters
        n_walkers (int): number of walkers per walker set
        n_steps (int): number of steps to take per walker
        walkers1 (array_like): starting point of group 1 of n_walkers
        walkers2 (array_like): starting point of group 2 of n_walkers

    Returns:
        array_like: posterior parameter chains
    """
    
    # initialize current state
    current_state1 = torch.tensor(walkers1, requires_grad=False)
    current_state2 = torch.tensor(walkers2, requires_grad=False)
    

    # initial target log prob for the walkers (and set any nans to -inf)...
    logp_current1 = log_prob(current_state1)
    logp_current2 = log_prob(current_state2)
    logp_current1 = torch.where(torch.isnan(logp_current1), torch.ones_like(logp_current1)*torch.log(torch.tensor(0.)), logp_current1)
    logp_current2 = torch.where(torch.isnan(logp_current2), torch.ones_like(logp_current2)*torch.log(torch.tensor(0.)), logp_current2)

    # holder for the whole chain
    chain = [torch.cat([current_state1, current_state2], dim=0)]

    
    
    # MCMC loop
    #with trange(1, n_steps) as t:
    if progress_bar:
        t = trange(1, n_steps)
    else:
        t = range(1, n_steps)
        
    for epoch in t:

        # first set of walkers:

        # proposals
        partners1 = current_state2[torch.randint(low=0, high=n_walkers, size=[n_walkers])] #torch.gather(current_state2, torch.randint(low=0, high=n_walkers, size=[n_walkers]))
        z1 = 0.5*(torch.rand([n_walkers])+1)**2 # need truncated normal heres
        proposed_state1 = partners1 + (z1*(current_state1 - partners1).T).T
        # mask out prior edges
        
        # target log prob at proposed points
        logp_proposed1 = log_prob(proposed_state1)
        
        # acceptance probability
        p_accept1 = torch.minimum(torch.ones((n_walkers)), z1**(n_params-1)*torch.exp(logp_proposed1 - logp_current1) )
        # accept or not
        accept1_ = (torch.rand([n_walkers],) <= p_accept1)
        accept1 = accept1_.float()

        # update the state
        current_state1 = ((current_state1).T*(1-accept1) + (proposed_state1).T*accept1).T
        logp_current1 = torch.where(accept1_, logp_proposed1, logp_current1)

        # second set of walkers:

        # proposals
        partners2 = current_state1[torch.randint(low=0, high=n_walkers, size=[n_walkers])] #torch.gather(current_state1, torch.randint(low=0, high=n_walkers, size=[n_walkers]))
        z2 = 0.5*(torch.rand([n_walkers])+1)**2
        proposed_state2 = partners2 + (z2*(current_state2 - partners2).T).T
        
        # target log prob at proposed points
        logp_proposed2 = log_prob(proposed_state2)
        logp_proposed2 = torch.where(torch.isnan(logp_proposed2), torch.ones_like(logp_proposed2)*torch.log(torch.tensor(0.)), logp_proposed2)

        # acceptance probability
        p_accept2 = torch.minimum(torch.ones(n_walkers), z2**(n_params-1)*torch.exp(logp_proposed2 - logp_current2) )

        # accept or not
        accept2_ = (torch.rand([n_walkers]) <= p_accept2)
        accept2 = accept2_.float()

        # update the state
        current_state2 = ((current_state2).T*(1-accept2) + (proposed_state2).T*accept2).T
        logp_current2 = torch.where(accept2_, logp_proposed2, logp_current2)

        # append to chain
        chain.append(torch.cat([current_state1, current_state2], dim=0))

    # stack up the chain
    chain = torch.stack(chain, dim=0)
    
    return chain


