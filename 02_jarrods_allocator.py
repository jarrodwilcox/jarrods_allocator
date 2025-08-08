#!/usr/bin/env python
# coding: utf-8

# Author:  Jarrod W. Wilcox
# Date: 08/01/2025
# License: MIT

# DEPENDENCIES

# In[1]:


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import cvxpy as cp 
import jax
import jax.numpy as jnp


# LOAD INPUT PRICE FILE

# In[2]:


def load_source(sourcefile):
    try:
        source_df=pd.read_csv(sourcefile)
        temp=source_df.get('Date')
        if not temp is None:
            source_df.index=temp 
            source_df=source_df.drop(columns=['Date'])
        return source_df
    except:
        print('NO SOURCE FOUND')
        return None


# CALCULATE RETURNS

# In[3]:


def calculate_returns(source_df):
    price_data=np.array(source_df.values,dtype='float64')
    price_data1=np.ones((price_data.shape[0],price_data.shape[1]))
    price_data1[1:]=price_data[:-1]
    returns=(price_data/price_data1)
    returns=returns[1:]-1. 
    returns_df=pd.DataFrame(returns,columns=source_df.columns,index=source_df.index[1:])   
    return(returns_df)


# FIND BEST ALLOCATION

# In[4]:


def find_best_allocation(rtns_df,lev,long_only,worst):
    rtns=rtns_df.values
    nrows,ncols=rtns.shape
    levreturn=(rtns*lev)
    
    xx=cp.Variable(ncols)
    if long_only:
        constraints =[sum(xx)==1, 0<=xx, xx<=1, worst <= levreturn @ xx ]
    else:
        constraints = [sum(xx)==1,worst <= levreturn @ xx ]
    objective=cp.Minimize(cp.sum(-cp.log1p(levreturn @ xx)))
    prob=cp.Problem(objective,constraints)
    result=prob.solve(solver=cp.CLARABEL,tol_feas=1e-7,tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_ktratio=1e-7, verbose=False) /nrows/lev
    xxvalue=xx.value #allocation
            
    if xxvalue is None:                
        print('WARNING!!!! cvxpy problem mappears not feasible.')
        return None
                
    prtns=np.dot(rtns,xxvalue)     
    alloc=xxvalue 

    return ('dummy',prtns,alloc,-result)


# LOAD ACTUAL ALLOCATION -- NON OPTIMAL

# In[5]:


def load_actual_alloc(tickers,allocationfile):
    print(' ')
    try:
        alloc_df=pd.read_csv(allocationfile)
        allocs=alloc_df.values[0]
        if (list(alloc_df.columns)==list(tickers)):
            return allocs
        else:
            print('APPROPRIATE ACTUAL ALLOCATIONS NOT FOUND')
            return None
    except:
        print('BAD ALLOCATION FILE')
        return None


# IMPLIED RETURNS ESTIMATOR -- NELDER

# In[6]:


def implied_dif_returns_nelder(rtns, lev, allocation, prtns, target_exputil):

    nrows, ncols = rtns.shape
    
    def objective(x):
        adjusted_rtns = rtns + x[None, :]
        port_rtns = adjusted_rtns @ allocation
        lev_port_rtns = lev * port_rtns
        
        if np.any(1 + lev_port_rtns <= 0):
            return 1e10
        
        exp_util = np.sum(np.log1p(lev_port_rtns)) / nrows / lev
        return (exp_util - target_exputil)**2
    
    x0 = np.zeros(ncols)
    result = minimize(objective, x0, method='Nelder-Mead', 
                     options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 20000})
    
    return result.x



# IMPLIED RETURNS ESTIMATOR -- SLSQP

# In[7]:


def implied_dif_returns_SLSQP(rtns, lev, w_ideal, target_exputil, allocation, alpha=0.5, niter=50, T=0.1, stepsize=0.05):
    """
    Finds implied return changes using a gradient-based basinhopping global optimizer.

    This function uses the basinhopping algorithm to search for a global minimum.
    It uses the efficient, gradient-based SLSQP method for its local search steps,
    combining global exploration with fast local convergence.

    Args:
        rtns (np.ndarray): (n_scenarios, n_assets) matrix of base return scenarios.
        lev (float): Leverage factor.
        w_ideal (np.ndarray): (n_assets,) vector of the original ideal portfolio weights.
        target_exputil (float): The target expected utility for the drifted portfolio.
        allocation (np.ndarray): (n_assets,) vector of current (drifted) portfolio weights.
        alpha (float): Weight for the profile matching objective (0 to 1).
        niter (int): Number of basin-hopping iterations.
        T (float): The "temperature" parameter for the accept test. Higher T allows
                   more exploration of the solution space.
        stepsize (float): The step size for the random displacement.

    Returns:
        np.ndarray: The vector of implied differences in expected returns.
    """
    # --- Input Validation (Identical to the original) ---
    assert rtns.ndim == 2, f"rtns must be a 2D array, but has {rtns.ndim} dimensions."
    assert allocation.ndim == 1, f"allocation must be a 1D vector, but has {allocation.ndim} dimensions."
    assert w_ideal.ndim == 1, f"w_ideal must be a 1D vector, but has {w_ideal.ndim} dimensions."

    nrows, ncols = rtns.shape

    assert allocation.shape[0] == ncols, f"Shape mismatch: rtns has {ncols} assets, but allocation has {allocation.shape[0]}."
    assert w_ideal.shape[0] == ncols, f"Shape mismatch: rtns has {ncols} assets, but w_ideal has {w_ideal.shape[0]}."

    assert np.isscalar(lev), "lev must be a scalar."
    assert np.isscalar(target_exputil), "target_exputil must be a scalar."
    assert np.isscalar(alpha), "alpha must be a scalar."

    # --- JAX Setup (Identical to the original) ---
    rtns_jax = jnp.array(rtns)
    w_current_jax = jnp.array(allocation)
    w_ideal_jax = jnp.array(w_ideal)

    @jax.jit
    def objective_jax(x):
        """
        The JAX-compiled objective function to be minimized.
        (This function is identical to the original).
        """
        adjusted_rtns = rtns_jax + x[jnp.newaxis, :]
        port_rtns_implied = adjusted_rtns @ w_current_jax
        lev_port_rtns = lev * port_rtns_implied
        safe_lev_port_rtns = jnp.maximum(-0.99, lev_port_rtns)
        exp_util = jnp.sum(jnp.log1p(safe_lev_port_rtns)) / nrows / lev
        utility_objective = (exp_util - target_exputil)**2
        
        ideal_contributions = rtns_jax * w_ideal_jax[jnp.newaxis, :]
        implied_contributions = adjusted_rtns * w_current_jax[jnp.newaxis, :]
        profile_matching_objective = jnp.mean((ideal_contributions - implied_contributions)**2)
        
        combined_objective = (1 - alpha) * utility_objective + alpha * profile_matching_objective
        epsilon = 1e-6
        bankruptcy_penalty = jnp.sum(jnp.maximum(0, -(1 + lev_port_rtns) + epsilon)**2)
        return combined_objective + 1e9 * bankruptcy_penalty

    # --- JAX Value & Gradient Wrappers (Identical to the original) ---
    value_and_grad_fn = jax.value_and_grad(objective_jax)

    def scipy_wrapper(x_np):
        value, grad = value_and_grad_fn(x_np)
        return float(value), np.array(grad)

    # --- Constraint Definition (This was commented out in the original) ---
    # It's good practice to include the constraint for economic interpretation.
    # The implied changes should be a zero-sum adjustment.
    sum_to_zero_constraint = {
        'type': 'eq',
        'fun': lambda x: np.sum(x),
        'jac': lambda x: np.ones_like(x)
    }

    # --- Set up the Local Minimizer (SLSQP) ---
    # This dictionary packages all the arguments for the local SLSQP search.
    minimizer_kwargs = {
        'method': 'SLSQP',
        'jac': True,  # IMPORTANT: Tells SLSQP that our wrapper provides the gradient
        'constraints': [sum_to_zero_constraint],
        'options': {'ftol': 1e-11, 'disp': False, 'maxiter': 1000}
    }

    # --- Initial Guess and Basin-Hopping Optimizer Call ---
    x0 = np.zeros(ncols)
    
    result = basinhopping(
        func=scipy_wrapper,
        x0=x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=niter,
        T=T,
        stepsize=stepsize,
        disp=False # Set to True to see progress of the global search
    )

    # The result object from basinhopping is slightly different.
    # We check the success of the best result found.
    if not result.lowest_optimization_result.success:
        message = result.lowest_optimization_result.message
        print(f"Warning: Optimizer may not have converged. Message: {message}")
        
    return result.x


# IMPLIED EXPECTED RETURN

# In[8]:


def find_implied_dif_expected_returns(returns_df, lev, walloc, exputil, allocation):
    rtns = returns_df.values
    #output = implied_dif_returns_nelder(rtns, lev, allocation, worst, exputil).T
    output = implied_dif_returns_SLSQP(rtns, lev, walloc, exputil, allocation).T
    return output


# PRINT PARAMETERS

# In[9]:


def print_parameters(sourcefile,sourcetype,Llist,long_only,worst,actual_alloc):
    print(' ')    
    print(f'{sourcefile=}')
    print(f'{sourcetype=}')
    print(f'{Llist=}')
    print(f'{long_only=}') 
    print(f'{worst=}')
    print(f'{actual_alloc=}')
    print(' ')
    return    


# MAIN PROGRAM

# In[10]:


def woptimize(params={}):

    sourcefile=params.get('sourcefile')
    sourcetype=params.get('sourcetype')    
    Llist=params.get('Llist')
    long_only=params.get('long_only')
    worst=params.get('worst')
    allocationfile=params.get('allocationfile')
    
    #record control parameters
    print_parameters(sourcefile,sourcetype,Llist,long_only,worst,allocationfile)
        
    #Read in Prices or Returns, based on sourcetype, adjusted for dividends and interest if possible
    if sourcetype=='PRICES':        
        #Calculate return matrix
        returns_df=calculate_returns(load_source(sourcefile))
    elif sourcetype=='RETURNS':
        returns_df=load_source(sourcefile)
    else:
        print('UNABLE TO DETERMINE SOURCE TYPE')
        raise
    print(returns_df.head())
    
    act_alloc = load_actual_alloc(returns_df.columns,allocationfile)
    print('ACTUAL ALLOCATION')
    print(pd.DataFrame(act_alloc[None,:],columns=returns_df.columns))
    print(' ')
    
    
    #log leveraged surplus optimizations
    big_exputil_df=pd.DataFrame(np.zeros((1,len(Llist))),columns=Llist)
    big_walloc=np.zeros((len(returns_df.columns),len(Llist)))
    big_walloc_df = pd.DataFrame(big_walloc,columns=Llist,index=returns_df.columns)
    big_implied_dif = np.zeros((len(returns_df.columns),len(Llist)))
    big_implied_dif_df = pd.DataFrame(big_implied_dif,columns=Llist,index=returns_df.columns)
    for lev in Llist:
        (error_code1, wpreturns,walloc,exputil) = find_best_allocation(returns_df,lev,long_only,worst)
        big_walloc_df[lev]=walloc
        big_exputil_df[lev]=exputil
        
        big_implied_dif_df[lev] = find_implied_dif_expected_returns(returns_df,lev,walloc,exputil,act_alloc)
         
    with pd.option_context('display.float_format', '{:,.5f}'.format):
        print(' ')
        print('OPTIMAL ALLOCATIONS')
        print(big_walloc_df)
        print(' ')
        print('EXPECTED UTILITIES')
        print(big_exputil_df)
        print(' ')
        print('IMPLIED DIFFERENCE IN EXPECTED RETURN')
        print(big_implied_dif_df)
    print(' ')        
    
    print('DONE!')
    
    return


# SET PARAMETERS, CALL ALLOCATION OPTIMIZATION, CALCULATE IMPLIED RETURNS FOR A NON-OPTIMAL ALLOCATION

# In[11]:


#set parameters
params=dict(
    sourcefile='DATA20/prices.csv',
    sourcetype='PRICES',
    Llist=[1,2,4,8],
    long_only=True,
    worst=(-0.99),
    allocationfile = 'test_allocation.csv'
    )

#run main program
optimizer_output=woptimize(params)

