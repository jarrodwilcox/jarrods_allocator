#!/usr/bin/env python
# coding: utf-8

# # Jarrod's Allocator
# 
# Author: Jarrod W. Wilcox  
# Date: 09/15/2025  
# License: MIT

# ## DEPENDENCIES

# In[1]:


import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp


# ## LOAD INPUT PRICE FILE

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


# ## CALCULATE RETURNS

# In[3]:


def calculate_returns(source_df):
    price_data=np.array(source_df.values,dtype='float64')
    price_data1=np.ones((price_data.shape[0],price_data.shape[1]))
    price_data1[1:]=price_data[:-1]
    returns=(price_data/price_data1)
    returns=returns[1:]-1. 
    returns_df=pd.DataFrame(returns,columns=source_df.columns,index=source_df.index[1:])   
    return(returns_df)


# ## FIND BEST ALLOCATION

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
        print('WARNING!!!! cvxpy problem appears not feasible.')
        return None
                
    prtns=np.dot(rtns,xxvalue)     
    alloc=xxvalue 

    return ('dummy',prtns,alloc,-result)


# ## LOAD ACTUAL ALLOCATION -- NON OPTIMAL

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


# ## IMPLIED RETURNS ESTIMATOR -- SLSQP

# In[6]:


def implied_dif_returns_SLSQP(rtns, lev, allocation, worst, target_exputil, norm_type, zero_sum_type=None):
    """
    Finds implied return adjustments using constrained optimization.
    
    Minimizes the sum of squared deviations (L2) or absolute deviations (L1) or in between them,
    subject to the constraint that the expected utility equals the target value.
    Additional optional constraints:  avoid worst case, set sum of return adjustments.
        These have to be turned on in the code body.
    
    Args:
        rtns (np.ndarray): (n_scenarios, n_assets) matrix of base return scenarios.
        lev (float): Leverage factor.
        allocation (np.ndarray): (n_assets,) vector of current portfolio weights.
        target_exputil (float): The target expected utility to match.
        zero_sum_type (str): Type of zero-sum constraint:
            None - no zero-sum constraint (original)
            'equal' - adjustments sum to zero (market neutral)
            'weighted' - allocation-weighted adjustments sum to zero (portfolio neutral)
        norm_type_list (str): 'L2' for sum of squares, 'L1' for sum of absolute values,
                        'L1.X' for general p-norm with p=1+X, 0<X<1
                
    Returns:
        np.ndarray: The vector of implied differences in expected returns.
    """
    nrows, ncols = rtns.shape
    
    if norm_type == 'L2':
        # Objective: minimize sum of squared deviations
        def objective(x):
            return np.sum(x**2)
        
        # Jacobian of objective (gradient)
        def objective_jac(x):
            return 2 * x
            
    elif norm_type == 'L1':
        # For L1 norm, we use a smooth approximation: sqrt(x^2 + eps)
        eps = 1e-8
        
        def objective(x):
            return np.sum(np.sqrt(x**2 + eps))
        
        def objective_jac(x):
            return x / np.sqrt(x**2 + eps)
            
    elif (norm_type.startswith('L') and norm_type[1:].replace('.', '').isdigit() and 1.0 < float(norm_type[1:]) < 2.0):
        # General Lp norm: sum(|x_i|^p)^(1/p), we minimize sum(|x_i|^p)
        eps = 1e-8  # For numerical stability
        p_norm=float(norm_type[1:])
        
        def objective(x):
            p_norm=float(norm_type[1:])
            return np.sum(np.abs(x + eps * np.sign(x))**p_norm)
        
        def objective_jac(x):
            # Gradient of |x|^p is p * |x|^(p-1) * sign(x)
            abs_x = np.abs(x) + eps
            return p_norm * abs_x**(p_norm - 1) * np.sign(x + eps * np.sign(x))   
            
    else:
        raise ValueError(f"Norm type not understood.")
    
            
    if (1==1):
        # Standard case for L2 and L1 smooth approximation
        # Constraint: expected utility must equal target
        def constraint(x):
            # Apply scaling if needed
           
            adjusted_rtns = rtns + x[None, :]
            port_rtns = adjusted_rtns @ allocation
            lev_port_rtns = lev * port_rtns
            
            # Check for invalid values
            min_lev_return = np.min(1 + lev_port_rtns)
            if min_lev_return <= 0:
                # Return a large constraint violation proportional to how negative we are
                return 1e6 * abs(min_lev_return)
            
            exp_util = np.sum(np.log1p(lev_port_rtns)) / nrows / lev
            return exp_util - target_exputil
        
        # Jacobian of constraint
        def constraint_jac(x):
            
            adjusted_rtns = rtns + x[None, :]
            port_rtns = adjusted_rtns @ allocation
            lev_port_rtns = lev * port_rtns
            
            # Check for valid domain
            if np.any(1 + lev_port_rtns <= 0):
                # Return a gradient that pushes away from invalid region
                return np.sign(allocation) * 1e6
            
            # Gradient with respect to each asset's return adjustment
            grad = np.zeros(ncols)
            denominator = 1 + lev_port_rtns
            
            for j in range(ncols):
                grad[j] = np.sum(allocation[j] * lev / denominator) / nrows / lev
            
            return grad
        
        # Set up bounds to prevent extreme adjustments
        bounds = [(-1.0, 1.0) for _ in range(ncols)]
        
        # Set up constraints for SLSQP
        constraints = [{'type': 'eq', 'fun': constraint, 'jac': constraint_jac}]
        
        # Add worst-case constraint if specified
        if worst is not None:
            def worst_constraint(x):
                adjusted_rtns = rtns + x[None, :]
                port_rtns = adjusted_rtns @ allocation
                lev_port_rtns = lev * port_rtns
                # All scenarios must be above worst-case threshold
                return np.min(lev_port_rtns) - worst
            
            def worst_constraint_jac(x):
                adjusted_rtns = rtns + x[None, :]
                port_rtns = adjusted_rtns @ allocation
                lev_port_rtns = lev * port_rtns
                # Find which scenario is the minimum
                min_idx = np.argmin(lev_port_rtns)
                # Gradient is just the allocation scaled by leverage for that scenario
                grad = np.zeros(ncols)
                for j in range(ncols):
                    grad[j] = allocation[j] * lev
                return grad
            
            constraints.append({
                'type': 'ineq',  # inequality constraint: worst_constraint >= 0
                'fun': worst_constraint,
                'jac': worst_constraint_jac
            })
        
        # Add zero-sum constraint if requested
        if zero_sum_type == 'equal':
            # Equal-weighted: sum of all adjustments = 0
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x),
                'jac': lambda x: np.ones(ncols)
            })
        elif zero_sum_type == 'weighted':
            # Allocation-weighted: sum of allocation * adjustment = 0
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.dot(allocation, x),
                'jac': lambda x: allocation
            })
        
        # Initial guess
        x0 = np.zeros(ncols)
    
    # Adjust tolerances based on leverage for better numerical stability
    ftol = 1e-10 if lev <= 4 else 1e-7
    maxiter = 2000 if lev <= 4 else 3000
    
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        jac=objective_jac,
        constraints=constraints,
        bounds=bounds,
        options={'ftol': ftol, 'maxiter': maxiter, 'disp': False}
    )
    
    if not result.success:
        print(f"Warning: SLSQP did not converge (lev={lev}, norm={norm_type}). Message: {result.message}")
        return result.x
    
    return result.x


# ## IMPLIED EXPECTED RETURN

# In[7]:


def find_implied_dif_expected_returns(returns_df, lev, exputil, allocation, norm_type):
    rtns = returns_df.values
    # Use specified norm type for implied returns
    output = implied_dif_returns_SLSQP(rtns, lev, allocation, worst=None, 
                                      target_exputil=exputil, zero_sum_type=None, 
                                      norm_type=norm_type)
    
    return output


# ## PRINT PARAMETERS

# In[8]:


def print_parameters(sourcefile,sourcetype,Llist,long_only,worst,actual_alloc,norm_type_list):
    print(' ')    
    print(f'{sourcefile=}')
    print(f'{sourcetype=}')
    print(f'{Llist=}')
    print(f'{long_only=}') 
    print(f'{worst=}')
    print(f'{actual_alloc=}')
    print(f'{norm_type_list=}')
    print(' ')
    return


# ## MAIN PROGRAM

# In[9]:


def woptimize(params={}):
    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False
    sourcefile=params.get('sourcefile')
    sourcetype=params.get('sourcetype')    
    Llist=params.get('Llist')
    long_only=params.get('long_only')
    worst=params.get('worst')
    allocationfile=params.get('allocationfile')
    norm_type_list= params.get('norm_type_list') 
    
    #record control parameters
    print_parameters(sourcefile,sourcetype,Llist,long_only,worst,allocationfile,norm_type_list)
        
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
    
    for norm_type in norm_type_list:
        print(' ')
        print('Analysis for Norm_type '+ norm_type)
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
        
            # Calculate implied return differences using specified norm
            big_implied_dif_df[lev] = find_implied_dif_expected_returns(returns_df,lev,exputil,act_alloc,norm_type)
         
        with pd.option_context('display.float_format', '{:,.5f}'.format):
            print(' ')
            print('OPTIMAL ALLOCATIONS')
            print(big_walloc_df)
            print(' ')
            print('EXPECTED UTILITIES')
            print(big_exputil_df)
            print(' ')
           
            print(f'IMPLIED DIFFERENCE IN EXPECTED RETURN (using {norm_type})')
            print(big_implied_dif_df)
            print(' ')
        
            # Show summary statistics
            print(f'SUMMARY: IMPLIED RETURN ADJUSTMENTS ({norm_type})')
            if norm_type == 'L1':
                norm_values = big_implied_dif_df.abs().sum(axis=0)
            elif norm_type[0]=='L' and is_float(norm_type[1:]):
                p_norm=float(norm_type[1:])
                norm_values = (big_implied_dif_df.abs()**p_norm).sum(axis=0)**(1/p_norm)
            elif norm_type=='L2':
                norm_values = np.sqrt((big_implied_dif_df**2).sum(axis=0))
            
        print(' ')        
    
    print('DONE!')
    
    return


# ## SET PARAMETERS, CALL ALLOCATION OPTIMIZATION, CALCULATE IMPLIED RETURNS FOR A NON-OPTIMAL ALLOCATION

# In[10]:


#set parameters for L2 norm (original)
params=dict(
    sourcefile='DATA20/prices.csv',
    sourcetype='PRICES',
    Llist=[1,2,4,8],
    long_only=True,
    worst=(-0.99),
    allocationfile = 'test_allocation.csv',
    norm_type_list=['L2','L1.5','L1.2','L1']
    )

optimizer_output=woptimize(params)


# In[ ]:




