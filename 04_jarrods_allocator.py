#!/usr/bin/env python
# coding: utf-8

# Author:  Jarrod W. Wilcox
# Date: 08/23/2025
# License: MIT

# DEPENDENCIES

# In[1]:


import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp 

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
        print('WARNING!!!! cvxpy problem appears not feasible.')
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


def implied_dif_returns_nelder(rtns, lev, allocation, worst, target_exputil):

    nrows, ncols = rtns.shape
    
    def objective(x):
        adjusted_rtns = rtns + x[None, :]
        port_rtns = adjusted_rtns @ allocation
        lev_port_rtns = lev * port_rtns
        
        # Check for bankruptcy
        if np.any(1 + lev_port_rtns <= 0):
            return 1e10
        
        # Check worst-case constraint (same as in allocation optimization)
        if worst is not None and np.any(lev_port_rtns < worst):
            return 1e10
        
        exp_util = np.sum(np.log1p(lev_port_rtns)) / nrows / lev
        return (exp_util - target_exputil)**2
    
    x0 = np.zeros(ncols)
    result = minimize(objective, x0, method='Nelder-Mead', 
                     options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 20000})
    
    return result.x



# IMPLIED RETURNS ESTIMATOR -- SLSQP

# In[7]:


def implied_dif_returns_SLSQP(rtns, lev, allocation, worst, target_exputil, zero_sum_type=None):
    """
    Finds implied return adjustments using constrained optimization.
    
    Minimizes the sum of squared deviations subject to the constraint that
    the expected utility equals the target value.
    
    Args:
        rtns (np.ndarray): (n_scenarios, n_assets) matrix of base return scenarios.
        lev (float): Leverage factor.
        allocation (np.ndarray): (n_assets,) vector of current portfolio weights.
        target_exputil (float): The target expected utility to match.
        zero_sum_type (str): Type of zero-sum constraint:
            None - no zero-sum constraint (original)
            'equal' - adjustments sum to zero (market neutral)
            'weighted' - allocation-weighted adjustments sum to zero (portfolio neutral)
        
    Returns:
        np.ndarray: The vector of implied differences in expected returns.
    """
    nrows, ncols = rtns.shape
    
    # Objective: minimize sum of squared deviations
    def objective(x):
        return np.sum(x**2)
    
    # Jacobian of objective (gradient)
    def objective_jac(x):
        return 2 * x
    
    # Constraint: expected utility must equal target
    def constraint(x):
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
    # Allow up to 100% return adjustment
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
    
    # Initial guess: no adjustments from priors used in optimization
    x0 = np.zeros(ncols)
    
    # First attempt with SLSQP
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        jac=objective_jac,
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 1e-10, 'maxiter': 2000, 'disp': False}
    )
    
    # If SLSQP fails, try with a different starting point or fallback to Nelder-Mead
    if not result.success:
        # Try with a small positive adjustment as initial guess
        x0_alt = np.ones(ncols) * 0.001
        result_alt = minimize(
            objective,
            x0_alt,
            method='SLSQP',
            jac=objective_jac,
            constraints=constraints,
            bounds=bounds,
            options={'ftol': 1e-10, 'maxiter': 2000, 'disp': False}
        )
        
        if result_alt.success or result_alt.fun < result.fun:
            result = result_alt
    
    if not result.success:
        print(f"Warning: SLSQP did not converge (lev={lev}). Message: {result.message}")
        # Fallback to Nelder-Mead if SLSQP fails
        print(f"Falling back to Nelder-Mead for lev={lev}")
        result.x = implied_dif_returns_nelder(rtns, lev, allocation, worst, target_exputil)
    
    return result.x






# IMPLIED EXPECTED RETURN

# In[8]:


def find_implied_dif_expected_returns(returns_df, lev, exputil, allocation):
    rtns = returns_df.values
    # Use minimum norm SLSQP without worst-case constraint for implied returns
    # The worst-case constraint is important for forward optimization but 
    # overly restrictive for the inverse problem
    output = implied_dif_returns_SLSQP(rtns, lev, allocation, worst=None, target_exputil=exputil, zero_sum_type=None)
    
    # Alternative approaches (kept for reference):
    # output = implied_dif_returns_nelder(rtns, lev, allocation, worst=None, exputil)  # Nelder-Mead
    # output = implied_dif_returns_SLSQP(rtns, lev, allocation, None, exputil, 'weighted')  # Zero-sum
    
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
        
        # Calculate implied return differences using minimum norm approach
        big_implied_dif_df[lev] = find_implied_dif_expected_returns(returns_df,lev,exputil,act_alloc)
         
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
        
        # Show summary statistics
        print('SUMMARY: IMPLIED RETURN ADJUSTMENTS')
        summary_df = pd.DataFrame({
            'L2 Norm': np.sqrt((big_implied_dif_df**2).sum(axis=0)),
            'Portfolio-Weighted Sum': [np.dot(act_alloc, big_implied_dif_df[lev]) for lev in Llist],
            'Max Adjustment': big_implied_dif_df.max(axis=0),
            'Min Adjustment': big_implied_dif_df.min(axis=0)
        }, index=Llist).T
        print(summary_df)
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

