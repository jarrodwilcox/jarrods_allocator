#!/usr/bin/env python
# coding: utf-8

# # N

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
    
    print(f'\nOptimizing for leverage={lev}...')
    result=prob.solve(solver=cp.CLARABEL,tol_feas=1e-7,tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_ktratio=1e-7, verbose=False) /nrows/lev
    
    # Print optimization details
    print(f'  Status: {prob.status}')
    print(f'  Objective value: {prob.value}')
    if hasattr(prob, 'solver_stats') and prob.solver_stats:
        print(f'  Solver iterations: {prob.solver_stats.num_iters if hasattr(prob.solver_stats, "num_iters") else "N/A"}')
    
    xxvalue=xx.value #allocation

    if xxvalue is None:                
        print('WARNING!!!! cvxpy problem appears not feasible.')
        print(f'  Problem status: {prob.status}')
        return None

    prtns=np.dot(rtns,xxvalue)     
    alloc=xxvalue 
    print(f'  Portfolio returns shape: {prtns.shape}')
    print(f'  Expected utility: {-result:.6f}')
    return ('dummy',prtns,xxvalue,-result)


# NELDER-MEAD IMPLIED RETURNS ESTIMATOR

# In[5]:


def nm_implied_dif_expected_returns(returns_df, lev, act_alloc, worst, target_exputil):

    rtns = returns_df.values
    nrows, ncols = rtns.shape

    def objective(x):
        # Add penalty for violating sum-to-zero constraint
        constraint_penalty = 1e6 * (np.sum(x))**2
        
        adjusted_rtns = rtns + x[np.newaxis, :]
        port_rtns = adjusted_rtns @ act_alloc
        lev_port_rtns = lev * port_rtns

        if np.any(lev_port_rtns <= worst):
            return 1e10

        exp_util = np.sum(np.log1p(lev_port_rtns)) / nrows / lev
        return (exp_util - target_exputil)**2 + constraint_penalty

    x0 = np.zeros(ncols) #starting point tends toward smallest deviations from prior optimal -- influences solution nm finds first
    result = minimize(objective, x0, method='Nelder-Mead', 
                     options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 20000})
    
    print(f'  NM sum of adjustments: {np.sum(result.x):.6f}')

    return (result.x).T


# CALCULATE ASSET VOLATILITIES

# In[6]:


def calculate_asset_volatilities(returns_df):
    """Calculate annualized volatility for each asset"""
    # Assuming daily returns, annualize with sqrt(252)
    daily_vols = returns_df.std()
    annual_vols = daily_vols * np.sqrt(252)
    return annual_vols


# LEAST-SQUARES IMPLIED RETURNS ESTIMATOR WITH VOLATILITY BOUNDS

# In[7]:


def ls_implied_dif_expected_returns(returns_df, act_alloc, prtns, sigma_bound=2.0):
    """
    Least squares optimization with volatility-scaled bounds on adjustments
    
    Parameters:
    - returns_df: DataFrame of returns
    - act_alloc: actual allocation (column vector)
    - prtns: target portfolio returns from optimal allocation
    - sigma_bound: number of standard deviations for bounds (default 2.0)
    """
    print(f'\nLeast-squares optimization with {sigma_bound}-sigma bounds')
    print('prtns.shape: ',prtns.shape)
    rtns = returns_df.values
    nrows, ncols = rtns.shape
    
    # Calculate asset volatilities
    asset_vols = calculate_asset_volatilities(returns_df)
    print(f'\nAsset volatilities (annualized):')
    for i, asset in enumerate(returns_df.columns):
        print(f'  {asset}: {asset_vols.iloc[i]:.4f}')
    
    # Ensure act_alloc is a 1D array for proper broadcasting
    act_alloc_1d = act_alloc.flatten()
    print(f'act_alloc shape: {act_alloc.shape}, flattened shape: {act_alloc_1d.shape}')

    # Define adjustment variable (one per asset)
    xxx = cp.Variable(ncols)
    
    # Calculate portfolio returns with actual allocation (baseline)
    constant_portrtn = rtns @ act_alloc  # Shape: (nrows, 1)
    print(f'constant_portrtn.shape: {constant_portrtn.shape}')
    
    # Calculate adjustment contribution
    adjustment_per_asset = cp.vstack([xxx for _ in range(nrows)])  # Shape: (nrows, ncols)
    adjustment_contribution = adjustment_per_asset @ act_alloc  # Shape: (nrows, 1)
    
    # Total adjusted portfolio returns
    act_prtn = constant_portrtn + adjustment_contribution
    
    # Difference from optimal returns
    dif_prtn = act_prtn - prtns
    
    # Minimize sum of squared differences
    objective = cp.Minimize(cp.sum_squares(dif_prtn))
    
    # Constraints: sum to zero AND volatility bounds
    constraints = [
        cp.sum(xxx) == 0,  # Sum to zero
        xxx >= -sigma_bound * asset_vols.values,  # Lower bounds
        xxx <= sigma_bound * asset_vols.values   # Upper bounds
    ]
    
    problem = cp.Problem(objective, constraints)
    
    # Solve with verbose output
    print('\nSolving optimization problem...')
    result = problem.solve(verbose=True)
    
    # Error checking
    print(f'\nOptimization status: {problem.status}')
    print(f'Optimal objective value: {problem.value}')
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"ERROR: Optimization failed with status: {problem.status}")
        return None
        
    if xxx.value is None:
        print("ERROR: No solution found (xxx.value is None)")
        return None
    
    # Print solution details with bounds check
    print(f'\nSolution (adjustments per asset):')
    print(f'{"Asset":<10} {"Adjustment":>12} {"Lower Bound":>12} {"Upper Bound":>12} {"% of Bound":>12}')
    print('-' * 60)
    for i, asset in enumerate(returns_df.columns):
        adj = xxx.value[i]
        lower = -sigma_bound * asset_vols.iloc[i]
        upper = sigma_bound * asset_vols.iloc[i]
        pct_of_bound = abs(adj) / (sigma_bound * asset_vols.iloc[i]) * 100
        print(f'{asset:<10} {adj:>12.6f} {lower:>12.6f} {upper:>12.6f} {pct_of_bound:>11.1f}%')
    
    # Check which constraints are binding
    binding_lower = np.abs(xxx.value + sigma_bound * asset_vols.values) < 1e-6
    binding_upper = np.abs(xxx.value - sigma_bound * asset_vols.values) < 1e-6
    n_binding = np.sum(binding_lower) + np.sum(binding_upper)
    
    print(f'\nConstraint analysis:')
    print(f'  Number of assets at lower bound: {np.sum(binding_lower)}')
    print(f'  Number of assets at upper bound: {np.sum(binding_upper)}')
    print(f'  Total binding constraints: {n_binding}')
    
    if n_binding > 0:
        print('  Assets at bounds:')
        for i, asset in enumerate(returns_df.columns):
            if binding_lower[i]:
                print(f'    {asset}: at lower bound')
            elif binding_upper[i]:
                print(f'    {asset}: at upper bound')
    
    # Verify the solution
    adjusted_returns = rtns + xxx.value[np.newaxis, :]
    final_portfolio_returns = adjusted_returns @ act_alloc
    print(f'\nVerification:')
    print(f'  Original portfolio returns range: [{constant_portrtn.min():.4f}, {constant_portrtn.max():.4f}]')
    print(f'  Adjusted portfolio returns range: [{final_portfolio_returns.min():.4f}, {final_portfolio_returns.max():.4f}]')
    print(f'  Target (optimal) returns range: [{prtns.min():.4f}, {prtns.max():.4f}]')
    print(f'  Sum of adjustments: {np.sum(xxx.value):.2e}')
    
    return xxx.value


# ECONOMIC FILTER FOR TRANSACTION COST ANALYSIS

# In[8]:


def economic_rebalancing_analysis(xxx_adjustments, current_alloc, optimal_alloc, 
                                  returns_df, transaction_costs=None, confidence_level=0.95):
    """
    Analyze which rebalancing trades are economically justified
    
    Parameters:
    - xxx_adjustments: implied return adjustments from optimization
    - current_alloc: current portfolio weights
    - optimal_alloc: optimal portfolio weights  
    - returns_df: historical returns for volatility calculation
    - transaction_costs: dict or scalar of costs (default 50bps)
    - confidence_level: for determining adjustment significance
    """
    
    # Default transaction costs (50 bps = 0.005)
    if transaction_costs is None:
        transaction_costs = 0.005
    
    # Convert scalar to dict if needed
    if np.isscalar(transaction_costs):
        tc_dict = {asset: transaction_costs for asset in returns_df.columns}
    else:
        tc_dict = transaction_costs
    
    # Calculate volatilities for normalization
    asset_vols = calculate_asset_volatilities(returns_df)
    
    # Calculate required trades
    trade_sizes = optimal_alloc - current_alloc.flatten()
    trade_costs = np.array([np.abs(trade_sizes[i]) * tc_dict[asset] 
                           for i, asset in enumerate(returns_df.columns)])
    
    # Normalize adjustments by volatility (Sharpe-like ratio)
    vol_normalized_adjustments = xxx_adjustments / asset_vols.values
    
    # Determine which trades are justified
    # Buffer based on volatility (1 standard error ≈ vol/sqrt(n))
    n_periods = len(returns_df)
    volatility_buffer = asset_vols.values / np.sqrt(n_periods) * 1.96  # 95% confidence
    
    # Trade is justified if adjustment exceeds cost + statistical buffer
    adjustment_magnitude = np.abs(xxx_adjustments)
    hurdle_rate = trade_costs + volatility_buffer
    should_trade = adjustment_magnitude > hurdle_rate
    
    # Calculate confidence in adjustments
    t_stats = adjustment_magnitude / (asset_vols.values / np.sqrt(n_periods))
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Asset': returns_df.columns,
        'Current_Weight': current_alloc.flatten(),
        'Optimal_Weight': optimal_alloc.flatten() if optimal_alloc.ndim > 1 else optimal_alloc,
        'Trade_Size': trade_sizes.flatten() if trade_sizes.ndim > 1 else trade_sizes,
        'Implied_Adj': xxx_adjustments.flatten() if xxx_adjustments.ndim > 1 else xxx_adjustments,
        'Vol_Norm_Adj': vol_normalized_adjustments.flatten() if vol_normalized_adjustments.ndim > 1 else vol_normalized_adjustments,
        'Annual_Vol': asset_vols.values,
        'Trade_Cost': trade_costs.flatten() if trade_costs.ndim > 1 else trade_costs,
        'Stat_Buffer': volatility_buffer.flatten() if volatility_buffer.ndim > 1 else volatility_buffer,
        'Hurdle_Rate': hurdle_rate.flatten() if hurdle_rate.ndim > 1 else hurdle_rate,
        'Should_Trade': should_trade.flatten() if should_trade.ndim > 1 else should_trade,
        't_stat': t_stats.flatten() if t_stats.ndim > 1 else t_stats,
        'Net_Benefit': (adjustment_magnitude - hurdle_rate).flatten() if (adjustment_magnitude - hurdle_rate).ndim > 1 else (adjustment_magnitude - hurdle_rate)
    })
    
    # Sort by net benefit
    results_df = results_df.sort_values('Net_Benefit', ascending=False)
    
    return results_df


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
    actual_alloc=params.get('actual_alloc')

    #record control parameters
    print_parameters(sourcefile,sourcetype,Llist,long_only,worst,actual_alloc)

    #Read in Prices or Returns, based on sourcetype, adjusted for dividends and interest if possible
    if sourcetype=='PRICES':        
        #Calculate return matrix
        returns_df=calculate_returns(load_source(sourcefile))
    elif sourcetype=='RETURNS':
        returns_df=load_source(sourcefile)
    else:
        print('UNABLE TO DETERMINE SOURCE TYPE')
        raise

    #log leveraged surplus optimizations
    big_exputil_df=pd.DataFrame(np.zeros((1,len(Llist))),columns=Llist)
    big_walloc=np.zeros((len(returns_df.columns),len(Llist)))
    big_walloc_df = pd.DataFrame(big_walloc,columns=Llist,index=returns_df.columns)
    nm_implied_dif = np.zeros((len(returns_df.columns),len(Llist)))
    nm_implied_dif_df = pd.DataFrame(nm_implied_dif,columns=Llist,index=returns_df.columns)
    ls_implied_dif_1sigma = np.zeros((len(returns_df.columns),len(Llist)))
    ls_implied_dif_1sigma_df = pd.DataFrame(ls_implied_dif_1sigma,columns=Llist,index=returns_df.columns)
    ls_implied_dif_2sigma = np.zeros((len(returns_df.columns),len(Llist)))
    ls_implied_dif_2sigma_df = pd.DataFrame(ls_implied_dif_2sigma,columns=Llist,index=returns_df.columns)

    act_alloc =[actual_alloc[a] for a in actual_alloc]  #safer data entry
    act_alloc = np.array(act_alloc).reshape(-1, 1) #convert to a vertical vector for matrix muliplication
    print('act_alloc: ',act_alloc)
    for lev in Llist:
        (error_code1,prtns,walloc,exputil) = find_best_allocation(returns_df,lev,long_only,worst)
        big_exputil_df[lev]=exputil
        big_walloc_df[lev]=walloc  #note: act_alloc - walloc also interesting
        nm_implied_dif_df[lev] = nm_implied_dif_expected_returns(returns_df,lev,act_alloc,worst,exputil)
        prtns=prtns.reshape(-1, 1) # column vector
        
        # Run least squares with 1-sigma bounds
        print(f'\n{"="*60}')
        print(f'LEAST SQUARES WITH 1-SIGMA BOUNDS (Leverage={lev})')
        print(f'{"="*60}')
        ls_implied_dif_1sigma_df[lev] = ls_implied_dif_expected_returns(returns_df,act_alloc,prtns,sigma_bound=1.0)
        
        # Run least squares with 2-sigma bounds
        print(f'\n{"="*60}')
        print(f'LEAST SQUARES WITH 2-SIGMA BOUNDS (Leverage={lev})')
        print(f'{"="*60}')
        ls_implied_dif_2sigma_df[lev] = ls_implied_dif_expected_returns(returns_df,act_alloc,prtns,sigma_bound=2.0)

    with pd.option_context('display.float_format', '{:,.5f}'.format):
        print(' ')
        print('OPTIMAL ALLOCATIONS')
        print(big_walloc_df)
        print(' ')
        print('EXPECTED UTILITIES')
        print(big_exputil_df)
        print(' ')
        print('Nelder-Mead Implied Expected Return Difference')
        print(nm_implied_dif_df)
        print(' ')
        print('Least-Squares Implied Expected Return Difference (1-sigma bounds)')
        print(ls_implied_dif_1sigma_df)
        print(' ')
        print('Least-Squares Implied Expected Return Difference (2-sigma bounds)')
        print(ls_implied_dif_2sigma_df)
    
    # Perform economic analysis for the median leverage case
    median_lev_idx = len(Llist) // 2
    median_lev = Llist[median_lev_idx]
    
    print('\n' + '='*80)
    print(f'ECONOMIC REBALANCING ANALYSIS (Leverage = {median_lev})')
    print('='*80)
    
    # Get the adjustments and optimal allocation for median leverage
    nm_adjustments = nm_implied_dif_df[median_lev].values
    ls_adjustments_1sigma = ls_implied_dif_1sigma_df[median_lev].values
    ls_adjustments_2sigma = ls_implied_dif_2sigma_df[median_lev].values
    optimal_weights = big_walloc_df[median_lev].values
    
    # Convert actual_alloc dict to array in same order as returns_df columns
    current_weights = np.array([actual_alloc[asset] for asset in returns_df.columns]).reshape(-1, 1)
    
    # Compare the approaches
    print('\n\nCOMPARISON OF APPROACHES:')
    comparison_df = pd.DataFrame({
        'Asset': returns_df.columns,
        'Annual_Vol': calculate_asset_volatilities(returns_df).values,
        'NM_Adj': nm_adjustments,
        'LS_1σ_Adj': ls_adjustments_1sigma,
        'LS_2σ_Adj': ls_adjustments_2sigma,
        'NM_vs_1σ': nm_adjustments - ls_adjustments_1sigma,
        'NM_vs_2σ': nm_adjustments - ls_adjustments_2sigma
    })
    
    with pd.option_context('display.float_format', '{:,.4f}'.format,
                          'display.max_columns', None,
                          'display.width', None):
        print(comparison_df.to_string(index=False))
        
        # Summary statistics
        print(f'\n\nSUMMARY STATISTICS:')
        print(f'  Nelder-Mead:')
        print(f'    Range: [{nm_adjustments.min():.4f}, {nm_adjustments.max():.4f}]')
        print(f'    Std dev: {nm_adjustments.std():.4f}')
        print(f'    Max |adjustment|/volatility: {np.max(np.abs(nm_adjustments)/comparison_df["Annual_Vol"].values):.2f}σ')
        
        print(f'\n  Least Squares (1-sigma):')
        print(f'    Range: [{ls_adjustments_1sigma.min():.4f}, {ls_adjustments_1sigma.max():.4f}]')
        print(f'    Std dev: {ls_adjustments_1sigma.std():.4f}')
        
        print(f'\n  Least Squares (2-sigma):')
        print(f'    Range: [{ls_adjustments_2sigma.min():.4f}, {ls_adjustments_2sigma.max():.4f}]')
        print(f'    Std dev: {ls_adjustments_2sigma.std():.4f}')
        
        # Economic analysis for 2-sigma case
        print('\n\nECONOMIC ANALYSIS (2-sigma bounds):')
        econ_results = economic_rebalancing_analysis(
            xxx_adjustments=ls_adjustments_2sigma,
            current_alloc=current_weights,
            optimal_alloc=optimal_weights,
            returns_df=returns_df,
            transaction_costs=0.005  # 50 bps default
        )
        
        # Display top recommendations
        print('\nTOP REBALANCING RECOMMENDATIONS:')
        print(econ_results[econ_results['Should_Trade']].to_string(index=False))
    
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
    actual_alloc={
    'VWSTX' : .0666,
    'XLE'   : .0667,
    'EWJ'   : .0666,
    'VWEHX' : .0666,
    'XLP'   : .0666,
    'VWAHX' : .0666,
    'VFIIX' : .0666,
    'XLV'   : .0666,
    'VUSTX' : .0666,
    'VWESX' : .0666,
    'XLY'   : .0666,
    'DIA'   : .0666,
    'XLK'   : .0666,
    'SPY'   : .0666,
    'VFITX' : .0666,
    }

    )

#run main program
optimizer_output=woptimize(params)


# In[ ]:


