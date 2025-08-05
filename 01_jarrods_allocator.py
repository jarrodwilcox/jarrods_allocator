#!/usr/bin/env python
# coding: utf-8

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
        print('WARNING!!!! cvxpy problem mappears not feasible.')
        return None
                
    prtns=np.dot(rtns,xxvalue)     
    alloc=xxvalue 

    return ('dummy',prtns,xxvalue,-result)


# LOAD ACTUAL ALLOCATION -- NON OPTIMAL

# In[5]:


def load_actual_alloc(tickers,allocationfile):
    print(' ')
    try:
        alloc_df=pd.read_csv(allocationfile)
        allocs=alloc_df.values[0]
        print('Input Allocation: ')
        print(allocs)
        if (list(alloc_df.columns)==list(tickers)):
            return allocs
        else:
            print('APPROPRIATE ACTUAL ALLOCATIONS NOT FOUND')
            return None
    except:
        print('BAD ALLOCATION FILE')
        return None


# IMPLIED RETURNS ESTIMATOR

# In[6]:


def implied_dif_returns_nelder(rtns, lev, allocation, worst, target_exputil):

    nrows, ncols = rtns.shape
    
    def objective(x):
        adjusted_rtns = rtns + x[np.newaxis, :]
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



# IMPLIED EXPECTED RETURN

# In[7]:


def find_implied_dif_expected_returns(returns_df, lev, worst, allocation, exputil):
    rtns = returns_df.values
    ncols=rtns.shape[1]
    output = implied_dif_returns_nelder(rtns, lev, allocation, worst, exputil).T
   
    return output


# PRINT PARAMETERS

# In[8]:


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

# In[9]:


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
    print('act_alloc: ',act_alloc)
    
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
        
        big_implied_dif_df[lev] = find_implied_dif_expected_returns(returns_df,lev,worst,act_alloc,exputil)
         
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

# In[10]:


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

