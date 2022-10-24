#!/usr/bin/env python
# coding: utf-8

import numpy   as np
import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import skewnorm


def forecast_paths(data, days, trials, min_max, risk_metric = 0.95, type = 'data'):
    if type == 'data':
        result = forecast_from_history(data, days, trials, risk_metric)
    if type == 'estimation':
        result = forecast_from_estimation(min_max, days, trials, risk_metric)
    return result


def forecast_from_history(history, days, trials, risk_metric = 0.95):
    
    log_return = np.log(1 + history.pct_change())
    mu         = log_return.mean()
    var        = log_return.var()        
    stdev      = log_return.std()
    
    drift  = mu - (0.5 * var)    
    z      = norm.ppf(np.random.rand(days, trials))    
    daily_returns  = np.exp(drift + stdev * z)  
    price_paths    = np.zeros_like(daily_returns)
    price_paths[0] = history.iloc[-1]
    
    for t in range(1, days):
        price_paths[t] = price_paths[t - 1] * daily_returns[t]
    
    output = pd.DataFrame(price_paths)
    result = output.iloc[-1].to_numpy()
    result = np.sort(result)
    n_min  = int(risk_metric * trials - 50)
    n_max  = int(risk_metric * trials + 50)
    result = result[n_min:n_max]
    result = np.round(result.mean(), 4)
    
    return result


def forecast_from_estimation(min_max, days, trials, risk_metric = 0.95):    
    
    df = pd.DataFrame(data = min_max)    
    df = df['col1']

    log_return = np.log(1 + df.pct_change())
    mu         = log_return.mean() / 365
    var        = log_return.var() / 365
    stdev      = log_return.std() / 365
    
    drift  = mu - (0.5 * var)
    z      = norm.ppf(np.random.rand(days, trials)) 
    daily_returns  = np.exp(drift + stdev * z)        
    price_paths    = np.zeros_like(daily_returns)
    price_paths[0] = df.iloc[0]
     
    for t in range(1, days):
        price_paths[t] = price_paths[t - 1] * daily_returns[t]
    
    output = pd.DataFrame(price_paths)
    result = output.iloc[-1].to_numpy()
    result = np.sort(result)
    n_min  = int(risk_metric * trials - 50)
    n_max  = int(risk_metric * trials + 50)
    result = result[n_min:n_max]
    result = np.round(result.mean(), 4)
    
    return result    


data = pd.read_csv('fx_rates_pln.csv')
data = data['fx']


initial_investment = 208000
initial_fx_rate    = 4.56
n_simulations      = 10000
alternative_rate   = 0.002


n_sim = n_simulations
seed  = 123
feature_list = ['sim_id', 'scenario', 'time', 'price', 'profit', 'fx_rate', 'npv_eur']
output_df    = pd.DataFrame(0.0, index = np.arange(n_sim), columns = feature_list)


for i in range(n_sim):
    
    set_seed = seed + i
    np.random.seed(set_seed)
    
    scen  = np.random.randint(1, 4)
    time  = np.random.randint(3, 13)
    price = np.round(np.random.triangular(9500, 11000, 13000, size = 1), 0)
    days  = time * 30
    
    fx_rate  = np.round(forecast_paths(data, days, 1000, {'col1': [0.098, 0.176, 0.251]}, 0.95, 'data'), 4)
    inf_rate = np.round(forecast_paths(data, days, 1000, {'col1': [0.098, 0.176, 0.251]}, 0.95, 'estimation'), 4)
    inf_rate = 1.0 + inf_rate
    
    compute_time     = time - 3    
    
    if scen == 1:
        extra_investment = 0
        paid_interest    = 1900 * time
        utilities_rent   = np.round(550 * compute_time * inf_rate, 0)
        utilities_own    = np.round(427 * compute_time * inf_rate, 0)
        if compute_time <= 6:
            rent_cost = compute_time * 1750
        else:
            rent_cost = (6 * 1750) + ((compute_time - 6) * 2150)
    if scen == 2:
        extra_investment = 87000
        paid_interest    = 1900 * time
        utilities_rent   = 0
        utilities_own    = np.round(677 * compute_time * inf_rate, 0)
        rent_cost        = 0
    if scen == 3:
        extra_investment = 102000
        paid_interest    = 1900 * time
        utilities_rent   = 0
        utilities_own    = np.round(677 * compute_time * inf_rate, 0)
        rent_cost        = 0    
    
    revenue       = 42.61 * price
    fixed_cost    = 233000 + 25000 + 15000 + 6900
    variable_cost = extra_investment + paid_interest + utilities_rent + utilities_own + rent_cost
    profit_loc    = revenue - fixed_cost - variable_cost
    
    profit_eur = profit_loc / fx_rate
    initial_i  = np.round(initial_investment / initial_fx_rate, 0)
    net_cash   = profit_eur - initial_i
    
    pv_denomin = (1 + (alternative_rate / 12)) ** time
    npv_eur    = np.round(net_cash / pv_denomin, 0)    
    
    output_df['sim_id'][i]     = i + 1
    output_df['scenario'][i]   = scen    
    output_df['time'][i]       = time
    output_df['price'][i]      = price    
    output_df['profit'][i]     = profit_loc
    output_df['fx_rate'][i]    = fx_rate    
    output_df['npv_eur'][i]    = npv_eur


sorted_df = output_df.sort_values(by = 'npv_eur', ascending = False)
new_index = np.arange(len(sorted_df['sim_id']))
sorted_df = sorted_df.set_index(new_index)
sorted_df['expected_loss'] = sorted_df['npv_eur'] * -1

n_crit = 0.95 * n_simulations
n_min  = n_crit - 0.01 * n_simulations
n_max  = n_crit + 0.01 * n_simulations

filter_range = np.arange(n_min, n_max + 1, 1)
result       = sorted_df.loc[filter_range, ]


a = sorted_df['expected_loss'].to_numpy()
prob_of_positive_result = np.count_nonzero(a < 0) / a.size

positive_npv = sorted_df[sorted_df['npv_eur'] > 0]

a = sorted_df['expected_loss'].to_numpy()
prob_not_loose_all = np.count_nonzero(a < 21000) / a.size


final_result = sorted_df[sorted_df['scenario'] == 1]
new_index    = np.arange(len(final_result['sim_id']))
final_result = final_result.set_index(new_index)
n_crit = np.round(0.95 * len(final_result['sim_id']), 0)
n_min  = int(n_crit - np.round(0.01 * len(final_result['sim_id']), 0))
n_max  = int(n_crit + np.round(0.01 * len(final_result['sim_id']), 0))


filter_range = np.arange(n_min, n_max + 1, 1)
final_result = final_result.loc[filter_range, ]
