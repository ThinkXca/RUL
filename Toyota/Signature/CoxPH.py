import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from scipy.integrate import trapz
import torch
import torchtuples as tt
from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from survival_evaluation import d_calibration, l1, one_calibration
import random
import statistics
from sklearn.metrics import roc_auc_score
from sksurv.metrics import cumulative_dynamic_auc

path = './ExtractedData/charge.csv'
d = pd.read_csv(path)

x_cols = d.iloc[:, :154]
event_col = ['event']
time_col = ['time']
cols_standardize = [f'feature {i}' for i in range(1, 155)]
cols_leave = []
CI = []
IBS = []
L1_hinge = []
L1_margin = []
AUC_scores = []

# Split the dataset (including training, validation, and test sets)
def train_test_split_with_val(df, stratify_colname='y', frac_train=0.7, frac_val=0.15, frac_test=0.15, random_state=None):
    # Check if the proportions sum up to 1.0
    if not np.isclose(frac_train + frac_val + frac_test, 1.0):
        raise ValueError(f'Proportions {frac_train}, {frac_val}, {frac_test} do not sum to 1.0')
    if stratify_colname not in df.columns:
        raise ValueError(f'{stratify_colname} is not a column in the dataframe')

    # Set random seed to ensure reproducibility
    if random_state:
        np.random.seed(random_state)

    # First, split into training and temporary sets (which include validation and test sets)
    df_train, df_temp = train_test_split(df, stratify=df[stratify_colname], train_size=frac_train, random_state=random_state)

    # Then, split the temporary set into validation and test sets
    df_val, df_test = train_test_split(df_temp, stratify=df_temp[stratify_colname], train_size=frac_val/(frac_val + frac_test), random_state=random_state)

    return df_train, df_val, df_test

# CoxPH model
def CoxPH_():
    df_train, df_val, df_test = train_test_split_with_val(d, 'event', frac_train=0.8, frac_val=0.05, frac_test=0.15, random_state=10)
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    x_mapper = DataFrameMapper(standardize + leave)
    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')

    in_features = x_train.shape[1]

    get_target = lambda df: (df['time'].values, df['event'].values)
    y_train = get_target(df_train)
    y_val = get_target(df_val)
    y_test = get_target(df_test)
    durations_test, events_test = y_test

    out_features = 1
    num_nodes = [16, 16]
    batch_norm = True
    dropout = 0.05
    output_bias = False

    # Build Neural Network
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                  dropout, output_bias=output_bias)

    model = CoxPH(net, tt.optim.Adam)

    batch_size = 128
    lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
    lr_finder.get_best_lr()
    model.optimizer.set_lr(0.01)

    epochs = 512
    callbacks = [tt.cb.EarlyStopping()]
    model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=(x_val, y_val), verbose=0, val_batch_size=batch_size)

    _ = model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_test)  # Predict Survival Function

    # Add Time Series 'time' to the surv DataFrame
    surv_df = pd.DataFrame(surv)

    # Extract Row and Column Names
    surv_df.index.name = 'time'
    surv_df.columns.name = 'survival_function'

    # Save Row and Column Names to CSV
    surv_df.to_csv("./output/charge_CoxPH.csv", index=True)


    survival_predictions = pd.Series(trapz(surv.values.T, surv.index), index=df_test.index)

    l1_hinge = l1(df_test.time, df_test.event, survival_predictions, l1_type='hinge')
    l1_margin = l1(df_test.time, df_test.event, survival_predictions, df_train.time, df_train.event, l1_type='margin')

    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    c_index = ev.concordance_td('antolini')
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 10)
    brier = ev.integrated_brier_score(time_grid)

    quantiles = np.percentile(df_test['time'], np.linspace(1, 99, 362))

    # Calculate AUC
    labels_train = np.array([(e, t) for e, t in zip(df_train['event'], df_train['time'])], dtype=[('event', 'bool'), ('time', 'float')])
    labels_test = np.array([(e, t) for e, t in zip(df_test['event'], df_test['time'])], dtype=[('event', 'bool'), ('time', 'float')])

    time_grid_train = np.unique(df_train['time'])
    
    auc_scores = []
    for eval_time in quantiles: 
        try:
            interp_time_index = np.argmin(np.abs(eval_time - time_grid_train))
            surv_values_at_eval_time = surv.iloc[interp_time_index].values
            estimated_risks = 1 - surv_values_at_eval_time

            if np.min(estimated_risks) == np.max(estimated_risks): 
                continue  

            auc = cumulative_dynamic_auc(labels_train, labels_test, estimated_risks, times=[eval_time])[0][0]
            if not np.isnan(auc) and not np.isinf(auc):
                auc_scores.append(auc)
        except Exception as e:
            print(f"AUC calculation failed: {e}, eval_time={eval_time}")

   

    return l1_hinge, l1_margin, c_index, brier, auc_scores, surv, df_test

def safe_stat(data):
    return round(statistics.mean(data), 3), round(statistics.stdev(data), 3) if len(data) > 1 else (0.0, 0.0)

def view_results(num_experiments):
    for i in range(num_experiments):
        l1_hinge, l1_margin, c_index, brier, auc_scores, surv, df_test = CoxPH_()
        L1_hinge.append(l1_hinge)
        L1_margin.append(l1_margin)
        CI.append(c_index)
        IBS.append(brier)
        AUC_scores.append(np.mean(auc_scores) if auc_scores else 0.5)  # set a default value of 0.5 (random level)
        auc_mean, auc_std = safe_stat(AUC_scores) 
       

    print('L1_hinge: {:.3f} ({:.3f})\n'.format(statistics.mean(L1_hinge), statistics.stdev(L1_hinge)))
    print('L1_margin: {:.3f} ({:.3f})\n'.format(statistics.mean(L1_margin), statistics.stdev(L1_margin)))
    print('CI: {:.3f} ({:.3f})\n'.format(statistics.mean(CI), statistics.stdev(CI)))
    print('IBS: {:.3f} ({:.3f})\n'.format(statistics.mean(IBS), statistics.stdev(IBS)))
    print(f'AUC: {auc_mean} ± {auc_std}')
    

    

    # d-calibration 
    d_cal = d_calibration(df_test.event, surv.iloc[6])
    print('d-calibration: \n')
    print('p_value: {:.4f}\n'.format(d_cal['p_value']))
    print('bin_proportions:')
    for i in d_cal['bin_proportions']:
        print(i)
    print('\n')
    print('censored_contributions:')
    for i in d_cal['censored_contributions']:
        print(i)
    print('\n')
    print('uncensored_contributions:')
    for i in d_cal['uncensored_contributions']:
        print(i)

    L1_hinge.clear()
    L1_margin.clear()
    CI.clear()
    IBS.clear()

view_results(2)
