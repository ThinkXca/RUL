import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import mean_squared_error, roc_auc_score
from scipy.integrate import trapz
import torch
import torchtuples as tt
from pycox.models import CoxCC
from pycox.evaluation import EvalSurv
from survival_evaluation import d_calibration, l1
import statistics
from pycox.models import LogisticHazard, PMF, DeepHitSingle, CoxPH, MTLR
from sksurv.metrics import cumulative_dynamic_auc

path = './ExtractedData/discharge.csv'
D = pd.read_csv(path)

x_cols = D.iloc[:, :154].columns.tolist() 
event_col = ['event']  
time_col = ['time'] 

D = D[x_cols + event_col + time_col]

d = D.copy()

cols_standardize = d.columns.values.tolist()
n_exp = 30

if len(cols_standardize) > 4: 
    cols_standardize.pop(1)
    cols_standardize.pop(4)

CI = []
IBS = []
L1_hinge = []
AUC_scores = []

def train_val_test_stratified_split(df, stratify_colname='y', frac_train=0.6, frac_val=0.15, frac_test=0.25, random_state=None):
    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % (frac_train, frac_val, frac_test))
    if stratify_colname not in df.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))
    X = df # Contains all columns.
    y = df[[stratify_colname]] # Dataframe of just the column on which to stratify.
    df_train, df_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=(1.0 - frac_train), random_state=random_state)
    relative_frac_test = frac_test / (frac_val + frac_test)
    if relative_frac_test == 1.0:
        df_val, df_test, y_val, y_test = [], df_temp, [], y_temp
    else:
        df_val, df_test, y_val, y_test = train_test_split(df_temp, y_temp, stratify=y_temp, test_size=relative_frac_test, random_state=random_state)
    assert len(df) == len(df_train) + len(df_val) + len(df_test)
    return df_train, df_val, df_test

CI = []
IBS = []
L1_hinge = []
L1_margin = []
for i in range(n_exp):
    df_train, df_val, df_test = train_val_test_stratified_split(d, 'event', frac_train=0.8, frac_val=0.05, frac_test=0.15, random_state=10)
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = []
    x_mapper = DataFrameMapper(standardize + leave)
    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')

    get_target = lambda df: (df['time'].values, df['event'].values)
    num_durations = 10
    labtrans = MTLR.label_transform(num_durations)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))

    #train = (x_train, y_train)
    val = (x_val, y_val)
    durations_test, events_test = get_target(df_test)

    in_features = x_train.shape[1]
    out_features = labtrans.out_features
    num_nodes = [32, 32]
    batch_norm = True
    dropout = 0.1

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)


    model = MTLR(net,tt.optim.Adam, duration_index=labtrans.cuts)

    batch_size = 256
    lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=6)
    #_ = lr_finder.plot()
    lr_finder.get_best_lr()
    model.optimizer.set_lr(0.01)

    epochs = 300
    callbacks = [tt.cb.EarlyStopping()]
    model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val, verbose=0)

    surv = model.interpolate(10).predict_surv_df(x_test)
    print(surv)

    surv_df = pd.DataFrame(surv)

    surv_df.index.name = 'time'
    surv_df.columns.name = 'survival_function'

    surv_df.to_csv("./output/discharge_MTLR.csv", index=True)

    #surv.iloc[:, :5].plot(drawstyle='steps-post')
    #plt.ylabel('S(t | x)')
    #_ = plt.xlabel('Time')

    surv = model.interpolate(10).predict_surv_df(x_test)
    survival_predictions = pd.Series(trapz(surv.values.T, surv.index), index=df_test.index)
    l1_hinge = l1(df_test.time, df_test.event, survival_predictions, l1_type = 'hinge')
    l1_margin = l1(df_test.time, df_test.event, survival_predictions, df_train.time, df_train.event, l1_type = 'margin')
    #surv.iloc[:, :5].plot(drawstyle='steps-post')
    #plt.ylabel('S(t | x)')
    #_ = plt.xlabel('Time')

    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    c_index = ev.concordance_td('antolini')
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    #ev.brier_score(time_grid).plot()
    brier = ev.integrated_brier_score(time_grid)
    #plt.ylabel('Brier score')
    #_ = plt.xlabel('Time')

    quantiles = np.percentile(df_test['time'], np.linspace(1, 99, 362))

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


    AUC_scores.append(np.mean(auc_scores) if auc_scores else 0.5) 

    CI.append(c_index)
    IBS.append(brier)
    L1_hinge.append(l1_hinge)
    L1_margin.append(l1_margin)

def safe_stat(data):
    return round(statistics.mean(data), 3), round(statistics.stdev(data), 3) if len(data) > 1 else (0.0, 0.0)
auc_mean, auc_std = safe_stat(AUC_scores) 


print('CI:', round(statistics.mean(CI), 3), round(statistics.stdev(CI), 3))
print('IBS:', round(statistics.mean(IBS), 3), round(statistics.stdev(IBS), 3))
print('L1_hinge:', round(statistics.mean(L1_hinge), 3), round(statistics.stdev(L1_hinge), 3))
print('L1_margin:', round(statistics.mean(L1_margin), 3), round(statistics.stdev(L1_margin), 3))
print(f'AUC: {auc_mean} ± {auc_std}')

print('d_calibration_p_value:', round(d_calibration(df_test.event, surv.iloc[6])['p_value'], 3))
print('d_calibration_bin_proportions:')
for i in d_calibration(df_test.event, surv.iloc[6])['bin_proportions']:
    print(i)
print('d_calibration_censored_contributions:')
for i in d_calibration(df_test.event, surv.iloc[6])['censored_contributions']:
    print(i)
print('d_calibration_uncensored_contributions:')
for i in d_calibration(df_test.event, surv.iloc[6])['uncensored_contributions']:
    print(i)