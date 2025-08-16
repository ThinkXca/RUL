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
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.models import LogisticHazard, PMF, DeepHitSingle, CoxPH, MTLR, CoxTime
from sksurv.metrics import cumulative_dynamic_auc

# Read and process the data
path = './ExtractedData/discharge.csv'
D = pd.read_csv(path)

# Configure column names
x_cols = D.iloc[:, :154].columns.tolist()  # The first 154 columns are features
event_col = ['event']  # Survival event column
time_col = ['time']  # Survival time column

# Data cleaning and column selection
D = D[x_cols + event_col + time_col]

# Create a copy of the DataFrame to avoid modifying the original data
d = D.copy()

# List of columns to standardize
cols_standardize = d.columns.values.tolist()
n_exp = 30

# Example of columns to remove (adjust based on actual data)
if len(cols_standardize) > 4:  # Avoid index out of range
    cols_standardize.pop(1)
    cols_standardize.pop(4)

# Model metrics
CI = []
IBS = []
L1_hinge = []



def train_val_test_stratified_split(df, stratify_colname='y', frac_train=0.6, frac_val=0.15, frac_test=0.25, random_state=None):
    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % (frac_train, frac_val, frac_test))
    if stratify_colname not in df.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))
    X = df
    y = df[[stratify_colname]]
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
AUC_scores = []
for i in range(n_exp):

    df_train, df_val, df_test = train_val_test_stratified_split(d, 'event', frac_train=0.80, frac_val=0.05, frac_test=0.15, random_state=10)
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = []
    x_mapper = DataFrameMapper(standardize + leave)
    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')
    in_features = x_train.shape[1]

    num_durations = 10
    labtrans = CoxTime.label_transform(num_durations)
    get_target = lambda df: (df['time'].values, df['event'].values)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))

    #train = (x_train, y_train)
    val = tt.tuplefy(x_val, y_val)
    durations_test, events_test = get_target(df_test)

    val.repeat(2).cat().shapes()

    in_features = x_train.shape[1]
    out_features = 1
    num_nodes = [32, 32]
    batch_norm = True
    dropout = 0.1

    net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)

    model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)

    batch_size = 256
    lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=2)
    #_ = lr_finder.plot()
    lr_finder.get_best_lr()
    model.optimizer.set_lr(0.01)

    epochs = 512
    callbacks = [tt.cb.EarlyStopping()]
    model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val, verbose=0, val_batch_size=batch_size)

    _ = model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_test)

    surv_df = pd.DataFrame(surv)

    surv_df.index.name = 'time'
    surv_df.columns.name = 'survival_function'

    surv_df.to_csv("./output/discharge_CoxTime.csv", index=True)

    survival_predictions = pd.Series(trapz(surv.values.T, surv.index), index=df_test.index)
    l1_hinge = l1(df_test.time, df_test.event, survival_predictions, l1_type = 'hinge')
    #l1_margin = l1(df_test.time, df_test.event, survival_predictions, df_train.time, df_train.event, l1_type = 'margin')

    #surv.iloc[:, :5].plot(drawstyle='steps-post')
    #plt.ylabel('S(t | x)')
    #_ = plt.xlabel('Time')

    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    c_index = ev.concordance_td('antolini')
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 10)
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

            if np.min(estimated_risks) == np.max(estimated_risks):  # Avoid invalid AUC
                continue  

            auc = cumulative_dynamic_auc(labels_train, labels_test, estimated_risks, times=[eval_time])[0][0]
            if not np.isnan(auc) and not np.isinf(auc):
                auc_scores.append(auc)
        except Exception as e:
            print(f"AUC 计算失败: {e}, eval_time={eval_time}")

    # Ensure AUC_scores is not empty
    AUC_scores.append(np.mean(auc_scores) if auc_scores else 0.5)  # Set default value to 0.5 (random level)

    # Store metrics
    CI.append(c_index)
    IBS.append(brier)
    L1_hinge.append(l1_hinge)
    
def safe_stat(data):
    return round(statistics.mean(data), 3), round(statistics.stdev(data), 3) if len(data) > 1 else (0.0, 0.0)
auc_mean, auc_std = safe_stat(AUC_scores) 

print('CI:', round(statistics.mean(CI), 3), '±', round(statistics.stdev(CI), 3))
print('IBS:', round(statistics.mean(IBS), 3), '±', round(statistics.stdev(IBS), 3))
print('L1_hinge:', round(statistics.mean(L1_hinge), 3), '±', round(statistics.stdev(L1_hinge), 3))
print(f'AUC: {auc_mean} ± {auc_std}')

# D-Calibration
d_calib = d_calibration(df_test['event'], surv.iloc[6])
print('d_calibration_p_value:', round(d_calib['p_value'], 3))
print('D-Calibration (bin proportions):', round(sum(d_calib['bin_proportions']), 3))
print('D-Calibration (censored contributions):', round(sum(d_calib['censored_contributions']), 3))
print('D-Calibration (uncensored contributions):', round(sum(d_calib['uncensored_contributions']), 3))


