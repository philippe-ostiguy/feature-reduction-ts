import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def is_sp500(df):
    last_date = df['ds'].max()
    last_value = float(df.loc[df['ds'] == last_date, 'value'].iloc[0])
    return 5400 <= last_value <= 5650

initial_model_train = None
for i, df in enumerate(processed_dataframes):
    if df['value'].isna().any():
        continue
    if is_sp500(df):
        initial_model_train = df
        break

TRAIN_SIZE = .90
START_DATE = '2018-10-01'
END_DATE = '2024-09-05'
initial_model_train = initial_model_train.rename(columns={'value': 'price'}).reset_index(drop=True)
initial_model_train['unique_id'] = 'SPY'
initial_model_train['price'] = initial_model_train['price'].astype(float)
initial_model_train['y'] = initial_model_train['price'].pct_change()

initial_model_train = initial_model_train[initial_model_train['ds'] > START_DATE].reset_index(drop=True)
combined_df_all = pd.concat([df.drop(columns=['ds']) for df in processed_dataframes], axis=1)
combined_df_all.columns = [f'value_{i}' for i in range(len(processed_dataframes))]
rows_to_keep = len(initial_model_train)
combined_df_all = combined_df_all.iloc[-rows_to_keep:].reset_index(drop=True)

train_size = int(len(initial_model_train)*TRAIN_SIZE)
initial_model_test = initial_model_train[train_size:]
initial_model_train = initial_model_train[:train_size]
combined_df_test = combined_df_all[train_size:]
combined_df_train = combined_df_all[:train_size]


import numpy as np
from statsmodels.tsa.stattools import adfuller
P_VALUE = 0.05

def replace_inf_nan(series):
    if np.isnan(series.iloc[0]) or np.isinf(series.iloc[0]):
        series.iloc[0] = 0
    mask = np.isinf(series) | np.isnan(series)
    series = series.copy()
    series[mask] = np.nan
    series = series.ffill()
    return series

def safe_convert_to_numeric(series):
    return pd.to_numeric(series, errors='coerce')

tempo_df = pd.DataFrame()
stationary_df_train = pd.DataFrame()
stationary_df_test = pd.DataFrame()

value_columns = [col for col in combined_df_all.columns if col.startswith('value_')]

transformations = ['first_diff', 'pct_change', 'log', 'identity']

def get_first_diff(numeric_series):
    return replace_inf_nan(numeric_series.diff())

def get_pct_change(numeric_series):
    return replace_inf_nan(numeric_series.pct_change())

def get_log_transform(numeric_series):
    return replace_inf_nan(np.log(numeric_series.replace(0, np.nan)))

def get_identity(numeric_series):
    return numeric_series

for index, val_col in enumerate(value_columns):
    numeric_series = safe_convert_to_numeric(combined_df_train[val_col])
    numeric_series_all = safe_convert_to_numeric(combined_df_all[val_col])

    if numeric_series.isna().all():
        continue

    valid_transformations = []

    tempo_df['first_diff'] = get_first_diff(numeric_series)
    tempo_df['pct_change'] = get_pct_change(numeric_series)
    tempo_df['log'] = get_log_transform(numeric_series)
    tempo_df['identity'] = get_identity(numeric_series)

    for transfo in transformations:
        tempo_df[transfo] = replace_inf_nan(tempo_df[transfo])
        series = tempo_df[transfo].dropna()

        if len(series) > 1 and not (series == series.iloc[0]).all():
            result = adfuller(series)
            if result[1] < P_VALUE:
                valid_transformations.append((transfo, result[0], result[1]))

    if valid_transformations:
        if any(transfo == 'identity' for transfo, _, _ in valid_transformations):
            chosen_transfo = 'identity'
        else:
            chosen_transfo = min(valid_transformations, key=lambda x: x[1])[0]

        if chosen_transfo == 'first_diff':
            stationary_df_train[val_col] = get_first_diff(numeric_series_all)
        elif chosen_transfo == 'pct_change':
            stationary_df_train[val_col] = get_pct_change(numeric_series_all)
        elif chosen_transfo == 'log':
            stationary_df_train[val_col] = get_log_transform(numeric_series_all)
        else:
            stationary_df_train[val_col] = get_identity(numeric_series_all)

    else:
        print(f"No valid transformation found for {val_col}")


stationary_df_test = stationary_df_train[train_size:]
stationary_df_train = stationary_df_train[:train_size]

initial_model_train = initial_model_train.iloc[1:].reset_index(drop=True)
stationary_df_train = stationary_df_train.iloc[1:].reset_index(drop=True)
last_train_index = stationary_df_train.index[-1]
stationary_df_test = stationary_df_test.loc[last_train_index + 1:].reset_index(drop=True)
initial_model_test = initial_model_test.loc[last_train_index + 1:].reset_index(drop=True)

CORR_COFF = .95
corr_matrix = stationary_df_train.corr().abs()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
high_corr = corr_matrix.where(mask).stack()
high_corr = high_corr[high_corr >= CORR_COFF]
unique_cols = set(high_corr.index.get_level_values(0)) | set(high_corr.index.get_level_values(1))
num_high_corr_cols = len(unique_cols)

print(f"\n{num_high_corr_cols}/{stationary_df_train.shape[1]} variables have ≥{int(CORR_COFF*100)}% "
      f"correlation with another variable.\n")


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
EXPLAINED_VARIANCE = .9
MIN_VARIANCE = 1e-10

X_train = stationary_df_train.values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pca = PCA(n_components=EXPLAINED_VARIANCE, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_scaled)

components_to_keep = pca.explained_variance_ > MIN_VARIANCE
X_train_pca = X_train_pca[:, components_to_keep]

pca_df_train = pd.DataFrame(
    X_train_pca,
    columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])]
)

X_test = stationary_df_test.values
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

X_test_pca = X_test_pca[:, components_to_keep]

pca_df_test = pd.DataFrame(
    X_test_pca,
    columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])]
)

print(f"\nOriginal number of features: {stationary_df_train.shape[1]}")
print(f"Number of components after PCA: {pca_df_train.shape[1]}\n")

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping
import lightning.pytorch as pl
import torch

pl.seed_everything(42)
max_encoder_length = 32
max_prediction_length = 1
VAL_SIZE = .2
VARIABLES_IMPORTANCE = .8
model_data_feature_sel = initial_model_train.join(stationary_df_train)
model_data_feature_sel = model_data_feature_sel.join(pca_df_train)

model_data_feature_sel['group'] = 'spy'
model_data_feature_sel['time_idx'] = range(len(model_data_feature_sel))

train_size_vsn = int((1-VAL_SIZE)*len(model_data_feature_sel))
train_data_feature = model_data_feature_sel[:train_size_vsn]
val_data_feature = model_data_feature_sel[train_size_vsn:]
unknown_reals_origin = [col for col in model_data_feature_sel.columns if col.startswith('value_')] + ['y']

timeseries_config = {
    "time_idx": "time_idx",
    "target": "y",
    "group_ids": ["group"],
    "max_encoder_length": max_encoder_length,
    "max_prediction_length": max_prediction_length,
    "time_varying_unknown_reals": unknown_reals_origin,
    "add_relative_time_idx": True,
    "add_target_scales": True,
    "add_encoder_length": True
}

training_ts = TimeSeriesDataSet(
    train_data_feature,
    **timeseries_config
)

if torch.cuda.is_available():
    accelerator = 'gpu'
    num_workers = 2
else :
    accelerator = 'auto'
    num_workers = 0

validation = TimeSeriesDataSet.from_dataset(training_ts, val_data_feature, predict=True, stop_randomization=True)
train_dataloader = training_ts.to_dataloader(train=True, batch_size=64, num_workers=num_workers)
val_dataloader = validation.to_dataloader(train=False, batch_size=64*5, num_workers=num_workers)

tft = TemporalFusionTransformer.from_dataset(
    training_ts,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    loss=QuantileLoss()
)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=5, verbose=False, mode="min")

trainer = pl.Trainer(max_epochs=20,  accelerator=accelerator, gradient_clip_val=.5, callbacks=[early_stop_callback])
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader

)

best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)


def get_top_encoder_variables(best_tft,interpretation):
    encoder_importances = interpretation["encoder_variables"]
    sorted_importances, indices = torch.sort(encoder_importances, descending=True)
    cumulative_importances = torch.cumsum(sorted_importances, dim=0)
    threshold_index = torch.where(cumulative_importances > VARIABLES_IMPORTANCE)[0][0]
    top_variables = [best_tft.encoder_variables[i] for i in indices[:threshold_index+1]]
    if 'relative_time_idx' in top_variables:
        top_variables.remove('relative_time_idx')
    return top_variables

interpretation= best_tft.interpret_output(raw_predictions.output, reduction="sum")
top_encoder_vars = get_top_encoder_variables(best_tft,interpretation)

print(f"\nOriginal number of features: {stationary_df_train.shape[1]}")
print(f"Number of features after Variable Selection Network (VSN): {len(top_encoder_vars)}\n")

from neuralforecast.models import TiDE
from neuralforecast import NeuralForecast

train_data = initial_model_train.join(stationary_df_train)
train_data = train_data.join(pca_df_train)
test_data = initial_model_test.join(stationary_df_test)
test_data = test_data.join(pca_df_test)

hist_exog_list_origin = [col for col in train_data.columns if col.startswith('value_')] + ['y']
hist_exog_list_pca = [col for col in train_data.columns if col.startswith('PC')] + ['y']
hist_exog_list_vsn = top_encoder_vars


tide_params = {
    "h": 1,
    "input_size": 32,
    "scaler_type": "robust",
    "max_steps": 500,
    "val_check_steps": 20,
    "early_stop_patience_steps": 5
}

model_original = TiDE(
    **tide_params,
    hist_exog_list=hist_exog_list_origin,
)


model_pca = TiDE(
    **tide_params,
    hist_exog_list=hist_exog_list_pca,
)

model_vsn = TiDE(
    **tide_params,
    hist_exog_list=hist_exog_list_vsn,
)

nf = NeuralForecast(
    models=[model_original, model_pca, model_vsn],
    freq='D'
)

val_size = int(train_size*VAL_SIZE)
nf.fit(df=train_data,val_size=val_size,use_init_models=True)

from tabulate import tabulate
y_hat_test_ret = pd.DataFrame()
current_train_data = train_data.copy()

y_hat_ret = nf.predict(current_train_data)
y_hat_test_ret = pd.concat([y_hat_test_ret, y_hat_ret.iloc[[-1]]])

for i in range(len(test_data) - 1):
    combined_data = pd.concat([current_train_data, test_data.iloc[[i]]])
    y_hat_ret = nf.predict(combined_data)
    y_hat_test_ret = pd.concat([y_hat_test_ret, y_hat_ret.iloc[[-1]]])
    current_train_data = combined_data

predicted_returns_original = y_hat_test_ret['TiDE'].values
predicted_returns_pca = y_hat_test_ret['TiDE1'].values
predicted_returns_vsn = y_hat_test_ret['TiDE2'].values

predicted_prices_original = []
predicted_prices_pca = []
predicted_prices_vsn = []

for i in range(len(predicted_returns_pca)):
    if i == 0:
        last_true_price = train_data['price'].iloc[-1]
    else:
        last_true_price = test_data['price'].iloc[i-1]
    predicted_prices_original.append(last_true_price * (1 + predicted_returns_original[i]))
    predicted_prices_pca.append(last_true_price * (1 + predicted_returns_pca[i]))
    predicted_prices_vsn.append(last_true_price * (1 + predicted_returns_vsn[i]))

true_values = test_data['price']
methods = ['Original','PCA', 'VSN']
predicted_prices = [predicted_prices_original,predicted_prices_pca, predicted_prices_vsn]

results = []

for method, prices in zip(methods, predicted_prices):
    mse = np.mean((np.array(prices) - true_values)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(np.array(prices) - true_values))

    results.append([method, mse, rmse, mae])

headers = ["Method", "MSE", "RMSE", "MAE"]
table = tabulate(results, headers=headers, floatfmt=".4f", tablefmt="grid")

print("\nPrediction Errors Comparison:")
print(table)

with open("prediction_errors_comparison.txt", "w") as f:
    f.write("Prediction Errors Comparison:\n")
    f.write(table)


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(train_data['ds'], train_data['price'], label='Training Data', color='blue')
plt.plot(test_data['ds'], true_values, label='True Prices', color='green')
plt.plot(test_data['ds'], predicted_prices_original, label='Predicted Prices', color='red')
plt.legend()
plt.title('SPY Price Forecast Using All Original Feature')
plt.xlabel('Date')
plt.ylabel('SPY Price')
plt.savefig('spy_forecast_chart_original.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(train_data['ds'], train_data['price'], label='Training Data', color='blue')
plt.plot(test_data['ds'], true_values, label='True Prices', color='green')
plt.plot(test_data['ds'], predicted_prices_pca, label='Predicted Prices', color='red')
plt.legend()
plt.title('SPY Price Forecast Using PCA Dimensionality Reduction')
plt.xlabel('Date')
plt.ylabel('SPY Price')
plt.savefig('spy_forecast_chart_pca.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(train_data['ds'], train_data['price'], label='Training Data', color='blue')
plt.plot(test_data['ds'], true_values, label='True Prices', color='green')
plt.plot(test_data['ds'], predicted_prices_vsn, label='Predicted Prices', color='red')
plt.legend()
plt.title('SPY Price Forecast Using VSN')
plt.xlabel('Date')
plt.ylabel('SPY Price')
plt.savefig('spy_forecast_chart_vsn.png', dpi=300, bbox_inches='tight')
plt.close()
