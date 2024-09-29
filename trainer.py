import requests
import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("API_KEY")
FRED_LIST= 'fred_daily_series_list.csv'

def get_daily_series(api_key):
    base_url = "https://api.stlouisfed.org/fred/tags/series"
    params = {
        "api_key": api_key,
        "tag_names": "daily",
        "file_type": "json",
        "limit": 1000,
        "offset": 0
    }

    all_series = []

    while True:
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if "seriess" in data:
                series_chunk = data["seriess"]
                all_series.extend(series_chunk)

                if len(series_chunk) < params["limit"]:
                    break
                params["offset"] += params["limit"]
            else:
                break
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
    return all_series



daily_series = get_daily_series(API_KEY)
df = pd.DataFrame(daily_series)
df = df[['id']]
df.to_csv(FRED_LIST, index=False)

import os
from io import StringIO

START_DATE = '2014-10-01'
END_DATE = '2024-09-05'
DATA_FOLDER = 'data'

def fetch_data(series_id):
    request = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    request += f"&cosd={START_DATE}"
    request += f"&coed={END_DATE}"
    try:
        response = requests.get(request)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text), parse_dates=True)
        df.rename(
            columns={
                df.columns[0]: 'ds',
                df.columns[1]: 'value',
            },
            inplace=True,
        )
        return series_id, df
    except requests.RequestException as e:
        print(f"Error fetching data for {series_id}: {e}")
        return series_id, None

df = pd.read_csv(FRED_LIST)
os.makedirs(DATA_FOLDER, exist_ok=True)
series_ids = df['id'].tolist()

for series_id in series_ids:
    series_id, data = fetch_data(series_id)
    if data is not None:
        filename = os.path.join(DATA_FOLDER, f"{series_id}.csv")
        data.to_csv(filename, index=False)


def count_csv_files(folder_path):
    csv_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            csv_count += 1
    return csv_count

num_csv_files = count_csv_files(DATA_FOLDER)


import pandas_market_calendars as mcal
from typing import Optional


def obtain_market_dates(start_date: str, end_date: str, market : Optional[str] = "NYSE") -> pd.DataFrame:
    nyse = mcal.get_calendar(market)
    market_open_dates = nyse.schedule(
        start_date=start_date,
        end_date=end_date,
    )
    return market_open_dates


market_dates = obtain_market_dates(START_DATE,END_DATE)

def replace_empty_data(df : pd.DataFrame) -> pd.DataFrame:
    mask = df.isin(["", ".", None])
    rows_to_remove = mask.any(axis=1)
    return df.loc[~rows_to_remove]


from typing import Union, Tuple
import logging
MAX_MISSING_DATA = 0.02

def handle_missing_data(
        data: pd.DataFrame,
        market_open_dates : pd.DataFrame,
        data_series : str
) -> Tuple[Union[None,pd.DataFrame], Union[pd.DataFrame, None]]:
    modified_data = data.copy()
    market_open_dates["count"] = 0
    date_counts = data['ds'].value_counts()

    market_open_dates["count"] = market_open_dates.index.map(
        date_counts
    ).fillna(0)

    missing_dates = market_open_dates.loc[
        market_open_dates["count"] < 1
    ]

    if not missing_dates.empty:
        max_count = (
            len(market_open_dates)
            * MAX_MISSING_DATA
        )

        if len(missing_dates) > max_count:
            logging.warning(
                f"For the asset {data_series} there are "
                f"{len(missing_dates)} data points missing, which is greater than the maximum threshold of "
                f"{MAX_MISSING_DATA * 100}%"
            )
            return pd.DataFrame(), None
        else:
            for date, row in missing_dates.iterrows():
                modified_data = insert_missing_date(
                    modified_data, date, 'ds'
                )
    return modified_data, missing_dates


def insert_missing_date(
        data: pd.DataFrame,
        date: str,
        date_column: str
) -> pd.DataFrame:
    date = pd.to_datetime(date)
    if date not in data[date_column].values:
        prev_date = (
            data[data[date_column] < date].iloc[-1]
            if not data[data[date_column] < date].empty
            else data.iloc[0]
        )
        new_row = prev_date.copy()
        new_row[date_column] = date
        data = (
            pd.concat([data, new_row.to_frame().T], ignore_index=True)
            .sort_values(by=date_column)
            .reset_index(drop=True)
        )
    return data


import glob
processed_dataframes = []
market_dates_only = market_dates.index.date

for csv_file in glob.glob(os.path.join(DATA_FOLDER, "*.csv")):

    df = pd.read_csv(csv_file)
    df['ds'] = pd.to_datetime(df['ds'])
    df_correct_dates = df[df['ds'].dt.date.isin(market_dates_only)]
    df_cleaned = replace_empty_data(df_correct_dates)
    processed_df, missing_dates = handle_missing_data(df_cleaned,market_dates,os.path.basename(csv_file).split('.')[0])
    if not processed_df.empty:
        processed_df['ds'] = pd.to_datetime(processed_df['ds'])
        if not missing_dates.empty:
            for missing_date in missing_dates.index:
                missing_date = pd.to_datetime(missing_date)
                if missing_date in processed_df['ds'].values:
                    continue

                previous_day_data = processed_df[processed_df['ds'] < missing_date].tail(1)

                if previous_day_data.empty:
                    new_row = pd.DataFrame({'ds': [missing_date], 'value': [0]})
                else:
                    new_row = previous_day_data.copy()
                    new_row['ds'] = missing_date

                processed_df = pd.concat([processed_df, new_row]).sort_values('ds').reset_index(drop=True)
        if 'SP500.csv' in csv_file:
            model_data = processed_df.rename(columns={'value': 'price'}).reset_index(drop=True)
        processed_dataframes.append(processed_df.reset_index(drop=True))

print(f"\nNumber of data series remaining after cleanup: {len(processed_dataframes)}\n")


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
EXPLAINED_VARIANCE = .9
MIN_VARIANCE = 1e-10

combined_df = pd.concat([df.set_index('ds') for df in processed_dataframes], axis=1)
combined_df.columns = [f'value_{i}' for i in range(len(processed_dataframes))]

X = combined_df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=EXPLAINED_VARIANCE, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)

X_pca = X_pca[:, pca.explained_variance_ > MIN_VARIANCE]


pca_df = pd.DataFrame(
    X_pca,
    columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
)

print(f"\nOriginal number of features: {combined_df.shape[1]}")
print(f"Number of components after PCA: {pca_df.shape[1]}\n")

model_data = model_data.join(pca_df)


from neuralforecast.models import TFT
from neuralforecast import NeuralForecast

TRAIN_SIZE = .90
model_data['unique_id'] = 'SPY'
model_data['price'] = model_data['price'].astype(float)
model_data['y'] = model_data['price'].pct_change()
model_data = model_data.iloc[1:]
hist_exog_list = [col for col in model_data.columns if col.startswith('PC')]

train_size = int(len(model_data) * TRAIN_SIZE)
train_data = model_data[:train_size]
test_data = model_data[train_size:]

model = TFT(
    h=1,
    input_size=24,
    hist_exog_list=hist_exog_list,
    scaler_type='robust',
    max_steps=20

)

nf = NeuralForecast(
    models=[model],
    freq='D'
)

nf.fit(df=model_data)


y_hat_test_ret = pd.DataFrame()
current_train_data = train_data.copy()

y_hat_ret = nf.predict(current_train_data)
y_hat_test_ret = pd.concat([y_hat_test_ret, y_hat_ret.iloc[[-1]]])

for i in range(len(test_data) - 1):
    combined_data = pd.concat([current_train_data, test_data.iloc[[i]]])
    y_hat_ret = nf.predict(combined_data)
    y_hat_test_ret = pd.concat([y_hat_test_ret, y_hat_ret.iloc[[-1]]])
    current_train_data = combined_data

predicted_returns = y_hat_test_ret['TFT'].values

predicted_prices_ret = []
for i, ret in enumerate(predicted_returns):
    if i == 0:
        last_true_price = train_data['price'].iloc[-1]
    else:
        last_true_price = test_data['price'].iloc[i-1]
    predicted_prices_ret.append(last_true_price * (1 + ret))


import matplotlib.pyplot as plt
true_values = test_data['price']

plt.figure(figsize=(12, 6))
plt.plot(train_data['ds'], train_data['price'], label='Training Data', color='blue')
plt.plot(test_data['ds'], true_values, label='True Prices', color='green')
plt.plot(test_data['ds'], predicted_prices_ret, label='Predicted Prices', color='red')
plt.legend()
plt.title('Basic SPY Stepwise Forecast using TFT')
plt.xlabel('Date')
plt.ylabel('SPY Price')
plt.savefig('spy_forecast_chart.png', dpi=300, bbox_inches='tight')
plt.close()
