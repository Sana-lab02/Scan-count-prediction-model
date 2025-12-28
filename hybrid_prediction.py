import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from prophet import Prophet


# load cleaned data
file_path = "File path"
df = pd.read_excel(file_path).dropna(how='all')

# expand date list
df_expanded = (
    df.dropna(subset=['date_list'])
    .assign(date=df['date_list'].str.split(r',\s*'))
    .explode('date')
)

df_expanded['date'] = pd.to_datetime(df_expanded['date'].str.strip(), errors='coerce')
df_expanded = df_expanded.dropna(subset=['date'])
df_expanded['year'] = df_expanded['date'].dt.year
df_expanded['month'] = df_expanded['date'].dt.month

df_expanded.rename(columns={'company_clean': 'retailer'}, inplace=True)
df_expanded['retailer'] = df_expanded['retailer'].str.strip().str.lower()


# create montly scan count
monthly_scans = (
    df_expanded.groupby(['retailer', 'year', 'month'])
    .size()
    .reset_index(name='scan_count')
)

# add full date for prophet
monthly_scans['ds'] = pd.to_datetime(
    monthly_scans['year'].astype(str) + '-' + monthly_scans['month'].astype(str) + '-01'
)

# forecast settings
forecast_months = [
    pd.Timestamp('2025-11-01'),
    pd.Timestamp('2025-12-01')
] + pd.date_range('2026-01-01', '2026-12-01', freq='MS').tolist()


# Create list
company_predictions = []


retailer_totals = df_expanded.groupby('retailer')['scan_count'].sum()
low_activity_retailers = retailer_totals[retailer_totals < 5].index.tolist()

# loop through companies
for retailer in monthly_scans['retailer'].unique():
    data = monthly_scans[monthly_scans['retailer'] == retailer].copy()
    data = data.sort_values('ds')
    n_months = len(data)

    # calculate rolling average
    rolling_avg = data['scan_count'].rolling(window=min(3, n_months), min_periods=1).mean().iloc[-1]

    if retailer in low_activity_retailers:
        print(f"Not enough data for {retailer}")

    if len(data) < 1:
        continue

    # if enough data use prophet
    if len(data) >= 12:
        try:
            prophet__data = data.rename(columns={'scan_count': 'y'})[['ds', 'y']]
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(prophet__data)

            future = pd.DataFrame({'ds': forecast_months})
            forecast = m.predict(future)

            # weighted hybrid model: 70% Prophet + 30% rolling average
            forecast['yhat'] = (0.7 * forecast['yhat'] + 0.3 * rolling_avg)
            forecast['yhat'] = np.clip(forecast['yhat'], 0, 2 * rolling_avg)
            forecast['yhat'] = forecast['yhat'].rolling(window=2, min_periods=1).mean()

            predictions = forecast[['ds', 'yhat']].copy()
            predictions.rename(columns={'yhat': 'predicted_scan_count'}, inplace=True)
            predictions['retailer'] = retailer
            predictions['model'] = 'Prophet+RollingAvg'

        except Exception as e:
            print(f"Prophet failed for {retailer}: {e}")
            continue
        
    # If not prophet fall back to linear regression
    elif n_months >= 6:
        X = np.arange(n_months).reshape(-1, 1)
        y = data['scan_count'].values
        model = LinearRegression()
        model.fit(X, y)

        future_steps = np.arange(len(data), len(data) + len(forecast_months)).reshape(-1, 1)
        preds = model.predict(future_steps)

        # Weighted averge; 60% regression + 40% rolling average
        preds = 0.6 * preds + 0.4 * rolling_avg

        predictions = pd.DataFrame({
            'ds': forecast_months,
            'predicted_scan_count': np.maximum(0, np.round(preds, 2)),
            'retailer': retailer,
            'model': 'LinearRegression'
        })

    else: 
        y = data['scan_count'].values
        n_months = len(y)
        
        # Ading weight to recent months
        weights = np.linspace(0.3, 1.0, n_months)
        weights /= weights.sum()
        weighted_avg = np.dot(y, weights)
        slope = 0

        if n_months > 1:
            X = np.arange(n_months).reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0] * 0.5

        preds = [max(0, round(weighted_avg + slope*i, 2)) for i in range(1, len(forecast_months) + 1)]
        predictions = pd.DataFrame({
            'retailer': retailer,
            'predicted_scan_count': preds,
            'ds': forecast_months,
            'model': 'RollingAvg+Trend'
        })

    # inactivity adjustment
    last_scan_date = data['ds'].max()
    latest_date = monthly_scans['ds'].max()
    months_since_last_scan = (latest_date.year - last_scan_date.year) * 12 + (latest_date.month - last_scan_date.month)

    scans_last_6m = data.loc[data['ds'] >= latest_date - pd.DateOffset(months=6), 'scan_count'].sum()
    scans_last_12m = data.loc[data['ds'] >= latest_date - pd.DateOffset(months=12), 'scan_count'].sum()

    if months_since_last_scan <= 3:
        activity_factor = 1.0
    elif months_since_last_scan <= 6:
        activity_factor = 0.6
    elif months_since_last_scan <= 9:
        activity_factor = 0.3
    else:
        activity_factor = 0.0

    #  decay for low scan volume
    if scans_last_12m <= 3:
        activity_factor *= 0.3
    elif scans_last_12m <= 6:
        activity_factor *= 0.6

    
    predictions['predicted_scan_count'] = (predictions['predicted_scan_count'] * activity_factor).round(2)
    predictions['predicted_scan_count'] = np.floor(predictions['predicted_scan_count'] + 0.5).astype(int)
    predictions['activity_factor'] = activity_factor

    predictions['retailer'] = retailer
    predictions['year'] = predictions['ds'].dt.year
    predictions['month'] = predictions['ds'].dt.month

    company_predictions.append(predictions)



# combine and export report
predictions_df = pd.concat(company_predictions, ignore_index=True)
export_path = "/Users/phood/Downloads/hybrid_montly_predictions.xlsx"
predictions_df.to_excel(export_path, index=False)

print(f"Hybrid predictions exported to {export_path}")


