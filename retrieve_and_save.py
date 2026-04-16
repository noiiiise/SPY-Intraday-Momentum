
from utils.download_data import fetch_alpaca_data, fetch_alpaca_dividends
import pandas as pd
from datetime import datetime, timedelta

symbol = 'SPY'

# Determine incremental start dates from existing data
intra_existing  = pd.read_csv('./data/spy_intra_data.csv')
daily_existing  = pd.read_csv('./data/spy_daily_data.csv')
div_existing    = pd.read_csv('./data/dividends.csv')

last_intra_date  = pd.to_datetime(intra_existing['caldt']).max()
last_daily_date  = pd.to_datetime(daily_existing['caldt']).max()
last_div_date    = pd.to_datetime(div_existing['caldt']).max() if not div_existing.empty else pd.Timestamp('2021-01-01')

intra_start  = (last_intra_date + timedelta(days=1)).strftime('%Y-%m-%d')
daily_start  = (last_daily_date + timedelta(days=1)).strftime('%Y-%m-%d')
div_start    = (last_div_date   + timedelta(days=1)).strftime('%Y-%m-%d')
end_date     = datetime.today().strftime('%Y-%m-%d')

print(f"Fetching intraday data from {intra_start} to {end_date}")
print(f"Fetching daily data from {daily_start} to {end_date}")
print(f"Fetching dividends from {div_start} to {end_date}")

# Fetch new data
spy_intra_new = fetch_alpaca_data(symbol, '1Min', intra_start, end_date)
spy_daily_new = fetch_alpaca_data(symbol, '1D',   daily_start, end_date)
dividends_new = fetch_alpaca_dividends(symbol, div_start, end_date)

# Append and save
if not spy_intra_new.empty:
    df_intra = pd.concat([intra_existing, spy_intra_new], ignore_index=True)
    df_intra.to_csv('./data/spy_intra_data.csv', index=False)
    print(f"Intraday data updated: {len(spy_intra_new)} new rows added.")
else:
    print("No new intraday data.")

if not spy_daily_new.empty:
    df_daily = pd.concat([daily_existing, spy_daily_new], ignore_index=True)
    df_daily.to_csv('./data/spy_daily_data.csv', index=False)
    print(f"Daily data updated: {len(spy_daily_new)} new rows added.")
else:
    print("No new daily data.")

if not dividends_new.empty:
    df_div = pd.concat([div_existing, dividends_new], ignore_index=True)
    df_div.to_csv('./data/dividends.csv', index=False)
    print(f"Dividends updated: {len(dividends_new)} new rows added.")
else:
    print("No new dividend data.")
