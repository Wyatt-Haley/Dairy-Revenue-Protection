import pandas as pd 
import requests
import zipfile
import io
from datetime import datetime

# Years to pull (adjust as needed)
years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Collect DataFrames from each year
dfs = []

# Mapping of praticecode to (year_offset, quarter)
praticecode_map = {
    801: (0, 10),  # Oct - Dec/Yr1 - Q4
    802: (1, 1),   # Jan - Mar/Yr2 - Q1
    803: (1, 4),   # Apr - Jun/Yr2 - Q2
    804: (1, 7),   # Jul - Sep/Yr2 - Q3
    805: (1, 10),  # Oct - Dec/Yr2 - Q4
    806: (2, 1),   # Jan - Mar/Yr3 - Q1
    807: (2, 4),   # Apr - Jun/Yr3 - Q2
    808: (2, 7),   # Jul - Sep/Yr3 - Q3
}

# Loop through each year and process
for year in years:
    url = f'https://pubfs-rma.fpac.usda.gov/pub/Web_Data_Files/Summary_of_Business/livestock_and_dairy_participation/drp_{year}_0002.zip'
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        filename = z.namelist()[0]
        with z.open(filename) as f:
            df = pd.read_csv(f, sep='|')

    # Assign column names
    df.columns = [
        "ryear", "cyear", "statecode", "stateabbreviation", "countycode", "countyname",
        "comcode", "comname", "insurancecode", "insurancename", "covtypecode",
        "coveragetypedesrip", "typecode", "typecodename", "praticecode", "praticecodename",
        "purchasedate", "cl", "pf", "CLWF", "CMWF", "bf", "p", "EEP", "EI", "declared", "sub",
        "tp", "pp", "liability", "indemnity"
    ]

    # Drop unwanted columns
    df.drop(columns=['ryear', 'comcode', 'comname', 'insurancecode', 'insurancename',
                     'covtypecode', 'coveragetypedesrip'], inplace=True, errors='ignore')

    # Remove rows with 99 or 999 codes
    df = df[(df['statecode'] != 99) & (df['countycode'] != 999)]

    # Add physical year and quarter
    df['pyear'] = df.apply(
        lambda row: row['cyear'] - 1 + praticecode_map.get(row['praticecode'], (None, None))[0], axis=1)
    df['qtr'] = df['praticecode'].map(lambda x: praticecode_map.get(x, (None, None))[1])

    # Convert to datetime
    df['purchasedate'] = pd.to_datetime(df['purchasedate'], errors='coerce')

    # Compute start of coverage
    def compute_coverage_start(row):
        offset, start_month = praticecode_map.get(row['praticecode'], (None, None))
        if offset is None or pd.isna(row['purchasedate']):
            return pd.NaT
        start_year = row['cyear'] - 1 + offset
        return datetime(start_year, start_month, 1)

    df['coverage_start_date'] = df.apply(compute_coverage_start, axis=1)

    # Calculate months between
    def months_between(start, end): 
        if pd.isna(start) or pd.isna(end):
            return None
        return (end.year - start.year) * 12 + (end.month - start.month)

    df['length'] = df.apply(
        lambda row: months_between(row['purchasedate'], row['coverage_start_date']), axis=1)

    # Size indicator
    df['mil'] = (df['declared'] >= 1_000_000).astype(int)

    # Coverage level categorization
    df['cl'] = 0
    df.loc[df['cl'] < 0.90, 'cl'] = 3
    df.loc[(df['cl'] >= 0.90) & (df['cl'] < 0.94), 'cl'] = 2
    df.loc[df['cl'] >= 0.95, 'cl'] = 1

    # Protection Factor categorization
    df['PF'] = 0
    df.loc[df['pf'] == 1.5, 'PF'] = 1
    df.loc[(df['pf'] > 1) & (df['pf'] < 1.5), 'PF'] = 2
    df.loc[df['pf'] == 1, 'PF'] = 3

    # Numeric encodings
    for col in ['cyear', 'qtr', 'length', 'statecode']:
        df[col + '_code'] = pd.factorize(df[col])[0] + 1

    # Weight variable
    df['wt'] = df['declared'] / 1_000_000

    # Append this year's processed df
    dfs.append(df)

# === Combine all years ===
combined_df = pd.concat(dfs, ignore_index=True)

# Save to CSV
combined_df.to_csv('drp_combined.csv', index=False)
