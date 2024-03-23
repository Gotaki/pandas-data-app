# %%

import pandas as pd
import numpy as np

def infer_and_convert_data_types(df, categorical_threshold=0.5):
    for col in df.columns:
        df_converted = pd.to_numeric(df[col], errors='coerce')
        if not df_converted.isna().all():
            df[col] = df_converted
            continue

        try:
            df[col] = pd.to_datetime(df[col], errors='raise', infer_datetime_format=True)
            continue
        except (ValueError, TypeError):
            pass

        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
            if df[col].nunique() / len(df[col]) < categorical_threshold:  
                df[col] = df[col].astype('category')
                df[col] = df[col].cat.add_categories([''])

        if df[col].dtype in ['float64', 'float32'] and 'j' in df[col].astype(str).str.lower().str.strip().tolist():
            df[col] = df[col].astype(complex)
        
        df[col].fillna('', inplace=True)
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        
    return df

# Test the function with your DataFrame
df = pd.read_csv('sample_data.csv')
print("Data types before inference:")
print(df.dtypes)

df = infer_and_convert_data_types(df)

print("\nData types after inference:")
print(df.dtypes)

# %%
