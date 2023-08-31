from sklearn.model_selection import train_test_split
import pandas as pd

patient_info_df = pd.read_csv("")
train_val_set, test_set = train_test_split(
    patient_info_df.index.values,
    test_size=0.15,
    stratify=patient_info_df['Recurrence_status'].values
)
patient_info_df.loc[train_val_set, 'data_subset'] = 'train/validation'
patient_info_df.loc[test_set, 'data_subset'] = 'test'
tile_info_df = tile_info_df.drop(
    columns='data_subset',
    errors='ignore'
).join(
    patient_info_df['data_subset'],
    on='patient_id'
)

train_val_mask = tile_info_df['data_subset'] != 'test'
train_set, val_set = train_test_split(
    tile_info_df.index.values[train_val_mask],
    train_size=0.882,
    stratify=tile_info_df['patient_id'].values[train_val_mask]
)
tile_info_df.loc[train_set, 'data_subset'] = 'train'
tile_info_df.loc[val_set, 'data_subset'] = 'validation'

tile_info_df.to_csv(tile_df_save_path)
print(f'Saved updated tile_info_df to "{tile_df_save_path}"')
patient_info_df.to_csv(patient_df_save_path)
print(f'Saved updated patient_info_df to "{patient_df_save_path}"')

tile_info_df.groupby('patient_id')['data_subset'].unique()

