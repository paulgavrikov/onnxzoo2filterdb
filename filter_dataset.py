import pandas as pd
import h5py
import os
import numpy as np


class FilterDataset:

    def __init__(self, data_ID, root=".", filters=True, tables=True):
        if tables:
            print("Loading tables ...")
            self.load_tables(data_ID, root)
            print("Done")
        if filters:    
            print("Loading filters ...")
            self.load_filters(data_ID, root)
            print("Done")
    
    def load_filters(self, data_ID, root):
        with h5py.File(os.path.join(root, f"{data_ID}/dataset.h5"), 'r') as hf:
            self.filters = hf["filters"][...].reshape(-1, 9)

        
    def load_tables(self, data_ID, root):
        print(" df_filter_info")
        df_filter_info = pd.read_csv(os.path.join(root, f"{data_ID}/filterinfo.csv"), 
                             dtype={'filter_id_start': 'int32',
                                    'filter_id_end': 'int32',
                                    'model': 'int32',
                                    'depth': 'int32',
                                    'conv_depth': 'int32'})

        df_filter_info["filter_ids"] = df_filter_info.apply(lambda r: np.arange(r.filter_id_start, r.filter_id_end, dtype=np.int32), axis=1)
        df_filter_info = df_filter_info.rename(columns={"model": "model_id"})
        df_filter_info.set_index("model_id", inplace=True)
        
        print(" df_meta")
        df_meta = pd.read_csv(os.path.join(root, f"{data_ID}/datastorage.meta.csv"))
        df_meta = df_meta[df_meta["(3, 3) filters"] > 0]  # remove entries that have no assoc. filters
        df_meta = df_meta.rename(columns={df_meta.columns[0]: "model_id" })

        print(" df_meta_spreadsheets")
        df_meta_spreadsheets = pd.read_csv(os.path.join(root, f"{data_ID}/meta.csv"), dtype={'Comment': 'object',
        'Dataset URL': 'object',
        'Visual Category': 'object',
        'Visual Category_micro': 'object',
        'Pretraining': 'object',
        'Selection': 'object',
        'Tracer Warning': 'object',
        'Unnamed: 20': 'object',
        'Unnamed: 21': 'object',
        'Unnamed: 25': 'object'})
        df_meta_spreadsheets["Name"] = df_meta_spreadsheets["Name"].astype(str)

        print(" merging meta")
        df_fused_meta = df_meta.merge(df_meta_spreadsheets, how="left", left_on="model", right_on="Name").set_index("model_id")
        
        filter_cols = [name for name in df_fused_meta.columns.values if ") filters" in name and not "(1, 1)" in name]
        df_fused_meta["total_filters"] = df_fused_meta[filter_cols].sum(axis=1)
        df_fused_meta["3x3_filter_share"] = df_fused_meta["(3, 3) filters"] / df_fused_meta["total_filters"]

        self.df_filter_info = df_filter_info
        self.df_fused_meta = df_fused_meta

        # self.df_full = df_filter_info.merge(df_fused_meta, how='left', left_on='model', right_on='model_id', suffixes=('_filter',''))