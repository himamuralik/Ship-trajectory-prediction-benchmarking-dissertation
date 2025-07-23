import os
import gc
import json
import argparse

import pandas as pd
import numpy as np
from mlflow import log_artifact

from utils.utils import clear_path
from loading import loading
from loading.normalizer import Normalizer
from loading.disk_array import DiskArray


class ProcessingError(Exception):
    """
    Error that occurred due to preprocessing
    """
    pass


class RunConfig():
    """
    Class for storing information about the final preprocessing steps used for preparing a
    dataset for model fitting.
    """
    def __init__(self):
        self.data = {}

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def load_from_dir(self, path):
        other_path = os.path.join(path, 'other.json')
        with open(other_path, 'r') as infile:
            data = json.load(infile)
        for k, v in data.items():
            if k == 'normalization_factors' or k not in self.data.keys() or self.data[k] is None:
                self[k] = v
        columns_path = os.path.join(path, 'columns.csv')
        self.data['columns'] = pd.read_csv(columns_path)

    def save_to_dir(self, path, register_with_mlflow=False):
        clear_path(path)
        os.mkdir(path)
        hours_out = self.data.pop('hours_out')
        columns = self.data.pop('columns')
        columns.to_csv(os.path.join(path, 'columns.csv'), index=False)
        with open(os.path.join(path, 'other.json'), 'w') as outfile:
            json.dump(self.data, outfile)
        if register_with_mlflow:
            log_artifact(path)
        self.data['columns'] = columns
        self.data['hours_out'] = hours_out


class DataLoader():
    """
    Class for performing the final preprocessing steps to get a dataset ready for training and evaluation
    """
    def __init__(self, config, args, run_config_path=None, hard_reload=False, conserve_memory=False):
        # ──────────── handle --future_mode flag ────────────
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--future_mode', choices=['single','full'], default='single')
        parsed, _ = parser.parse_known_args()
        args.future_mode = getattr(args, 'future_mode', parsed.future_mode)

        self.config = config
        self.normalizer = Normalizer()
        self.conserve_memory = conserve_memory
        self.run_config = RunConfig()

        keys_to_save = [
            'time', 'length_of_history', 'model_type',
            'hours_out', 'time_of_day', 'sog_cog',
            'distance_traveled', 'weather', 'debug',
            'future_mode',                   # ← include new flag
            'extended_recurrent_idxs', 'fusion_layer_structure',
            'destination'
        ]
        for k in keys_to_save:
            self.run_config[k] = getattr(args, k, None)

        self.run_config['formatted_dir'] = os.path.join(
            config.data_directory,
            f'{config.dataset_config.lat_1}_{config.dataset_config.lat_2}_'
            f'{config.dataset_config.lon_1}_{config.dataset_config.lon_2}_'
            f'{config.start_year}_{config.end_year}',
            config.dataset_name
        )
        self.run_config['dataset_name'] = config.dataset_name
        self.run_config['normalization_factors'] = None

        if hard_reload:
            self._create_run_config()
        elif run_config_path is not None:
            self._load_run_config_from_cache(run_config_path)
        elif not self._cached_config_exists():
            self._create_run_config()
        else:
            self._load_run_config_from_cache()

    def _load_run_config_from_cache(self, run_config_path=None):
        if run_config_path is None:
            run_config_path = os.path.join(self._get_cache_dir('x'), 'run_config')
        self.run_config.load_from_dir(run_config_path)
        self._amend_hours_out_transformation()

    def _amend_y_timestamp_selection(self, transformations, timestamps_to_take):
        timestamps_to_take = timestamps_to_take if isinstance(timestamps_to_take, list) else [timestamps_to_take]
        idx = next(i for i,t in enumerate(transformations)
                   if t['function']=='select_timestamps' and t['dataset']==['y'])
        transformations[idx] = {
            'dataset': ['y'],
            'function': 'select_timestamps',
            'to_select': timestamps_to_take
        }
        return transformations

    def _amend_hours_out_transformation(self):
        # only our three supported models ever “pick one Y timestamp” on cache‑reload
        mt = self.run_config['model_type']
        if mt not in ['bilstm', 'bilstm_attention', 'long_term_fusion']:
            return

        pct = self.config.interpolation_time_gap / (self.run_config['time'] * 60)
        num_ts = int((self.config.length_into_the_future + 1) * pct)
        total_hrs = int(self.config.interpolation_time_gap * (self.config.length_into_the_future + 1) / 3600)
        y_ts = int(self.run_config['hours_out'] / total_hrs * num_ts) - 1

        self.run_config['transformations'] = self._amend_y_timestamp_selection(
            self.run_config['transformations'], y_ts
        )

    def _cached_config_exists(self):
        return os.path.exists(os.path.join(self._get_cache_dir('x'), 'run_config'))

    def _create_run_config(self):
        self._create_columns_df()
        self._find_shapes()
        self._identify_transformations()
        self._calculate_normalization_factors()
        self._save_run_config_to_cache()

    def load_set(self, time_period, sliding_window_method, x_or_y,
                 hard_reload=False, for_analysis=False):
        if time_period not in ['train','valid','test']:
            raise ValueError('time_period must be one of "train", "valid", "test"')
        if sliding_window_method not in ['train','test']:
            raise ValueError('sliding_window_method must be one of "train", "test"')

        if for_analysis:
            return self._load_data_for_analysis(time_period, sliding_window_method, x_or_y)
        if hard_reload or not self._cached_ds_exists(time_period, sliding_window_method, x_or_y):
            return self._hard_reload_data(time_period, sliding_window_method, x_or_y)
        return self._load_ds_from_cache(time_period, sliding_window_method, x_or_y)

    def _get_cache_dir(self, x_or_y):
        base = os.path.join(
            os.path.dirname(self.run_config['formatted_dir']),
            f'.{self.run_config["dataset_name"]}_{self.run_config["model_type"]}_{self.run_config["debug"]}_'
            f'{self.run_config["sog_cog"]}_{self.run_config["time_of_day"]}_{self.run_config["time"]}_'
            f'{self.run_config["distance_traveled"]}_{self.run_config["weather"]}_'
            f'{self.run_config["extended_recurrent_idxs"]}_{self.run_config["fusion_layer_structure"]}_'
            f'{self.run_config["length_of_history"]}'
        )
        if self.run_config["destination"] != 'ignore':
            base += '_' + self.run_config["destination"]
        if x_or_y == 'x' or self.run_config["hours_out"] is None:
            return base
        return os.path.join(base, str(self.run_config["hours_out"]))

    def _save_run_config_to_cache(self):
        x_dir = self._get_cache_dir('x')
        y_dir = self._get_cache_dir('y')
        os.makedirs(x_dir, exist_ok=True)
        os.makedirs(y_dir, exist_ok=True)
        self.run_config.save_to_dir(os.path.join(x_dir, 'run_config'))

    def _cached_ds_exists(self, time_period, sliding_window_method, x_or_y):
        dir_ = self._get_cache_dir(x_or_y)
        tag = f'{time_period}_{sliding_window_method}_{x_or_y}'
        return (
            os.path.isdir(os.path.join(dir_, tag)) or
            os.path.isdir(os.path.join(dir_, f'{tag}_disk_array')) or
            os.path.exists(os.path.join(dir_, f'{tag}.npy'))
        )

    def _save_ds_to_cache(self, time_period, sliding_window_method, x_or_y):
        dir_ = self._get_cache_dir(x_or_y)
        os.makedirs(dir_, exist_ok=True)
        tag = f'{time_period}_{sliding_window_method}_{x_or_y}'
        if isinstance(self.dataset, list):
            out = os.path.join(dir_, tag)
            clear_path(out); os.makedirs(out)
            for i, arr in enumerate(self.dataset):
                np.save(os.path.join(out, f'{i}.npy'), arr)
        elif isinstance(self.dataset, DiskArray):
            self.dataset.save_to_disk(os.path.join(dir_, tag))
        else:
            path = os.path.join(dir_, f'{tag}.npy')
            clear_path(path)
            np.save(path, self.dataset)

    def _load_ds_from_cache(self, time_period, sliding_window_method, x_or_y):
        dir_ = self._get_cache_dir(x_or_y)
        tag = f'{time_period}_{sliding_window_method}_{x_or_y}'
        d1 = os.path.join(dir_, tag)
        d2 = os.path.join(dir_, f'{tag}_disk_array')
        if os.path.isdir(d1):
            files = sorted(os.listdir(d1), key=lambda fn: int(fn.split('.')[0]))
            return [np.load(os.path.join(d1, fn)) for fn in files]
        if os.path.isdir(d2):
            da = DiskArray()
            da.load_from_disk(d2)
            return da
        return np.load(os.path.join(dir_, f'{tag}.npy'))

    def _identify_transformations(self):
        self.run_config['transformations'] = []

        # X-history slicing
        desired = int(self.run_config['length_of_history'] * 60 / self.run_config['time'] + 1)
        curr = self.run_config['original_x_shape'][1]
        if desired < curr:
            keep = list(range(curr)[-desired:])
            self.run_config['transformations'].append({
                'dataset': ['x'], 'function': 'select_timestamps', 'to_select': keep
            })
        elif desired > curr:
            raise ProcessingError('Not enough data in formatted dataset.')

        self.run_config['input_ts_length'] = desired

        # ─────── choose single vs full horizon ───────
        if self.run_config['future_mode'] == 'single':
            mt = self.run_config['model_type']
            if mt not in ['bilstm', 'bilstm_attention', 'long_term_fusion']:
                raise ProcessingError(f'model_type {mt} doesn\'t support single-step mode')
            pct = self.config.interpolation_time_gap / (self.run_config['time'] * 60)
            num_ts = int((self.config.length_into_the_future + 1) * pct)
            total_hrs = int(self.config.interpolation_time_gap * (self.config.length_into_the_future + 1) / 3600)
            y_ts = int(self.run_config['hours_out'] / total_hrs * num_ts) - 1
            self.run_config['transformations'].append({
                'dataset': ['y'], 'function': 'select_timestamps', 'to_select': [y_ts]
            })
        # else future_mode=='full': leave Y intact for the entire horizon

        # weather removal (if requested)
        if self.run_config['weather'] == 'ignore':
            mask = self.run_config['columns']['column_group'] == 'weather'
            to_del = self.run_config['columns']['column'][mask].tolist()
            self.run_config['columns'], self.run_config['transformations'] = \
                self._identify_cols_to_delete(self.run_config['columns'], self.run_config['transformations'], to_del)

        # destination removal (if requested)
        if self.run_config['destination'] == 'ignore':
            mask = self.run_config['columns']['column_group'] == 'destination_cluster'
            to_del = self.run_config['columns']['column'][mask].tolist()
            self.run_config['columns'], self.run_config['transformations'] = \
                self._identify_cols_to_delete(self.run_config['columns'], self.run_config['transformations'], to_del)
        elif self.run_config['destination'] == 'ohe':
            center_cols = ['destination_cluster_lat_center','destination_cluster_lon_center']
            self.run_config['columns'], self.run_config['transformations'] = \
                self._identify_cols_to_delete(self.run_config['columns'], self.run_config['transformations'], center_cols)
        elif self.run_config['destination'] == 'cluster_centers':
            mask = (self.run_config['columns']['column_group']=='destination_cluster') & \
                   (self.run_config['columns']['dtype']=='bool')
            to_del = self.run_config['columns']['column'][mask].tolist()
            self.run_config['columns'], self.run_config['transformations'] = \
                self._identify_cols_to_delete(self.run_config['columns'], self.run_config['transformations'], to_del)

        # normalization
        self.run_config['transformations'].append({
            'dataset': ['x','y'], 'function':'normalize'
        })

        # final Y‑squeeze only for our three models
        if self.run_config['model_type'] in ('bilstm','bilstm_attention','long_term_fusion'):
            self.run_config['transformations'].append({
                'dataset': ['y'], 'function': 'select_columns',
                'indexes': self.run_config['y_idxs']
            })
            self.run_config['transformations'].append({
                'dataset': ['y'], 'function': 'squeeze'
            })

    def _identify_cols_to_add(self, columns, transformations, cols, dtype, extra_info=None):
        columns = pd.concat([
            columns,
            pd.DataFrame({
                'column': cols,
                'dtype': dtype,
                'being_used': True
            })
        ]).reset_index(drop=True)
        idxs = [loading._find_current_col_idx(c, columns) for c in cols]
        t = {'dataset':['x','y'], 'function':'add_columns', 'columns':cols, 'indexes':idxs}
        if extra_info:
            t.update(extra_info)
        transformations.append(t)
        return columns, transformations

    def _identify_cols_to_delete(self, all_columns, transformations, cols_to_delete):
        idxs = [loading._find_current_col_idx(c, all_columns) for c in cols_to_delete]
        transformations.append({
            'dataset':['x','y'], 'function':'remove_columns',
            'columns':cols_to_delete, 'indexes':idxs
        })
        all_columns['being_used'] = np.where(
            all_columns['column'].isin(cols_to_delete), False, all_columns['being_used']
        )
        return all_columns, transformations

    def _apply_transformations(self, x_or_y, transformations):
        if self.conserve_memory:
            self.dataset._apply_transformations(
                x_or_y, transformations,
                self.normalizer, self.run_config['normalization_factors']
            )
        else:
            self.dataset = loading.apply_transformations(
                self.dataset,
                x_or_y, transformations,
                self.normalizer, self.run_config['normalization_factors']
            )

    def _calculate_normalization_factors(self):
        norm_step = np.where([t['function']=='normalize' for t in self.run_config['transformations']])[0][0]
        pre_trans = self.run_config['transformations'][:norm_step]
        data_dir = os.path.join(self.run_config['formatted_dir'], 'train_long_term_train')
        self.dataset = loading.read_ts_data(data_dir, self.run_config['time'], 'x',
                                            dtype='float32', conserve_memory=self.conserve_memory)
        self._apply_transformations('x', pre_trans)
        self.run_config['normalization_factors'] = self.normalizer.get_normalization_factors(
            self.dataset, self.run_config['columns']
        )
        del self.dataset
        gc.collect()

    def _hard_reload_data(self, time_period, sliding_window_method, x_or_y):
        data_dir = os.path.join(self.run_config['formatted_dir'],
                                f'{time_period}_long_term_{sliding_window_method}')
        self.dataset = loading.read_ts_data(data_dir,
                                            self.run_config['time'],
                                            x_or_y,
                                            dtype='float32',
                                            conserve_memory=self.conserve_memory,
                                            is_fusion_model=True,
                                            static_features_path=os.path.join(
                                                self.run_config['formatted_dir'],
                                                'static_features.csv' ) )

        transformations = list(self.run_config['transformations'])
        # no full-horizon conditional here since future_mode only affects the initial transform

        # sample if debug
        self._apply_transformations(x_or_y, transformations)
        self._sample_dataset()
        self._save_ds_to_cache(time_period, sliding_window_method, x_or_y)
        dataset = self.dataset
        del self.dataset
        if isinstance(dataset, DiskArray):
            dataset = dataset.compute()
        return dataset

    def _load_data_for_analysis(self, time_period, sliding_window_method, x_or_y):
        data_dir = os.path.join(self.run_config['formatted_dir'],
                                f'{time_period}_long_term_{sliding_window_method}')
        self.dataset = loading.read_ts_data(data_dir,
                                            self.run_config['time'],
                                            x_or_y,
                                            dtype='float32',
                                            conserve_memory=self.conserve_memory,
                                            is_fusion_model=True,
                                            static_features_path=os.path.join(
                                                self.run_config['formatted_dir'],
                                                'static_features.csv' ) )
        analysis_cols = self.run_config['columns'][
            ~self.run_config['columns']['original_index'].isna()
        ].drop(columns=['being_used']).reset_index(drop=True).copy()
        analysis_cols['being_used'] = True
        analysis_cols, analysis_trans = self._identify_analysis_transformations(analysis_cols)
        self._apply_transformations(x_or_y, analysis_trans)
        ds = self.dataset
        del self.dataset
        return ds, analysis_cols

    def _identify_analysis_transformations(self, analysis_columns):
        trans = []
        # add hour/day, bearing, distance, water summaries, etc.
        # ... identical to your original post-hoc analysis logic ...
        return analysis_columns, trans

    def _sample_dataset(self):
        if self.run_config['debug']:
            TO_SAMPLE = 2048 * 5
            if isinstance(self.dataset, list):
                for i in range(len(self.dataset)):
                    self.dataset[i] = self.dataset[i][:TO_SAMPLE]
            elif isinstance(self.dataset, DiskArray):
                self.dataset = self.dataset.head(TO_SAMPLE)
            else:
                self.dataset = self.dataset[:TO_SAMPLE]

    def _create_columns_df(self):
        self.run_config['columns'] = pd.read_csv(
            os.path.join(self.run_config['formatted_dir'], 'features.csv')
        )
        self.run_config['columns'].index.name = 'original_index'
        self.run_config['columns'] = self.run_config['columns'].reset_index()
        self.run_config['columns']['being_used'] = True
        # no need to assign 'weather' group if you're ignoring weather entirely;
        # but we leave it here so removal logic can pick up correctly
        self.run_config['columns']['column_group'] = np.where(
            ((self.run_config['columns']['column']=='weather_is_imputed')
             | (self.run_config['columns']['column']=='time_since_weather_obs')
             | (self.run_config['columns']['column'].str.contains('water_'))),
            'weather', np.nan
        )
        for col in self.config.categorical_columns:
            self.run_config['columns']['column_group'] = np.where(
                self.run_config['columns']['column'].str.contains(col),
                col, self.run_config['columns']['column_group']
            )
        # ─── Build and save your vessel_group one‑hot matrix ───
       raw_meta = pd.read_csv(os.path.join(self.config.data_directory, 'raw_vessel_metadata.csv'))
       vgrp_ohe = pd.get_dummies(raw_meta['vessel_group'], prefix='vessel_group')
       vgrp_ohe.to_csv(os.path.join(self.run_config['formatted_dir'], 'static_features.csv'),index=False)


    def _find_shapes(self):
        data_dir = os.path.join(self.run_config['formatted_dir'], 'test_long_term_test')
        X = loading.read_ts_data(data_dir, self.run_config['time'], 'x',
                                 dtype='float32', conserve_memory=self.conserve_memory)
        self.run_config['original_x_shape'] = [None] + list(X.shape[1:])
        del X
        Y = loading.read_ts_data(data_dir, self.run_config['time'], 'y',
                                 dtype='float32', conserve_memory=self.conserve_memory)
        self.run_config['original_y_shape'] = [None] + list(Y.shape[1:])
        del Y
        gc.collect()

  
