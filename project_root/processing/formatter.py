
class Formatter(ProcessingStep):
    def __init__(self):
        super().__init__()
        if 'mmsi' in config.static_columns:
            logging.warning("MMSI should not be in static_columns - removing it")
            config.static_columns = [col for col in config.static_columns if col != 'mmsi']
        self._define_directories(
            from_name='windowed_with_stride_3' + ('_debug' if args.debug else ''),
            to_name='formatted_data_with__stride_3' + ('_debug' if args.debug else '')
        )
        self._initialize_logging(args.save_log, 'format_data')

        logging.info(f'categorical_columns used are {config.categorical_columns}')
        self.dataset_names = [
            'train_long_term_train',
            'test_long_term_test',
            'valid_long_term_train',
            'valid_long_term_test'
        ]

    def load(self):
        for dataset_name in self.dataset_names:
            dataset_path = os.path.join(self.from_dir, f'{dataset_name}.parquet')
            self.datasets[dataset_name] = dd.read_parquet(dataset_path)
            tmp_dir = os.path.join(self.from_dir, f'.tmp_{dataset_name}.parquet')
            if not os.path.exists(tmp_dir):
                self._repartition_ds(dataset_name)
            self.datasets[dataset_name] = dd.read_parquet(tmp_dir)
        logging.info('File paths have been specified for dask')

    def save(self):
        conserve_memory = True
        clear_path(self.to_dir)
        os.mkdir(self.to_dir)
        self.features.to_csv(os.path.join(self.to_dir, 'features.csv'))

        for dataset_name in self.dataset_names:
            set_dir = os.path.join(self.to_dir, dataset_name)
            os.mkdir(set_dir)

            if not conserve_memory:
                self.datasets[dataset_name] = self.datasets[dataset_name].compute()

            for new_time_gap in config.time_gaps:
                x_idxs = np.arange(0,
                                   config.length_of_history,
                                   new_time_gap / config.interpolation_time_gap, dtype=int)

                if 'long_term' in dataset_name:
                    timesteps_into_the_future = config.length_of_history + config.length_into_the_future + 1
                else:
                    raise ValueError('Unknown Dataset')

                next_gap = config.length_of_history + new_time_gap / config.interpolation_time_gap - 1
                y_idxs = np.arange(next_gap,
                                   timesteps_into_the_future,
                                   new_time_gap / config.interpolation_time_gap, dtype=int)

                num_minutes = int(new_time_gap / 60)

                if conserve_memory:
                    npart = self.datasets[dataset_name].npartitions
                    x_dir = os.path.join(set_dir, f'{num_minutes}_min_time_gap_x')
                    os.mkdir(x_dir)
                    x_len = 0
                    for i in range(npart):
                        x_path = os.path.join(set_dir, f'{num_minutes}_min_time_gap_x', f'{i}.npy')
                        data = self.datasets[dataset_name].partitions[i].compute()[:, x_idxs, :]
                        gc.collect()
                        np.save(x_path, data)
                        x_len += len(data)
                        x_shape = list(data.shape)
                        del data
                        gc.collect()
                    x_shape[0] = x_len
                    logging.info(f'For {dataset_name}, the shape of {num_minutes}_min_time_gap_x is {x_shape}')
                else:
                    x_path = os.path.join(set_dir, f'{num_minutes}_min_time_gap_x.npy')
                    X = self.datasets[dataset_name][:, x_idxs, :]
                    np.save(x_path, X)
                    logging.info(f'For {dataset_name}, the shape of {num_minutes}_min_time_gap_x.npy is {X.shape}')
                    del X

                if conserve_memory:
                    y_dir = os.path.join(set_dir, f'{num_minutes}_min_time_gap_y')
                    os.mkdir(y_dir)
                    y_len = 0
                    for i in range(npart):
                        y_path = os.path.join(set_dir, f'{num_minutes}_min_time_gap_y', f'{i}.npy')
                        data = self.datasets[dataset_name].partitions[i].compute()[:, y_idxs, :]
                        gc.collect()
                        y_len += len(data)
                        np.save(y_path, data)
                        y_shape = list(data.shape)
                        del data
                        gc.collect()
                    y_shape[0] = y_len
                    logging.info(f'For {dataset_name}, the shape of {num_minutes}_min_time_gap_y is {y_shape}')
                else:
                    y_path = os.path.join(set_dir, f'{num_minutes}_min_time_gap_y.npy')
                    Y = self.datasets[dataset_name][:, y_idxs, :]
                    np.save(y_path, Y)
                    logging.info(f'For {dataset_name}, the shape of {num_minutes}_min_time_gap_y.npy is {Y.shape}')
                    del Y

            logging.info(f'{dataset_name} set saved to directory {set_dir}')
            del self.datasets[dataset_name]
        self._clear_tmp_files()

    def _reshape_partition(self, partition: pd.DataFrame, into_the_future):
        partition = partition.to_numpy()
        num_ts = config.length_of_history + into_the_future
        partition = partition[[range(idx, idx + num_ts) for idx in range(0, len(partition), num_ts)]]
        partition = np.stack(partition)
        return partition

    def _one_hot(self, dataset_name):
        for col in config.categorical_columns:
            if col not in self.datasets[dataset_name].columns:
                raise KeyError(f"Column '{col}' not found in dataset '{dataset_name}'")
            self.datasets[dataset_name][col] = self.datasets[dataset_name][col].astype('category')

        self.datasets[dataset_name] = self.datasets[dataset_name].categorize()
        encoder = OneHotEncoder()
        encoder = encoder.fit(self.datasets[dataset_name][config.categorical_columns])

        if not hasattr(self, 'column_order'):
            self.column_order = []
            for col, levels in zip(config.categorical_columns, encoder.categories_):
                for level in levels:
                    self.column_order += [f'{col}_{level}']

        transformed = encoder.transform(self.datasets[dataset_name][config.categorical_columns])
        transformed = transformed[self.column_order].astype('bool')

        for col in config.categorical_columns:
            categories = self.datasets[dataset_name][col].cat.categories.to_list()
            logging.info(f'Using {len(categories)} values for column {col} in {dataset_name} set: {categories}')

        self.datasets[dataset_name] = self.datasets[dataset_name].drop(config.categorical_columns, axis=1)
        return encoder

    def extract_static_features(self, dataset_name):
        static_df = (
        self.datasets[dataset_name][['mmsi'] + config.static_columns]  # Include MMSI for grouping
        .drop_duplicates(subset=['mmsi'])
        .drop(columns=['mmsi'])  # Explicitly remove MMSI before saving
        .compute() )
        save_path = os.path.join(self.to_dir, dataset_name, 'static_features.csv')
        static_df.to_csv(save_path, index=False)
        logging.info(f'Saved static features for {dataset_name} with shape {static_df.shape}')

    def calculate(self):
        for dataset_name in self.dataset_names:
            # Add this debug check:
            logging.debug(f"Columns before processing in {dataset_name}: {self.datasets[dataset_name].columns.tolist()}")
            assert 'mmsi' in self.datasets[dataset_name].columns, "MMSI column missing!"
            self._one_hot(dataset_name)
            self.extract_static_features(dataset_name)
            self.features = self.datasets[dataset_name].dtypes.astype(str)
            self.features = self.features.replace('Sparse[bool, False]', 'bool')
            self.features.name = 'dtype'
            self.features.index.name = 'column'

            if 'long_term' in dataset_name:
                timesteps_into_the_future = config.length_into_the_future + 1
            else:
                raise ValueError('Unknown Dataset')

            out_meta = [(i, z) for i, z in self.datasets[dataset_name].dtypes.items()]
            self.datasets[dataset_name] = self.datasets[dataset_name].map_partitions(
                self._reshape_partition, timesteps_into_the_future, meta=out_meta
            )
        logging.info(f'Calculation methods have been defined for Dask')

    def _repartition_ds(self, dataset_name):
        dataset_path = os.path.join(self.from_dir, f'{dataset_name}.parquet')
        tmp_dir = os.path.join(self.from_dir, f'.tmp_{dataset_name}.parquet')
        MAX_PARTITION_SIZE = '4000MB'
        self.datasets[dataset_name] = self.datasets[dataset_name].repartition(partition_size=MAX_PARTITION_SIZE,
                                                                              force=True)
        divisions = [self.datasets[dataset_name].partitions[i].index.min().compute() for i in
                     range(self.datasets[dataset_name].npartitions)]
        divisions += [self.datasets[dataset_name].partitions[
                          self.datasets[dataset_name].npartitions - 1].index.max().compute() + 1]

        self.datasets[dataset_name] = dd.read_parquet(dataset_path)
        self.datasets[dataset_name] = self.datasets[dataset_name].repartition(
            divisions=divisions,
            force=True)
        dd.to_parquet(self.datasets[dataset_name], tmp_dir, schema='infer')
        del self.datasets[dataset_name]

    def _clear_tmp_files(self):
        for dataset_name in self.dataset_names:
            tmp_dir = os.path.join(self.from_dir, f'.tmp_{dataset_name}.parquet')
            clear_path(tmp_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', choices=datasets.keys())
    parser.add_argument('-l', '--log_level', type=int, default=2, choices=[0, 1, 2, 3, 4])
    parser.add_argument('-s', '--save_log', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    config.set_log_level(args.log_level)
    config.dataset_config = datasets[args.dataset_name]

    dask.config.set(scheduler='single-threaded')

    formatter = Formatter()
    formatter.load()
    for dataset_name in formatter.dataset_names:
        print(f" Columns in {dataset_name}:")
        print(formatter.datasets[dataset_name].columns)
    formatter.calculate()
    formatter.save()
