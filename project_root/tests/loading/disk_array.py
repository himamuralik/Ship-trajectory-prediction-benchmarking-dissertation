import tempfile
import numpy as np
import gc
import os
import atexit
from loading import loading
from utils.utils import total_system_ram

class DiskArray():
    """
    Light weight class for processing large data sets in chunks

    Saves chunks as tmp files on disk, so it only has to have one chunk in memory at a time.

    Chunks can contain multiple arrays, meaning a DiskArray can be equivalent to 2+ numpy arrays (although these
    arrays will need to have the same number of rows).

    Keeps track of the complete size of all chunks, and if the chunks are small enough to fit in memory all together,
    can convert itself to a numpy array or list of numpy arrays
    """
    def __init__(self):
        self.temp_paths = []
        self.shape = None
        self.nbytes = 0
        self.axis = 0
        self.static_features = None
        self.excluded_columns = None  # Track indices of weather/destination columns to exclude
        atexit.register(self.exit_handler, self)

    def set_excluded_columns(self, column_indices):
        """Set which column indices should be excluded (weather/destination columns)"""
        self.excluded_columns = column_indices

    def attach_static_features(self, static_data):
        """Associate static features with the DiskArray"""
        if self.shape is not None and len(static_data) != self.shape[0]:
            raise ValueError("Static feature count doesn't match main data")
        self.static_features = static_data

    def add_array(self, array):
        """Add array with column filtering"""
        if self.excluded_columns is not None:
            array = self._filter_excluded_columns(array)
        self._update_basic_info(array)
        
        self.temp_paths.append(tempfile.TemporaryDirectory())
        self._save_partition_to_path(self.temp_paths[-1].name, array)
        del array
        gc.collect()

    def _filter_excluded_columns(self, data):
        """Remove excluded columns from data chunks"""
        if self.excluded_columns is None:
            return data
        if isinstance(data, list):
            return [np.delete(d, self.excluded_columns, axis=1) for d in data]
        return np.delete(data, self.excluded_columns, axis=1)

    def _save_partition_to_path(self, path, data):
        """
        Save one of the DiskArray's chunks to a specific path

        The data object can be a numpy array or list of numpy arrays.

        :param path: The path to save to
        :param data: The chunk to be saved
        :return:
        """
        if type(data) != list:
            data = [data]
        for i, d in enumerate(data):
            if not os.path.exists(path):
                os.mkdir(path)
            save_path = os.path.join(path, f'{i}.npy')
            np.save(save_path, d)

    def _load_partition_from_path(self, path):
        """
        Load a chunk from disk

        :param path: The path to save to
        :return:
        """
        files = os.listdir(path)
        files = np.array(files)[np.argsort([int(n.split('.')[0]) for n in files])].tolist()
        data = []
        for f in files:
            data += [np.load(os.path.join(path, f))]
        if len(data) == 1:
            data = data[0]
        return data

    def __getitem__(self, item):
        """
        Load a chunk by index

        :param item: index of chunk to load
        :return:
        """
        return self._load_partition_from_path(self.temp_paths[item].name)

    def compute(self):
        """Convert to numpy array with static features if available"""
        if self.nbytes > (total_system_ram() * 0.95):
            raise MemoryError('Not enough memory to load the dataset')
            
        data = [self._load_partition_from_path(t.name) for t in self.temp_paths]
        
        # Handle main data concatenation
        if isinstance(data[0], list):
            joined_data = [np.concatenate([d[i] for d in data], axis=self.axis) 
                          for i in range(len(data[0]))]
        else:
            joined_data = np.concatenate(data, axis=self.axis)
        
        # Return format depends on static features presence
        if self.static_features is not None:
            if isinstance(joined_data, list):
                return joined_data + [self.static_features]
            return (joined_data, self.static_features)
        return joined_data

    def head(self, n):
        """
        Return the first n rows of DiskArray

        Will return two numpy arrays if DiskArray contains two arrays

        :param n: Number of rows
        :return:
        """
        sampled = 0
        head_data = None
        for t in self.temp_paths:
            data = self._load_partition_from_path(t.name)
            if type(data) == list:
                t_len = len(data[0])
                to_sample = min(t_len, n - sampled)
                data = [d[:to_sample] for d in data]
                if head_data is None:
                    head_data = data
                else:
                    head_data = [np.concatenate([hd, d], axis=self.axis) for hd, d in zip(head_data, data)]
            else:
                t_len = len(data)
                to_sample = min(t_len, n - sampled)
                data = data[:to_sample]
                if head_data is None:
                    head_data = data
                else:
                    head_data = np.concatenate([head_data, data], axis=self.axis)
            sampled += to_sample
            if sampled == n:
                break
            assert sampled < n

        if sampled < n:
            raise UserWarning('Dataset does not contain the desired number of records. Entire dataset returned')

        return head_data

    def save_to_disk(self, dir):
        """
        Save DiskArray to a path

        If the size of the disk array is less than 35% of total system ram, the array will be converted to
        numpy arrays beforehand. Otherwise the chunks will be saved individually

        :param dir:
        :return:
        """
        if self.nbytes / total_system_ram() < 0.35:
            data = self.compute()
            if type(data) == list:
                os.mkdir(dir)
                for i, set in enumerate(data):
                    path = os.path.join(dir, f'{i}.npy')
                    np.save(path, set)
            else:
                path = dir + '.npy'
                np.save(path, data)
        else:
            dir += '_disk_array'
            if not os.path.exists(dir):
                os.mkdir(dir)
            for i, t in enumerate(self.temp_paths):
                data = self._load_partition_from_path(t.name)
                t_dir = os.path.join(dir, str(i))
                self._save_partition_to_path(t_dir, data)
                    
    def load_from_disk(self, dir):
        """
        Load a disk array from a save path

        :param dir:
        :return:
        """
        partitions = os.listdir(dir)
        for p in partitions:
            path = os.path.join(dir, p)
            data = self._load_partition_from_path(path)
            self.add_array(data)

    def __del__(self):
        """
        Delete self

        :return:
        """
        self.close()

    def __len__(self):
        """
        Number of rows

        :return:
        """
        return self.shape[0]

    def _calculate_min_max(self):
        """
        Calculate the minimum/maximum value for each column in DiskArray

        Used for calculating normalization factors

        :return:
        """
        num_columns = self.shape[-1]
        mins = np.array(np.ones((num_columns,)) * np.inf)
        maxes = np.array(np.ones((num_columns,)) * -np.inf)

        for t in self.temp_paths:
            data = self._load_partition_from_path(t.name)
            partition_mins = data.min(axis=0)
            partition_maxes = data.max(axis=0)
            while len(partition_mins.shape) > 1:
                partition_mins = partition_mins.min(axis=0)
                partition_maxes = partition_maxes.max(axis=0)
            mins = np.min([mins, partition_mins], axis=0)
            maxes = np.max([maxes, partition_maxes], axis=0)
            del data
            gc.collect()

        return mins, maxes

    def _update_basic_info(self, data):
        """
        Update summary information for DiskArray

        Keeps track of the total number of bytes of all chunks, as well as the total shape

        :param data: New chunk
        :return:
        """
        if type(data) == list:
            self.nbytes += np.sum([a.nbytes for a in data]).astype(int)
            if self.shape == None:
                self.shape = [a.shape for a in data]
            else:
                for i, (s, a) in enumerate(zip(self.shape, data)):
                    s = list(s)
                    s[self.axis] += a.shape[self.axis]
                    self.shape[i] = tuple(s)
        else:
            self.nbytes += data.nbytes
            if self.shape == None:
                self.shape = data.shape
            else:
                self.shape = list(self.shape)
                self.shape[self.axis] += data.shape[self.axis]
                self.shape = tuple(self.shape)

    def _apply_transformations(self, x_or_y, transformations, normalizer, normalization_factors):
        """Apply transformations while maintaining static features"""
        self.nbytes = 0
        self.shape = None
        
        # Save static features temporarily
        temp_static = self.static_features
        
        for t in self.temp_paths:
            array = self._load_partition_from_path(t.name)
            array = loading.apply_transformations(
                array, x_or_y, transformations, 
                normalizer, normalization_factors
            )
            self._update_basic_info(array)
            self._save_partition_to_path(t.name, array)
            del array
            gc.collect()
        
        # Restore static features
        self.static_features = temp_static

    def close(self):
        """Clean up including static features"""
        for t in self.temp_paths:
            t.cleanup()
        self.temp_paths = []
        self.shape = None
        self.nbytes = 0
        self.axis = 0
        self.static_features = None
        self.excluded_columns = None

    def exit_handler(self, _):
        """
        Wrapper for 'close' method, which can be called by atexit

        Used for handling a SIGINT

        :param _:
        :return:
        """
        self.close()
