import numpy as np
import tensorflow as tf
import gc
from loading.disk_array import DiskArray


class DataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for keras modeling with support for DiskArray and static features
    """
    def __init__(self, X, Y, batch_size=512, shuffle=True):
        """
        If data length is not divisible by batch size, will keep out a random set of rows each round (who make up the
        remainder that don't fit into a full batch)

        :param X: input sequences. Can contain multiple arrays or DiskArray
        :param Y: output sequences or DiskArray
        :param batch_size: size of each batch
        :param shuffle: Whether to shuffle data after each epoch
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Handle X input (can be list, numpy array or DiskArray)
        if isinstance(X, DiskArray):
            self.X_diskarray = X
            self.X_unified = X.compute()  # Load into memory
            if isinstance(self.X_unified, tuple):
                # Handle case where DiskArray has static features
                self.X_unified, self.static_features = self.X_unified
                 # Add the validation check here
                 if X.static_features is not None:
                if len(X.static_features) != len(self.X_unified):
                    raise ValueError(
                        f"Static features count ({len(X.static_features)}) "
                        f"doesn't match main data ({len(self.X_unified)})"
                    )
            else:
                self.static_features = None
        else:
            self.X_diskarray = None
            if isinstance(X, list):
                self.X_unified = [xset.copy() for xset in X]
            else:
                self.X_unified = [X]
            self.static_features = None

        # Handle Y input
        if isinstance(Y, DiskArray):
            self.Y_diskarray = Y
            self.Y_unified = Y.compute()
        else:
            self.Y_diskarray = None
            self.Y_unified = Y.copy()

        self.num_X_sets = len(self.X_unified) if isinstance(self.X_unified, list) else 1
        self.X_split = None
        self.Y_split = None
        self.data_is_split = False
        
        # Get complete length from first X array
        if isinstance(self.X_unified, list):
            self.complete_len = len(self.X_unified[0])
        else:
            self.complete_len = len(self.X_unified)
            
        self.batch_len = int(np.floor(self.complete_len / self.batch_size))
        self.split_indexes = np.arange(0, self.complete_len, self.batch_size)[1:]
        self.on_epoch_end()

    def __len__(self):
        """
        :return: Number of batches per epoch
        """
        return self.batch_len

    def __getitem__(self, index):
        """
        Return one batch of data with static features if available

        :param index: Index of batch to retrieve
        :return: tuple (input, output) where input may include static features
        """
        return self.__data_generation(index)

    def on_epoch_begin(self):
        """
        Split data into batches at the beginning of epoch

        :return:
        """
        if not self.data_is_split:
            # Split X data
            if isinstance(self.X_unified, list):
                self.X_split = [np.split(xset, self.split_indexes, axis=0) for xset in self.X_unified]
            else:
                self.X_split = [np.split(self.X_unified, self.split_indexes, axis=0)]
                
            # Split Y data
            self.Y_split = np.split(self.Y_unified, self.split_indexes, axis=0)
            
            # Clear unified data to save memory
            self.X_unified = None
            self.Y_unified = None
            self.data_is_split = True

    def on_epoch_end(self):
        """
        Shuffle dataset at the end of an epoch while preserving static features
        """
        if self.data_is_split:
            # Reconstruct unified arrays from splits
            if isinstance(self.X_split[0], list):
                # Multiple X arrays case
                X_info = [[list(xset[0].shape), xset[0].dtype] for xset in self.X_split]
                for i in range(len(X_info)):
                    X_info[i][0][0] = self.complete_len
                self.X_unified = [np.empty(shape, dtype=dtype) for shape, dtype in X_info]
                
                for i in range(self.num_X_sets):
                    self.X_unified[i][:] = np.nan
                    
                for xset_idx, start_index in zip(range(len(self.X_split[0])), 
                                             range(0, self.complete_len, self.batch_size)):
                    for i in range(self.num_X_sets):
                        self.X_unified[i][start_index:start_index+self.batch_size] = self.X_split[i][xset_idx]
                        self.X_split[i][xset_idx] = None
            else:
                # Single X array case
                shape = list(self.X_split[0][0].shape)
                shape[0] = self.complete_len
                self.X_unified = np.empty(shape, dtype=self.X_split[0][0].dtype)
                
                for xset_idx, start_index in zip(range(len(self.X_split[0])), 
                                             range(0, self.complete_len, self.batch_size)):
                    self.X_unified[start_index:start_index+self.batch_size] = self.X_split[0][xset_idx]
                    self.X_split[0][xset_idx] = None

            self.Y_unified = np.concatenate(self.Y_split, axis=0)
            self.Y_split = None
            self.X_split = None
            self.data_is_split = False

        # Generate shuffled indexes
        self.indexes = np.arange(self.complete_len)
        if self.shuffle:
            np.random.shuffle(self.indexes)

        # Apply shuffle to data
        if isinstance(self.X_unified, list):
            self.X_unified = [xset[self.indexes] for xset in self.X_unified]
        else:
            self.X_unified = self.X_unified[self.indexes]
            
        self.Y_unified = self.Y_unified[self.indexes]
        
        # Apply shuffle to static features if they exist
        if self.static_features is not None:
            self.static_features = self.static_features.iloc[self.indexes]

        # Split data again for next epoch
        self.X_split = [
            np.split(xset, self.split_indexes, axis=0) for xset in (self.X_unified if isinstance(self.X_unified, list) 
                                                                   else [self.X_unified])
        ]
        self.Y_split = np.split(self.Y_unified, self.split_indexes, axis=0)
        self.X_unified = None
        self.Y_unified = None
        self.data_is_split = True
        gc.collect()

    def __data_generation(self, index):
        """
        Retrieve a batch by index with static features if available

        :param index: Index of batch
        :return: tuple (input, output) where input may include static features
        """
        output = self.Y_split[index]
        
        # Handle input generation
        if self.num_X_sets == 1:
            input = self.X_split[0][index]
        else:
            input = {f'input_{i+1}': xset[index] for i, xset in enumerate(self.X_split)}
        
        # Add static features if they exist
        if self.static_features is not None:
            batch_start = index * self.batch_size
            batch_end = batch_start + self.batch_size
            static_batch = self.static_features.iloc[batch_start:batch_end]
            
            if isinstance(input, dict):
                input['static_features'] = static_batch.values
            else:
                # For single input case, return as tuple with static features
                input = (input, static_batch.values)
                
        return input, output

    def __del__(self):
        """Clean up resources"""
        if self.X_diskarray is not None:
            self.X_diskarray.close()
        if self.Y_diskarray is not None:
            self.Y_diskarray.close()
        gc.collect()
