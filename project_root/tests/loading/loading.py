import os

from loading.disk_array import DiskArray
import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit

def read_ts_data(directory, time_gap, x_or_y, dtype=None, conserve_memory=False, is_fusion_model=False, static_features_path=None):
    """
    Read a dataset from disk with enhanced support for fusion models and static features
    
    :param directory: Directory where data is stored
    :param time_gap: Minutes between AIS messages
    :param x_or_y: 'x' or 'y' dataset
    :param dtype: Datatype to read as
    :param conserve_memory: Use DiskArray if True
    :param is_fusion_model: Whether this is for a fusion model (loads static features)
    :param static_features_path: Optional override path for static features
    :return: Loaded dataset (with static features if fusion model)
    """
    # Main data loading
    path = os.path.join(directory, f'{time_gap}_min_time_gap_{x_or_y}.npy')
    
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        if dtype is not None:
            data = data.astype(dtype)
    else:
        path = os.path.join(directory, f'{time_gap}_min_time_gap_{x_or_y}')
        files = os.listdir(path)
        files = sorted(files, key=lambda x: int(x.split('.')[0]))
        
        if dtype is None:
            dtype = np.load(os.path.join(path, files[0]), allow_pickle=True).dtype
            
        if conserve_memory:
            data = DiskArray()
            for f in files:
                arr = np.load(os.path.join(path, f), allow_pickle=True).astype(dtype)
                data.add_array(arr)
        else:
            data = np.concatenate([np.load(os.path.join(path, f), allow_pickle=True).astype(dtype) 
                           for f in files])

    # Handle static features for fusion models
    if is_fusion_model and x_or_y == 'x':
        static_path = static_features_path or os.path.join(directory, 'static_features.csv')
        if os.path.exists(static_path):
            try:
                static_features = pd.read_csv(static_path)
                
                # Ensure proper alignment with main data
                if len(static_features) != len(data):
                    raise ValueError(
                        f"Static features length ({len(static_features)}) "
                        f"doesn't match main data ({len(data)})"
                    )
                    
                if conserve_memory:
                    if isinstance(data, DiskArray):
                        data.attach_static_features(static_features)
                    else:
                        return (data, static_features)
                else:
                    return (data, static_features)
                    
            except Exception as e:
                logging.error(f"Error loading static features: {str(e)}")
                raise
        else:
            logging.warning(f"No static features found at {static_path}")
            if is_fusion_model:  # Only warn if actually expecting static features
                logging.warning("Proceeding without static features for fusion model")

    return data

def add_distance_traveled(X, lat_lon_idxs, dt_idx):
    """
    Calculate the distance traveled from first to last timestamp (in km)
    Compatible with both regular arrays and DiskArray chunks
    
    :param X: Input dataset (numpy array or DiskArray chunk)
    :param lat_lon_idxs: Indices of [lat, lon] columns 
    :param dt_idx: Index to insert distance at
    :return: Dataset with distance added
    """
    # Ensure we have valid lat/lon indices
    if len(lat_lon_idxs) != 2:
        raise ValueError("lat_lon_idxs must contain exactly 2 elements [lat_idx, lon_idx]")
    
    # Calculate distances
    start_location = X[:, 0, lat_lon_idxs]
    end_location = X[:, -1, lat_lon_idxs]
    distance_traveled = haversine_vector(start_location, end_location, Unit.KILOMETERS)
    
    # Reshape to match time dimension
    distance_traveled = np.stack([distance_traveled] * X.shape[1], axis=1)
    
    # Insert new column
    return np.insert(X, dt_idx, distance_traveled, axis=2)

def add_stats(X, col_idx, which):
    """
    Add a set of summary statistics to a dataset

    Works over the first index, which should represent timestamps

    :param X: Dataset to insert into
    :param col_idx: Index of column to summarize
    :param which: Either 'median' or 'min_median_max'
    :return:
    """
    if which == 'min_median_max':
        add_stat(X, col_idx, 'min')

    add_stat(X, col_idx, 'median')

    if which == 'min_median_max':
        add_stat(X, col_idx, 'max')

    return X


def add_stat(X, col_idx, which):
    """
    Add a summary statistic to a dataset

    Works over the first index, which should represent timestamps

    :param X: The dataset to add to
    :param col_idx: Column to be summarized
    :param which: either 'min', 'median' or 'max'
    :return:
    """
    if which == 'min':
        mins = np.stack([X[:,:,col_idx].min(axis=1)] * X.shape[1],axis=1)
        min_idx = X.shape[2]
        X = np.insert(X, min_idx, mins, axis=2)
    elif which == 'median':
        medians = np.stack([np.median(X[:,:,col_idx],axis=1)] * X.shape[1],axis=1)
        median_idx = X.shape[2]
        X = np.insert(X, median_idx, medians, axis=2)
    elif which == 'max':
        maxes = np.stack([X[:,:,col_idx].max(axis=1)] * X.shape[1],axis=1)
        max_idx = X.shape[2]
        X = np.insert(X, max_idx, maxes, axis=2)

    return X

def _calc_time_stats_1d(data, stat, timezone='US/Pacific'):
    """
    Extract time component from unix timestamps (optimized for bulk processing)
    
    Args:
        data: 1D array of unix timestamps
        stat: Time attribute to extract ('hour', 'day_of_week', etc.)
        timezone: Target timezone (defaults to US/Pacific)
    
    Returns:
        Extracted time components as numpy array
    """
    # Vectorized conversion for better performance
    timestamps = pd.to_datetime(data * 1e9, unit='ns')
    return (
        timestamps
        .tz_localize('UTC')
        .tz_convert(timezone)
        .to_series()
        .dt.__getattribute__(stat)
        .to_numpy()
    )

def add_time_stats(data, datetime_col, hour_idx, dow_idx):
    """
    Add hour/dayofweek features while preserving array structure
    
    Args:
        data: Input array (numpy or DiskArray compatible)
        datetime_col: Index of timestamp column
        hour_idx: Insert position for hour feature
        dow_idx: Insert position for day-of-week feature
    
    Returns:
        Array with time features added
    """
    # Single pass timestamp conversion
    timestamps = data[..., datetime_col]
    hour = _calc_time_stats_1d(timestamps.flatten(), 'hour').reshape(timestamps.shape)
    day_of_week = _calc_time_stats_1d(timestamps.flatten(), 'dayofweek').reshape(timestamps.shape)
    
    # Insert new features
    data = np.insert(data, hour_idx, hour, axis=-1)
    return np.insert(data, dow_idx, day_of_week, axis=-1)
  
def split_X_for_fusion(X, recurrent_idxs, static_features=None):
    """
    Split input dataset for fusion models with proper static feature handling
    
    Args:
        X: Input dataset (3D array or tuple with static features)
        recurrent_idxs: Columns to keep in recurrent part
        static_features: Static features DataFrame (optional)
    
    Returns:
        List containing:
        - recurrent_part (3D array)
        - dense_part (2D array)
        - static_features (if provided)
    """
    # Handle case where static features are passed separately (from DiskArray)
    if isinstance(X, tuple) and len(X) == 2:
        X, static_features = X
    
    # Split main features
    recurrent_part = X[:, :, recurrent_idxs]
    dense_part = X[:, -1, :]
    dense_part = np.delete(dense_part, recurrent_idxs, axis=-1)
    
    # Prepare return values
    result = [recurrent_part, dense_part]
    
    # Include static features if provided
    if static_features is not None:
        # Verify static features alignment
        if len(static_features) != len(X):
            raise ValueError("Static features length doesn't match input data")
        result.append(static_features.values if hasattr(static_features, 'values') else static_features)
    
    return result

def _find_current_col_idx(col, columns):
    """
    Find the current index of a column accounting for removed columns.
    Handles both regular features and static features for fusion models.

    Args:
        col: Name of column to find
        columns: DataFrame with 'column' and 'being_used' columns
        
    Returns:
        int: Current column index after removals
        
    Raises:
        ValueError: If column doesn't exist or was removed
    """
    try:
        # Find column position in original DataFrame
        idx_in_columns_df = columns['column'].tolist().index(col)
        
        # Check if column was removed
        if not columns['being_used'].iloc[idx_in_columns_df]:
            raise ValueError(
                f"Column '{col}' is not available - it may have been removed "
                "(e.g., weather/destination columns) or is a static feature "
                "not included in this transformation"
            )
            
        # Calculate current index accounting for removed columns
        return int(columns['being_used'][:idx_in_columns_df].sum())
        
    except ValueError as e:
        if str(e) == f"'{col}' is not in list":
            raise ValueError(
                f"Column '{col}' not found in dataset. Available columns: "
                f"{columns[columns['being_used']]['column'].tolist()}"
            ) from e
        raise

def _add_columns(dataset, transformation, is_fusion=False):
    """
    Add columns to dataset with special handling for:
    - Fusion model static features (vessel group)
    - Time statistics
    - SOG/COG metrics
    - Distance traveled
    
    Args:
        dataset: Input data (numpy array or DataFrame)
        transformation: Transformation config dict
        is_fusion: Whether processing fusion model data
        
    Returns:
        Modified dataset with added columns
    """
    # Skip vessel group conversion for fusion models (handled as static feature)
    if transformation['columns'] == ['vessel_group'] and is_fusion:
        return dataset
        
    # Handle time stats (hour/day_of_week)
    if transformation['columns'] == ['hour', 'day_of_week']:
        return add_time_stats(
            dataset,
            transformation['base_datetime_idx'],
            *transformation['indexes']
        )
        
    # Handle SOG/COG metrics
    elif any(x in transformation['columns'] for x in ['sog_median', 'cog_median']):
        for col, idx in zip(transformation['columns'], transformation['indexes']):
            base_col, stat = col.split('_')
            dataset = add_stat(
                dataset,
                transformation[f'{base_col}_index'],
                stat
            )
        return dataset
        
    # Handle distance traveled
    elif transformation['columns'] == ['distance_traveled']:
        return add_distance_traveled(
            dataset,
            lat_lon_idxs=transformation['lat_lon_idxs'],
            dt_idx=transformation['indexes'][0]
        )
        
    return dataset
  def apply_transformations(dataset, x_or_y, transformations, normalizer, normalization_factors):
    """
    Apply transformations to dataset with enhanced handling for:
    - Weather/Destination Cluster Removal
    - Fusion model static features
    - Vessel group as static feature
    
    Maintains all original functionality while removing weather-specific processing
    """
    # Track if we're processing a fusion model
    is_fusion = any(t['function'] == 'split_for_fusion' for t in transformations)
    
    for transformation in transformations:
        if x_or_y in transformation['dataset']:
            # Skip any weather-related transformations if they exist
            if 'weather' in transformation.get('columns', []) or \
               'destination_cluster' in transformation.get('columns', []):
                continue
                
            if transformation['function'] == 'select_timestamps':
                assert type(dataset) == np.ndarray
                ts_to_select = transformation['to_select']
                dataset = dataset.take(indices=ts_to_select, axis=1)
                
            elif transformation['function'] == 'add_columns':
                # Skip vessel group categorical conversion for fusion models
                if is_fusion and transformation['columns'] == ['vessel_group']:
                    continue
                dataset = _add_columns(dataset, transformation)
                
            elif transformation['function'] == 'remove_columns':
                dataset = _remove_columns(dataset, transformation)
                
            elif transformation['function'] == 'normalize':
                assert type(dataset) == np.ndarray
                dataset = normalizer.normalize_data(dataset, normalization_factors)
                
            elif transformation['function'] == 'split_for_fusion':
                assert type(dataset) == np.ndarray
                dataset = split_X_for_fusion(dataset, transformation['indexes'])
                
            elif transformation['function'] == 'select_columns':
                assert type(dataset) == np.ndarray
                cols_to_select = transformation['indexes']
                dataset = dataset.take(indices=cols_to_select, axis=2)
                
            elif transformation['function'] == 'squeeze':
                assert type(dataset) == np.ndarray
                dataset = dataset.squeeze()
                
            elif transformation['function'] == 'convert_to_pandas':
                assert type(dataset) == np.ndarray
                if len(dataset.shape) == 3:
                    trajs = dataset[:, :, transformation['lat_lon_idxs']]
                    lats = [trajs[i, :, 0].tolist() for i in range(len(trajs))]
                    lons = [trajs[i, :, 1].tolist() for i in range(len(trajs))]
                    dataset = pd.DataFrame(dataset[:,-1,:], columns=transformation['df_columns'])
                    dataset['lats'] = lats
                    dataset['lons'] = lons
                else:
                    dataset = pd.DataFrame(dataset, columns=transformation['df_columns'])

    return dataset



def get_bearing(lat1, long1, lat2, long2):
    """Original implementation preserved"""
    dLon = (long2 - long1)
    x = np.cos(np.radians(lat2)) * np.sin(np.radians(dLon))
    y = (np.cos(np.radians(lat1))
         * np.sin(np.radians(lat2))
         - np.sin(np.radians(lat1))
         * np.cos(np.radians(lat2))
         * np.cos(np.radians(dLon)))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)
    brng %= 360
    return brng


def _remove_columns(dataset, transformation):
    """Modified to preserve static features"""
    # Skip removal of static features if they exist
    if hasattr(dataset, 'static_features') and \
       any(col in dataset.static_features.columns for col in transformation.get('columns', [])):
        return dataset
        
    # Original removal logic
    if type(dataset) == pd.DataFrame:
        return dataset.drop(columns=transformation['columns'])
    else:
        return np.delete(dataset, transformation['indexes'], axis=-1)


def _add_columns(dataset, transformation):
    """Modified to handle static features appropriately"""
    # Skip weather-related columns
    if any('weather' in col for col in transformation.get('columns', [])):
        return dataset
        
    # Skip destination cluster processing
    if 'destination_cluster' in transformation.get('columns', []):
        return dataset
        
    # Original processing for other columns
    if transformation['columns'] == ['hour', 'day_of_week']:
        assert type(dataset) == np.ndarray
        dataset = add_time_stats(dataset,
                               transformation['base_datetime_idx'],
                               *transformation['indexes'])
    elif 'sog_median' in transformation['columns']:
        assert type(dataset) == np.ndarray
        for col, idx in zip(transformation['columns'], transformation['indexes']):
            col, summary = col.split('_')
            dataset = add_stat(dataset,
                             transformation[f'{col}_index'],
                             summary)
    elif transformation['columns'] == ['distance_traveled']:
        assert type(dataset) == np.ndarray
        dataset = add_distance_traveled(dataset,
                                      lat_lon_idxs=transformation['lat_lon_idxs'],
                                      dt_idx=transformation['indexes'][0])
    elif transformation['columns'] == ['mean_bearing_angle','std_bearing_angle']:
        assert type(dataset) == np.ndarray
        lats = dataset[...,transformation['lat_idx']]
        lons = dataset[...,transformation['lon_idx']]
        angles = []
        for i in range(lats.shape[1] - 1):
            angles += [get_bearing(lats[...,i], lons[...,i],
                       lats[...,i+1], lons[...,i+1])]
        angles = np.stack(angles).T
        for c, idx in zip(transformation['columns'], transformation['indexes']):
            if c == 'mean_bearing_angle':
                stat = np.mean(angles, axis=1)
            elif c == 'std_bearing_angle':
                stat = np.std(angles, axis=1)
            stat = np.stack([stat] * dataset.shape[1]).T
            dataset = np.insert(dataset, idx, stat, axis=-1)
    elif 'sog_bin' in transformation['columns']:
        assert type(dataset) == pd.DataFrame
        for (original_c, bin_size, bin_cutoff), new_c in zip(transformation['bins'], transformation['columns']):
            binned = (dataset[original_c] // bin_size * bin_size).astype(int)
            if bin_cutoff is not None:
                binned = np.where(binned < bin_cutoff,
                                '[' + binned.astype(str) + ', ' + (binned + bin_size).astype(str) + ')',
                                f'[{bin_cutoff}+]')
            else:
                binned = '[' + binned.astype(str) + ', ' + (binned + bin_size).astype(str) + ')'
            dataset[new_c] = binned

    return dataset
