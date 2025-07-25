import pandas as pd
import logging

# Base directory for data
global data_directory
data_directory = '/home/isaac/data/'

# Years used for benchmarking: 2015-2019 (as in the paper)
global start_year
start_year = 2015
global end_year
end_year = 2019
global years
years = range(start_year, end_year + 1)

# Data download URL
global base_url
base_url = 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/'

# Preprocessing and filtering parameters
global new_trajectory_time_gap
new_trajectory_time_gap = 120 * 60

global sog_cutoff
sog_cutoff = 30
global empirical_speed_cutoff
empirical_speed_cutoff = 40

# Interpolation gap (5 min)
global interpolation_time_gap
interpolation_time_gap = 5 * 60

# Length of history/prediction (3 hours)
global length_of_history
length_of_history = int(3 * 60 * 60 / interpolation_time_gap) + 1
global length_into_the_future
length_into_the_future = int(3 * 60 * 60 / interpolation_time_gap) - 1

global min_track_length
min_track_length = (length_of_history + length_into_the_future) * interpolation_time_gap

# Vessel types to keep
global vessel_types
vessel_types = [
    'cargo',
    'passenger',
    'fishing',
    'tug tow',
    'tanker',
    'pleasure craft or sailing',
]

# Statuses to keep
global desired_statuses
desired_statuses = [
    'under way sailing',
    'under way using engine',
    'undefined'
]

# Only 30-min time gap for benchmarking
global time_gaps
time_gaps = [30 * 60]

# Vessel group as the categorical column (for static input/fusion)
global categorical_columns
categorical_columns = ['vessel_group']

# ---- FEATURE LISTS ----

# Dynamic columns (fed to RNN/sequence model)
global dynamic_columns
dynamic_columns = ['lat', 'lon', 'sog', 'cog', 'year', 'day', 'month']

# Static columns (fed to fusion/dense input) - for fusion models
global static_columns
static_columns = [
    'month', 'year'
]

# Preprocessing files
global statuses
statuses = pd.read_csv('navigational_statuses.csv')
types = pd.read_csv('vessel_types.csv')

# Function to set log level
def set_log_level(level):
    global log_level
    if level == 0:
        log_level = logging.CRITICAL
    elif level == 1:
        log_level = logging.ERROR
    elif level == 2:
        log_level = logging.WARNING
    elif level == 3:
        log_level = logging.INFO
    elif level == 4:
        log_level = logging.DEBUG
