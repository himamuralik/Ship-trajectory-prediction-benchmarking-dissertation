import logging

import pandas as pd

# Set a base directory to use for data storage. You should change this value.
global data_directory
data_directory = '/home/isaac/data/'


global dataset_config
# Define start/end years to look at. Currently, the earliest supported year is 2015. Data prior to 2015 uses a different
# url and will also need to be preprocessed slightly differently - check the ais_data_faq_from_marine_cadastre.pdf in
# resources_and_information for details.
global start_year
start_year = 2015
assert start_year >= 2015

global end_year
end_year = 2019

global years
years = range(start_year, end_year + 1)

# Url to download data from. This should not be changed.
global base_url
base_url = 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/'

# Trajectory processing parameters
global new_trajectory_time_gap
new_trajectory_time_gap = 120 * 60  # 2 hours in seconds

# Speed filters
global sog_cutoff
sog_cutoff = 30  # knots
global empirical_speed_cutoff
empirical_speed_cutoff = 40  # knots

# Interpolation parameters
global interpolation_time_gap
interpolation_time_gap = 5 * 60  # 5 minutes in seconds

# Prediction window settings
global length_of_history
length_of_history = int(3 * 60 * 60 / interpolation_time_gap) + 1  # 3 hours
global length_into_the_future
length_into_the_future = int(3 * 60 * 60 / interpolation_time_gap) - 1  # 3 hours
global min_track_length
min_track_length = (length_of_history + length_into_the_future) * interpolation_time_gap

# ========== KEY CHANGES START HERE ==========
# Static features (unchanging per vessel track)
global static_columns
static_columns = ['vessel_type']

# Dynamic categorical features (one-hot encoded)
global categorical_columns
categorical_columns = ['vessel_group']

# Dynamic numerical features
global dynamic_columns
dynamic_columns = ['lat', 'lon', 'sog', 'cog', 'heading']
# ========== KEY CHANGES END HERE ==========

# Vessel types to include
global vessel_types
vessel_types = [
    'cargo',
    'passenger',
    'fishing',
    'tug tow',
    'tanker',
    'pleasure craft or sailing',
]

# Navigation statuses to include
global desired_statuses
desired_statuses = [
    'under way sailing',
    'under way using engine',
    'undefined'
]

# Time gaps for evaluation
global time_gaps
time_gaps = [30 * 60]  # 30 minutes in seconds

# Reference data files
global statuses
statuses = pd.read_csv('config/navigation_statuses.csv')
global types
types = pd.read_csv('config/vessel_type_codes.csv')

# Log level configuration
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
