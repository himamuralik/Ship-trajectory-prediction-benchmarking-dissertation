
import shutil
import urllib

import pandas as pd
import utm
import re

from calendar import monthrange
from dateutil import rrule
from datetime import datetime

def get_zones_from_coordinates(corner_1, corner_2):
    """
    Get UTM zones to download, based on lat/lon coordinates

    :param corner_1: Lat/lon pair (lat, lon)
    :param corner_2: Lat/lon pair (lat, lon)
    :return: list of UTM zones to download
    """
    _, _, zone_1, _ = utm.from_latlon(*corner_1)
    _, _, zone_2, _ = utm.from_latlon(*corner_2)
    
    if zone_1 > 19 or zone_2 > 19:
        raise ValueError("One or both coordinates are outside the MarineCadastre.gov supported UTM zones (1–19).")
    
    return list(range(min(zone_1, zone_2), max(zone_1, zone_2) + 1))

# ---- Usage for New York only ----
ny_config = datasets['new_york']
zones = get_zones_from_coordinates(ny_config.corner_1, ny_config.corner_2)

print("UTM zones to download:", zones)  # Expected: [18,19]

def get_file_specifier(year, month, zone_num=None, day_num=None, extension="csv"):
    """Generate MarineCadastre.gov AIS data filename based on year and month.
    
    Args:
        year: Integer year (2015-2019)
        month: Integer month (1-12)
        zone_num: Required for 2015-2017 data, the zone number (1-20)
        day_num: Required for 2018+ data, the day of month (1-31)
        extension: File extension (default "csv")
        
    Returns:
        String filename in MarineCadastre.gov format
        
    Raises:
        AssertionError: If required parameters are missing for the given year
    """
    if year <= 2017:
        assert zone_num is not None, "Zone number required for 2015-2017 data"
        assert 1 <= zone_num <= 20, "Zone number must be between 1-20"
        specifier = f"AIS_{year}_{month:02d}_Zone{zone_num:02d}.{extension}"
    else:
        assert day_num is not None, "Day number required for 2018+ data"
        assert 1 <= day_num <= 31, "Day number must be between 1-31"
        specifier = f"AIS_{year}_{month:02d}_{day_num:02d}.{extension}"
    return specifier


def get_info_from_specifier(file_name):
    """
    Split the file specifier into its constituent info

    The file specifier contains the year, month, zone/day, and file extension for the file in question. This splits up
    a file specifier into these parts. Whether the third piece of information is the zone or day depends on what year
    the file is from (2015-2017 will contain the zone, while 2018+ will contain the day, as this is how the files are
    organized on MarineCadastre.gov).

    :param file_name: The filename to parse (e.g., "AIS_2015_01_Zone18.csv" or "AIS_2018_01_01.csv")
    :return: tuple of (year, month, zone_or_day, extension)
    :raises: ValueError if the filename format is not recognized
    """
    # Match 2015-2017 format: AIS_YYYY_MM_ZoneZZ.csv
    zone_match = re.search(r'AIS_([0-9]{4})_([0-9]{2})_Zone([0-9]{2})\.(.+)', file_name)
    if zone_match:
        year = int(zone_match.group(1))
        if year > 2017:
            raise ValueError(f"Zone format not valid for year {year} (should be day format)")
        return int(zone_match.group(1)), int(zone_match.group(2)), int(zone_match.group(3)), zone_match.group(4)
    
    # Match 2018+ format: AIS_YYYY_MM_DD.csv
    day_match = re.search(r'AIS_([0-9]{4})_([0-9]{2})_([0-9]{2})\.(.+)', file_name)
    if day_match:
        year = int(day_match.group(1))
        if year <= 2017:
            raise ValueError(f"Day format not valid for year {year} (should be zone format)")
        return int(day_match.group(1)), int(day_match.group(2)), int(day_match.group(3)), day_match.group(4)
    
    raise ValueError('File name format not recognized. Expected either: '
                   '"AIS_YYYY_MM_ZoneZZ.ext" (2015-2017) or '
                   '"AIS_YYYY_MM_DD.ext" (2018+)')
def all_specifiers(zones, years, extension, dir=None):
    """
    Get all file specifiers for the relevant zones and years

    A specifier is a string formatted something like '2017/AIS_2017_01_Zone01.zip'

    :param zones: The UTM zones to look at
    :param years: The years to consider
    :param extension: The file extension to use for the specifier
    :param dir: The directory, if one is desired at the start of the specifier,
    :return: paths
    """
    specifiers = []
    if dir is not None:
        paths = []
    for year in years:
        if year in (2015, 2016, 2017):
            for month in range(1, 13):
                for zone in zones:
                    specifier = get_file_specifier(year, month, zone, extension)
                    specifiers.append(specifier)

                    if dir is not None:
                        path = os.path.join(dir, specifier)
                        paths.append(path)
        elif year in (2018, 2019, 2020, 2021):
            for dt in rrule.rrule(rrule.DAILY,
                                  dtstart=datetime.strptime(f'{year}-01-01', '%Y-%m-%d'),
                                  until=datetime.strptime(f'{year}-12-31', '%Y-%m-%d')):
                specifier = get_file_specifier(dt.year, dt.month, dt.day, extension)
                specifiers.append(specifier)

                if dir is not None:
                    path = os.path.join(dir, specifier)
                    paths.append(path)

    if dir is not None:
        all_zym = {'paths': paths, 'specifiers': specifiers}
    else:
        all_zym = {'specifiers': specifiers}

    return all_zym


def pd_append(values):
    """
    Append values together into a pandas series

    values should be a list of different things to append together, e.g. the first item might be the integer, the second
    a pd.Series of integers.

    :param values: A list of different things to append together
    :return:
    """
    v1 = values[0]
    if len(values) > 2:
        if type(v1) == pd.Series:
            series = pd.concat([
                v1,
                pd_append(values[1:])
            ]).reset_index(drop=True)
        else:
            series = pd.concat([
                pd.Series([v1]),
                pd_append(values[1:])
            ]).reset_index(drop=True)
    elif len(values) == 2:
        v2 = values[1]
        if type(v1) == pd.Series:
            series = pd.concat([
                v1,
                pd.Series([v2])
            ]).reset_index(drop=True)
        elif type(v2) == pd.Series:
            series = pd.concat([
                pd.Series([v1]),
                v2
            ]).reset_index(drop=True)
        else:
            series = pd.concat([
                pd.Series([v1]),
                pd.Series([v2])
            ]).reset_index(drop=True)
    return series


def to_snake_case(name):
    """
    Convert a string to snake case

    :param name: Name to convert
    :return: Converted name
    """
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()


def clear_path(path):
    """
    Delete any files or directories from a path

    :param path: Path to remove
    :return:
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
def get_min_max_times(specifier):
    """
    Get the first/last possible time for AIS messages contained in a file

    :param specifier: file information
    :return: tuple of (min_time, max_time) as pandas Timestamps
    :raises: ValueError if year is not supported
    """
    year, month, zone_or_day, extension = get_info_from_specifier(specifier)
    
    if year <= 2017:
        # Zone-based files (monthly)
        min_time = pd.to_datetime(f'{year}-{month}-01 00:00:00')
        _, last_day = monthrange(year, month)
        max_time = pd.to_datetime(f'{year}-{month}-{last_day} 23:59:59')
    elif 2018 <= year <= 2021:
        # Day-based files (daily)
        day = zone_or_day
        min_time = pd.to_datetime(f'{year}-{month}-{day} 00:00:00')
        max_time = pd.to_datetime(f'{year}-{month}-{day} 23:59:59')
    else:
        raise ValueError(f'Year {year} not supported (must be 2015-2021)')

    return min_time, max_time

  
