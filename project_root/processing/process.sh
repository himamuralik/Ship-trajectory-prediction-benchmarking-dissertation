## RUN AFTER dataset_config.py HAS BEEN UPDATED:
python downloader.py new_york -l 4 -s &&
python cleaner.py   new_york -l 4 -s --seed 47033218 --memory conserve &&
python interpolator.py new_york -l 4 -s &&
#  (no current_*/destination_appender steps)
## RUN ONLY AFTER SETTING DBSCAN PARAMETER VALUES
#python destination_appender.py new_york -l 4 -s    # still unused
#python current_appender.py    new_york -l 4 -s    # still unused
python sliding_window.py      new_york -l 4 -s &&
python formatter.py           new_york -l 4 -s
