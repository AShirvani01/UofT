# CSV compiler

"""
Script that compiles all csv's in a given directory, extracts desired columns,
and imports to SQL database.
"""


# Import libraries/packages/modules
import pandas as pd
import os
from sqlalchemy import create_engine, types as t
import time


# Track runtime
start = time.time()
count = 0


# Connect to SQL db
file_path = r'~\drive-failure'
file_list = os.walk(file_path)
engine = create_engine('mysql+mysqlconnector://root:~@localhost:~/hard_drive',
                       echo=False)


# Iterate over csv's in directory
for root, _, files in file_list:
    for name in files:
        print(f'Importing {name}')
        df = pd.read_csv(rf'{root}\{name}')
        df = df[['date', 'serial_number', 'model', 'capacity_bytes', 'failure']]

        # Import to MySQL
        table_name = 'hd_failure'
        df.to_sql(name=table_name, con=engine, if_exists='append', index=False,
                  dtype={
                      'date': t.DATE(),
                      'serial_number': t.NVARCHAR(length=255),
                      'model': t.NVARCHAR(length=255),
                      'capacity_bytes': t.BIGINT(),
                      'failure': t.BOOLEAN()
                      }
                  )
        count += 1

end = time.time()
print(f'{count} Files Imported Successfully')
print(f'Runtime: {round(end-start,1)} sec')
