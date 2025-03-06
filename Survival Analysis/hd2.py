# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:07:41 2024

@author: abtin
"""

import pandas as pd
import os
from sqlalchemy import create_engine
import sqlalchemy.types as t
import time

start = time.time()
count = 0

file_path = r'~\drive-failure'
file_list = os.walk(file_path)
engine = create_engine('mysql+mysqlconnector://root:~@localhost:~/hard_drive', echo=False)
        
for root, _, files in file_list:
    for name in files:
        print(f'Importing {name}')
        df = pd.read_csv(rf'{root}\{name}')
        df = df[['date','serial_number','model','capacity_bytes','failure']]
        
        
        #Import to MySQL
        table_name = 'hd_failure'
        df.to_sql(name=table_name, con=engine, if_exists='append', index=False,
                  dtype={
                      'date': t.DATE(),
                      'serial_number': t.NVARCHAR(length = 255),
                      'model': t.NVARCHAR(length = 255),
                      'capacity_bytes': t.BIGINT(),
                      'failure': t.BOOLEAN()
                      }
                  )
        count += 1

end = time.time()
print(f'{count} Files Imported Successfully')
print(f'Runtime: {round(end-start,1)} sec')