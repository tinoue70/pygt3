#!/usr/bin/env python
import numpy as np
from pygt3file import GT3File, GT3Header
from datetime import datetime, timedelta

f = GT3File('testgt3', mode='wb')

epoch = datetime(1,1,1,0,0,0)
init_date = datetime(2018,6,19,0,0,0)
init_time = (init_date-epoch).days*24

date = init_date
time = init_time
deltaT = 3  # hour

d = np.zeros(shape=(1,32,64), dtype='>f4')

f.current_header = GT3Header(
    dset='test',
    item='hoge', unit='-', title='test data',
    date=date, utim='HOUR', time=time,
    aitm1='GLON64', astr1=1, aend1=64,
    aitm2='GGLA32', astr2=1, aend2=32,
    aitm3='SFC1', astr3=1, aend3=1,
)

f.set_current_data(d)

f.write_one_header()
f.write_one_data()

for i in range(11):
    time = time + deltaT
    date = date + timedelta(hours=deltaT)
    f.current_header.date = date
    f.current_header.time = time

    f.write_one_header()
    f.write_one_data()

f.close()
