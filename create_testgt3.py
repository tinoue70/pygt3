#!/usr/bin/env python
import numpy as np
# import pandas as pd
from pygt3file import GT3File, GT3Header
# import sys


init_date = "20180619 000000"

date = init_date
deltaT = 3  # hour

d = np.zeros(shape=(1,32,64), dtype='>f4')

with GT3File('testgt3', mode='wb') as f:
    f.current_header = GT3Header(
        dset='test',
        item='hoge', unit='-', title='test data',
        date=date, utim='HOUR',
        aitm1='GLON64', astr1=1, aend1=64,
        aitm2='GGLA32', astr2=1, aend2=32,
        aitm3='SFC1', astr3=1, aend3=1,
    )

    f.set_current_data(d)

    f.write_one_header()
    f.write_one_data()

    for i in range(11):
        time = f.current_header.time + deltaT
        f.current_header.set_time_date(time=time)

        f.write_one_header()
        f.write_one_data()


