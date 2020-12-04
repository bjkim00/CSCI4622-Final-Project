import pandas as pd
import numpy as np
from DataHandler import DataHandler
dh = DataHandler(beg=2010, split_by_pos=True, offset=3, ignore_na = False, fill_mean = True)
print(dh.y_train)
print(dh.info_train)