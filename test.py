import pandas as pd
import numpy as np
from DataHandler import DataHandler
dh = DataHandler(beg = 2015, split_by_pos=True)
print(dh.names_train)