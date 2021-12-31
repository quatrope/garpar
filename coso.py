import garpar as gp

import numpy as np

pf = gp.datasets.RissoNormal().make_portfolio()

pf._weights = np.array([0.5, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0])


import pypfopt

import pandas as pd
