import garpar as gp

import numpy as np

pf = gp.datasets.RissoNormal().make_portfolio(days=252)

pf._weights = np.array([0.5, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0])

zaraza = pf.div.mrc()
# import ipdb; ipdb.set_trace()

import pypfopt

import pandas as pd
