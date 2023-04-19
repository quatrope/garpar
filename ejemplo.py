from garpar.datasets import *

coso = load_MERVAL().refresh_entropy()


fff = MultiSector(
    {"foo": RissoNormal(), "faa": RissoUniform()}
).make_portfolio(price=20)

import ipdb

ipdb.set_trace()
