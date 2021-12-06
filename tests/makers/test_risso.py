# This file is part of the
#   Garpar Project (https://github.com/nluczywo/GARPAR).
# Copyright (c) 2021, Juan Cabral, Nadia Luczywo
# License: MIT
#   Full Text: https://github.com/nluczywo/GARPAR/blob/master/LICENSE

from garpar.makers import risso

import numpy as np

import pandas as pd

import pytest


def test_risso_argnearest():
    assert risso.argnearest([0.1, -0.98], 0) == 0
    assert risso.argnearest([0.1, -0.98], -0.99) == 1
    assert risso.argnearest([0.1, -0.10], 0.1) == 0


@pytest.mark.parametrize(
    "windows_size, h",
    [
        (1, [0.0, 0.0]),
        (2, [0.0, 1.0, 0.0]),
        (3, [0.0, 0.91830, 0.91830, 0.0]),
        (4, [0.0, 0.81127, 1.0, 0.81127, 0.0]),
        (5, [0.0, 0.72193, 0.97095, 0.97095, 0.72193, 0.0]),
        (6, [0.0, 0.65002, 0.91830, 1.0, 0.91830, 0.65002, 0.0]),
        (7, [0.0, 0.59167, 0.86312, 0.98522, 0.98522, 0.86312, 0.59167, 0.0]),
    ],
)
def test_RissoNormal_candidate_entropy(windows_size, h):
    maker = risso.RissoNormal()
    me, _ = maker.risso_candidate_entropy(windows_size)
    assert np.allclose(me, h, atol=1e-05)


@pytest.mark.parametrize("windows_size", [0, -1])
def test_RissoNormal_candidate_entropy_le0(windows_size):
    maker = risso.RissoNormal()
    with pytest.raises(ValueError):
        maker.risso_candidate_entropy(windows_size)


def test_RissoNormal_make_stock_price():
    maker = risso.RissoNormal()
    assert maker.make_stock_price(100, True, maker.random_state) < 100
    assert maker.make_stock_price(100, False, maker.random_state) > 100
    assert maker.make_stock_price(0, False, maker.random_state) == 0
    assert maker.make_stock_price(0, True, maker.random_state) == 0


@pytest.mark.parametrize(
    "windows_size, loss_prob",
    [
        (1, 0.),
        (2, 0.),
        (3, 0.33333),
        (4, 0.25000),
        (5, 0.20000),
        (6, 0.16667),
        (7, 0.14286),
    ],
)
def test_RissoNormal_window_loss_probability(windows_size, loss_prob):
    maker = risso.RissoNormal()
    result = maker.get_window_loss_probability(windows_size, 0.5)
    assert np.allclose(result, loss_prob, atol=1e-05)


# @pytest.mark.parametrize(
#     "windows_size, sequence",
#     [
#         (1, [True]),
#         (2, [False, True]),
#         (3, [False, True, False]),
#         (4, [False, True, False, True]),
#         (5, [True, False, True, False, True]),
#         (6, [True, False, True, False, True, False]),
#         (7, [True, False, True, False, True, False, True]),
#     ],
# )
# def test_loss_sequence(windows_size, sequence):
#
#     result = gp.loss_sequence(
#         loss_probability=0.33, windows_size=windows_size, seed=10
#     )
#     assert np.all(result == sequence)


def test_RissoNormal_make_market():

    expected = pd.DataFrame(
        {
            "window": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "day": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            "stock_0_price": [
                100.12784040316728,
                100.44408299551087,
                100.46088415301516,
                101.31392808058874,
                100.43453010572591,
                100.8032808898084,
                101.7621634906374,
                102.64061379194467,
                102.69053970293092,
                102.87540206647618,
            ],
            "stock_1_price": [
                99.63455593563592,
                99.22182332403993,
                98.79100232103205,
                96.6493547201616,
                97.05576973654621,
                96.21561325958368,
                95.39113204389244,
                94.74053925606773,
                95.48379342727118,
                96.02694769557638,
            ],
        }
    )

    maker = risso.RissoNormal(random_state=42)
    result = maker.make_market(
        window_number=2, window_size=5, stock_number=2
    )
    pd.testing.assert_frame_equal(result, expected, atol=1e-10)
