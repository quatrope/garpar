#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file is part of the
#   Garpar Project (https://github.com/nluczywo/GARPAR).
# Copyright (c) 2021, Nadia Luczywo
# License: MIT
#   Full Text: https://github.com/nluczywo/GARPAR/blob/master/LICENSE

"""
Created on Wed Sep  1 15:09:17 2021

@author: nadia
"""


import numpy as np

import pandas as pd


def risso_candidate_entropy(windows_size):
    if windows_size <= 0:
        raise ValueError("'windows_size' must be > 0")

    loss_probability = np.linspace(0.0, 1.0, num=windows_size + 1)

    # Se corrigen probabilidades porque el cálculo de la entropía trabaja con
    # logaritmo y el logaritmo de cero no puede calcularse
    epsilon = np.finfo(loss_probability.dtype).eps
    loss_probability[0] = epsilon
    loss_probability[-1] = 1 - epsilon

    # Calcula entropy
    first_part = loss_probability * np.log2(loss_probability)
    second_part = (1 - loss_probability) * np.log2(1 - loss_probability)

    modificated_entropy = -1 * (first_part + second_part)
    return modificated_entropy, loss_probability


def argnearest(arr, v):
    diff = np.abs(np.subtract(arr, v))
    idx = np.argmin(diff)
    return idx


def loss_sequence(windows_size, loss_probability, seed=None):
    random = np.random.default_rng(seed=seed)
    probability_win = 1 - loss_probability
    sequence = random.choice(
        [True, False], size=windows_size, p=[loss_probability, probability_win]
    )
    if random.choice([True, False]):
        sequence = ~sequence
    return sequence


def make_stock_price(price, loss, seed=None):
    if price == 0.0:
        return 0.0
    random = np.random.default_rng(seed=seed)
    sign = -1 if loss else 1
    day_return = sign * np.abs(random.normal(0, 1))
    new_price = price + day_return
    return 0.0 if new_price < 0 else new_price


def make_stock(
    window_number, windows_size, loss_probability, price, seed=None
):
    random = np.random.default_rng(seed=seed)
    rows = []
    for window in range(window_number):
        lsequence = loss_sequence(windows_size, loss_probability, random)
        for day, loss in enumerate(lsequence):
            price = make_stock_price(price, loss, random)
            row = {"window": window, "day": day, "price": price}
            rows.append(row)
    return pd.DataFrame(rows)


def make_market(
    window_number=100,
    windows_size=5,
    aproximate_entropy=0.5,
    stock_number=100,
    price=100,
    seed=None,
):

    random = np.random.default_rng(seed=seed)

    if isinstance(price, (int, float)):
        price = np.full(stock_number, price, dtype=float)
    elif len(price) != stock_number:
        raise ValueError(f"The q of price must be equal {stock_number}")

    h_candidates, loss_probabilities = risso_candidate_entropy(windows_size)
    idx = argnearest(h_candidates, aproximate_entropy)
    loss_probability = loss_probabilities[idx]

    stocks, initial_prices = [], {}
    for stock_idx, stock_price in enumerate(price):

        stock_df = make_stock(
            window_number, windows_size, loss_probability, stock_price, random
        )
        if stocks:
            del stock_df["window"], stock_df["day"]
        stock_df.rename(
            columns={"price": f"stock_{stock_idx}_price"}, inplace=True
        )
        stocks.append(stock_df)

        initial_prices[f"stock_{stock_idx}"] = stock_price

    stock_df = pd.concat(stocks, axis=1)
    stock_df.attrs["initial_prices"] = pd.Series(initial_prices)

    return stock_df
