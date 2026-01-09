# Whats new?

<!-- BODY -->

---

## Version *1.5.1* - 2026-01

- Added Python 3.14 support and compatibility.
- Updated Risso datasets return calculation for improved accuracy.
- Adjusted test expected values across multiple test modules.
- Updated CI and publish workflow references.

---

## Version *1.5.0* - 2025-02

- First stable relase
- Implemented `MVOptimizer` with new mean-variance optimization models.
- Added support for the Markowitz model (`Markowitz` class) for portfolio optimization.
- Introduced `UtilitiesAccessor` with tracking error and quadratic utility calculations.
- Implemented HDF5 storage capabilities in `garpar_io.py` for saving and loading stock sets.
- Added `StocksSet.to_dataframe()` method for easier data extraction.
- Introduced new pruning methods (`weights_prune` and `delisted_prune`) to optimize datasets.
- Refactored `StocksSet` to improve weight and entropy handling.
- Improved metadata management with `GARPAR_METADATA_KEY`.
- Optimized covariance and correlation calculations for better performance.
- Resolved issues with missing data in covariance matrix calculations.
- Fixed incorrect behavior in `to_hdf5()` when handling metadata attributes.
- Addressed rounding errors in weight calculations for portfolio optimization.
- Deprecated older portfolio optimization methods in favor of new mean-variance models.
- Removed redundant utility functions now covered by `UtilitiesAccessor`.

For a full list of changes, visit: [Garpar Repository](https://github.com/quatrope/garpar).



## Version *1.0*

Pre relase don't use.
