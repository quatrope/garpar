from .base import OptimizerABC, MeanVarianceFamilyMixin

class Markowitz(MeanVarianceFamilyMixin, OptimizerABC):
    """Classic Markowitz model.

    This method implements the Classic Model Markowitz 1952 in Mansini, R.,
    WLodzimierz, O., and Speranza, M. G. (2015). Linear and mixed
    integer programming for portfolio optimization. Springer and EURO: The
    Association of European Operational Research Societies
    """

    def _get_optimizer(self, pf):
        """Get the pypfopt EfficientFrontier optimizer.

        Parameters
        ----------
        pf : Portfolio
            The portfolio to optimize.

        Returns
        -------
        pypfopt.EfficientFrontier
            The configured optimizer.
        """
        expected_returns = pf.ereturns(self.returns, **self.returns_kw)
        cov_matrix = pf.covariance(self.covariance, **self.covariance_kw)
        weight_bounds = self.weight_bounds
        optimizer = pypfopt.EfficientFrontier(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            weight_bounds=weight_bounds,
        )
        return optimizer

    def _coerce_target_return(self, pf):
        """Coerce the target return.

        Parameters
        ----------
        pf : Portfolio
            The portfolio to optimize.

        Returns
        -------
        float
            The coerced target return.
        """
        if self.target_return is None:
            returns = pf.as_returns().to_numpy().flatten()
            returns = returns[returns != 0]
            returns = returns[~np.isnan(returns)]
            return np.min(np.abs(returns))
        return self.target_return

    def _calculate_weights(self, pf):
        """Calculate the optimal weights for the portfolio.

        Parameters
        ----------
        pf : Portfolio
            The portfolio to optimize.

        Returns
        -------
        tuple
            A tuple containing the optimal weights and optimizer metadata.
        """
        optimizer = self._get_optimizer(pf)
        target_return = self._coerce_target_return(pf)
        market_neutral = self._get_market_neutral()
        # market_neutral = self.market_neutral

        weights_dict = optimizer.efficient_return(
            target_return, market_neutral=market_neutral
        )
        weights = [weights_dict[stock] for stock in pf.stocks]

        optimizer_metadata = {
            "name": type(self).__name__,
            "target_return": target_return,
        }

        return weights, optimizer_metadata