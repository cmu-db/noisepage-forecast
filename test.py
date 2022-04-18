from scipy import stats
import numpy as np


class Dist(stats.rv_continuous):
    def __init__(self, a, b, name, pred, quantiles):
        super(Dist, self).__init__(a=a, b=b, name=name)
        self.pred = pred
        self.quantiles = quantiles

    def _cdf(self, x):
        conditions = [x <= self.pred[0]]
        for k in range(self.pred.shape[0] - 1):
            conditions.append(self.pred[k] <= x <= self.pred[k + 1])
        choices = self.quantiles
        return np.select(conditions, choices, default=0)

if __name__ == "__main__":
    pred = np.arange(101)
    quantiles = np.arange(0, 101, 10)
    dist = Dist(a=pred[0], b=pred[-1], name="deterministic", pred=pred, quantiles=quantiles)

    for i in range(10):
        dist.rvs()