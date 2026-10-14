from weighted_poisson_binomial import WeightedPoissonBinomial
import numpy as np


p = [0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4 , 0.39,
       0.38, 0.36, 0.34, 0.32, 0.3 ]
w = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 8, 10]

model = WeightedPoissonBinomial(p, w)

rng = np.random.default_rng(42)
obs = (rng.random(model.n) < np.array(p)).tolist()
rep = model.summarize(obs)

print(obs)
print("观测布尔序列     =", "".join("1" if b else "0" for b in obs))
print(f"观测总贡献 s_obs = {rep['observed_contribution']:.4f}")
print(f"P(S = s_obs)     = {rep['pmf']:.6e}")
print(f"CDF P(S<=s_obs)  = {rep['cdf']:.6f}")
print(f"百分位           = {rep['percentile']:.4f}%")
print("=" * 64)