from weighted_poisson_binomial import WeightedPoissonBinomial
from weighted_poisson_binomial_saddle import percentile_of_outcome
import numpy as np
from time import time


p = [0.493224, 0.483425, 0.472345, 0.460987, 0.452362, 0.449809, 0.4323091, 0.429872, 0.4132457, 0.49891, 0.399893, 0.3898272, 0.3609723, 0.3412345, 0.3243645, 0.398612 ]*8
w = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 8, 10]*8
rng = np.random.default_rng(42)


# ===== 1st 
model = WeightedPoissonBinomial(p, w)
obs = (rng.random(model.n) < np.array(p)).tolist()

# timeit
t0 = time()
rep = model.summarize(obs)
t1 = time()

print("观测布尔序列     =", "".join("1" if b else "0" for b in obs))
print(f"观测总贡献 s_obs = {rep['observed_contribution']:.4f}")
print(f"P(S = s_obs)     = {rep['pmf']:.6e}")
print(f"CDF P(S<=s_obs)  = {rep['cdf']:.6f}")
print(f"百分位           = {rep['percentile']:.4f}%")
print(f"计算耗时         = {t1-t0:.6f} 秒")
print("=" * 64)

# ===== saddle
t0 = time()
rep = percentile_of_outcome(p, w, obs)
t1 = time()

print(f"Saddlepoint 百分位 = {rep:.4f}")
print(f"计算耗时         = {t1-t0:.6f} 秒")
