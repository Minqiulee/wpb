"""加权泊松多项分布 · 测试案例：准确性校验 + 本地计算时间基准。

场景：n = 100 次独立试验，每次 K = 6 种互斥结果；随机生成 50 个独立
"过程"（即 50 组观测，每组按真实概率抽样得到各试验的结果索引），
对每个过程评估其总贡献在分布中的 CDF / 百分位。

准确性校验（三重）：
1. 矩校验：FFT 反算的 E[S]、Var[S] 与解析式逐位对齐。
2. 蒙特卡洛独立对照：抽样 N_mc 次完整过程，用经验 CDF 与 FFT 解析 CDF
   逐一比对（二者共享同一批样本，仅阈值不同）。
3. PIT 均匀性：若模型正确，观测的 CDF 应近似服从 Uniform(0, 1)，
   用 Kolmogorov–Smirnov 统计量检验。

时间基准：
- fit 首次（含 numba JIT 编译）与 warm 后的纯计算时间；
- 50 次 summarize 的总耗时与单次平均。
"""

from __future__ import annotations

import time

import numpy as np

from weighted_poisson_multinomial import WeightedPoissonMultinomial

# --------------------------------------------------------------------------- #
#  参数
# --------------------------------------------------------------------------- #
N = 100          # 试验数
K = 6            # 每试验结果数
N_PROC = 50      # 模拟过程数（观测组数）
N_MC = 100_000_000   # 蒙特卡洛抽样次数（经验 CDF 对照用）

rng = np.random.default_rng(20261014)

# 概率矩阵：每行 Dirichlet(1,...,1)（均匀单纯形）抽样，天然行和为 1
p = rng.dirichlet(np.ones(K), size=N)          # (N, K)
# 权重矩阵：0..10 的整数（允许 0 表示某结果零贡献）
w = rng.integers(0, 11, size=(N, K)).astype(np.float64)


# --------------------------------------------------------------------------- #
#  fit：首次（含 JIT 编译）与 warm 后（纯计算）分别计时
# --------------------------------------------------------------------------- #
t0 = time.perf_counter()
model = WeightedPoissonMultinomial(p, w)
t_fit_first = time.perf_counter() - t0

t0 = time.perf_counter()
model2 = WeightedPoissonMultinomial(p, w)
t_fit_warm = time.perf_counter() - t0

print("=" * 68)
print("加权泊松多项分布 · 测试案例")
print("=" * 68)
print(f"试验数 n           = {model.n}")
print(f"每试验结果数 K     = {model.K}")
print(f"格点步长 step      = {model.step}")
print(f"FFT 长度 M         = {model.M}")
print(f"最大可能贡献       = {model.max_k * model.step:.1f}")
print(f"蒙特卡洛抽样次数   = {N_MC:,}")
print("-" * 68)


# --------------------------------------------------------------------------- #
#  模拟 50 个过程：按真实概率抽样观测，评估 CDF / 百分位
# --------------------------------------------------------------------------- #
cdf_p = np.cumsum(p, axis=1)  # (N, K) 累积概率，抽样用

obs_list = []
s_obs_list = []
fft_cdf_list = []
for _ in range(N_PROC):
    obs = np.array([
        int(np.searchsorted(cdf_p[i], rng.random())) for i in range(N)
    ])
    obs_list.append(obs)
    s_obs_list.append(model._obs_to_k(obs))

t0 = time.perf_counter()
for obs in obs_list:
    model.summarize(obs)
t_summarize_total = time.perf_counter() - t0

for obs, s in zip(obs_list, s_obs_list):
    fft_cdf_list.append(model.cdf_of(obs))


# --------------------------------------------------------------------------- #
#  蒙特卡洛独立对照：抽样 N_MC 次完整过程，与 FFT CDF 逐观测比对
# --------------------------------------------------------------------------- #
t0 = time.perf_counter()
idx = np.empty((N_MC, N), dtype=np.int64)
for i in range(N):
    idx[:, i] = np.searchsorted(cdf_p[i], rng.random(N_MC))
mc_samples = w[np.arange(N)[None, :], idx].sum(axis=1)  # (N_MC,)
mc_sorted = np.sort(mc_samples)
t_mc = time.perf_counter() - t0

mc_cdf_list = [float(np.searchsorted(mc_sorted, s, side="right")) / N_MC
               for s in s_obs_list]

fft_cdf = np.array(fft_cdf_list)
mc_cdf = np.array(mc_cdf_list)
diff = np.abs(fft_cdf - mc_cdf)


# --------------------------------------------------------------------------- #
#  PIT 均匀性（Kolmogorov–Smirnov）
# --------------------------------------------------------------------------- #
def _ks_uniform(x: np.ndarray) -> tuple[float, float]:
    """对样本 x 做 KS 检验，检验其是否来自 Uniform(0,1)。

    返回 (D 统计量, 渐近 p 值)。D = sup |F_n(t) - t|。
    """
    xs = np.sort(x)
    n = x.size
    f = (np.arange(1, n + 1)) / n
    d_plus = np.max(f - xs)
    d_minus = np.max(xs - np.arange(0, n) / n)
    D = max(d_plus, d_minus)
    # Kolmogorov 分布渐近 p 值（取前 100 项）
    z = D * np.sqrt(n)
    pval = 0.0
    for k in range(1, 101):
        term = (-1.0) ** (k - 1) * np.exp(-2.0 * (k * z) ** 2)
        pval += term
        if abs(term) < 1e-15:
            break
    pval = 2.0 * pval
    return D, max(0.0, min(1.0, pval))


D_ks, p_ks = _ks_uniform(fft_cdf)


# --------------------------------------------------------------------------- #
#  输出
# --------------------------------------------------------------------------- #
print("【时间基准】")
print(f"  fit 首次（含 JIT 编译） : {t_fit_first * 1e3:9.2f} ms")
print(f"  fit warm（纯计算）      : {t_fit_warm * 1e3:9.2f} ms")
print(f"  50 次 summarize 总耗时  : {t_summarize_total * 1e6:9.2f} µs")
print(f"  单次 summarize 平均     : {t_summarize_total / N_PROC * 1e6:9.2f} µs")
print(f"  蒙特卡洛抽样+排序       : {t_mc * 1e3:9.2f} ms")
print("-" * 68)

print("【准确性 · 矩校验（FFT PMF vs 解析式）】")
a_mean, n_mean, a_var, n_var = model.check_moments()
print(f"  解析 E[S]   = {a_mean:.12f}")
print(f"  数值 E[S]   = {n_mean:.12f}   误差 = {abs(a_mean - n_mean):.3e}")
print(f"  解析 Var[S] = {a_var:.12f}")
print(f"  数值 Var[S] = {n_var:.12f}   误差 = {abs(a_var - n_var):.3e}")
print("-" * 68)

print("【准确性 · 蒙特卡洛 CDF 对照（50 个过程）】")
print(f"  最大绝对误差  = {diff.max():.4f}")
print(f"  平均绝对误差  = {diff.mean():.4f}")
print(f"  MC 经验 CDF 标准误 ≈ {0.5 / np.sqrt(N_MC):.4f}（对照基准）")
print("-" * 68)

print("【准确性 · PIT 均匀性（KS 检验）】")
print(f"  观测 CDF 均值 = {fft_cdf.mean():.4f}（均匀分布应为 0.5）")
print(f"  KS D 统计量   = {D_ks:.4f}")
print(f"  KS 渐近 p 值  = {p_ks:.4f}")
print("-" * 68)

# 抽样展示前 8 个观测的对照
print("【观测对照（前 8 个）】")
print(f"  {'#':>2}  {'贡献 s_obs':>10}  {'FFT CDF':>10}  {'MC CDF':>10}  {'Δ':>8}")
for i in range(8):
    print(f"  {i:>2}  {s_obs_list[i]:>10.1f}  {fft_cdf[i]:>10.6f}  "
          f"{mc_cdf[i]:>10.6f}  {diff[i]:>8.4f}")
print("=" * 68)

# --------------------------------------------------------------------------- #
#  结论判定
# --------------------------------------------------------------------------- #
# 用相对误差判定（方差绝对量级较大时，FFT 累积舍入误差 ~ M·eps·Var，绝对阈值无意义）
rel_mean = abs(a_mean - n_mean) / max(abs(a_mean), 1e-12)
rel_var = abs(a_var - n_var) / max(abs(a_var), 1e-12)
ok_moments = rel_mean < 1e-9 and rel_var < 1e-9
ok_mc = diff.max() < 5 * 0.5 / np.sqrt(N_MC)  # 5σ 内
ok_pit = p_ks > 0.01
print("判定：")
print(f"  [{'PASS' if ok_moments else 'FAIL'}] 矩校验")
print(f"  [{'PASS' if ok_mc else 'FAIL'}] 蒙特卡洛 CDF 对照（最大误差 < 5σ）")
print(f"  [{'PASS' if ok_pit else 'FAIL'}] PIT 均匀性（p > 0.01）")
print(f"  => 整体 {'通过' if all([ok_moments, ok_mc, ok_pit]) else '未通过'}")
