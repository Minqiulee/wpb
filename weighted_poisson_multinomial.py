"""
加权泊松多项分布（Weighted Poisson Multinomial Distribution）建模与计算模块。

问题定义
--------
设 n 个独立试验，第 i 个试验有 K_i 种互斥结果（一般各试验结果数相同，均为 K），
结果 c 出现的概率为 p[i, c]（0 <= p[i, c] <= 1，且对每个 i 有 Σ_c p[i, c] = 1），
该结果对总贡献的权重为 w[i, c]（一般取非负实数）。

每次试验 i 恰好落入一个结果 c，用指示变量 X[i, c] ∈ {0, 1} 表示，且 Σ_c X[i, c] = 1。

总贡献随机变量：  S = Σ_i Σ_c  w[i, c] · X[i, c]

目标：给定一次观测（各试验结果索引序列 obs，或等价的 one-hot 矩阵），
计算该观测总贡献落在整体随机过程分布中的累积分布位置（CDF 值 + 百分位）。

核心算法：傅立叶变换（单位根法 / 特征函数法）
--------------------------------------------
S 的概率生成函数（PGF）为

        G(z) = Π_i [ Σ_c p[i, c] · z^{w[i, c]} ]

在单位圆 M 个等分点 ω_j = exp(2π i j / M) 上采样 PGF 得到
        g_j = G(ω_j) ,  j = 0, ..., M-1

则 PMF 恰为 g 的离散傅立叶逆变换（归一化系数 1/M）：

        P(S = k) = (1/M) · Σ_j g_j · ω_j^{-jk}  =  IFFT(g)[k]

权重处理与泊松二项分布版一致：w 为小数时先统一缩放到整数格点（取公分母的最小公倍数），
使 z^{w} 成为 z^{int(w)}，缩放仅改变格点间距，PMF/CDF 取值与百分位不变。

与加权泊松二项分布的关系：当 K=2 且第二结果权重 w[i, 1] = 0 时，本模型退化为
加权泊松二项分布（p[i, 0] 即成功概率）。故本模块是后者的严格推广。

数值精度 / 误差来源（详见 docstring 末尾）
"""

from __future__ import annotations

import math
from fractions import Fraction
from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np

# Numba 为可选加速依赖：存在则用 @njit 编译 PGF 采样循环，否则回退到
# 纯 NumPy 广播实现。二者数值结果一致（同一套浮点算法）。
try:
    from numba import njit, prange

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - 环境无 numba 时优雅降级
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def _deco(fn):
            return fn

        return _deco

    def prange(x):  # 无 numba 时退化为普通 range
        return range(x)


@njit(cache=True, fastmath=True, parallel=True)
def _pgf_sample_numba(p: np.ndarray, w_int: np.ndarray, M: int) -> np.ndarray:
    """Numba 核：在 M 个单位根上累乘 PGF。

    与纯 NumPy 版 `_pgf_sample_vectorized` 数值等价，仅实现方式不同。
    G[j] = Π_i [ Σ_c p[i,c] · e^{-2πi j m[i,c] / M} ]，j = 0..M-1。
    """
    n = p.shape[0]
    K = p.shape[1]
    G = np.ones(M, dtype=np.complex128)
    for i in range(n):
        for j in prange(M):
            acc = 0.0 + 0.0j
            for c in range(K):
                pv = p[i, c]
                if pv == 0.0:
                    continue
                m = w_int[i, c]
                if m == 0:
                    acc += pv  # z^0 = 1，贡献常数 pv
                else:
                    acc += pv * np.exp(-2j * np.pi * j * m / M)
            G[j] *= acc
    return G


def _pgf_sample_vectorized(p: np.ndarray, w_int: np.ndarray, M: int) -> np.ndarray:
    """纯 NumPy 版 PGF 采样（无 Numba 时的回退 / 交叉校验对照）。

    逐试验循环（外层），内层用 (K, M) 广播一次算出该试验对全部格点的因子，
    再逐点累乘。内存 O(K·M)，避免 n·K·M 的三维中间数组。
    """
    j = np.arange(M, dtype=np.float64)
    angles = -2.0 * np.pi * j / M  # shape (M,)
    n = p.shape[0]
    G = np.ones(M, dtype=np.complex128)
    for i in range(n):
        # factor = Σ_c p[i,c] · e^{-2πi j m[i,c]/M}，形状 (M,)
        phase = np.exp(1j * angles[None, :] * w_int[i, :, None])  # (K, M)
        factor = p[i] @ phase  # (M,)
        G *= factor
    return G


@dataclass
class WeightedPoissonMultinomial:
    """加权泊松多项分布模型。

    参数
    ----
    p : Sequence[Sequence[float]]
        各试验、各结果的出现概率，形状 (n, K)。要求 0 <= p[i,c] <= 1，
        且每一行（同一试验的全部结果）概率之和恒为 1。
    w : Sequence[Sequence[float]]
        各试验、各结果的权重，形状 (n, K)，非负实数。体现各结果对
        总贡献的大小。
    step : float, 可选
        格点缩放步长。若为 None，则由 w 的有理近似自动推导最小公倍格点。
        显式给出可控制格点分辨率与 M（FFT 长度）。
    use_numba : bool, 默认 True
        优先用 Numba JIT 采样；环境无 numba 时自动回退纯 NumPy。

    属性（拟合后）
    -------------
    n          : 试验数
    K          : 每试验的结果数
    p_         : 归一化后的概率矩阵（每行和为 1）
    w_         : 原始权重矩阵
    w_int_     : 缩放后的整数权重（真实权重 = w_int_ * step）
    step       : 实际格点步长
    max_k      : 最大可能整数格点贡献 = Σ_i max_c w_int[i,c]
    M          : FFT 长度（2 的幂，>= max_k + 1）
    pmf_       : 概率质量函数数组，长度 M；pmf_[k] = P(整数贡献 = k)
    cdf_       : 累积分布函数数组，长度 M；cdf_[k] = P(整数贡献 <= k)
    support_   : 非零概率格点索引数组
    """

    def __init__(
        self,
        p: Sequence[Sequence[float]],
        w: Sequence[Sequence[float]],
        step: Optional[float] = None,
        use_numba: bool = True,
    ):
        self._use_numba = use_numba and _HAS_NUMBA
        p = np.asarray(p, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        if p.ndim != 2 or p.shape != w.shape:
            raise ValueError("p 与 w 必须是形状相同的二维序列 (n, K)")
        if p.shape[0] == 0 or p.shape[1] == 0:
            raise ValueError("至少需要一次试验，且每次试验至少一个结果")

        self.n = int(p.shape[0])
        self.K = int(p.shape[1])

        # ---- 概率校验 + 归一化：保证每一行概率之和恒为 1 ----
        if np.any(np.isnan(p)):
            raise ValueError("p 中存在 NaN")
        if np.any((p < 0) | (p > 1)):
            raise ValueError("p[i, c] 必须落在 [0, 1]")
        row_sums = p.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError("每个试验至少需要一个正概率的结果")
        if np.any(np.abs(row_sums - 1.0) > 1e-6):
            raise ValueError(
                "p 每一行（同一试验各结果的概率）之和必须为 1，"
                f"实际偏差最大 {float(np.max(np.abs(row_sums - 1.0))):.3g}"
            )
        # 归一化消除浮点累积偏差，严格保证行和为 1
        self.p_ = p / row_sums[:, None]

        if np.any(np.isnan(w)) or np.any(w < 0):
            raise ValueError("w[i, c] 必须为非负实数")
        self.w_ = w.copy()

        # ---- 权重整数化：w_int = round(w / step) ----
        if step is None:
            step = self._auto_step(w)
        if not (step > 0):
            raise ValueError("step 必须为正数")

        w_int = np.rint(w / step).astype(np.int64)
        residual = np.max(np.abs(w - w_int * step))
        if residual > 1e-6 * max(1.0, float(np.max(w))):
            raise ValueError(
                f"权重无法在该 step={step} 下精确整数化（最大残差 {residual:.3g}）。"
                f"请显式传入更小的 step 或更大分母。"
            )
        self.step = float(step)
        self.w_int_ = w_int

        # 最大贡献：每次试验独立取权重最大的结果
        self.max_k = int(w_int.max(axis=1).sum())
        self._fit()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _auto_step(w: np.ndarray) -> float:
        """由权重的小数表示推导最小公倍格点。

        对每个非零权重取其 Fraction 分母，取全部分母的最小公倍数 L，
        则 step = 1 / L 保证所有 w[i, c] 都能精确落到整数格点上。
        权重全为 0 时退化为 step=1。
        """
        L = 1
        for wi in w.flat:
            if wi == 0:
                continue
            f = Fraction(wi).limit_denominator(1_000_000)
            L = math.lcm(L, f.denominator)
        return 1.0 / L if L > 1 else 1.0

    def _fit(self) -> None:
        """单位根法：采样 PGF -> IFFT -> PMF/CDF。"""
        # FFT 长度：2 的幂且 > max_k
        M = 1
        while M <= self.max_k:
            M <<= 1
        self.M = int(M)

        # 单位根 omega = exp(-2π i / M)（负指数采样）。
        # pmf[k] = (1/M) Σ_j G(ω^j) ω^{-jk}，用负角度采样 G(e^{-2πi j/M})，
        # 再配合 np.fft.ifft（相位 +2πi jk/M，与 ω^{-jk} 抵消）。误用正角度
        # 会导致系数循环移位 + 折叠（混叠），PMF 失真。
        if self._use_numba:
            G = _pgf_sample_numba(self.p_, self.w_int_, self.M)
        else:
            G = _pgf_sample_vectorized(self.p_, self.w_int_, self.M)

        pmf = np.conj(np.fft.fft(np.conj(G))) / M
        pmf = np.real(pmf)
        pmf = np.clip(pmf, 0.0, None)
        pmf /= pmf.sum()  # 归一化，抵消累积舍入
        self.pmf_ = pmf

        self.cdf_ = np.cumsum(pmf)
        self.support_ = np.nonzero(pmf > 1e-12)[0]

    # ------------------------------------------------------------------ #
    #  查询接口
    # ------------------------------------------------------------------ #
    def _obs_to_k(self, obs: Sequence) -> int:
        """把观测序列映射到整数格点贡献 k = round(Σ_i w[i, obs_i] / step)。

        支持两种观测表示：
        - (n,) 整型结果索引（每项取值 0..K-1）；
        - (n, K) one-hot 布尔/数值矩阵（每行恰一个 1）。
        """
        obs = np.asarray(obs)
        if obs.ndim == 2:
            if obs.shape != (self.n, self.K):
                raise ValueError(f"one-hot 观测形状应为 ({self.n}, {self.K})，得到 {obs.shape}")
            obs = obs.argmax(axis=1)
        obs = obs.astype(np.int64)
        if obs.shape != (self.n,):
            raise ValueError(f"obs 长度应为 {self.n}，得到 {obs.shape}")
        if np.any((obs < 0) | (obs >= self.K)):
            raise ValueError(f"obs 中的结果索引必须落在 [0, {self.K})")
        contrib = self.w_[np.arange(self.n), obs].sum()
        return int(round(float(contrib / self.step)))

    def cdf_of(self, obs: Sequence) -> float:
        """观测序列对应的累积概率 P(S <= s_obs)。"""
        k = self._obs_to_k(obs)
        if k < 0:
            return 0.0
        if k >= self.M:
            return 1.0
        return float(self.cdf_[k])

    def pmf_of(self, obs: Sequence) -> float:
        """观测序列对应总贡献的概率质量 P(S = s_obs)。"""
        k = self._obs_to_k(obs)
        if k < 0 or k >= self.M:
            return 0.0
        return float(self.pmf_[k])

    def percentile_of(self, obs: Sequence) -> float:
        """观测结果在分布中的百分位（0~100）。

        定义为 CDF(s_obs) * 100，即"不超过该观测"的概率占比。
        """
        return self.cdf_of(obs) * 100.0

    def summarize(self, obs: Sequence) -> dict:
        """一次观测的完整报告：贡献值、PMF、CDF、百分位。"""
        k = self._obs_to_k(obs)
        s_obs = k * self.step
        return {
            "observed_contribution": float(s_obs),
            "grid_index_k": k,
            "pmf": self.pmf_of(obs),
            "cdf": self.cdf_of(obs),
            "percentile": self.percentile_of(obs),
            "max_contribution": float(self.max_k * self.step),
            "n_trials": self.n,
            "n_outcomes_per_trial": self.K,
            "fft_length_M": self.M,
        }

    # ------------------------------------------------------------------ #
    #  解析统计量（独立于 FFT，用于交叉校验）
    # ------------------------------------------------------------------ #
    def mean(self) -> float:
        """E[S] = Σ_i Σ_c p[i,c] w[i,c]。"""
        return float((self.p_ * self.w_).sum())

    def variance(self) -> float:
        """Var[S] = Σ_i [ Σ_c p[i,c] w[i,c]^2 - (Σ_c p[i,c] w[i,c])^2 ]。

        单次多项试验贡献的方差为 E[w^2] - E[w]^2，各试验独立故直接求和。
        """
        e1 = (self.p_ * self.w_).sum(axis=1)               # Σ_c p w
        e2 = (self.p_ * self.w_ ** 2).sum(axis=1)          # Σ_c p w^2
        return float((e2 - e1 ** 2).sum())

    def std(self) -> float:
        return math.sqrt(self.variance())

    def check_moments(self) -> tuple[float, float, float, float]:
        """用 PMF 数值反算期望/方差，与解析式对比，评估 FFT 数值误差。

        返回 (解析期望, 数值期望, 解析方差, 数值方差)。
        """
        k = np.arange(self.M, dtype=np.float64)
        num_mean = float((k * self.pmf_).sum()) * self.step
        num_var = float((k * k * self.pmf_).sum()) * self.step ** 2 - num_mean ** 2
        return self.mean(), num_mean, self.variance(), num_var


# ===================================================================== #
#  FFT 精度与数值误差来源说明
# ===================================================================== #
_ACCURACY_NOTES = r"""
FFT 精度与数值误差来源：

1. 单位根采样误差（主导，随 M 增大而减小）
   单位根法把连续支撑离散到 M 个格点。若 M < 真实最大贡献+1，会发生
   "混叠（aliasing）"——支撑尾部的概率折叠回低频段，导致 PMF 严重失真。
   本实现强制 M 为 2 的幂且 M > max_k（= Σ_i max_c w_int[i,c]），
   从根上避免混叠。

2. 权重整数化误差
   w[i,c] 为小数时需 step 缩放，step 不当会引入量化误差。本实现用
   Fraction 自动取公倍 step，并校验残差 < 1e-6，超出即报错。

3. 复数乘法累积舍入误差
   PGF 是 n 个复数因子连乘（每个因子又是 K 项求和），累积相对误差约
   n·eps 量级。n 很大时建议改用对数累加以稳定幅值。

4. IFFT 的浮点误差 + 负数/虚部残留
   逆变换后理论值应为非负实数，实际会出现 ~1e-15 量级的虚部与负值。
   实现中对实部做 clip>=0 并重新归一化。

5. 相位条件数
   当存在极接近 0 或 1 的 p[i,c]，或因子接近单位圆奇异点时相对误差放大。
   本实现跳过 p=0 的结果、对 w=0 的结果退化为常数项，避免无谓复数运算。

交叉校验：check_moments() 用 PMF 反算 E 与 Var，与解析式比对，
若二者差超过 1e-9，说明格点或精度配置有问题。
"""


# ===================================================================== #
#  验证示例：固定期望值与方差
# ===================================================================== #
def _validation_example() -> None:
    # 4 次试验、每次 3 种结果；权重体现不同结果的贡献差异
    p = [
        [0.50, 0.30, 0.20],
        [0.25, 0.50, 0.25],
        [0.60, 0.20, 0.20],
        [0.10, 0.40, 0.50],
    ]
    w = [
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [1.0, 1.0, 5.0],
        [0.0, 2.5, 1.5],
    ]

    model = WeightedPoissonMultinomial(p, w)

    # 解析期望与方差
    pp = np.asarray(p, dtype=np.float64)
    ww = np.asarray(w, dtype=np.float64)
    a_mean = float((pp * ww).sum())
    e1 = (pp * ww).sum(axis=1)
    e2 = (pp * ww ** 2).sum(axis=1)
    a_var = float((e2 - e1 ** 2).sum())

    a_mean_, n_mean, a_var_, n_var = model.check_moments()

    print("=" * 64)
    print("加权泊松多项分布 · FFT（单位根法）验证示例")
    print("=" * 64)
    print(f"试验数 n        = {model.n}")
    print(f"每试验结果数 K  = {model.K}")
    print(f"格点步长 step   = {model.step}")
    print(f"FFT 长度 M      = {model.M}")
    print(f"最大可能贡献     = {model.max_k * model.step:.4f}")
    print("-" * 64)
    print(f"解析期望 E[S]   = {a_mean_:.12f}")
    print(f"数值期望 (PMF)  = {n_mean:.12f}")
    print(f"解析方差 Var[S] = {a_var_:.12f}")
    print(f"数值方差 (PMF)  = {n_var:.12f}")
    print(f"期望相对误差     = {abs(a_mean_ - n_mean) / max(a_mean_, 1e-12):.3e}")
    print(f"方差相对误差     = {abs(a_var_ - n_var) / max(a_var_, 1e-12):.3e}")
    print("-" * 64)

    # 构造一个观测：随机给定各试验的结果索引
    rng = np.random.default_rng(42)
    obs = [int(rng.choice(model.K, p=row)) for row in pp]
    rep = model.summarize(obs)
    print("观测结果索引     =", obs)
    print(f"观测总贡献 s_obs = {rep['observed_contribution']:.4f}")
    print(f"P(S = s_obs)     = {rep['pmf']:.6e}")
    print(f"CDF P(S<=s_obs)  = {rep['cdf']:.6f}")
    print(f"百分位           = {rep['percentile']:.4f}%")
    print("=" * 64)

    # 断言校验
    assert abs(a_mean_ - n_mean) < 1e-9, "期望校验失败"
    assert abs(a_var_ - n_var) < 1e-9, "方差校验失败"
    assert abs(model.pmf_.sum() - 1.0) < 1e-12, "PMF 归一化失败"
    assert 0.0 <= rep["cdf"] <= 1.0
    print("✔ 校验通过：期望/方差与解析式一致，PMF 归一化成立。")


# ===================================================================== #
#  边界情况行为说明 + 自测
# ===================================================================== #
def _edge_cases() -> None:
    print("\n" + "=" * 64)
    print("边界情况处理")
    print("=" * 64)

    # (1) 确定性试验：某结果概率为 1，贡献恒为该结果权重
    # (2) 结果权重为 0：该项退化为常数 p（不随 z 变化）
    # (3) one-hot 观测与索引观测等价
    m = WeightedPoissonMultinomial(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        [[3.0, 0.0], [0.0, 2.0], [4.0, 1.0]],
    )
    print("[确定性 / 零权重 混合]")
    print(f"  解析期望 = {m.mean():.4f} "
          f"(1*3 + 0.5*0 + 0.5*2 + 1*1 = {1*3 + 0.5*0 + 0.5*2 + 1*1:.1f})")
    print(f"  PMF 非零格点 = {m.support_.tolist()}")

    # one-hot 与索引两种观测应得到相同结果
    obs_idx = [0, 1, 1]
    obs_onehot = [[1, 0], [0, 1], [0, 1]]
    print("[one-hot 与索引观测等价]")
    print(f"  索引观测 cdf = {m.cdf_of(obs_idx):.6f}")
    print(f"  one-hot cdf = {m.cdf_of(obs_onehot):.6f}")

    # (4) 退化为加权泊松二项分布：K=2 且第二结果权重为 0
    from weighted_poisson_binomial import WeightedPoissonBinomial
    p1 = [0.3, 0.6, 0.9]
    w1 = [1.0, 2.0, 5.0]
    ref = WeightedPoissonBinomial(p1, w1)
    deg = WeightedPoissonMultinomial(
        [[pi, 1 - pi] for pi in p1],
        [[wi, 0.0] for wi in w1],
    )
    print("[退化到加权泊松二项分布]")
    print(f"  参考期望 = {ref.mean():.6f}, 退化模型期望 = {deg.mean():.6f}")
    print(f"  参考方差 = {ref.variance():.6f}, 退化模型方差 = {deg.variance():.6f}")
    assert abs(ref.mean() - deg.mean()) < 1e-12
    assert abs(ref.variance() - deg.variance()) < 1e-12

    # (5) 非法输入
    bads = [
        ([[0.6, 0.6], [0.5, 0.5]], [[1, 2], [1, 2]]),  # 行和 != 1
        ([[-0.1, 1.1], [0.5, 0.5]], [[1, 2], [1, 2]]),  # 越界
        ([[0.0, 1.0], [0.5, 0.5]], [[1, 2]]),           # 形状不匹配
        ([[0.5, 0.5], [0.5, 0.5]], [[1, -2], [1, 2]]),  # 负权重
    ]
    for bp, bw in bads:
        try:
            WeightedPoissonMultinomial(bp, bw)
            print("[非法输入] 未抛错（BUG）")
        except ValueError as e:
            print(f"[非法输入] 正确抛错: {e}")

    # (6) 小数权重自动格点
    m4 = WeightedPoissonMultinomial(
        [[0.3, 0.7], [0.6, 0.4]],
        [[0.5, 1.25], [1.0, 2.0]],
    )
    print(f"[小数权重] 自动 step={m4.step}, 期望={m4.mean():.4f} "
          f"(解析 {0.3*0.5+0.7*1.25+0.6*1.0+0.4*2.0:.4f})")
    print("=" * 64)


if __name__ == "__main__":
    _validation_example()
    _edge_cases()
