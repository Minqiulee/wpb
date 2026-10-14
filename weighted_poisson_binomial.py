"""
加权泊松二项分布（Weighted Poisson Binomial Distribution）建模与计算模块。

问题定义
--------
设 n 个独立伯努利试验，第 i 个试验成功概率为 p_i（0 <= p_i <= 1），
成功时对总贡献的权重为 w_i（一般取非负实数）。

总贡献随机变量：  S = Σ_{i=1}^{n}  w_i · X_i ,   X_i ~ Bernoulli(p_i)

目标：给定一次观测（各试验成功/失败的布尔序列 obs），
计算该观测总贡献落在整体随机过程分布中的累积分布位置（CDF 值 + 百分位）。

核心算法：傅立叶变换（单位根法 / 特征函数法）
--------------------------------------------
S 的概率生成函数（PGF）为

        G(z) = Π_{i} [ (1 - p_i) + p_i · z^{w_i} ]

在单位圆 M 个等分点 ω_j = exp(2π i j / M) 上采样 PGF 得到
        g_j = G(ω_j) ,  j = 0, ..., M-1

则 PMF 恰为 g 的离散傅立叶逆变换（归一化系数 1/M）：

        P(S = k) = (1/M) · Σ_j g_j · ω_j^{-jk}  =  IFFT(g)[k]

由于 g_j 是各因子乘积、可直接 O(n·M) 算，再 IFFT O(M log M)，
总复杂度 O(n·M log M)，远优于朴素卷积 O(n·M^2)。

权重处理：w_i 可能是小数，需先统一缩放到整数格点（乘以分母的公倍数），
使 z^{w_i} 成为 z^{int(w_i)}，才能在单位根上采样。缩放仅改变支撑的格点间距，
PMF/CDF 的取值与百分位不变（我们返回整数格点 k 到其真实贡献 k·step 的映射）。

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

    与纯 NumPy 广播版 `_pgf_sample_vectorized` 数值等价，仅实现方式不同。
    G[0..M-1] = Π_i [ (1-p_i) + p_i · e^{-2πi j m_i / M} ]，j = 0..M-1。
    """
    n = p.shape[0]
    G = np.ones(M, dtype=np.complex128)
    for i in range(n):
        pi = p[i]
        if pi == 0.0:
            continue
        mi = w_int[i]
        if pi == 1.0:
            for j in prange(M):
                G[j] *= np.exp(-2j * np.pi * j * mi / M)
            continue
        if mi == 0:
            continue
        for j in prange(M):
            G[j] *= (1.0 - pi) + pi * np.exp(-2j * np.pi * j * mi / M)
    return G


def _pgf_sample_vectorized(p: np.ndarray, w_int: np.ndarray, M: int) -> np.ndarray:
    """纯 NumPy 广播版 PGF 采样（无 Numba 时的回退 / 交叉校验对照）。

    一次性广播所有 (i, j) 项再按 i 累乘，以内存换 Python 循环。
    """
    j = np.arange(M, dtype=np.float64)
    angles = -2.0 * np.pi * j / M  # shape (M,)

    # 过滤退化因子，避免无谓复数运算，语义与原循环一致
    mask = (p > 0.0) & (p < 1.0) & (w_int != 0)
    p_e = p[mask]
    m_e = w_int[mask]

    G = np.ones(M, dtype=np.complex128)
    if p_e.size > 0:
        # (1-p_i) + p_i·e^{-2πi j m_i/M}，形状 (n_e, M)
        phase = np.exp(1j * angles[None, :] * m_e[:, None])
        factors = (1.0 - p_e)[:, None] + p_e[:, None] * phase
        G = factors.prod(axis=0)

    # p_i == 1 的确定性因子：z^{m_i}，相位为 -2πi j m_i/M
    det = w_int[(p == 1.0) & (w_int != 0)]
    if det.size > 0:
        det_phase = np.exp(1j * angles[None, :] * det[:, None])
        G = G * det_phase.prod(axis=0)

    return G


@dataclass
class WeightedPoissonBinomial:
    """加权泊松二项分布模型。

    参数
    ----
    p : Sequence[float]
        各试验成功概率，取值 [0, 1]。允许 0（恒失败）或 1（恒成功）。
    w : Sequence[float]
        各试验权重，非负实数。体现各试验对"总成功次数"贡献的大小。
        权重为 0 表示该试验对总贡献无影响（等效于忽略该试验的取值波动，
        但注意：若 p_i=1 且 w_i=0，仍对贡献无影响，见边界说明）。
    step : float, 可选
        格点缩放步长。若为 None，则由 w 的有理近似自动推导最小公倍格点。
        显式给出可控制格点分辨率与 M（FFT 长度）。

    属性（拟合后）
    -------------
    n          : 试验数
    p_         : 归一化后的成功概率（内部裁剪到 [0,1]）
    w_int_     : 缩放后的整数权重（真实权重 = w_int_ * step）
    step       : 实际格点步长
    max_k      : 最大可能整数格点贡献 = Σ w_int_（对应真实贡献 = max_k*step）
    M          : FFT 长度（2 的幂，>= max_k + 1）
    pmf_       : 概率质量函数数组，长度 M；pmf_[k] = P(整数贡献 = k)
    cdf_       : 累积分布函数数组，长度 M；cdf_[k] = P(整数贡献 <= k)
    support_   : 非零概率格点索引数组
    """

    def __init__(
        self,
        p: Sequence[float],
        w: Sequence[float],
        step: Optional[float] = None,
        use_numba: bool = True,
    ):
        # use_numba：优先用 Numba JIT 采样循环；为 False 或环境无 numba 时
        # 回退到纯 NumPy 广播实现。二者数值一致。
        self._use_numba = use_numba and _HAS_NUMBA
        p = np.asarray(p, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        if p.shape != w.shape or p.ndim != 1:
            raise ValueError("p 与 w 必须是一维且等长的序列")
        if len(p) == 0:
            raise ValueError("至少需要一次试验")

        self.n = int(len(p))
        # 裁剪到 [0,1]，NaN 或越界抛错
        if np.any(np.isnan(p)):
            raise ValueError("p 中存在 NaN")
        if np.any((p < 0) | (p > 1)):
            raise ValueError("p_i 必须落在 [0, 1]")
        if np.any(np.isnan(w)) or np.any(w < 0):
            raise ValueError("w_i 必须为非负实数")

        self.p_ = p.copy()
        self.w_ = w.copy()

        # ---- 权重整数化：w_int = round(w / step) ----
        if step is None:
            step = self._auto_step(w)
        if not (step > 0):
            raise ValueError("step 必须为正数")

        w_int = np.rint(w / step).astype(np.int64)
        # 验证量化误差可忽略（相对误差 < 1e-6）
        residual = np.max(np.abs(w - w_int * step))
        if residual > 1e-6 * max(1.0, float(np.max(w))):
            raise ValueError(
                f"权重无法在该 step={step} 下精确整数化（最大残差 {residual:.3g}）。"
                f"请显式传入更小的 step 或更大分母。"
            )
        self.step = float(step)
        self.w_int_ = w_int

        self.max_k = int(w_int.sum())
        self._fit()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _auto_step(w: np.ndarray) -> float:
        """由权重的小数表示推导最小公倍格点。

        对每个非零权重取其 Fraction 分母，取分母的最小公倍数 L，
        则 step = 1 / L 保证所有 w_i 都能精确落到整数格点上。
        权重全为 0 时退化为 step=1。
        """
        L = 1
        for wi in w:
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

        # 单位根 omega = exp(-2π i / M)（负指数采样）
        # 关键：pmf[k] = (1/M) Σ_j G(ω^j) ω^{-jk}，其中 ω = e^{-2πi/M}。
        # 因此须用负角度采样 G(e^{-2πi j/M})，再配合 np.fft.ifft
        # （其相位为 +2πi jk/M，恰与 ω^{-jk} 抵消）。若误用正角度，
        # 会因相位不匹配导致系数循环移位 + 折叠（混叠），PMF 失真。
        #
        # PGF 采样：G(ω^j) = Π_i [ (1-p_i) + p_i · ω^{j·m_i} ]。
        # 该累乘是 n×M 的逐元素热点，用 Numba JIT（或 NumPy 广播）加速。
        # 逆 FFT 恢复 PMF（用共轭 + 正 FFT + 共轭 实现 IFFT，规避 scipy 依赖）
        if self._use_numba:
            G = _pgf_sample_numba(self.p_, self.w_int_, self.M)
        else:
            G = _pgf_sample_vectorized(self.p_, self.w_int_, self.M)

        pmf = np.conj(np.fft.fft(np.conj(G))) / M
        # 取实部；理论应为非负实数，因浮点误差可能有 -1e-16 量级虚部/负值
        pmf = np.real(pmf)
        pmf = np.clip(pmf, 0.0, None)
        pmf /= pmf.sum()  # 归一化，抵消累积舍入
        self.pmf_ = pmf

        self.cdf_ = np.cumsum(pmf)
        self.support_ = np.nonzero(pmf > 1e-12)[0]

    # ------------------------------------------------------------------ #
    #  查询接口
    # ------------------------------------------------------------------ #
    def _obs_to_k(self, obs: Sequence[bool]) -> int:
        """把布尔观测序列映射到整数格点贡献 k = round(Σ w_i·x_i / step)。"""
        obs = np.asarray(obs, dtype=bool)
        if obs.shape != (self.n,):
            raise ValueError(f"obs 长度应为 {self.n}，得到 {len(obs)}")
        return int(round(float((self.w_ * obs).sum() / self.step)))

    def cdf_of(self, obs: Sequence[bool]) -> float:
        """观测序列对应的累积概率 P(S <= s_obs)。"""
        k = self._obs_to_k(obs)
        if k < 0:
            return 0.0
        if k >= self.M:
            return 1.0
        return float(self.cdf_[k])

    def pmf_of(self, obs: Sequence[bool]) -> float:
        """观测序列对应总贡献的概率质量 P(S = s_obs)。"""
        k = self._obs_to_k(obs)
        if k < 0 or k >= self.M:
            return 0.0
        return float(self.pmf_[k])

    def percentile_of(self, obs: Sequence[bool]) -> float:
        """观测结果在分布中的百分位（0~100）。

        定义为 CDF(s_obs) * 100，即"不超过该观测"的概率占比。
        边界：s_obs 低于最小支撑 -> 0；高于最大支撑 -> 100。
        """
        return self.cdf_of(obs) * 100.0

    def summarize(self, obs: Sequence[bool]) -> dict:
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
            "fft_length_M": self.M,
        }

    # ------------------------------------------------------------------ #
    #  解析统计量（独立于 FFT，用于交叉校验）
    # ------------------------------------------------------------------ #
    def mean(self) -> float:
        """E[S] = Σ w_i p_i。"""
        return float((self.w_ * self.p_).sum())

    def variance(self) -> float:
        """Var[S] = Σ w_i^2 p_i (1 - p_i)。"""
        return float((self.w_ ** 2 * self.p_ * (1.0 - self.p_)).sum())

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
   本实现强制 M 为 2 的幂且 M > max_k，从根上避免混叠。

2. 权重整数化误差
   w_i 是小数时，需 step 缩放。若 step 选择不当（分母取不够大），
   round(w/step) 引入量化误差，等价于把真实权重"拉扯"到邻近格点，
   使均值/方差产生 O(step) 的偏移。本实现用 Fraction 自动取公倍 step，
   并校验残差 < 1e-6，超出即报错。

3. 复数乘法累积舍入误差
   PGF 是 n 个复数因子连乘，每步相对误差约 1 个 eps（≈2.2e-16），
   n 步后约 n·eps 量级。n 很大（如 >1e4）时建议改用对数累加
   （log(1-p+p·e^{iθ})）以稳定幅值。

4. IFFT 的浮点误差 + 负数/虚部残留
   逆变换后理论值应为非负实数，实际会出现 ~1e-15 量级的虚部与负值。
   实现中对实部做 clip>=0 并重新归一化。

5. 相位条件数
   当存在极接近 0 或 1 的 p_i，或极小的 w_i 使因子接近单位圆奇异点，
   (1-p)+p·e^{iθ} 的幅值可能极接近 0，取对数/连乘都会放大相对误差。
   本实现单独处理 p=0 与 p=1 的退化因子，避免无谓的复数运算。

交叉校验：check_moments() 用 PMF 反算 E 与 Var，与解析式比对，
若二者差超过 1e-9，说明格点或精度配置有问题。
"""


# ===================================================================== #
#  验证示例：固定期望值与方差
# ===================================================================== #
def _validation_example() -> None:
    # 16 个离散有理小数（0.01~0.2），权重体现不同试验贡献
    p = [0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4,
        0.39, 0.38, 0.36, 0.34, 0.32, 0.3 ]
    w = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 8, 10]

    model = WeightedPoissonBinomial(p, w)

    # 解析期望与方差
    mean = sum(pi * wi for pi, wi in zip(p, w))
    var = sum(wi ** 2 * pi * (1 - pi) for pi, wi in zip(p, w))

    a_mean, n_mean, a_var, n_var = model.check_moments()

    print("=" * 64)
    print("加权泊松二项分布 · FFT（单位根法）验证示例")
    print("=" * 64)
    print(f"试验数 n        = {model.n}")
    print(f"格点步长 step   = {model.step}")
    print(f"FFT 长度 M      = {model.M}")
    print(f"最大可能贡献     = {model.max_k * model.step:.4f}")
    print("-" * 64)
    print(f"解析期望 E[S]   = {a_mean:.12f}")
    print(f"数值期望 (PMF)  = {n_mean:.12f}")
    print(f"解析方差 Var[S] = {a_var:.12f}")
    print(f"数值方差 (PMF)  = {n_var:.12f}")
    print(f"期望相对误差     = {abs(a_mean - n_mean) / max(a_mean, 1e-12):.3e}")
    print(f"方差相对误差     = {abs(a_var - n_var) / max(a_var, 1e-12):.3e}")
    print("-" * 64)

    # 构造一个观测：随机给定布尔序列
    rng = np.random.default_rng(42)
    obs = (rng.random(model.n) < np.array(p)).tolist()
    rep = model.summarize(obs)
    print("观测布尔序列     =", "".join("1" if b else "0" for b in obs))
    print(f"观测总贡献 s_obs = {rep['observed_contribution']:.4f}")
    print(f"P(S = s_obs)     = {rep['pmf']:.6e}")
    print(f"CDF P(S<=s_obs)  = {rep['cdf']:.6f}")
    print(f"百分位           = {rep['percentile']:.4f}%")
    print("=" * 64)

    # 断言校验
    assert abs(a_mean - n_mean) < 1e-9, "期望校验失败"
    assert abs(a_var - n_var) < 1e-9, "方差校验失败"
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

    # (1) p_i = 0（恒失败）：因子恒 1，等价于该试验被忽略
    # (2) p_i = 1（恒成功）：因子 = z^{w_i}，贡献恒为 w_i（确定性）
    # (3) 权重 w_i = 0：因子 = (1-p)+p = 1，对分布无任何影响
    m = WeightedPoissonBinomial([0.0, 1.0, 0.5, 0.5], [3.0, 4.0, 0.0, 2.0])
    print("[p=0 / p=1 / w=0 混合]")
    print(f"  解析期望 = {m.mean():.4f} (0*3 + 1*4 + 0.5*0 + 0.5*2 = {0*3 + 1*4 + 0.5*0 + 0.5*2:.1f})")
    print(f"  PMF 非零格点 = {m.support_.tolist()}")

    # (4) 观测结果超出可能取值范围：返回 0 或 1，不报错
    print("[观测超出范围]")
    # p 全为 1：贡献恒为 5（确定性），分布只支撑在 s=5
    m2 = WeightedPoissonBinomial([1.0, 1.0], [2.0, 3.0])
    print(f"  观测贡献=5(即全成功) cdf = {m2.cdf_of([True, True]):.4f} (应为 1.0)")
    print(f"  观测贡献=0(分布外)   cdf = {m2.cdf_of([False, False]):.4f} "
          f"(应为 0.0：该观测不可能出现，落入支撑之下)")

    # (5) 全 0 概率：S 恒为 0
    m3 = WeightedPoissonBinomial([0.0, 0.0], [1.0, 2.0])
    print(f"[全 p=0] pmf[0] = {m3.pmf_[0]:.4f} (应为 1.0，退化为单位质量)")

    # (6) 非法输入
    for bad in [([-0.1, 0.5], [1, 1]), ([0.5, 1.5], [1, 1]), ([0.5], [1, 2])]:
        try:
            WeightedPoissonBinomial(bad[0], bad[1])
            print("[非法输入] 未抛错（BUG）")
        except ValueError as e:
            print(f"[非法输入] 正确抛错: {e}")

    # (7) 权重小数自动格点
    m4 = WeightedPoissonBinomial([0.3, 0.6], [0.5, 1.25])
    print(f"[小数权重] 自动 step={m4.step}, w_int={m4.w_int_.tolist()}, "
          f"期望={m4.mean():.4f} (解析 {0.3*0.5+0.6*1.25:.4f})")
    print("=" * 64)


if __name__ == "__main__":
    _validation_example()
    _edge_cases()
