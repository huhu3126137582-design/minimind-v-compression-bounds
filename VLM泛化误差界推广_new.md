<style>
/* 统一显示公式：公式主体居中，编号以正文栏右边界为准对齐。 */
.katex-display,
mjx-container[display="true"],
.MathJax_Display,
.MathJax_SVG_Display,
.MathJax_CHTML_Display {
  box-sizing: border-box;
  display: block;
  position: relative;
  width: 100%;
  max-width: 100%;
  margin: 1em auto !important;
  text-align: center !important;
  overflow-x: auto;
  overflow-y: hidden;
}

/* 定义、引理和证明中的公式也使用完整的正文栏宽度。 */
.definition,
.lemma,
.proof {
  box-sizing: border-box;
  display: block;
  width: 100%;
  max-width: 100%;
}
.definition .katex-display,
.lemma .katex-display,
.proof .katex-display,
.definition mjx-container[display="true"],
.lemma mjx-container[display="true"],
.proof mjx-container[display="true"],
.definition .MathJax_Display,
.lemma .MathJax_Display,
.proof .MathJax_Display {
  width: 100% !important;
  max-width: 100% !important;
  margin-left: auto !important;
  margin-right: auto !important;
  text-align: center !important;
}
.katex-display > .katex {
  box-sizing: border-box;
  display: block;
  position: relative;
  width: 100%;
  text-align: center;
}
.katex-display > .katex > .katex-html {
  display: block;
  position: relative;
  width: 100%;
}
.katex-display > .katex > .katex-html > .tag,
.katex-display .tag {
  position: absolute !important;
  right: 0 !important;
  margin-left: 0.75em;
  text-align: right;
}

/* MathJax 的带编号公式占满正文栏，编号因此落在同一条右对齐线上。 */
mjx-container[display="true"] mjx-mtable[width="full"],
mjx-container[display="true"] mjx-labels {
  width: 100% !important;
}
mjx-container[display="true"] mjx-label {
  text-align: right !important;
}

/* 兼容仍使用 MathJax 2 的 Markdown 预览器。 */
.MathJax_Display {
  position: relative !important;
}
.MathJax_Display .MathJax_EqNumber,
.MathJax_SVG_Display .MathJax_EqNumber,
.MathJax_CHTML_Display .MathJax_EqNumber {
  position: absolute !important;
  right: 0 !important;
  left: auto !important;
  width: auto !important;
  text-align: right !important;
}

/* Markdown 与原生 HTML 表格统一水平居中。 */
table {
  display: table !important;
  width: auto;
  max-width: 100%;
  margin: 1em auto !important;
  border-collapse: collapse;
}
table th,
table td {
  text-align: center !important;
}
</style>

# 从大语言模型到视觉语言模型的非空泛化误差界

**图像–文本预训练的压缩界、子采样界及完整推导**

## 摘要

本文将 *Non-Vacuous Generalization Bounds for Large Language Models* 中的有限假设压缩界推广到视觉语言模型（Vision–Language Model, VLM）的图像–文本预训练。研究对象是条件自回归模型 $p_h(Y\mid I)$，其中图像作为条件输入，caption token 作为预测目标。证明以独立图像或独立图像簇为统计样本单位，通过 prediction smoothing 使条件负对数似然有界，再利用 Hoeffding 不等式、按压缩先验加权的 union bound 和 prefix-free 模型编码，得到可实际计算的 VLM 非空泛化界。本文进一步给出只评估训练集子样本时的严格界、多个 caption 共享同一图像时的分组处理、Top-$k$ token error 界、任意有限精度平滑参数的 prefix-free 选择成本以及非空性的显式条件。该结果尤其适用于冻结视觉编码器和语言底座、仅用低维子空间训练跨模态 Projector 的 VLM alignment pretraining。

## 目录

- [研究范围与核心思路](#研究范围与核心思路)
- [数据、模型与风险定义](#数据模型与风险定义)
  - [图像–caption 数据分布](#图像caption-数据分布)
  - [VLM 结构](#vlm-结构)
  - [条件 BPD 风险](#条件-bpd-风险)
- [Prediction smoothing 与损失有界性](#prediction-smoothing-与损失有界性)
  - [任意有限精度平滑参数的 prefix-free 编码](#任意有限精度平滑参数的-prefix-free-编码)
- [带先验的统一 VLM 泛化界](#带先验的统一-vlm-泛化界)
- [从 VLM 压缩长度构造先验](#从-vlm-压缩长度构造先验)
  - [固定 side information](#固定-side-information)
  - [Prefix-free 模型编码](#prefix-free-模型编码)
- [V-SubLoRA 参数化与实际编码长度](#v-sublora-参数化与实际编码长度)
  - [模块化低维参数化](#模块化低维参数化)
  - [单个参数块的量化编码](#单个参数块的量化编码)
- [训练风险子采样界](#训练风险子采样界)
- [共享图像的多个 caption](#sec:multi-caption)
  - [按图像身份分簇，而不是按数据行计数](#按图像身份分簇而不是按数据行计数)
  - [簇内多个 caption 的有界损失](#簇内多个-caption-的有界损失)
- [Top-k token error 泛化界](#top-k-token-error-泛化界)
- [超参数选择成本](#超参数选择成本)
- [非空性的显式条件](#非空性的显式条件)
- [不同训练范围下的码长](#不同训练范围下的码长)
- [与 MININTP 结构分析的关系](#与-minintp-结构分析的关系)
- [最终主定理：显式体现 VLM 的数据与模块结构](#最终主定理显式体现-vlm-的数据与模块结构)
- [适用范围与结论](#适用范围与结论)

## 研究范围与核心思路

本文只研究 VLM 的图像–caption 预训练，不研究视觉指令微调、多轮视觉问答、 下游任务迁移或分布外泛化。给定图像 $I$ 和固定 caption prompt $q_0$，模型以 自回归方式学习
<a id="eq:conditional-factorization"></a>

$$
p_h(Y\mid I,q_0)
    =
    \prod_{t=1}^{T}
    p_h(y_t\mid I,q_0,y_{<t}). \tag{1}
$$

由于 $q_0$ 是固定的，后文为简化记号将其省略。

推广的证明主线为

$$
\boxed{
    \begin{gathered}
    \text{条件自回归风险}
    \longrightarrow
    \text{prediction smoothing}
    \longrightarrow
    \text{有界损失}
    \\
    \longrightarrow
    \text{按压缩先验加权的统一界}
    \longrightarrow
    \text{可计算的 VLM certificate}
    \end{gathered}.}
$$

与纯文本 LLM 相比，视觉输入不会改变 Hoeffding 集中不等式本身；它主要改变 以下三项：

- 独立样本由文档变为图像–caption 对或图像簇；
- 经验风险变为 image-conditioned caption NLL；
- 模型描述长度需要包含可训练的视觉编码器、Projector 和语言模型参数。

## 数据、模型与风险定义

### 图像–caption 数据分布

设图像空间为 $\mathcal X$，离散词表为

$$
\mathcal{V}=\{1,\ldots,V\}, \tag{2}
$$

其中 $V=|\mathcal{V}|$。一条图像–文本预训练样本记为

$$
Z=(I,Y),\qquad
    I\in\mathcal X,\qquad
    Y=(y_1,\ldots,y_T)\in\mathcal{V}^T, \tag{3}
$$

且 $1\le T\le T_{\max}$。

**假设 1** (图像级独立同分布). 训练集
<a id="eq:iid-pairs"></a>

$$
S_N=\{Z_i=(I_i,Y_i)\}_{i=1}^{N}
    \overset{\mathrm{i.i.d.}}{\sim}\mathcal{D}^N \tag{4}
$$

由图像–caption 总体分布 $\mathcal{D}$ 独立抽取。若同一图像有多个 caption， 则应使用第  节的图像簇定义，不能把共享同一图像的 多条记录直接视为相互独立。*

### VLM 结构

设 VLM 由视觉编码器、跨模态 Projector 和自回归语言模型组成：

$$
\begin{aligned}
    E_\psi &: \mathcal X\to\mathbb{R}^{M\times d_v},
\end{aligned} \tag{5}
$$

$$
\begin{aligned}
    P_\omega &: \mathbb{R}^{M\times d_v}\to\mathbb{R}^{M\times d},
\end{aligned} \tag{6}
$$

$$
\begin{aligned}
    G_\theta &: \mathbb{R}^{M\times d}\times\mathcal{V}^{t-1}
    \to\Delta^{V-1}.
\end{aligned} \tag{7}
$$

其中 $M$ 是视觉 token 数，$\Delta^{V-1}$ 是 $V$ 维概率单纯形。给定图像 $I$，

$$
V_I=E_\psi(I),\qquad Z_I=P_\omega(V_I). \tag{8}
$$

令

$$
h=(\psi,\omega,\theta) \tag{9}
$$

表示完整 VLM，则第 $t$ 个 caption token 的条件概率为

$$
p_h(y_t\mid I,y_{<t})
    =
    G_\theta(Z_I,y_{<t})_{y_t}. \tag{10}
$$

整个 caption 的条件概率即式 <a href="#eq:conditional-factorization">(1)</a>。

### 条件 BPD 风险

原始的图像条件 bits-per-dimension（BPD）损失定义为
<a id="eq:raw-bpd"></a>

$$
\ell(h;I,Y)
    =
    -\frac{1}{T}
    \sum_{t=1}^{T}
    \log_2 p_h(y_t\mid I,y_{<t}). \tag{11}
$$

对应的总体风险为

$$
R(h)
    =
    \mathbb{E}_{(I,Y)\sim\mathcal{D}}
    [\ell(h;I,Y)], \tag{12}
$$

经验风险为

$$
\widehat R_N(h)
    =
    \frac{1}{N}\sum_{i=1}^{N}\ell(h;I_i,Y_i). \tag{13}
$$

式 <a href="#eq:raw-bpd">(11)</a> 中应当只统计 caption 标签位置。固定 prompt、视觉 placeholder、padding 和其他 label 为 $-100$ 的位置不属于预测风险。 此外，应当先对每个 caption 内部的有效 token 求平均，再对图像样本求平均。 如果直接把整个数据集的所有 token 混合平均，长 caption 会获得更大权重， 所对应的就不再是图像级总体风险。

## Prediction smoothing 与损失有界性

原始 NLL 没有有限上界：当 $p_h(y_t\mid I,y_{<t})\to0$ 时，$-\log_2p_h(y_t\mid I,y_{<t})\to+\infty$，因此不能直接应用 Hoeffding 不等式。

**定义 1** (Prediction-smoothed VLM). 给定 $\alpha\in(0,1)$，定义
<a id="eq:prediction-smoothing"></a>

$$
p_{h,\alpha}(v\mid I,y_{<t})
    =
    (1-\alpha)p_h(v\mid I,y_{<t})
    +\frac{\alpha}{V},
    \qquad v\in\mathcal{V}. \tag{14}
$$

相应的平滑 caption BPD 损失为
<a id="eq:smoothed-loss"></a>

$$
\ell_\alpha(h;I,Y)
    =
    -\frac{1}{T}
    \sum_{t=1}^{T}
    \log_2p_{h,\alpha}(y_t\mid I,y_{<t}). \tag{15}
$$

<a id="lem:bounded-loss"></a>

**引理 1** (平滑 VLM 损失的取值区间). 对任意 $h,I,Y$，有

$$
\ell_\alpha(h;I,Y)\in[a_\alpha,b_\alpha], \tag{16}
$$

其中

$$
\begin{aligned}
    a_\alpha
    &=
    -\log_2\left(1-\alpha+\frac{\alpha}{V}\right),
\end{aligned} \tag{17}
$$

$$
\begin{aligned}
    b_\alpha
    &=
    \log_2\frac{V}{\alpha},
\end{aligned} \tag{18}
$$

区间宽度为
<a id="eq:delta-alpha"></a>

$$
\boxed{
    \Delta_\alpha
    =
    b_\alpha-a_\alpha
    =
    \log_2\left(
    1+\frac{(1-\alpha)V}{\alpha}
    \right).} \tag{19}
$$

*Proof.* 因为 $0\le p_h(y_t\mid I,y_{<t})\le1$，由式 <a href="#eq:prediction-smoothing">(14)</a>，

$$
\frac{\alpha}{V}
    \le
    p_{h,\alpha}(y_t\mid I,y_{<t})
    \le
    1-\alpha+\frac{\alpha}{V}. \tag{20}
$$

函数 $-\log_2x$ 在 $(0,1]$ 上单调递减，所以

$$
-\log_2\left(1-\alpha+\frac{\alpha}{V}\right)
    \le
    -\log_2p_{h,\alpha}(y_t\mid I,y_{<t})
    \le
    \log_2\frac{V}{\alpha}. \tag{21}
$$

式 <a href="#eq:smoothed-loss">(15)</a> 是上述 token 损失的平均，故仍落在相同区间。 最后，

$$
\begin{aligned}
    b_\alpha-a_\alpha
    &=
    \log_2\frac{V}{\alpha}
    +
    \log_2\left(1-\alpha+\frac{\alpha}{V}\right)
\end{aligned} \tag{22}
$$

$$
\begin{aligned}
    &=
    \log_2\left[
    \frac{V}{\alpha}
    \left(1-\alpha+\frac{\alpha}{V}\right)
    \right]
\end{aligned} \tag{23}
$$

$$
\begin{aligned}
    &=
    \log_2\left(
    1+\frac{(1-\alpha)V}{\alpha}
    \right).
\end{aligned} \tag{24}
$$

 ◻

定义平滑总体风险和经验风险：
<a id="eq:population-risk"></a>
<a id="eq:empirical-risk"></a>

$$
\begin{aligned}
    R_\alpha(h)
    &=
    \mathbb{E}_{Z\sim\mathcal{D}}[\ell_\alpha(h;Z)],
\end{aligned} \tag{25}
$$

$$
\begin{aligned}
    \widehat R_{\alpha,N}(h)
    &=
    \frac{1}{N}\sum_{i=1}^{N}\ell_\alpha(h;Z_i).
\end{aligned} \tag{26}
$$

### 任意有限精度平滑参数的 prefix-free 编码

为了允许在观察数据后选择 prediction smoothing 参数，同时不预先指定有限 候选网格，定义所有有限二进制精度参数组成的可数集合
<a id="eq:finite-precision-alpha-set"></a>

$$
\mathcal A_{\mathrm{fin}}
    =
    \left\{
    \frac{m}{2^b}:
    b\in\mathbb N,\quad
    1\le m\le 2^b-1,\quad
    m\text{ 为奇数}
    \right\}. \tag{27}
$$

要求 $m$ 为奇数使每个参数具有唯一表示。例如 $1/2$ 只表示为 $1/2^1$，而不再表示为 $2/2^2$。集合 $\mathcal A_{\mathrm{fin}}$ 不是有限网格；它包含任意精度的有限二进制小数， 并且在 $(0,1)$ 中稠密。

对 $\alpha=m/2^b\in\mathcal A_{\mathrm{fin}}$，令
<a id="eq:alpha-odd-index"></a>

$$
r=\frac{m-1}{2}. \tag{28}
$$

由于 $m$ 为奇数且 $1\le m\le2^b-1$，有

$$
r\in\{0,\ldots,2^{b-1}-1\}. \tag{29}
$$

先使用 Elias gamma code 编码正整数 $b$，再使用恰好 $b-1$ bits 编码 $r$。解码时由 $m=2r+1$ 恢复分子。Elias gamma code 的长度为

$$
L_\Gamma(b)
    =
    2\left\lfloor\log_2 b\right\rfloor+1, \tag{30}
$$

因此 $\alpha$ 的总描述长度为
<a id="eq:alpha-code-length"></a>

$$
\boxed{
    L_\alpha(\alpha)
    =
    b+2\left\lfloor\log_2 b\right\rfloor.} \tag{31}
$$

<a id="lem:alpha-kraft"></a>

**引理 2** (有限精度平滑参数的 Kraft 条件). 上述 $\alpha$ 编码是 prefix-free 的，并满足
<a id="eq:alpha-kraft"></a>

$$
\sum_{\alpha\in\mathcal A_{\mathrm{fin}}}
    2^{-L_\alpha(\alpha)}
    \le 1. \tag{32}
$$

因此
<a id="eq:alpha-prior"></a>

$$
q(\alpha)=2^{-L_\alpha(\alpha)} \tag{33}
$$

可以作为与训练数据无关的次概率先验。*

*Proof.* Elias gamma code 是正整数上的 prefix-free code。解码器先自定界地读出 $b$， 随后读取恰好 $b-1$ 个 bit 得到 $r$，再令 $m=2r+1$，所以两个部分的串联 仍然是 prefix-free 的。当 $b=1$ 时，第二部分长度为零，唯一可能的 $r=0$ 对应 $\alpha=1/2$，因此解码仍然唯一。对每个固定的 $b$，满足式  <a href="#eq:finite-precision-alpha-set">(27)</a> 的奇数 $m$ 恰有 $2^{b-1}$ 个，故

$$
\begin{aligned}
    \sum_{\alpha\in\mathcal A_{\mathrm{fin}}}
    2^{-L_\alpha(\alpha)}
    &=
    \sum_{b=1}^{\infty}
    2^{b-1}
    2^{-(b-1)}
    2^{-L_\Gamma(b)}
\end{aligned} \tag{34}
$$

$$
\begin{aligned}
    &=
    \sum_{b=1}^{\infty}2^{-L_\Gamma(b)}
    \le1,
\end{aligned} \tag{35}
$$

其中最后一步使用了 Elias gamma code 的 Kraft 不等式。与直接使用 $b$ bits 编码奇数 $m$ 相比，该编码恰好节省一 bit，并消除了未使用的一半码空间。 ◻

## 带先验的统一 VLM 泛化界

设 $\mathcal{H}$ 是量化后计算机能够表示的 VLM 假设集合。由于模型最终由有限 bit string 表示，$\mathcal{H}$ 至多是可数集合。给定与训练集独立的先验质量函数

$$
\pi:\mathcal{H}\to(0,1],
    \qquad
    \sum_{h\in\mathcal{H}}\pi(h)\le1. \tag{36}
$$

<a id="thm:prior-bound"></a>

**定理 1** (模型与有限精度平滑参数的联合统一界). 对任意 $\delta\in(0,1)$，以至少 $1-\delta$ 的概率，对所有 $h\in\mathcal{H}$ 和所有 $\alpha\in\mathcal A_{\mathrm{fin}}$ 同时成立
<a id="eq:prior-bound"></a>

$$
\boxed{
    R_\alpha(h)
    \le
    \widehat R_{\alpha,N}(h)
    +
    \Delta_\alpha
    \sqrt{
    \frac{
    \ln\frac{1}{\pi(h)}
    +
    L_\alpha(\alpha)\ln2
    +
    \ln\frac{1}{\delta}
    }{2N}
    }.} \tag{37}
$$

因此，即使最终模型 $h=h(S_N)$ 和平滑参数 $\alpha=\alpha(S_N)$ 依赖完整训练集，也可以在观察数据后代入该界。*

*Proof.* 先固定一个与数据无关的二元组 $(h,\alpha)\in\mathcal{H}\times\mathcal A_{\mathrm{fin}}$。令

$$
X_i=\ell_\alpha(h;Z_i). \tag{38}
$$

根据图像级 IID 假设，$X_1,\ldots,X_N$ 相互独立；由 引理 ，

$$
X_i\in[a_\alpha,b_\alpha],
    \qquad
    b_\alpha-a_\alpha=\Delta_\alpha. \tag{39}
$$

同时

$$
\mathbb{E}[X_i]=R_\alpha(h),
    \qquad
    \frac{1}{N}\sum_{i=1}^{N}X_i
    =
    \widehat R_{\alpha,N}(h). \tag{40}
$$

由单侧 Hoeffding 不等式，对任意 $t>0$，
<a id="eq:hoeffding-fixed-h-alpha"></a>

$$
\mathbb{P}\left(
    R_\alpha(h)-\widehat R_{\alpha,N}(h)\ge t
    \right)
    \le
    \exp\left(
    -\frac{2Nt^2}{\Delta_\alpha^2}
    \right). \tag{41}
$$

定义联合次概率质量
<a id="eq:joint-model-alpha-prior"></a>

$$
\widetilde\pi(h,\alpha)
    =
    \pi(h)q(\alpha)
    =
    \pi(h)2^{-L_\alpha(\alpha)}. \tag{42}
$$

由 $\sum_h\pi(h)\le1$ 和引理 ，
<a id="eq:joint-prior-kraft"></a>

$$
\sum_{h\in\mathcal{H}}
    \sum_{\alpha\in\mathcal A_{\mathrm{fin}}}
    \widetilde\pi(h,\alpha)
    \le1. \tag{43}
$$

对不同二元组选取不同的偏差阈值 $t_{h,\alpha}$，使

$$
\exp\left(
    -\frac{2Nt_{h,\alpha}^2}{\Delta_\alpha^2}
    \right)
    =
    \widetilde\pi(h,\alpha)\delta. \tag{44}
$$

解得

$$
t_{h,\alpha}
    =
    \Delta_\alpha
    \sqrt{
    \frac{
    \ln(1/\pi(h))
    +L_\alpha(\alpha)\ln2
    +\ln(1/\delta)
    }{2N}
    }. \tag{45}
$$

因此，由式 <a href="#eq:hoeffding-fixed-h-alpha">(41)</a>，

$$
\mathbb{P}\left(
    R_\alpha(h)>
    \widehat R_{\alpha,N}(h)+t_{h,\alpha}
    \right)
    \le
    \widetilde\pi(h,\alpha)\delta. \tag{46}
$$

定义坏事件

$$
A_{h,\alpha}=
    \left\{
    R_\alpha(h)>
    \widehat R_{\alpha,N}(h)+t_{h,\alpha}
    \right\}. \tag{47}
$$

对可数集合 $\mathcal{H}\times\mathcal A_{\mathrm{fin}}$ 使用 union bound：

$$
\begin{aligned}
    \mathbb{P}\left(
    \bigcup_{h\in\mathcal{H}}
    \bigcup_{\alpha\in\mathcal A_{\mathrm{fin}}}
    A_{h,\alpha}
    \right)
    &\le
    \sum_{h\in\mathcal{H}}
    \sum_{\alpha\in\mathcal A_{\mathrm{fin}}}
    \mathbb{P}(A_{h,\alpha})
\end{aligned} \tag{48}
$$

$$
\begin{aligned}
    &\le
    \delta
    \sum_{h\in\mathcal{H}}
    \sum_{\alpha\in\mathcal A_{\mathrm{fin}}}
    \widetilde\pi(h,\alpha)
\end{aligned} \tag{49}
$$

$$
\begin{aligned}
    &\le
    \delta.
\end{aligned} \tag{50}
$$

因此，以至少 $1-\delta$ 的概率，对所有 $(h,\alpha)$ 同时有 $R_\alpha(h)\le\widehat R_{\alpha,N}(h)+t_{h,\alpha}$，即式 <a href="#eq:prior-bound">(37)</a>。

最后，虽然学习算法输出的 $h(S_N)$ 以及证书计算选出的 $\alpha(S_N)$ 都可以依赖训练集，但上述成功事件已经对 *所有* $(h,\alpha)$ 同时成立，所以可以合法地代入数据依赖的二元组。 这里的 simultaneous guarantee 比“对每个预先固定的 $\alpha$ 分别以 $1-\delta$ 概率成立”更强；后者不足以支持观察数据后的参数选择。 ◻

## 从 VLM 压缩长度构造先验

### 固定 side information

令 $A$ 表示在抽取当前训练集前已固定的 side information，例如：

- tokenizer、图像预处理算法和 VLM 架构；
- 冻结的视觉编码器与冻结的语言模型；
- 随机子空间投影的生成算法与固定随机 seed；
- 量化、算术编码和模型重构算法。

<a id="assump:prior-independence"></a>

**假设 2** (先验独立性). $A$ 必须与当前训练样本 $S_N$ 独立。若冻结底座在当前图像或 caption 上训练过， 或根据当前训练数据选择了 checkpoint，则不能无条件把该底座免费放入 $A$。*

<a id="lem:frozen-side-information"></a>

**引理 3** (冻结底座作为合法的条件 side information). 令

$$
A=
    \bigl(
    E_{\psi_0},G_{\theta_0},
    \text{tokenizer},\text{图像预处理},\text{重构算法}
    \bigr) \tag{51}
$$

为确定性对象，或由独立于当前预训练集的外部数据和随机性生成的随机对象。 若对每个固定的 $A=a$，模型描述都是 prefix-free 的，则以 $A$ 为条件建立的 压缩泛化界也以同样的置信度无条件成立。因此，冻结的视觉编码器和语言底座 可以不计入当前任务的模型码长；但当前数据上学习的 Projector、LoRA 更新以及 根据当前数据选择的底座 checkpoint 必须编码。*

*Proof.* 由 $A\mathrel{\perp\!\!\!\perp}S_N$，对几乎处处的 $a$ 都有

$$
\mathcal L(S_N\mid A=a)=\mathcal{D}^N. \tag{52}
$$

固定 $A=a$ 后，Hoeffding 不等式、union bound 和 Kraft 不等式的证明完全 适用，故相应成功事件 $\mathcal E_a$ 满足

$$
\mathbb{P}(\mathcal E_a\mid A=a)\ge 1-\delta. \tag{53}
$$

对 $A$ 积分并使用全概率公式，

$$
\mathbb{P}(\mathcal E_A)
    =
    \mathbb{E}_A\!\left[
    \mathbb{P}(\mathcal E_A\mid A)
    \right]
    \ge 1-\delta. \tag{54}
$$

若 $A$ 依赖 $S_N$，第一步的条件分布等式通常不再成立，因而上述论证失效。 ◻

### Prefix-free 模型编码

设量化 VLM $h$ 在给定 $A$ 时具有长度为 $K(h\mid A)$ 的 prefix-free 描述。由 Kraft 不等式，

$$
\sum_{h\in\mathcal{H}}2^{-K(h\mid A)}\le1. \tag{55}
$$

因此可以定义

$$
\pi(h\mid A)=2^{-K(h\mid A)}. \tag{56}
$$

于是

$$
\ln\frac{1}{\pi(h\mid A)}
    =
    K(h\mid A)\ln2. \tag{57}
$$

<a id="cor:compression-bound"></a>

**推论 1** (可编码平滑参数的 VLM 压缩泛化界). 在假设  下，以至少 $1-\delta$ 的概率， 对所有 $h$ 和 $\alpha\in\mathcal A_{\mathrm{fin}}$ 同时有
<a id="eq:compression-bound"></a>

$$
\boxed{
    R_\alpha(h)
    \le
    \widehat R_{\alpha,N}(h)
    +
    \Delta_\alpha
    \sqrt{
    \frac{
    [K(h\mid A)+L_\alpha(\alpha)]\ln2
    +\ln(1/\delta)
    }{2N}
    }.} \tag{58}
$$

*Proof.* 在定理  中代入 $\pi(h\mid A)=2^{-K(h\mid A)}$ 即可。等价地，模型码与 $\alpha$ 码的串联给出联合 prefix-free 长度
<a id="eq:joint-code-length"></a>

$$
K_{\mathrm{joint}}(h,\alpha\mid A)
    =K(h\mid A)+L_\alpha(\alpha). \tag{59}
$$

 ◻

## V-SubLoRA 参数化与实际编码长度

### 模块化低维参数化

对视觉编码器、Projector 和语言模型，可以分别使用
<a id="eq:v-sublora"></a>

$$
\begin{aligned}
    \psi
    &=
    \psi_0+\operatorname{LoRA}_V(P_Vu_V),
\end{aligned} \tag{60}
$$

$$
\begin{aligned}
    \omega
    &=
    \omega_0+P_Pu_P,
\end{aligned} \tag{61}
$$

$$
\begin{aligned}
    \theta
    &=
    \theta_0+\operatorname{LoRA}_L(P_Lu_L).
\end{aligned} \tag{62}
$$

其中：

- $\psi_0,\omega_0,\theta_0$ 是预先固定的初始化或预训练权重；
- $P_V,P_P,P_L$ 由固定 seed 生成，不依赖当前训练集；
- $u_V,u_P,u_L$ 是真正需要学习和编码的 intrinsic vectors。

冻结某一模块时，对应的 $u$ 不存在，码长贡献为零。

### 单个参数块的量化编码

考虑任一可训练参数块

$$
u_b\in\mathbb{R}^{d_b},
    \qquad b\in\{V,P,L\}. \tag{63}
$$

设使用 $Q_b$ 个量化中心

$$
\mathcal C_b=\{c_{b1},\ldots,c_{bQ_b}\}. \tag{64}
$$

定义量化 symbol

$$
s_{bj}
    =
    \arg\min_{q\in[Q_b]}|u_{bj}-c_{bq}|, \tag{65}
$$

量化参数为

$$
\widehat u_{bj}=c_{b,s_{bj}}. \tag{66}
$$

第 $q$ 个 symbol 的 count 和经验概率分别为

$$
n_{bq}
    =
    \sum_{j=1}^{d_b}\mathbf{1}\{s_{bj}=q\},
    \qquad
    \widehat p_{bq}
    =
    \frac{n_{bq}}{d_b}. \tag{67}
$$

经验 symbol entropy 为

$$
H_2(\widehat{\bm p}_b)
    =
    -\sum_{q=1}^{Q_b}
    \widehat p_{bq}\log_2\widehat p_{bq}. \tag{68}
$$

给定 count table，算术编码 symbol sequence 可使用不超过

$$
C_{\mathrm{symbol},b}
    \le
    \left\lceil
    d_bH_2(\widehat{\bm p}_b)
    \right\rceil+1 \tag{69}
$$

bits。若 codebook center 用 FP16 表示，则

$$
C_{\mathrm{codebook},b}=16Q_b. \tag{70}
$$

由于 $0\le n_{bq}\le d_b$，保守地逐个编码 count 最多需要

$$
C_{\mathrm{count},b}
    =
    Q_b\left\lceil\log_2(d_b+1)\right\rceil \tag{71}
$$

bits。因此参数块 $b$ 的原始描述长度满足
<a id="eq:block-code-length"></a>

$$
\boxed{
    C_b
    \le
    \left\lceil
    d_bH_2(\widehat{\bm p}_b)
    \right\rceil
    +1
    +16Q_b
    +Q_b\left\lceil\log_2(d_b+1)\right\rceil.} \tag{72}
$$

全部模块的原始描述长度为
<a id="eq:raw-code-length"></a>

$$
C_{\mathrm{raw}}
    =
    C_V+C_P+C_L+C_{\mathrm{hp}}, \tag{73}
$$

其中 $C_{\mathrm{hp}}$ 是除 $\alpha$ 之外的超参数选择成本；$\alpha$ 将按 式 <a href="#eq:alpha-code-length">(31)</a> 单独编码，以免重复计费。通过 self-delimiting integer code 编码消息长度，存在与模型无关的常数 $c_0$，使 prefix-free 长度满足
<a id="eq:prefix-code-length"></a>

$$
\boxed{
    K(h\mid A)
    \le
    C_{\mathrm{raw}}
    +2\log_2(C_{\mathrm{raw}}+1)
    +c_0.} \tag{74}
$$

将式 <a href="#eq:prefix-code-length">(74)</a> 代入式 <a href="#eq:compression-bound">(58)</a>， 得到完全可计算的 VLM 界：
<a id="eq:computable-full-bound"></a>

$$
\boxed{
    R_\alpha(h)
    \le
    \widehat R_{\alpha,N}(h)
    +
    \Delta_\alpha
    \sqrt{
    \frac{
    \left[
    C_{\mathrm{raw}}
    +2\log_2(C_{\mathrm{raw}}+1)+c_0
    +L_\alpha(\alpha)
    \right]\ln2
    +\ln(1/\delta)
    }{2N}
    }.} \tag{75}
$$

**注 1** (证书约束的对象). 式 <a href="#eq:computable-full-bound">(75)</a> 约束的是量化后模型 $\widehat h$ 的风险， 而不是量化前模型的风险。因此必须在量化后模型上重新计算 $\widehat R_{\alpha,N}(\widehat h)$，不能把量化前训练损失代入。*

## 训练风险子采样界

在大规模数据上计算完整训练风险可能非常昂贵。模型训练并量化后，从 $N$ 个 训练图像簇中有放回均匀抽取

$$
J_1,\ldots,J_n
    \overset{\mathrm{i.i.d.}}{\sim}
    \operatorname{Uniform}\{1,\ldots,N\}, \tag{76}
$$

并定义
<a id="eq:subsample-risk"></a>

$$
\widehat R_{\alpha,n}^{\mathrm{sub}}(h)
    =
    \frac{1}{n}
    \sum_{j=1}^{n}
    \ell_\alpha(h;Z_{J_j}). \tag{77}
$$

<a id="thm:subsample-bound"></a>

**定理 2** (允许子采样后选择有限精度 $\alpha$ 的 VLM 压缩界). 令 $\delta_1,\delta_2\in(0,1)$ 且 $\delta_1+\delta_2=\delta$。 假设模型 $h=h(S_N)$ 在生成子采样索引之前已经确定，但允许在观察同一批 子采样损失后选择 $\alpha$。则以至少 $1-\delta$ 的概率，对所有 $\alpha\in\mathcal A_{\mathrm{fin}}$ 同时有
<a id="eq:subsample-compression-bound"></a>

$$
\begin{aligned}
    R_\alpha(h)
    \le\;&
    \widehat R_{\alpha,n}^{\mathrm{sub}}(h)
    \\\\
    &+
    \Delta_\alpha
    \sqrt{
    \frac{
    [K(h\mid A)+L_\alpha(\alpha)]\ln2
    +\ln(1/\delta_1)
    }{2N}
    }
    \\\\
    &+
    \Delta_\alpha
    \sqrt{
    \frac{
    L_\alpha(\alpha)\ln2+\ln(1/\delta_2)
    }{2n}
    }.
\end{aligned} \tag{78}
$$

*Proof.* 由推论 ，以至少 $1-\delta_1$ 的概率，下面的 事件对所有 $h$ 和 $\alpha\in\mathcal A_{\mathrm{fin}}$ 同时成立：
<a id="eq:first-event"></a>

$$
R_\alpha(h)
    \le
    \widehat R_{\alpha,N}(h)
    +
    \Delta_\alpha
    \sqrt{
    \frac{
    [K(h\mid A)+L_\alpha(\alpha)]\ln2
    +\ln(1/\delta_1)
    }{2N}
    }. \tag{79}
$$

现在条件于完整训练集 $S_N$ 以及在生成子采样索引前已经确定的模型 $h$。 即使 $h=h(S_N)$ 依赖训练数据，一旦条件对象固定，$h$ 和对每个 $\alpha\in\mathcal A_{\mathrm{fin}}$ 的有限总体

$$
\ell_\alpha(h;Z_1),\ldots,\ell_\alpha(h;Z_N) \tag{80}
$$

都是固定值。由于 $J_j$ 独立均匀采样，

$$
\begin{aligned}
    \mathbb{E}\left[
    \ell_\alpha(h;Z_{J_j})\mid S_N,h
    \right]
    &=
    \frac{1}{N}\sum_{i=1}^{N}\ell_\alpha(h;Z_i)
\end{aligned} \tag{81}
$$

$$
\begin{aligned}
    &=
    \widehat R_{\alpha,N}(h).
\end{aligned} \tag{82}
$$

同时 $\ell_\alpha(h;Z_{J_j})\in[a_\alpha,b_\alpha]$。先固定一个 $\alpha$。条件 Hoeffding 不等式给出，对任意 $\eta>0$，
<a id="eq:subsample-fixed-alpha-hoeffding"></a>

$$
\mathbb{P}\left\{
    \widehat R_{\alpha,N}(h)
    -\widehat R_{\alpha,n}^{\mathrm{sub}}(h)
    >\eta
    \,\middle|\,S_N,h
    \right\}
    \le
    \exp\left(-\frac{2n\eta^2}{\Delta_\alpha^2}\right). \tag{83}
$$

为不同的 $\alpha$ 取
<a id="eq:subsample-alpha-threshold"></a>

$$
\eta_\alpha
    =
    \Delta_\alpha
    \sqrt{
    \frac{
    L_\alpha(\alpha)\ln2+\ln(1/\delta_2)
    }{2n}
    }. \tag{84}
$$

代入式 <a href="#eq:subsample-fixed-alpha-hoeffding">(83)</a> 得到相应失败概率至多为

$$
\delta_2 2^{-L_\alpha(\alpha)}. \tag{85}
$$

对可数集合 $\mathcal A_{\mathrm{fin}}$ 使用 union bound，并使用式  <a href="#eq:alpha-kraft">(32)</a>，条件失败概率不超过

$$
\delta_2
    \sum_{\alpha\in\mathcal A_{\mathrm{fin}}}
    2^{-L_\alpha(\alpha)}
    \le\delta_2. \tag{86}
$$

因此，以至少 $1-\delta_2$ 的条件概率，下式对所有 $\alpha$ 同时成立：
<a id="eq:second-event"></a>

$$
\widehat R_{\alpha,N}(h)
    \le
    \widehat R_{\alpha,n}^{\mathrm{sub}}(h)
    +
    \Delta_\alpha
    \sqrt{
    \frac{
    L_\alpha(\alpha)\ln2+\ln(1/\delta_2)
    }{2n}
    }. \tag{87}
$$

该条件概率结论对每个 $(S_N,h)$ 都成立，因此由全概率公式，无条件失败概率 也不超过 $\delta_2$。特别地，因为成功事件对所有 $\alpha$ 同时成立， 可以在观察 $J_1,\ldots,J_n$ 以及所有子采样损失之后选择使最终证书最小的 $\alpha$。 将式 <a href="#eq:second-event">(87)</a> 代入式 <a href="#eq:first-event">(79)</a>，并对两个失败 事件使用 union bound：

$$
\mathbb{P}(E_1^c\cup E_2^c)
    \le\delta_1+\delta_2=\delta, \tag{88}
$$

即可得到式 <a href="#eq:subsample-compression-bound">(78)</a>。 ◻

可沿用原 LLM 工作中的分配

$$
s=\frac{n}{N+n},
    \qquad
    \delta_1=s\delta,
    \qquad
    \delta_2=(1-s)\delta, \tag{89}
$$

得到
<a id="eq:paper-style-subsample-bound"></a>

$$
\begin{aligned}
    R_\alpha(h)
    \le\;&
    \widehat R_{\alpha,n}^{\mathrm{sub}}(h)
    \\\\
    &+
    \Delta_\alpha
    \sqrt{
    \frac{
    [K(h\mid A)+L_\alpha(\alpha)]\ln2
    +\ln\frac{1}{s\delta}
    }{2N}
    }
    \\\\
    &+
    \Delta_\alpha
    \sqrt{
    \frac{
    L_\alpha(\alpha)\ln2
    +\ln\frac{1}{(1-s)\delta}
    }{2n}
    }.
\end{aligned} \tag{90}
$$

任意满足 $\delta_1+\delta_2=\delta$ 的分配都有效，也可以数值优化两者， 使右侧最小。

**注 2** (先选 $\alpha$、后抽认证子样本). 若使用一批独立 pilot sample 选择 $\alpha$，并在 $\alpha$ 确定后才生成用于 式 <a href="#eq:subsample-risk">(77)</a> 的认证索引，则第二层条件 Hoeffding 无需再对 $\alpha$ 做 union bound。此时式 <a href="#eq:subsample-compression-bound">(78)</a> 的 最后一个根号中可以删除 $L_\alpha(\alpha)\ln2$。本文采用更保守也更灵活的 版本：允许直接使用同一批认证子样本选择 $\alpha$。*

## 共享图像的多个 caption

### 按图像身份分簇，而不是按数据行计数

令原始表共有 $m$ 行。使用在查看训练损失之前固定的去重规则 $g:\mathcal X\to\mathcal U$，例如数据源中的 image ID、内容哈希，或预先固定 阈值的感知哈希。定义
<a id="eq:image-equivalence"></a>

$$
r\sim r'
    \quad\Longleftrightarrow\quad
    g(I_r)=g(I_{r'}). \tag{91}
$$

将所有行划分为等价类 $\mathcal G_1,\ldots,\mathcal G_{N_{\mathrm{eff}}}$。 第 $i$ 个独立统计样本不是一行，而是完整图像簇
<a id="eq:image-cluster"></a>

$$
Z_i=
    \left(
    \widetilde I_i,
    \{Y_r:r\in\mathcal G_i\}
    \right),
    \qquad
    i=1,\ldots,N_{\mathrm{eff}}, \tag{92}
$$

其中 $\widetilde I_i$ 代表该等价类中的同一图像内容。于是
<a id="eq:effective-image-sample-size"></a>

$$
\boxed{
    N_{\mathrm{eff}}
    =
    \#\{\text{去重后、相互独立的图像簇}\}
    \le m.} \tag{93}
$$

只有在
<a id="eq:iid-image-clusters"></a>

$$
Z_1,\ldots,Z_{N_{\mathrm{eff}}}
    \overset{\mathrm{i.i.d.}}{\sim}\mathcal{D}_{\mathrm{cluster}} \tag{94}
$$

成立时，后续界中的样本量才可取 $N_{\mathrm{eff}}$。来自同一视频、同一连拍、 同一患者或同一文档的图像即使像素不同也可能相关；此时应进一步合并到同一 上层簇，或另行采用依赖样本的集中不等式，不能仅凭图像文件数宣称独立。

### 簇内多个 caption 的有界损失

若第 $i$ 个图像簇有 $k_i=|\mathcal G_i|$ 个 caption，记为

$$
Y_{i1},\ldots,Y_{ik_i}, \tag{95}
$$

定义第 $j$ 个 caption 的损失

$$
\ell_\alpha(h;I_i,Y_{ij})
    =
    -\frac{1}{T_{ij}}
    \sum_{t=1}^{T_{ij}}
    \log_2
    p_{h,\alpha}
    (y_{ijt}\mid I_i,y_{ij,<t}), \tag{96}
$$

图像簇损失为
<a id="eq:cluster-loss"></a>

$$
\ell_\alpha(h;Z_i)
    =
    \frac{1}{k_i}
    \sum_{j=1}^{k_i}
    \ell_\alpha(h;I_i,Y_{ij}). \tag{97}
$$

由于每个 caption 损失都属于 $[a_\alpha,b_\alpha]$，其平均仍属于同一 区间。因此簇内 caption 可以任意相关，Hoeffding 不等式只要求不同 图像簇之间满足式 <a href="#eq:iid-image-clusters">(94)</a>。前面所有定理和证明保持 不变，但其中的 $N$ 必须解释为

$$
\boxed{N=N_{\mathrm{eff}},} \tag{98}
$$

而不是 parquet 行数、caption 总数或视觉 token 总数。

## Top-k token error 泛化界

定义图像条件 Top-$k$ token error：

$$
\ell_k(h;I,Y)
    =
    \frac{1}{T}
    \sum_{t=1}^{T}
    \mathbf{1}\left\{
    y_t\notin
    \operatorname{TopK}\left(
    p_h(\cdot\mid I,y_{<t})
    \right)
    \right\}. \tag{99}
$$

显然

$$
0\le\ell_k(h;I,Y)\le1. \tag{100}
$$

Top-$k$ 损失本身不依赖 $\alpha$，因此无需编码平滑参数。仅对模型码重复 定理  中的 Hoeffding 与加权 union-bound 论证，并令 损失区间宽度 $\Delta=1$，得到
<a id="eq:topk-full-bound"></a>

$$
\boxed{
    R_k(h)
    \le
    \widehat R_{k,N}(h)
    +
    \sqrt{
    \frac{
    K(h\mid A)\ln2+\ln(1/\delta)
    }{2N}
    }.} \tag{101}
$$

子采样版本为
<a id="eq:topk-subsample-bound"></a>

$$
\begin{aligned}
    R_k(h)
    \le\;&
    \widehat R_{k,n}^{\mathrm{sub}}(h)
    +
    \sqrt{
    \frac{
    K(h\mid A)\ln2+\ln(1/\delta_1)
    }{2N}
    }
    +
    \sqrt{
    \frac{\ln(1/\delta_2)}{2n}
    }.
\end{aligned} \tag{102}
$$

Prediction smoothing 不改变 token 概率排序，因为对任意 $a,b\in\mathcal{V}$，

$$
p_{h,\alpha}(a)-p_{h,\alpha}(b)
    =
    (1-\alpha)\left[p_h(a)-p_h(b)\right]. \tag{103}
$$

因此除原模型概率恰好并列外，平滑模型和原模型具有相同的 Top-$k$ 集合。

## 超参数选择成本

实际计算会搜索 $\alpha$、intrinsic dimension、LoRA rank、量化级别、 视觉 token 数和模型规模。若这些选择是在观察训练结果后完成的，就必须将 选择编码进假设。本文不再要求 $\alpha$ 来自预先给定的有限网格，而是使用 式 <a href="#eq:alpha-code-length">(31)</a> 对最终选中的有限精度值单独编码。

对于其余离散超参数，仍可设预先给定的有限候选集合

$$
d\in\mathcal D,\qquad
    r\in\mathcal R,\qquad
    Q\in\mathcal Q. \tag{104}
$$

使用均匀先验时，可取
<a id="eq:hyperparameter-bits"></a>

$$
\begin{aligned}
    C_{\mathrm{hp}}
    =\;&
    \left\lceil\log_2|\mathcal D|\right\rceil
    +
    \left\lceil\log_2|\mathcal R|\right\rceil
    +
    \left\lceil\log_2|\mathcal Q|\right\rceil
    \\\\
    &+
    C_{\mathrm{architecture}}
    +
    C_{\mathrm{module\ choice}}.
\end{aligned} \tag{105}
$$

模型总联合码长为

$$
K_{\mathrm{joint}}(h,\alpha\mid A)
    =K(h\mid A)+L_\alpha(\alpha), \tag{106}
$$

其中 $C_{\mathrm{hp}}$ 已包含在 $K(h\mid A)$ 中，而 $L_\alpha(\alpha)$ 不再包含于 $C_{\mathrm{hp}}$。

实际计算时可以只枚举一个有限搜索子集，例如所有 $b\le b_{\max}$ 的候选值， 但统计定理对整个可数无限集合 $\mathcal A_{\mathrm{fin}}$ 同时成立。因此搜索 子集的大小不再产生额外的 $\log_2|\mathcal A|$ 成本，只需支付最终选中值的 $L_\alpha(\alpha)$。提高精度 $b$ 可以更细致地优化 certificate，同时会增加 描述长度，二者的折中由界本身自动体现。

## 非空性的显式条件

均匀随机预测器满足

$$
p_{\mathrm{unif}}(y_t)=\frac{1}{V}, \tag{107}
$$

其 BPD 为

$$
R_{\mathrm{unif}}=\log_2V. \tag{108}
$$

prediction smoothing 后均匀预测器仍为均匀分布。因此当
<a id="eq:non-vacuous-condition"></a>

$$
\boxed{
    \operatorname{Bound}_{\mathrm{VLM}}<\log_2V
    } \tag{109}
$$

时，BPD bound 是 non-vacuous 的。

对定理 ，定义子采样误差项

$$
\varepsilon_{\mathrm{sub}}
    =
    \Delta_\alpha
    \sqrt{
    \frac{
    L_\alpha(\alpha)\ln2+\ln(1/\delta_2)
    }{2n}
    }, \tag{110}
$$

以及相对于均匀预测的剩余空间

$$
G
    =
    \log_2V
    -
    \widehat R_{\alpha,n}^{\mathrm{sub}}(h)
    -
    \varepsilon_{\mathrm{sub}}. \tag{111}
$$

首先必须有 $G>0$。若 $G>0$，非空性要求

$$
\Delta_\alpha
    \sqrt{
    \frac{
    [K(h\mid A)+L_\alpha(\alpha)]\ln2
    +\ln(1/\delta_1)
    }{2N}
    }
    <G. \tag{112}
$$

平方并整理可得允许的最大模型码长：
<a id="eq:max-code-length"></a>

$$
\boxed{
    K(h\mid A)
    <
    \frac{
    2NG^2/\Delta_\alpha^2
    -
    \ln(1/\delta_1)
    }{\ln2}
    -L_\alpha(\alpha).} \tag{113}
$$

式 <a href="#eq:max-code-length">(113)</a> 可以直接用于选择 Projector intrinsic dimension：$d_P$ 太小会增加经验风险，$d_P$ 太大则增加 $K(h\mid A)$， 最佳 certificate 通常出现在两者的折中点。

## 不同训练范围下的码长

<div style="display: flex; justify-content: center; width: 100%; overflow-x: auto;">
<table align="center" style="display: table; width: auto; max-width: 100%; margin: 1em auto; border-collapse: collapse; text-align: center;">
  <thead>
    <tr>
      <th style="padding: 0.35em 0.8em; text-align: center;">训练范围</th>
      <th style="padding: 0.35em 0.8em; text-align: center;">需要编码的参数</th>
      <th style="padding: 0.35em 0.8em; text-align: center;">码长</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 0.35em 0.8em;">仅训练 Projector</td>
      <td style="padding: 0.35em 0.8em;"><var>u<sub>P</sub></var></td>
      <td style="padding: 0.35em 0.8em;"><var>K<sub>P</sub> + C<sub>hp</sub></var></td>
    </tr>
    <tr>
      <td style="padding: 0.35em 0.8em;">Projector + LLM SubLoRA</td>
      <td style="padding: 0.35em 0.8em;"><var>u<sub>P</sub>, u<sub>L</sub></var></td>
      <td style="padding: 0.35em 0.8em;"><var>K<sub>P</sub> + K<sub>L</sub> + C<sub>hp</sub></var></td>
    </tr>
    <tr>
      <td style="padding: 0.35em 0.8em;">视觉编码器 + Projector + LLM</td>
      <td style="padding: 0.35em 0.8em;"><var>u<sub>V</sub>, u<sub>P</sub>, u<sub>L</sub></var></td>
      <td style="padding: 0.35em 0.8em;"><var>K<sub>V</sub> + K<sub>P</sub> + K<sub>L</sub> + C<sub>hp</sub></var></td>
    </tr>
  </tbody>
</table>
</div>

第一阶段最推荐冻结视觉编码器和语言底座，只训练

$$
\omega=\omega_0+P_Pu_P. \tag{114}
$$

这时模型码长主要由低维 $u_P$ 决定，最有希望在有限图像数下得到非空界。 若直接量化完整 Projector 或训练完整视觉编码器，码长可能比样本量大得多， 从而使式 <a href="#eq:compression-bound">(58)</a> 变为空界。

## 与 MININTP 结构分析的关系

上述压缩界已经是独立、完整且可计算的统计保证，不需要对 Transformer covering number 或 token mixing 作额外假设。MININTP 的设置更适合用来 解释模型结构如何影响经验风险和可压缩性。

可将 VLM 的 representation learner 写为

$$
H_{\mathrm{VLM}}
    =
    H_L\circ
    \operatorname{Concat}
    \left(
    P_\omega\circ E_\psi(I),
    E_T(Y_{<t})
    \right), \tag{115}
$$

token predictor 写为

$$
g_\rho(h_t)=\operatorname{softmax}(W_\rho h_t). \tag{116}
$$

形式上可以研究

$$
\mathfrak R
    \left(
    \ell\circ\mathcal G\circ
    \mathcal H_{\mathrm{VLM}}
    \right)
    \lesssim
    G_\ell G_g
    \mathfrak R(\mathcal H_{\mathrm{VLM}})
    +
    G_\ell
    \mathfrak R(\mathcal G\circ\widehat h), \tag{117}
$$

并进一步按视觉编码器、Projector 和语言 decoder 分解

$$
\mathfrak R(\mathcal H_{\mathrm{VLM}})
    \lesssim
    \mathcal C_L
    +
    L_L\mathcal C_P
    +
    L_LL_P\mathcal C_V. \tag{118}
$$

冻结视觉编码器时，$\mathcal E_V$ 是 singleton class，其学习复杂度可以 消失；但视觉特征范数和视觉 token 数仍会进入 Projector 与 decoder 的 covering-number 常数。

必须区分

$$
m_y=\text{caption 中实际预测的 token 数} \tag{119}
$$

和

$$
s=M+m_{\mathrm{prompt}}+m_y
    =\text{Transformer 总输入长度}. \tag{120}
$$

视觉 token 是同一图像内部的条件表示，不是独立监督样本，因此不能把有效 样本量写成 $NM$。另外，MININTP 对文本 token 使用的平稳 $\phi$-mixing 假设不能直接施加到“连续视觉 token + caption token”的完整 序列上。如需 token-level refined bound，只能对给定图像条件下的 caption 生成过程提出额外 mixing 或 martingale 假设。

Rademacher 界和压缩界应作为两条独立的复杂度分析。不能简单把两者的复杂度 项相加，否则会重复惩罚模型复杂度。本文以压缩界作为最终 non-vacuous certificate，以结构复杂度分析解释 $N$、caption 长度、视觉 token 数、 Projector 宽度和可训练参数量对 certificate 的影响。

## 最终主定理：显式体现 VLM 的数据与模块结构

<a id="thm:final-vlm-bound"></a>

**定理 3** (可编码平滑参数与模块化码长控制的 VLM 预训练界). 考虑结构为
<a id="eq:final-vlm-structure"></a>

$$
p_h(y_t\mid I,y_{<t})
    =
    G_\theta\!\left(
    P_\omega(E_\psi(I)),y_{<t}
    \right)_{y_t} \tag{121}
$$

的图像条件自回归 VLM。假设：

1. *原始图像–文本行按照式 <a href="#eq:image-equivalence">(91)</a> 预先去重并分簇， 得到 $N_{\mathrm{eff}}$ 个满足式 <a href="#eq:iid-image-clusters">(94)</a> 的独立 图像簇；簇内 caption 可以相关，簇损失按式 <a href="#eq:cluster-loss">(97)</a> 取平均；*
2. *prediction smoothing 参数 $\alpha\in\mathcal A_{\mathrm{fin}}$ 按式  <a href="#eq:alpha-code-length">(31)</a> 编码，不要求来自预先给定的有限网格；*
3. *冻结的视觉编码器、冻结的语言底座、图像预处理、随机投影 seed 和解码算法共同组成 $A$，且 $A$ 与当前图像簇训练集独立；*
4. *可训练模块集合为 $\mathcal T\subseteq\{V,P,L\}$，其中 $V,P,L$ 分别表示视觉编码器、Projector 和语言模型； 每个 $b\in\mathcal T$ 按式 <a href="#eq:v-sublora">(62)</a> 进行低维参数化并量化， 冻结模块不产生当前任务的参数码长；*
5. *量化后模型记为 $\widehat h$，其所有数据依赖选择均包含在 prefix-free 描述中；*
6. *模型确定后，从 $N_{\mathrm{eff}}$ 个训练图像簇中有放回均匀抽取 $n$ 个簇，计算 $\widehat R_{\alpha,n}^{\mathrm{sub}}(\widehat h)$，并允许在观察这批子采样损失后选择使证书最小的 $\alpha$。*

对每个可训练模块 $b\in\mathcal T$，令
<a id="eq:final-module-cost"></a>

$$
\overline C_b
    =
    \left\lceil
    d_bH_2(\widehat{\bm p}_b)
    \right\rceil
    +1
    +16Q_b
    +Q_b\left\lceil\log_2(d_b+1)\right\rceil, \tag{122}
$$

并定义
<a id="eq:final-vlm-raw-cost"></a>
<a id="eq:final-vlm-prefix-cost"></a>

$$
\begin{aligned}
    C_{\mathrm{VLM}}
    &=
    C_{\mathrm{hp}}
    +
    \sum_{b\in\mathcal T}\overline C_b,
\end{aligned} \tag{123}
$$

$$
\begin{aligned}
    K_{\mathrm{VLM}}
    &=
    C_{\mathrm{VLM}}
    +2\log_2(C_{\mathrm{VLM}}+1)
    +c_0.
\end{aligned} \tag{124}
$$

这里 $C_{\mathrm{hp}}$ 必须编码任何根据当前数据选择的视觉 token 数 $M$、 Projector 宽度、LoRA rank、量化级别和可训练模块集合，但不再包含 $\alpha$；模型与平滑参数的联合码长上界为
<a id="eq:final-joint-code-cost"></a>

$$
K_{\mathrm{joint}}
    =K_{\mathrm{VLM}}+L_\alpha(\alpha). \tag{125}
$$

则对任意 $\delta_1,\delta_2\in(0,1)$，以至少 $1-\delta_1-\delta_2$ 的联合概率，
<a id="eq:final-vlm-bound"></a>

$$
\begin{aligned}
    R_\alpha(\widehat h)
    \le\;&
    \widehat R_{\alpha,n}^{\mathrm{sub}}(\widehat h)
    \\\\
    &+
    \underbrace{\log_2\left(
    1+\frac{(1-\alpha)V}{\alpha}
    \right)}_{\Delta_\alpha}
    \sqrt{
    \frac{
    [K_{\mathrm{VLM}}+L_\alpha(\alpha)]\ln2
    +\ln(1/\delta_1)
    }{2N_{\mathrm{eff}}}
    }
    \\\\
    &+
    \log_2\left(
    1+\frac{(1-\alpha)V}{\alpha}
    \right)
    \sqrt{
    \frac{
    L_\alpha(\alpha)\ln2+\ln(1/\delta_2)
    }{2n}
    }.
\end{aligned} \tag{126}
$$

式 <a href="#eq:final-vlm-bound">(126)</a> 对所有 $\alpha\in\mathcal A_{\mathrm{fin}}$ 同时成立，因而可以在观察数据后选择其 右端最小的 $\alpha$。若该最小右端严格小于均匀预测基线 $\log_2V$，则它是 有实际意义的 non-vacuous certificate；仅小于值域上界 $b_\alpha=\log_2(V/\alpha)$ 则只表示优于逐点平凡上界。*

*Proof.* **步骤 1：明确两层随机性与独立样本单位。** 记
<a id="eq:final-proof-training-space"></a>

$$
N=N_{\mathrm{eff}},\qquad
    S_N=(Z_1,\ldots,Z_N)
    \sim\mathcal{D}_{\mathrm{cluster}}^N. \tag{127}
$$

训练集的随机性来自 $N$ 个独立图像簇；同一簇内的图像副本和多个 caption 可以任意相关。训练和量化算法在观察 $(A,S_N)$ 后输出

$$
\widehat h
    =
    \mathsf{Alg}(A,S_N). \tag{128}
$$

模型确定后，再独立生成子采样索引
<a id="eq:final-proof-index-space"></a>

$$
J_1,\ldots,J_n
    \overset{\mathrm{i.i.d.}}{\sim}
    \operatorname{Uniform}\{1,\ldots,N\}. \tag{129}
$$

因此结论中的联合概率取在 $(A,S_N,J_1,\ldots,J_n)$ 上。式  <a href="#eq:final-proof-training-space">(127)</a> 的样本量是图像簇数 $N$，不是 caption 行数，也不是视觉 token 数 $NM$。平滑参数可以进一步写成
<a id="eq:final-proof-alpha-selector"></a>

$$
\widehat\alpha
    =
    \mathsf{Select}(A,S_N,\widehat h,J_1,\ldots,J_n). \tag{130}
$$

后续两层事件都将对所有 $\alpha\in\mathcal A_{\mathrm{fin}}$ 同时控制，从而 允许这种数据依赖选择。

**步骤 2：证明 caption 损失和图像簇损失具有同一个有限值域。** 对任意量化 VLM $h$、图像条件和正确 token，prediction smoothing 给出

$$
p_{h,\alpha}(y_t\mid I,y_{<t})
    =
    (1-\alpha)p_h(y_t\mid I,y_{<t})
    +\frac{\alpha}{V}. \tag{131}
$$

由于原始概率属于 $[0,1]$，

$$
\frac{\alpha}{V}
    \le
    p_{h,\alpha}(y_t\mid I,y_{<t})
    \le
    1-\alpha+\frac{\alpha}{V}. \tag{132}
$$

函数 $x\mapsto-\log_2x$ 单调递减，所以每个 token 的 smoothed NLL 满足
<a id="eq:final-proof-token-range"></a>

$$
a_\alpha
    \le
    -\log_2p_{h,\alpha}(y_t\mid I,y_{<t})
    \le
    b_\alpha, \tag{133}
$$

其中

$$
a_\alpha
    =
    -\log_2\left(1-\alpha+\frac{\alpha}{V}\right),
    \qquad
    b_\alpha
    =
    \log_2\frac{V}{\alpha}. \tag{134}
$$

一条 caption 的损失是式 <a href="#eq:final-proof-token-range">(133)</a> 中若干数的平均， 图像簇损失又是簇内若干 caption 损失的平均，故对每个 $Z_i$，
<a id="eq:final-proof-cluster-range"></a>

$$
a_\alpha
    \le
    \ell_\alpha(h;Z_i)
    \le
    b_\alpha. \tag{135}
$$

这里不需要簇内 caption 相互独立。该区间的宽度为
<a id="eq:final-proof-delta"></a>

$$
\begin{aligned}
    \Delta_\alpha
    &=
    b_\alpha-a_\alpha
    \\\\
    &=
    \log_2\frac{V}{\alpha}
    +
    \log_2\left(1-\alpha+\frac{\alpha}{V}\right)
    \\\\
    &=
    \log_2\left(
    \frac{V(1-\alpha)+\alpha}{\alpha}
    \right)
    =
    \log_2\left(
    1+\frac{(1-\alpha)V}{\alpha}
    \right).
\end{aligned} \tag{136}
$$

对任意固定二元组 $(h,\alpha)$，定义图像簇总体风险和完整训练风险
<a id="eq:final-proof-population-risk"></a>
<a id="eq:final-proof-full-risk"></a>

$$
\begin{aligned}
    R_\alpha(h)
    &=
    \mathbb{E}_{Z\sim\mathcal{D}_{\mathrm{cluster}}}
    [\ell_\alpha(h;Z)],
\end{aligned} \tag{137}
$$

$$
\begin{aligned}
    \widehat R_{\alpha,N}(h)
    &=
    \frac{1}{N}\sum_{i=1}^{N}
    \ell_\alpha(h;Z_i).
\end{aligned} \tag{138}
$$

**步骤 3：条件于冻结底座，并构造模型–平滑参数联合先验。** 固定 $A=a$。因为 $A\mathrel{\perp\!\!\!\perp}S_N$，
<a id="eq:final-proof-conditioning"></a>

$$
\mathcal L(S_N\mid A=a)
    =
    \mathcal{D}_{\mathrm{cluster}}^N, \tag{139}
$$

所以冻结视觉编码器和冻结语言底座不会改变步骤 1 的 i.i.d. 假设。令 $\mathcal H_a$ 是给定 $A=a$ 后所有可能被解码的量化 VLM 构成的可数集合。

对每个可训练模块 $b\in\mathcal T$，量化 symbol sequence、算术编码的至多 一 bit 冗余、$Q_b$ 个 FP16 codebook center 和 $Q_b$ 个 count 分别需要

$$
\left\lceil
    d_bH_2(\widehat{\bm p}_b)
    \right\rceil,\qquad
    1,\qquad
    16Q_b,\qquad
    Q_b\left\lceil\log_2(d_b+1)\right\rceil \tag{140}
$$

bits。因此它们的和正是式 <a href="#eq:final-module-cost">(122)</a> 中的 $\overline C_b$。连接所有可训练模块的描述并加入 $C_{\mathrm{hp}}$，得到

$$
C_{\mathrm{VLM}}
    =
    C_{\mathrm{hp}}
    +
    \sum_{b\in\mathcal T}\overline C_b. \tag{141}
$$

再用 self-delimiting integer code 编码消息总长度，可取一个与模型无关的 常数 $c_0$，使实际 prefix-free 码长满足
<a id="eq:final-proof-code-domination"></a>

$$
K(h\mid a)
    \le
    C_{\mathrm{VLM}}(h)
    +2\log_2(C_{\mathrm{VLM}}(h)+1)
    +c_0
    =
    K_{\mathrm{VLM}}(h). \tag{142}
$$

由 prefix-free 性和 Kraft 不等式，
<a id="eq:final-proof-kraft"></a>

$$
\sum_{h\in\mathcal H_a}
    2^{-K(h\mid a)}
    \le1. \tag{143}
$$

对任意 $\alpha=m/2^b\in\mathcal A_{\mathrm{fin}}$，令 $r=(m-1)/2$，将 Elias gamma 编码的 $b$ 与固定 $b-1$ bits 的 $r$ 串联； 解码时令 $m=2r+1$。其码长为

$$
L_\alpha(\alpha)
    =b+2\lfloor\log_2b\rfloor. \tag{144}
$$

由引理 ，
<a id="eq:final-proof-alpha-kraft"></a>

$$
\sum_{\alpha\in\mathcal A_{\mathrm{fin}}}
    2^{-L_\alpha(\alpha)}
    \le1. \tag{145}
$$

模型码和 $\alpha$ 码均为 prefix-free，因此两者的串联也是 prefix-free，且 联合码长为
<a id="eq:final-proof-joint-code"></a>

$$
K(h,\alpha\mid a)
    =K(h\mid a)+L_\alpha(\alpha). \tag{146}
$$

相应的联合 Kraft 和满足
<a id="eq:final-proof-joint-kraft"></a>

$$
\begin{aligned}
    &\sum_{h\in\mathcal H_a}
    \sum_{\alpha\in\mathcal A_{\mathrm{fin}}}
    2^{-K(h\mid a)-L_\alpha(\alpha)}
    \\\\
    &\qquad=
    \left(\sum_{h\in\mathcal H_a}2^{-K(h\mid a)}\right)
    \left(\sum_{\alpha\in\mathcal A_{\mathrm{fin}}}
    2^{-L_\alpha(\alpha)}\right)
    \le1.
\end{aligned} \tag{147}
$$

这里 $M$、Projector 宽度和可训练模块集合等数据依赖选择仍编码进 $C_{\mathrm{hp}}$，而 $\alpha$ 由 $L_\alpha(\alpha)$ 单独计费，不在 $C_{\mathrm{hp}}$ 中重复编码。

**步骤 4：对所有模型与所有有限精度 $\alpha$ 同时建立完整训练集界。** 先固定任意 $(h,\alpha)\in\mathcal H_a\times\mathcal A_{\mathrm{fin}}$。由式  <a href="#eq:final-proof-training-space">(127)</a> 和  <a href="#eq:final-proof-cluster-range">(135)</a>，随机变量 $\ell_\alpha(h;Z_1),\ldots,\ell_\alpha(h;Z_N)$ 独立、同分布并位于宽度为 $\Delta_\alpha$ 的区间内。单侧 Hoeffding 不等式给出：对任意 $\varepsilon>0$，
<a id="eq:final-proof-hoeffding"></a>

$$
\mathbb{P}\left\{
    R_\alpha(h)-\widehat R_{\alpha,N}(h)
    >
    \varepsilon
    \,\middle|\,A=a
    \right\}
    \le
    \exp\left(
    -\frac{2N\varepsilon^2}{\Delta_\alpha^2}
    \right). \tag{148}
$$

为每个 $(h,\alpha)$ 选择
<a id="eq:final-proof-epsilon-h-alpha"></a>

$$
\varepsilon_{h,\alpha}
    =
    \Delta_\alpha
    \sqrt{
    \frac{
    [K(h\mid a)+L_\alpha(\alpha)]\ln2
    +\ln(1/\delta_1)
    }{2N}
    }. \tag{149}
$$

将式 <a href="#eq:final-proof-epsilon-h-alpha">(149)</a> 代入式  <a href="#eq:final-proof-hoeffding">(148)</a>，得到
<a id="eq:final-proof-weighted-failure"></a>

$$
\begin{aligned}
    \mathbb{P}\left\{
    R_\alpha(h)-\widehat R_{\alpha,N}(h)
    >
    \varepsilon_{h,\alpha}
    \,\middle|\,A=a
    \right\}
    &\le
    \exp\left(
    -[K(h\mid a)+L_\alpha(\alpha)]\ln2
    -\ln\frac{1}{\delta_1}
    \right)
    \\\\
    &=
    \delta_1\,
    2^{-K(h\mid a)}2^{-L_\alpha(\alpha)}.
\end{aligned} \tag{150}
$$

对整个可数集合 $\mathcal H_a\times\mathcal A_{\mathrm{fin}}$ 使用 union bound，并使用式  <a href="#eq:final-proof-joint-kraft">(147)</a>，
<a id="eq:final-proof-union"></a>

$$
\begin{aligned}
    &\mathbb{P}\left\{
    \exists(h,\alpha)\in
    \mathcal H_a\times\mathcal A_{\mathrm{fin}}:
    R_\alpha(h)-\widehat R_{\alpha,N}(h)
    >
    \varepsilon_{h,\alpha}
    \,\middle|\,A=a
    \right\}
    \\\\
    &\qquad\le
    \delta_1
    \sum_{h\in\mathcal H_a}
    \sum_{\alpha\in\mathcal A_{\mathrm{fin}}}
    2^{-K(h\mid a)}2^{-L_\alpha(\alpha)}
    \le\delta_1.
\end{aligned} \tag{151}
$$

因此，以条件概率至少 $1-\delta_1$，上述不等式对所有 $(h,\alpha)$ 同时成立。正因为结论同时覆盖整个可数的有限精度参数集合， 才可以在观察训练集后代入数据依赖模型 $\widehat h=\mathsf{Alg}(a,S_N)$，并在之后选择 $\alpha$。再由式 <a href="#eq:final-proof-code-domination">(142)</a> 中 $K(\widehat h\mid a)\le K_{\mathrm{VLM}}(\widehat h)$，得到
<a id="eq:final-proof-first-layer"></a>

$$
R_\alpha(\widehat h)
    \le
    \widehat R_{\alpha,N}(\widehat h)
    +
    \Delta_\alpha
    \sqrt{
    \frac{
    [K_{\mathrm{VLM}}+L_\alpha(\alpha)]\ln2
    +\ln(1/\delta_1)
    }{2N}
    }. \tag{152}
$$

而且式 <a href="#eq:final-proof-first-layer">(152)</a> 对所有 $\alpha\in\mathcal A_{\mathrm{fin}}$ 同时成立。 式 <a href="#eq:final-proof-union">(151)</a> 对几乎处处的 $a$ 都成立。对 $A$ 积分， 由全概率公式可知式 <a href="#eq:final-proof-first-layer">(152)</a> 的无条件失败概率 仍不超过 $\delta_1$。

**步骤 5：用独立子采样估计完整训练风险。** 条件于 $\sigma(A,S_N,\widehat h)$。先固定任意 $\alpha\in\mathcal A_{\mathrm{fin}}$，令

$$
X_s
    =
    \ell_\alpha(\widehat h;Z_{J_s}),
    \qquad s=1,\ldots,n. \tag{153}
$$

由式 <a href="#eq:final-proof-index-space">(129)</a>，$X_1,\ldots,X_n$ 在该条件下 独立同分布；由式 <a href="#eq:final-proof-cluster-range">(135)</a>，它们属于 $[a_\alpha,b_\alpha]$，而且其条件期望恰好是完整训练风险：
<a id="eq:final-proof-subsample-mean"></a>

$$
\begin{aligned}
    \mathbb{E}[X_s\mid A,S_N,\widehat h]
    &=
    \frac{1}{N}\sum_{i=1}^{N}
    \ell_\alpha(\widehat h;Z_i)
    \\\\
    &=
    \widehat R_{\alpha,N}(\widehat h).
\end{aligned} \tag{154}
$$

同时

$$
\frac{1}{n}\sum_{s=1}^{n}X_s
    =
    \widehat R_{\alpha,n}^{\mathrm{sub}}(\widehat h). \tag{155}
$$

对这些条件随机变量应用单侧 Hoeffding 不等式，对任意 $\eta>0$ 有

$$
\mathbb{P}\left\{
    \widehat R_{\alpha,N}(\widehat h)
    -
    \widehat R_{\alpha,n}^{\mathrm{sub}}(\widehat h)
    >
    \eta
    \,\middle|\,
    A,S_N,\widehat h
    \right\}
    \le
    \exp\left(
    -\frac{2n\eta^2}{\Delta_\alpha^2}
    \right). \tag{156}
$$

为每个 $\alpha$ 取
<a id="eq:final-proof-subsample-alpha-threshold"></a>

$$
\eta_\alpha
    =
    \Delta_\alpha
    \sqrt{
    \frac{
    L_\alpha(\alpha)\ln2+\ln(1/\delta_2)
    }{2n}
    }, \tag{157}
$$

则固定 $\alpha$ 的条件失败概率至多为

$$
\delta_2 2^{-L_\alpha(\alpha)}. \tag{158}
$$

对所有 $\alpha\in\mathcal A_{\mathrm{fin}}$ 使用 union bound，并应用式  <a href="#eq:final-proof-alpha-kraft">(145)</a>，可得
<a id="eq:final-proof-subsample-alpha-union"></a>

$$
\begin{aligned}
    &\mathbb{P}\left\{
    \exists\alpha\in\mathcal A_{\mathrm{fin}}:
    \widehat R_{\alpha,N}(\widehat h)
    -\widehat R_{\alpha,n}^{\mathrm{sub}}(\widehat h)
    >\eta_\alpha
    \,\middle|\,A,S_N,\widehat h
    \right\}
    \\\\
    &\qquad\le
    \delta_2
    \sum_{\alpha\in\mathcal A_{\mathrm{fin}}}
    2^{-L_\alpha(\alpha)}
    \le\delta_2.
\end{aligned} \tag{159}
$$

再次用全概率公式消去条件，得到无条件失败概率至多为 $\delta_2$、且对所有 $\alpha$ 同时成立的事件
<a id="eq:final-proof-second-layer"></a>

$$
\widehat R_{\alpha,N}(\widehat h)
    \le
    \widehat R_{\alpha,n}^{\mathrm{sub}}(\widehat h)
    +
    \Delta_\alpha
    \sqrt{
    \frac{
    L_\alpha(\alpha)\ln2+\ln(1/\delta_2)
    }{2n}
    }. \tag{160}
$$

这里必须先确定 $\widehat h$，再独立生成 $J_1,\ldots,J_n$；若反过来利用 这批子样本选择模型，就还必须在第二层对所有候选模型同时控制。与模型不同， $\alpha$ 可以在观察这批子样本后选择，因为式  <a href="#eq:final-proof-subsample-alpha-union">(159)</a> 已经对所有可编码 $\alpha$ 同时成立，其选择成本正是第二个偏差项中的 $L_\alpha(\alpha)\ln2$。

**步骤 6：合并两个高概率事件。** 令 $\mathcal E_1$ 和 $\mathcal E_2$ 分别表示式  <a href="#eq:final-proof-first-layer">(152)</a> 与  <a href="#eq:final-proof-second-layer">(160)</a> 对所有可编码 $\alpha$ 同时成立。前述推导给出

$$
\mathbb{P}(\mathcal E_1^c)\le\delta_1,
    \qquad
    \mathbb{P}(\mathcal E_2^c)\le\delta_2. \tag{161}
$$

两个事件不必相互独立。由 union bound，

$$
\mathbb{P}(\mathcal E_1\cap\mathcal E_2)
    \ge
    1-\delta_1-\delta_2. \tag{162}
$$

在交事件上，将式 <a href="#eq:final-proof-second-layer">(160)</a> 代入式  <a href="#eq:final-proof-first-layer">(152)</a>，再使用 $N=N_{\mathrm{eff}}$ 和式  <a href="#eq:final-proof-delta">(136)</a>，正好得到式 <a href="#eq:final-vlm-bound">(126)</a>，而且该式 对所有 $\alpha\in\mathcal A_{\mathrm{fin}}$ 同时成立。因此可在交事件上定义
<a id="eq:final-alpha-selection"></a>

$$
\widehat\alpha
    \in
    \arg\min_{\alpha\in\mathcal A_{\mathrm{search}}}
    \operatorname{Bound}_{\mathrm{VLM}}(\alpha), \tag{163}
$$

其中 $\mathcal A_{\mathrm{search}}$ 可以是任意数据依赖或预先给定的可计算子集， 并将 $\widehat\alpha$ 代入同一个界。理论上不需要预先限制有限网格；实际程序 只需为了有限计算时间枚举有限多个候选值。

最后，由式 <a href="#eq:final-proof-cluster-range">(135)</a>， $R_\alpha(\widehat h)\le b_\alpha$ 永远成立，所以右端小于 $b_\alpha$ 表示优于损失值域给出的逐点平凡界。 另一方面，均匀预测器在 smoothing 前后都具有 BPD $\log_2V$。因此本文采用 更有实际意义的标准：只有当选中 $\widehat\alpha$ 后的右端严格小于 $\log_2V$ 时，才称其为 non-vacuous certificate。 ◻

**注 3** (哪些量真正体现了 VLM 特性). 式 <a href="#eq:final-vlm-bound">(126)</a> 的集中工具仍是 Hoeffding 不等式，但证书并非 简单地把图像写进条件变量：*

1. *图像去重、共享 caption 和跨图像相关性决定 $N_{\mathrm{eff}}$，而不是文本行数；*
2. *视觉编码器、Projector 和 LLM 的训练范围决定 $\mathcal T$ 以及模块化码长 $K_{\mathrm{VLM}}$；*
3. *冻结视觉底座能否免费使用由 $A\mathrel{\perp\!\!\!\perp}S_{N_{\mathrm{eff}}}$ 决定；*
4. *视觉 token 数 $M$ 只描述单个图像内部的条件表示，不能将样本量 扩大为 $N_{\mathrm{eff}}M$；若 $M$、Projector 宽度或模块结构经过 数据依赖选择，则其选择成本进入 $C_{\mathrm{hp}}$。*
5. *平滑参数不受预设有限网格限制；最终选择的有限精度 $\alpha$ 通过 $L_\alpha(\alpha)$ 计费。若使用同一认证子样本选择 $\alpha$，该码长还必须进入第二层子采样偏差项。*

## 适用范围与结论

定理  保证的是：

- 量化后、prediction-smoothed VLM 的同分布条件 caption 风险；
- 对独立新图像–caption 样本的期望 BPD；
- 条件于训练集独立的冻结视觉和语言底座的 alignment pretraining 泛化能力；
- 在支付显式描述长度后，允许从可数无限的有限精度参数集合中进行 数据依赖的 prediction smoothing 参数选择。

它不自动保证未平滑模型的无界 NLL、分布外视觉泛化、组合泛化、幻觉率或 下游 VQA accuracy。若要研究这些指标，必须另外引入尾部假设、domain discrepancy、任务迁移或结构化损失。

对 MiniMind-V 风格的模型，最可行的第一阶段是冻结视觉编码器和 LLM， 将两层 Projector 写成 $\omega=\omega_0+P_Pu_P$，量化并算术编码 $u_P$，再按独立 image ID 抽样计算 caption-only smoothed BPD。其理论 certificate 正是 式 <a href="#eq:final-vlm-bound">(126)</a>。

## 参考文献

1. S. Lotfi, M. Finzi, Y. Kuang, T. G. J. Rudner, M. Goldblum, and A. G. Wilson. *Non-Vacuous Generalization Bounds for Large Language Models*. [https://arxiv.org/abs/2312.17173](https://arxiv.org/abs/2312.17173), 2023.
2. Z. Li, X. Jiang, L. Liu, X. Zhang, H. Chen, and F. Zheng. *On the Generalization Ability of Next-Token-Prediction Pretraining*. Proceedings of the 42nd International Conference on Machine Learning, PMLR 267, 2025. [https://proceedings.mlr.press/v267/li25ao.html](https://proceedings.mlr.press/v267/li25ao.html).
3. S. Lotfi et al. *SubLoRA Bounds for LLMs: official implementation*. [https://github.com/Sanaelotfi/SubLoRA-bounds-for-LLMs](https://github.com/Sanaelotfi/SubLoRA-bounds-for-LLMs).
4. Z. Li et al. *MININTP: code for next-token-prediction generalization experiments*. [https://github.com/lizhihao-leo/MININTP](https://github.com/lizhihao-leo/MININTP).