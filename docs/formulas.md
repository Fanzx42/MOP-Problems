# Mathematical definitions

All formulas minimize every objective. Indices in this document start at **1**;
NumPy implementations start at 0. Let `n` be the number of variables and `M` the
number of objectives. Bounds and defaults are recorded in the catalogue. Sources
and conventions are listed in [references](references.md).

## ZDT1

$n=30$ by default, $x_i\in[0,1]$.

$$f_1=x_1,\quad g=1+\frac9{n-1}\sum_{i=2}^n x_i,
\quad f_2=g\left(1-\sqrt{f_1/g}\right).$$

The Pareto set has $x_{2:n}=0$; the front is $f_2=1-\sqrt{f_1}$.

## ZDT2

Use ZDT1's bounds, $f_1$ and $g$, but
$f_2=g[1-(f_1/g)^2]$. The front is $f_2=1-f_1^2$, $0\le f_1\le1$.

## ZDT3

Use ZDT1's bounds, $f_1$ and $g$, but

$$f_2=g\left[1-\sqrt{f_1/g}-(f_1/g)\sin(10\pi f_1)\right].$$

The nondominated front consists of five segments, not the full $g=1$ curve.
The sampler uses these approximate intervals of $f_1$:

```text
[0,            0.0830015349]
(0.1822287280, 0.2577623634]
(0.4093136748, 0.4538821041]
(0.6183967944, 0.6525117038]
(0.8233317983, 0.8518328654]
```

## ZDT4

$n=10$ by default; $x_1\in[0,1]$, $x_{2:n}\in[-5,5]$.

$$f_1=x_1,\quad g=1+10(n-1)+\sum_{i=2}^n[x_i^2-10\cos(4\pi x_i)],
\quad f_2=g(1-\sqrt{f_1/g}).$$

The global front is the same as ZDT1; the distance function is multimodal.

## ZDT6

$n=10$ by default and $x_i\in[0,1]$.

$$f_1=1-e^{-4x_1}\sin^6(6\pi x_1),\quad
g=1+9\left(\frac1{n-1}\sum_{i=2}^n x_i\right)^{1/4},\quad
f_2=g[1-(f_1/g)^2].$$

The front uses $f_1\in[f_{1,\min},1]$, not $[0,1]$;
$f_{1,\min}\approx0.2807753191$ is attained at
$x_1=\arctan(9\pi)/(6\pi)$. The fractional power is not differentiable at zero.

## DTLZ shared definitions

Let $k=n-M+1\ge1$, $z=x_{M:n}$, and all variables lie in $[0,1]$.
For angles $\theta_1,\ldots,\theta_{M-1}$, define

$$s_1=\prod_{j=1}^{M-1}\cos\theta_j,\qquad
s_i=\left(\prod_{j=1}^{M-i}\cos\theta_j\right)\sin\theta_{M-i+1}
\quad (i=2,\ldots,M).$$

Empty products equal 1. Unless stated otherwise, $M=3$ and $k=10$ by default.

## DTLZ1

Default $k=5$. Define

$$g=100\left[k+\sum_{x\in z}((x-0.5)^2-\cos(20\pi(x-0.5)))\right].$$

$$f_1=\tfrac12(1+g)\prod_{j=1}^{M-1}x_j,\qquad
f_i=\tfrac12(1+g)\left(\prod_{j=1}^{M-i}x_j\right)(1-x_{M-i+1})
\quad (i=2,\ldots,M).$$

At $z=0.5$, the front satisfies $\sum_i f_i=0.5$.

## DTLZ2

$g=\sum_{x\in z}(x-0.5)^2$, $\theta_j=\pi x_j/2$, and $f_i=(1+g)s_i$.
At $z=0.5$, the front is the nonnegative unit sphere $\sum_i f_i^2=1$.

## DTLZ3

Use DTLZ2's angular mapping and DTLZ1's multimodal $g$, with default $k=10$.
The global front is the nonnegative unit sphere.

## DTLZ4

Use DTLZ2's $g$ and $\theta_j=\pi x_j^{\alpha}/2$, default $\alpha=100$.
The front remains spherical but the decision-to-front distribution is biased.

## DTLZ5

Use DTLZ2's $g$ and $f_i=(1+g)s_i$ with

$$\theta_1=\pi x_1/2,\qquad
\theta_j=\frac{\pi(1+2g x_j)}{4(1+g)},\quad j=2,\ldots,M-1.$$

The reference sampler uses $g=0$, giving $\theta_{2:M-1}=\pi/4$.

## DTLZ6

Use DTLZ5's angular mapping, replacing $g$ with $\sum_{x\in z}x^{0.1}$.
The reference set is reached at $z=0$; the fractional power needs boundary care.

## DTLZ7

Default $k=20$.

$$f_i=x_i\quad(i<M),\quad g=1+\frac9k\sum_{x\in z}x,\qquad
f_M=(1+g)\left[M-\sum_{i=1}^{M-1}\frac{f_i}{1+g}(1+\sin(3\pi f_i))\right].$$

Note the additional **$1+g$**, even though $g$ itself starts at 1.
This release evaluates the problem but does not provide a front sampler for its
disconnected nondominated subset.

## SCH

Schaffer N.1: $x\in[-1000,1000]$, $f_1=x^2$, $f_2=(x-2)^2$.
The Pareto set is $[0,2]$. Some libraries use a smaller box; this repository's
bound convention is explicit.

## FON

Fonseca–Fleming, default $n=3$, $x_i\in[-4,4]$:

$$f_1=1-\exp\left[-\sum_{i=1}^n(x_i-1/\sqrt n)^2\right],\qquad
f_2=1-\exp\left[-\sum_{i=1}^n(x_i+1/\sqrt n)^2\right].$$

The Pareto set has every component equal to $a\in[-1/\sqrt n,1/\sqrt n]$.
This is the `Fonseca2` variant in MOEA Framework, not its different `Fonseca` problem.

## POL

For $x_1,x_2\in[-\pi,\pi]$, set

$$B_1=0.5\sin x_1-2\cos x_1+\sin x_2-1.5\cos x_2,$$
$$B_2=1.5\sin x_1-\cos x_1+2\sin x_2-0.5\cos x_2.$$

Let $A_1=B_1(1,2)$ and $A_2=B_2(1,2)$. Then

$$f_1=1+(A_1-B_1)^2+(A_2-B_2)^2,\qquad
f_2=(x_1+3)^2+(x_2+1)^2.$$

We minimize these positive values. A library that maximizes their negatives is
equivalent; copying the signs without copying objective directions is incorrect.

## BNH

$x_1\in[0,5]$, $x_2\in[0,3]$:

$$f_1=4(x_1^2+x_2^2),\qquad f_2=(x_1-5)^2+(x_2-5)^2,$$
$$G_1=(x_1-5)^2+x_2^2-25,\qquad
G_2=7.7-(x_1-8)^2-(x_2+3)^2.$$

Both constraints must be $\le0$. These are raw residuals; pymoo divides them by
25 and 7.7, respectively.

## TNK

$x_1,x_2\in[0,\pi]$, $F=(x_1,x_2)$:

$$G_1=1+0.1\cos(16\operatorname{atan2}(x_1,x_2))-x_1^2-x_2^2,$$
$$G_2=(x_1-0.5)^2+(x_2-0.5)^2-0.5.$$

The argument order is intentional: the original angle is $\arctan(x_1/x_2)$.
`atan2` handles the axes; at the infeasible origin we define the angle as zero.
pymoo scales $G_2$ by 2 and uses a small positive lower bound for $x_2$.

## OSY

Bounds are $l=(0,0,1,0,1,0)$ and $u=(10,10,5,6,5,10)$.

$$f_1=-[25(x_1-2)^2+(x_2-2)^2+(x_3-1)^2+(x_4-4)^2+(x_5-1)^2],
\qquad f_2=\sum_{i=1}^6x_i^2.$$

The six residuals, all $\le0$, are

$$G=(2-x_1-x_2,\ x_1+x_2-6,\ x_2-x_1-2,\ x_1-3x_2-2,
\ (x_3-3)^2+x_4-4,\ 4-(x_5-3)^2-x_6).$$

pymoo divides these residuals by $(2,6,2,2,4,4)$.

## SRN

$x_1,x_2\in[-20,20]$:

$$f_1=2+(x_1-2)^2+(x_2-1)^2,\qquad f_2=9x_1-(x_2-1)^2,$$
$$G_1=x_1^2+x_2^2-225,\qquad G_2=x_1-3x_2+10.$$

The raw residuals are feasible when $G_1,G_2\le0$.
