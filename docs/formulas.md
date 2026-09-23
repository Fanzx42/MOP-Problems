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

For $M\le3$, the reference sampler uses $g=0$, giving $\theta_{2:M-1}=\pi/4$.
For $M>3$, that curve can omit nondominated tradeoffs with $g>0$; only objective
evaluation is supported, not a complete-front sampler.

## DTLZ6

Use DTLZ5's angular mapping, replacing $g$ with $\sum_{x\in z}x^{0.1}$.
For $M\le3$, the reference set is reached at $z=0$; the fractional power needs
boundary care. The same $M>3$ sampler restriction as DTLZ5 applies.

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

## WFG shared definitions

Source: [Huband et al.](https://doi.org/10.1109/TEVC.2005.861417);
versioned numerical comparison: [pymoo 0.6.2](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/problems/many/wfg.py).
Normalize $y_i=x_i/(2i)$, with $x_i\in[0,2i]$. There are $k$ position variables,
$l=n-k>0$ distance variables and $M\ge2$ objectives. Require $k$ to be a positive
multiple of $M-1$, and even $l$ for WFG2/3. Defaults are $M=3$, $k=4$, $l=20$.
With other objective counts, default $k=4$ for $M=2$, otherwise $k=2(M-1)$.
The following scalar functions act elementwise:

$$s_L(y)=\frac{|y-0.35|}{|\lfloor0.35-y\rfloor+0.35|},\qquad b_P(y)=y^{0.02}.$$

$$b_F(y)=0.8+\min(0,\lfloor y-0.75\rfloor)\frac{0.8(0.75-y)}{0.75}
-\min(0,\lfloor0.85-y\rfloor)\frac{0.2(y-0.85)}{0.15}.$$

For $a=0.98/49.98$, define the parameter-dependent bias

$$b_D(y,u)=y^{0.02+49.98[a-(1-2u)|\lfloor0.5-u\rfloor+a|]}.$$

For deceptive shift parameters $A=0.35$, $B=0.001$, $C=0.05$:

$$s_D(y)=1+(|y-A|-B)\left[
\frac{\lfloor y-A+B\rfloor(1-C+(A-B)/B)}{A-B}
+\frac{\lfloor A+B-y\rfloor(1-C+(1-A-B)/B)}{1-A-B}+\frac1B\right].$$

For multimodal shift $q=|y-0.35|/[2(\lfloor0.35-y\rfloor+0.35)]$:

$$s_M(y,B)=\frac{1+\cos(122\pi(0.5-q))+4Bq^2}{B+2}.$$

Weighted reduction is $r_S(y,w)=\sum_iw_iy_i/\sum_iw_i$; uniform weights give
the mean. Nonseparable reduction of a block of length $p$ and degree $A$ is

$$r_N(y,A)=\frac{\sum_{j=1}^p[y_j+\sum_{d=1}^{A-1}|y_j-y_{1+((j+d-1)\bmod p)}|]}
{(p/A)\lceil A/2\rceil[1+2A-2\lceil A/2\rceil]}.$$

Final reduction gives $t_1,\ldots,t_M$: split the first $k$ variables into $M-1$
equal consecutive groups and use the entire remaining tail for the last group.
Set $z_i=\max(t_M,A_i)(t_i-0.5)+0.5$ for $i<M$ and
$f_i=t_M+2i\,h_i(z)$. Usually every $A_i=1$; WFG3 uses $(1,0,\ldots,0)$.
With $p=M-i$, the shapes are

$$h_i^{linear}=\prod_{j=1}^{p}z_j
\begin{cases}1&i=1\\1-z_{p+1}&i>1,\end{cases}$$

$$h_i^{convex}=\prod_{j=1}^{p}(1-\cos(\pi z_j/2))
\begin{cases}1&i=1\\1-\sin(\pi z_{p+1}/2)&i>1,\end{cases}$$

$$h_i^{concave}=\prod_{j=1}^{p}\sin(\pi z_j/2)
\begin{cases}1&i=1\\\cos(\pi z_{p+1}/2)&i>1.\end{cases}$$

Only floating-point excursions just outside $[0,1]$ are corrected; positive tiny
values are retained. WFG1's exponent 0.02 can amplify rounding near a distance
minimum. No WFG analytical-front sampler is exposed in this release.

## WFG1

Apply $s_L$ to the tail, then $b_F$ to that tail and $b_P$ to all variables.
Use weighted reductions with original variable weights $w_j=2j$ and convex
shapes, replacing the final shape by
$h_M=1-z_1-\cos(10\pi z_1+\pi/2)/(10\pi)$.

## WFG2

Apply $s_L$ to the tail, reduce its consecutive pairs using $r_N(\cdot,2)$,
then apply uniform group reductions. Use convex shapes, replacing the final
shape by $h_M=1-z_1\cos^2(5\pi z_1)$. Its front is disconnected.

## WFG3

Use WFG2's transformations and reductions, but linear shapes and
$A=(1,0,\ldots,0)$. For $M>2$ this yields a degenerate front.

## WFG4

Apply $s_M(y,10)$ to all normalized variables, then uniform group reductions
and concave shapes.

## WFG5

Apply $s_D$ to all normalized variables, then uniform group reductions
and concave shapes.

## WFG6

Apply $s_L$ to the tail. Reduce each final group with $r_N$ using its full
length as the degree. Use concave shapes.

## WFG7

For $i\le k$, apply $b_D(y_i,\operatorname{mean}(y_{i+1:n}))$ using the
unmodified normalized suffix. Apply $s_L$ to the tail, uniform group reductions
and concave shapes.

## WFG8

For $i>k$, apply $b_D(y_i,\operatorname{mean}(y_{1:i-1}))$ using the
**unmodified** normalized prefix, without cascading updates. Then apply $s_L$
to the tail, uniform group reductions and concave shapes.

## WFG9

For $i<n$, apply $b_D$ with the original suffix mean. Apply $s_D$ to the first
$k$ values and $s_M(y,95)$ to the tail. Use full-degree nonseparable reductions
and concave shapes.

## UF shared definitions

Source: [CEC 2009 report CES-487](https://al-roomi.org/multimedia/CEC_Database/CEC2009/MultiObjectiveEA/CEC2009_MultiObjectiveEA_TechnicalReport.pdf).
Defaults are $n=30$. UF1–7 require $n\ge3$, have two objectives and use
$J_1=\{3,5,\ldots\}\cap[2,n]$, $J_2=\{2,4,\ldots\}\cap[2,n]$.
Let $t=x_1$, $\phi_j=6\pi t+j\pi/n$ and normally $y_j=x_j-\sin\phi_j$.
Define penalties for a nonempty index set $J$:

$$Q_J(y)=\frac2{|J|}\sum_{j\in J}y_j^2,\qquad
P_J(y)=\frac2{|J|}\left(4\sum_{j\in J}y_j^2
-2\prod_{j\in J}\cos(20\pi y_j/\sqrt j)+2\right).$$

Except UF3/4, two-objective bounds are $t\in[0,1]$, $x_{2:n}\in[-1,1]$.
The fixed constants below select the competition variants.

## UF1

$F=(t+Q_{J_1},\,1-\sqrt t+Q_{J_2})$.
The front is $(u,1-\sqrt u)$ for $u\in[0,1]$.

## UF2

Use UF1's objectives with
$a_j=0.3t^2\cos(24\pi t+4j\pi/n)+0.6t$.
For odd $j$, $y_j=x_j-a_j\cos\phi_j$; for even $j$,
$y_j=x_j-a_j\sin\phi_j$. The front is the same as UF1.

## UF3

All variables lie in $[0,1]$.
Set $y_j=x_j-t^{[1+3(j-2)/(n-2)]/2}$ and use
$F=(t+P_{J_1},\,1-\sqrt t+P_{J_2})$.

## UF4

Use $x_{2:n}\in[-2,2]$ and $h(y)=|y|/[1+\exp(2|y|)]$:
$F=(t+2\operatorname{mean}_{J_1}h(y),\,1-t^2+2\operatorname{mean}_{J_2}h(y))$.
The front is $(u,1-u^2)$.

## UF5

Use $h(y)=2y^2-\cos(4\pi y)+1$ and
$b(t)=0.15|\sin(20\pi t)|$ ($N=10$, $\epsilon=0.1$):

$$F=(t+b+2\operatorname{mean}_{J_1}h(y),\,
1-t+b+2\operatorname{mean}_{J_2}h(y)).$$

The front contains the 21 discrete points $(i/20,1-i/20)$, $i=0,\ldots,20$.
There is no arbitrary-size front sampler for this finite set.

## UF6

Use $b(t)=\max(0,0.7\sin(4\pi t))$ ($N=2$, $\epsilon=0.1$):
$F=(t+b+P_{J_1},\,1-t+b+P_{J_2})$.
The front lies on $(u,1-u)$ with
$u\in\{0\}\cup[1/4,1/2]\cup[3/4,1]$. Its sampler is not implemented.

## UF7

$F=(t^{0.2}+Q_{J_1},\,1-t^{0.2}+Q_{J_2})$.
The front is $(u,1-u)$ for $u\in[0,1]$.

## UF8

UF8–10 have three objectives and require $n\ge5$:
$x_1,x_2\in[0,1]$, $x_{3:n}\in[-2,2]$.
Use $J_1=\{4,7,\ldots\}$, $J_2=\{5,8,\ldots\}$,
$J_3=\{3,6,\ldots\}$, truncated at $n$, and
$y_j=x_j-2x_2\sin(2\pi x_1+j\pi/n)$.

$$F=(\cos(\pi x_1/2)\cos(\pi x_2/2)+Q_{J_1},\,
\cos(\pi x_1/2)\sin(\pi x_2/2)+Q_{J_2},\,
\sin(\pi x_1/2)+Q_{J_3}).$$

The front is the nonnegative unit sphere.

## UF9

Use UF8's groups, bounds and residuals.
Set $b=\max(0,1.1[1-4(2x_1-1)^2])$ ($\epsilon=0.1$):

$$F=((b+2x_1)x_2/2+Q_{J_1},\,
(b-2x_1+2)x_2/2+Q_{J_2},\,1-x_2+Q_{J_3}).$$

The disconnected nondominated subset has no sampler in this release.

## UF10

Use UF8's objectives but replace every $Q_J$ by
$2\operatorname{mean}_{j\in J}[4y_j^2-\cos(8\pi y_j)+1]$.
The front remains the nonnegative unit sphere.

## Omni-test

For $n\ge2$, $x_i\in[0,6]$:

$$f_1=\sum_i\sin(\pi x_i),\qquad f_2=\sum_i\cos(\pi x_i).$$

For each vector $b\in\{0,1,2\}^n$, one Pareto subset is
$x_i=1+t+2b_i$ with the **same** $t\in[0,1/2]$ for all coordinates.
These $3^n$ subsets share the negative-quadrant circular front of radius $n$.
The front sampler returns objective values; it does not enumerate decision sets.

## SYM-PART

This selects the simple pymoo 0.6.2 variant: $a=1$, $b=c=10$ and box
$[-100,100]^2$. For coordinates $(u,v)$, define

$$t_u=\operatorname{clip}(\operatorname{sign}(u)\lceil(|u|-6)/12\rceil,-1,1),
\quad t_v=\operatorname{clip}(\operatorname{sign}(v)\lceil(|v|-5)/10\rceil,-1,1).$$

Set $p=u-10t_u$, $q=v-10t_v$:
$F=((p+1)^2+q^2,(p-1)^2+q^2)$.
The nine Pareto subsets are $(u,v)=(10i+t,10j)$ for
$i,j\in\{-1,0,1\}$ and $t\in[-1,1]$.

## SYM-PART-Rotated

Apply SYM-PART to $u=x_1\cos\theta+x_2\sin\theta$,
$v=-x_1\sin\theta+x_2\cos\theta$, default $\theta=\pi/4$.
The Pareto subsets rotate counterclockwise by $\theta$; the objectives and front
are unchanged. The rotated-distorted variant is not included.

## C-DTLZ shared definitions

Use the named DTLZ base objectives with $M\ge2$, $n\ge M$ and box $[0,1]^n$.
All five variants default to $M=3$, $n=M+9$; this explicitly uses $k=10$
also for the DTLZ1-based variants. All residuals below require $G\le0$.
There is no constrained-front sampler: a base DTLZ front must not be assumed
feasible or Pareto-optimal for a constrained variant.

## C1-DTLZ1

Use DTLZ1 objectives and
$G=2\sum_{i=1}^{M-1}f_i+f_M/0.6-1$.

## C1-DTLZ3

Use DTLZ3 objectives. With $s=\sum_i f_i^2$,
$G=-(s-16)(s-r^2)$. Default radius is $9$ for $M<5$,
$12.5$ for $5\le M\le12$, otherwise $15$.
A finite positive radius can be supplied explicitly.

## C2-DTLZ2

Use DTLZ2 objectives and

$$G=\min\left\{\min_i\left[(f_i-1)^2+\sum_{j\ne i}f_j^2-r^2\right],
\ \sum_i(f_i-1/\sqrt M)^2-r^2\right\}.$$

Default $r=0.2$ for $M=2$, $0.4$ for $M=3$, otherwise $0.5$.
A finite positive radius can be supplied explicitly.

## C3-DTLZ1

Use DTLZ1 objectives and $M$ residuals:
$G_i=1-2f_i-\sum_{j\ne i}f_j$.

## C3-DTLZ4

Use DTLZ4 objectives, default $\alpha=100$, and $M$ residuals:
$G_i=1-f_i^2/4-\sum_{j\ne i}f_j^2$.

## Welded-Beam

This is the **pymoo 0.6.2 two-objective, four-constraint variant**.
Variables $(h,\ell,t,b)$ are in inches, with bounds
$(0.125,0.1,0.1,0.125)$ to $(5,10,10,5)$.
Cost and end deflection are minimized:

$$f_1=1.10471h^2\ell+0.04811tb(14+\ell),\qquad
f_2=2.1952/(bt^3).$$

For a 6000 lbf load and 14 inch overhang:

$$R=\tfrac12\sqrt{\ell^2+(h+t)^2},\quad
M_b=6000(14+\ell/2),\quad
J=\sqrt2h\ell[\ell^2/12+(h+t)^2/4],$$

$$\tau_1=6000/(\sqrt2h\ell),\quad \tau_2=M_bR/J,\quad
\tau=\sqrt{\tau_1^2+\tau_2^2+\tau_1\tau_2\ell/R},$$

$$\sigma=504000/(bt^2),\qquad
P_c=64746.022(1-0.0282346t)tb^3,$$

$$G=(\tau-13600,\ \sigma-30000,\ h-b,\ 6000-P_c).$$

Stresses are in psi, deflection in inches and buckling load in lbf.
pymoo normalizes these four residuals by $(13600,30000,4.875,6000)$;
this repository returns the raw values. This selected software variant uses
the $J$ coefficient $\sqrt2$: other engineering formulations use $2\sqrt2$,
which changes shear stresses and the feasible set. They must not be compared as
identical benchmarks. No extra cost or deflection constraint is added.
