"""Private forward-mode chain rules for the benchmark kernels.

This is deliberately a small NumPy subset, not a general AD framework. Unsupported
operations fail rather than discarding derivatives. The last axis carries seeds;
all preceding axes have the same meaning as in the ordinary formula evaluation.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


def _near_zero(x):
    return np.abs(x) <= 32 * np.finfo(float).eps


class Jet(NDArrayOperatorsMixin):
    __array_priority__ = 1000

    def __init__(self, value, derivative, regular=True):
        self.v = np.asarray(value, dtype=float)
        self.d = np.asarray(derivative, dtype=float)
        self.r = np.broadcast_to(regular, self.v.shape).copy()

    @classmethod
    def variables(cls, x):
        return cls(x.copy(), np.broadcast_to(np.eye(x.shape[1]), (*x.shape, x.shape[1])).copy())

    @property
    def shape(self):
        return self.v.shape

    @property
    def ndim(self):
        return self.v.ndim

    @property
    def T(self):
        axes = tuple(reversed(range(self.ndim)))
        return Jet(self.v.T, self.d.transpose(*axes, self.ndim), self.r.T)

    def __len__(self):
        return len(self.v)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        # Expand ellipsis before appending the protected derivative axis.
        if any(k is Ellipsis for k in key):
            used = sum(k is not None and k is not Ellipsis for k in key)
            expanded = []
            for k in key:
                expanded.extend([slice(None)] * (self.ndim - used) if k is Ellipsis else [k])
            key = tuple(expanded)
        return Jet(self.v[key], self.d[(*key, slice(None))], self.r[key])

    def __setitem__(self, key, other):
        other = constant(other, self.d.shape[-1])
        key = key if isinstance(key, tuple) else (key,)
        self.v[key], self.d[(*key, slice(None))], self.r[key] = other.v, other.d, other.r

    def __array__(self, dtype=None, copy=None):
        raise TypeError("A benchmark kernel attempted to discard its derivative; use supported ops")

    def copy(self):
        return Jet(self.v.copy(), self.d.copy(), self.r.copy())

    def sum(self, axis=None, keepdims=False):
        axes = (
            tuple(range(self.ndim))
            if axis is None
            else (
                tuple(a % self.ndim for a in axis) if isinstance(axis, tuple) else axis % self.ndim
            )
        )
        return Jet(
            self.v.sum(axis=axis, keepdims=keepdims),
            self.d.sum(axis=axes, keepdims=keepdims),
            self.r.all(axis=axis, keepdims=keepdims),
        )

    def mean(self, axis=None, keepdims=False):
        count = self.v.size if axis is None else self.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / count

    def prod(self, axis=None, keepdims=False):
        if axis is None:
            raise NotImplementedError("AD product requires an explicit axis")
        axis %= self.ndim
        # Prefix/suffix products work at zero without division by the factors.
        vals = np.moveaxis(self.v, axis, 0)
        ders = np.moveaxis(self.d, axis, 0)
        prefix = np.ones_like(vals)
        suffix = np.ones_like(vals)
        if len(vals) > 1:
            prefix[1:] = np.cumprod(vals[:-1], axis=0)
            suffix[:-1] = np.cumprod(vals[:0:-1], axis=0)[::-1]
        derivative = ((prefix * suffix)[..., None] * ders).sum(axis=0)
        if keepdims:
            derivative = np.expand_dims(derivative, axis)
        return Jet(
            self.v.prod(axis=axis, keepdims=keepdims),
            derivative,
            self.r.all(axis=axis, keepdims=keepdims),
        )

    def max(self, axis=None, keepdims=False):
        if axis is None:
            raise NotImplementedError("AD max requires an explicit axis")
        axis %= self.ndim
        index = self.v.argmax(axis=axis, keepdims=True)
        value = np.take_along_axis(self.v, index, axis)
        derivative = np.take_along_axis(self.d, index[..., None], axis)
        ties = _near_zero(self.v - value)
        same = np.all(_near_zero(self.d - derivative), axis=-1)
        regular = np.all(~ties | (same & self.r), axis=axis, keepdims=True)
        if not keepdims:
            value, derivative, regular = (
                np.squeeze(a, axis=axis) for a in (value, derivative, regular)
            )
        return Jet(value, derivative, regular)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs and method == "__call__":
            targets = kwargs.pop("out")
            if len(targets) != 1 or not isinstance(targets[0], Jet):
                return NotImplemented
            result = self.__array_ufunc__(ufunc, method, *inputs, **kwargs)
            target = targets[0]
            target.v[...], target.d[...], target.r[...] = result.v, result.d, result.r
            return target
        if method != "__call__" or kwargs:
            return NotImplemented
        n = self.d.shape[-1]
        args = [constant(a, n) for a in inputs]
        a = args[0]
        v, d, r = a.v, a.d, a.r
        if ufunc in (np.less, np.less_equal, np.greater, np.greater_equal, np.equal, np.not_equal):
            return ufunc(*[a.v for a in args])
        if ufunc is np.matmul:
            b = args[1]
            deriv = np.stack([a.d[..., i] @ b.v + a.v @ b.d[..., i] for i in range(n)], -1)
            # Preserve batch-local regularity rather than contaminating every
            # batch row if one matrix row has a nonsmooth entry.
            bad_left = (~a.r).astype(int) @ np.ones_like(b.v, dtype=int)
            bad_right = np.ones_like(a.v, dtype=int) @ (~b.r).astype(int)
            regular = (bad_left == 0) & (bad_right == 0)
            return Jet(a.v @ b.v, deriv, regular)
        if len(args) == 2:
            b = args[1]
            shape = np.broadcast_shapes(a.shape, b.shape)
            av, bv = np.broadcast_to(v, shape), np.broadcast_to(b.v, shape)
            ad, bd = np.broadcast_to(d, (*shape, n)), np.broadcast_to(b.d, (*shape, n))
            regular = np.broadcast_to(r, shape) & np.broadcast_to(b.r, shape)
            if ufunc is np.add:
                derivative = ad + bd
            elif ufunc is np.subtract:
                derivative = ad - bd
            elif ufunc is np.multiply:
                derivative = ad * bv[..., None] + bd * av[..., None]
            elif ufunc is np.divide:
                derivative = (ad - bd * (av / bv)[..., None]) / bv[..., None]
            elif ufunc in (np.power, np.float_power):
                value = av**bv
                # Avoid 0 * inf for constant bases/exponents (e.g. integer powers).
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    base_factor = np.where(bv == 0, 0, bv * av ** (bv - 1))
                    exp_factor = np.where((av == 0) & (bv > 0), 0, value * np.log(av))
                    derivative = np.where(ad == 0, 0, ad * base_factor[..., None])
                    derivative += np.where(bd == 0, 0, bd * exp_factor[..., None])
                regular &= ~((av == 0) & (bv > 0) & (bv < 1))
                return Jet(value, derivative, regular)
            elif ufunc in (np.minimum, np.maximum):
                choose = av <= bv if ufunc is np.minimum else av >= bv
                derivative = np.where(choose[..., None], ad, bd)
                regular = np.where(choose, np.broadcast_to(r, shape), np.broadcast_to(b.r, shape))
                regular &= ~_near_zero(av - bv) | np.all(_near_zero(ad - bd), axis=-1)
            elif ufunc is np.arctan2:
                derivative = (bv[..., None] * ad - av[..., None] * bd) / (av**2 + bv**2)[..., None]
                regular &= ~((av == 0) & (bv <= 0))
            else:
                return NotImplemented
            return Jet(ufunc(av, bv), derivative, regular)
        if ufunc is np.negative:
            factor = -np.ones_like(v)
        elif ufunc is np.positive:
            factor = np.ones_like(v)
        elif ufunc is np.sin:
            factor = np.cos(v)
        elif ufunc is np.cos:
            factor = -np.sin(v)
        elif ufunc in (np.exp, np.expm1):
            factor = np.exp(v)
        elif ufunc is np.sqrt:
            factor = 0.5 / np.sqrt(v)
            r = r & (v > 0)
        elif ufunc in (np.abs, np.absolute):
            factor = np.sign(v)
            r = r & (~_near_zero(v) | np.all(d == 0, axis=-1))
        elif ufunc in (np.floor, np.ceil, np.sign):
            factor = np.zeros_like(v)
            at_break = _near_zero(v) if ufunc is np.sign else _near_zero(v - np.rint(v))
            r = r & ~at_break
        elif ufunc is np.square:
            factor = 2 * v
        else:
            return NotImplemented
        with np.errstate(invalid="ignore"):
            derivative = np.where(d == 0, 0, d * factor[..., None])
        return Jet(ufunc(v), derivative, r)

    def __array_function__(self, func, types, args, kwargs):
        if func in (np.sum, np.mean, np.prod, np.max, np.amax):
            name = "max" if func in (np.max, np.amax) else func.__name__
            return getattr(args[0], name)(*args[1:], **kwargs)
        if func in (np.column_stack, np.stack, np.concatenate):
            items = [constant(a, self.d.shape[-1]) for a in args[0]]
            if func is np.column_stack:
                items = [a[:, None] if a.ndim == 1 else a for a in items]
                return Jet(
                    np.concatenate([a.v for a in items], axis=1),
                    np.concatenate([a.d for a in items], axis=1),
                    np.concatenate([a.r for a in items], axis=1),
                )
            axis = kwargs.get("axis", args[1] if len(args) > 1 else 0)
            axis %= items[0].ndim + (func is np.stack)
            return Jet(*[func([getattr(a, k) for a in items], axis=axis) for k in ("v", "d", "r")])
        if func is np.roll:
            a, shift = args[:2]
            axis = kwargs.get("axis", args[2] if len(args) > 2 else None)
            if axis is None:
                raise NotImplementedError("AD roll requires an explicit axis")
            axis %= a.ndim
            return Jet(*[np.roll(getattr(a, k), shift, axis=axis) for k in ("v", "d", "r")])
        if func is np.where:
            condition, left, right = args
            a, b = (constant(x, self.d.shape[-1]) for x in (left, right))
            return Jet(
                np.where(condition, a.v, b.v),
                np.where(np.asarray(condition)[..., None], a.d, b.d),
                np.where(condition, a.r, b.r),
            )
        if func is np.clip:
            a, lo, hi = args
            return np.minimum(np.maximum(a, lo), hi)
        return NotImplemented


def constant(value, n):
    if isinstance(value, Jet):
        return value
    v = np.asarray(value, dtype=float)
    return Jet(v, np.zeros((*v.shape, n)))
