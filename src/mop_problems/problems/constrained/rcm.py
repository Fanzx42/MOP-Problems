"""Continuous RCM engineering equations (Kumar et al., 2021).

RCM5/14 use the author's continuous relaxation of disk counts. Original-code
equation variants are intentional and documented in the per-problem catalogue.
Power-system kernels split real/imaginary parts for exact forward derivatives.
"""

import json
from functools import lru_cache
from importlib.resources import files

import numpy as np

from ...core import Problem


@lru_cache(maxsize=1)
def _constants():
    return json.loads(files("mop_problems").joinpath("data", "rcm_constants.json").read_text())


class _RCM(Problem):
    number: int

    def __init__(self):
        p = _constants()[str(self.number)]
        super().__init__(
            f"RCM{self.number}",
            p["n_var"],
            p["n_obj"],
            p["lower"],
            p["upper"],
            n_ieq_constr=p["n_ieq_constr"],
            n_eq_constr=p["n_eq_constr"],
        )

    def _evaluate(self, X):
        p = {key: np.asarray(value) for key, value in _constants()[str(self.number)].items()}
        i = self.number
        if 30 <= i <= 35:
            angles = X * np.pi / 180
            k, s = p["k"].ravel(), p["s"].ravel()
            harmonic = np.column_stack([np.sum(s * np.cos(v * angles), axis=1) for v in k])
            f1 = np.sqrt(np.sum(harmonic**2 / k**4, axis=1) / np.sum(1 / k**4))
            f2 = (np.sum(s * np.cos(angles), axis=1) - p["modulation"]) ** 2
            return np.column_stack((f1, f2)), X[:, :-1] - X[:, 1:] + 1e-6, None
        if i <= 46:
            return self._network(X, p)
        if i <= 49:
            return self._droop(X, p)
        f1 = np.sum(p["a"] + p["b"] * X + p["c"] * X**2, axis=1)
        f2 = np.sum(p["alpha"] + p["beta"] * X + p["gamma"] * X**2, axis=1)
        loss = np.sum((X @ p["B"]) * X, axis=1)
        return np.column_stack((f1, f2)), None, (X.sum(axis=1) - 1200 - loss)[:, None]

    def _network(self, X, p):
        i = self.number
        zero = X[:, 0] * 0
        if i < 40:
            vr = np.column_stack((zero + 1, zero - 0.5, zero - 0.5, X[:, :12]))
            vi = np.column_stack((zero, zero - np.sqrt(3) / 2, zero + np.sqrt(3) / 2, X[:, 12:24]))
            mapping = {6: 24, 14: 25}
        else:
            vr = np.column_stack((zero + 1, X[:, :13]))
            vi = np.column_stack((zero, X[:, 13:26]))
            mapping = {1: 26, 2: 27, 5: 28, 7: 29}
        count = vr.shape[1]
        pg = np.column_stack([X[:, mapping[j]] if j in mapping else zero for j in range(count)])
        qg = np.column_stack(
            [X[:, mapping[j] + len(mapping)] if j in mapping else zero for j in range(count)]
        )
        ir, ii = vr @ p["G"].T - vi @ p["B"].T, vr @ p["B"].T + vi @ p["G"].T
        pr, qi = vr * ir + vi * ii, vi * ir - vr * ii
        start, sign = (3, -1) if i < 40 else (1, 1)
        h = np.column_stack(
            (
                (pr - pg + sign * p["P"].ravel())[:, start:],
                (qi - qg + sign * p["Q"].ravel())[:, start:],
            )
        )
        active, reactive = pr.sum(axis=1), qi.sum(axis=1)
        if i < 40:
            r1, q1 = ir[:, :3].sum(axis=1), ii[:, :3].sum(axis=1)
            r2 = ir[:, 0] - 0.5 * (ir[:, 1] + ir[:, 2]) + np.sqrt(3) / 2 * (ii[:, 1] - ii[:, 2])
            q2 = ii[:, 0] - 0.5 * (ii[:, 1] + ii[:, 2]) + np.sqrt(3) / 2 * (ir[:, 2] - ir[:, 1])
            imbalance = np.sqrt(r1 * r1 + q1 * q1) + np.sqrt(r2 * r2 + q2 * q2)
            columns = {
                36: (imbalance, active),
                37: (imbalance, reactive),
                38: (reactive, active),
                39: (imbalance, active, reactive),
            }[i]
        else:
            dev = np.sum((1 - np.sqrt(vr[:, 1:] ** 2 + vi[:, 1:] ** 2)) ** 2, axis=1)
            fuel = np.sum(
                X[:, 26:30] * [1.75, 1, 3.25, 3]
                + X[:, 26:30] ** 2 * [0.0175, 0.0625, 0.00834, 0.025],
                axis=1,
            )
            # Published source RCM40/41 uses conj(I(2)) for the slack reactive term.
            reactive_i2 = -ii[:, 1] + qi[:, 1:].sum(axis=1)
            columns = {
                40: (active, reactive_i2),
                41: (active, reactive_i2, dev),
                42: (active, dev),
                43: (fuel, active),
                44: (fuel, active, reactive),
                45: (fuel, active, dev),
                46: (fuel, active, reactive, dev),
            }[i]
        return np.column_stack(columns), None, h

    def _droop(self, X, p):
        zero = X[:, 0] * 0
        vr = np.column_stack((X[:, 11] + 1e-5, X[:, :5]))
        vi = np.column_stack((zero, X[:, 5:10]))
        w = X[:, 10]
        ir, ii = [zero for _ in range(6)], [zero for _ in range(6)]
        for a, b, r, reactance, ground, tap in p["L"]:
            a, b = int(a) - 1, int(b) - 1
            den = r * r + (w * reactance) ** 2
            g, s = r / den, -w * reactance / den
            dr, di = vr[:, a] / tap - vr[:, b], vi[:, a] / tap - vi[:, b]
            real, imag = g * dr - s * di, s * dr + g * di
            ir[a] = ir[a] + real / tap - ground * vi[:, a]
            ii[a] = ii[a] + imag / tap + ground * vr[:, a]
            ir[b] = ir[b] - real - ground * vi[:, b]
            ii[b] = ii[b] - imag + ground * vr[:, b]
        ir, ii = np.column_stack(ir), np.column_stack(ii)
        magnitude2 = vr * vr + vi * vi
        magnitude = np.sqrt(magnitude2)
        pc = np.column_stack((zero, zero, zero, X[:, 12:15]))
        qc = np.column_stack((zero, zero, zero, X[:, 15:18]))
        ps = pc * (1 - w[:, None]) - p["P"][:, 0] * (magnitude / p["P"][:, 4]) ** p["P"][:, 5]
        qs = qc * (1 - magnitude) - p["Q"][:, 0] * (magnitude / p["Q"][:, 4]) ** p["Q"][:, 5]
        h = np.column_stack(
            (ir - (ps * vr + qs * vi) / magnitude2, ii - (ps * vi - qs * vr) / magnitude2)
        )
        active, reactive = ps.sum(axis=1), qs.sum(axis=1)
        dev = np.sum((1 - magnitude) ** 2, axis=1)
        columns = {47: (active, reactive), 48: (active, dev), 49: (active, reactive, dev)}[
            self.number
        ]
        return np.column_stack(columns), None, h


class RCM2(_RCM):
    number = 2

    def _evaluate(self, X):
        f, g, _h = [None] * 2, [None] * 5, [None] * 0
        d1 = X[:, 0]
        d2 = X[:, 1]
        d3 = X[:, 2]
        b = X[:, 3]
        L = X[:, 4]
        rho1 = 100
        rho2 = 2770
        rho3 = 7780
        E1 = 1.6
        E2 = 70
        c1 = 500
        c2 = 1500
        c3 = 800
        mu = 2 * b * (rho1 * d1 + rho2 * (d2 - d1) + rho3 * (d3 - d2))
        EI = (2 * b / 3) * (E1 * d1**3 + E2 * (d2**3 - d1**3) + rho3 * (d3 - d2))
        f[0] = (-np.pi) / (2 * L) ** 2 * (np.abs(EI / mu)) ** 0.5
        f[1] = 2 * b * L * (c1 * d1 + c2 * (d2 - d1) + c3 * (d3 - d2))
        g[0] = mu * L - 2800
        g[1] = d1 - d2
        g[2] = d2 - d1 - 0.15
        g[3] = d2 - d3
        g[4] = d3 - d2 - 0.01
        return np.column_stack(f), np.column_stack(g), None


class RCM3(_RCM):
    number = 3

    def _evaluate(self, X):
        f, g, _h = [None] * 2, [None] * 3, [None] * 0
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        f[0] = x1 * (16 + x3**2) ** (0.5) + x2 * (1 + x3**2) ** (0.5)
        f[1] = (20 * (16 + x3**2) ** (0.5)) / (x3 * x1)
        g[0] = f[0] - 0.1
        g[1] = f[1] - 1e5
        g[2] = (80 * (1 + x3**2) ** (0.5)) / (x3 * x2) - 1e5
        return np.column_stack(f), np.column_stack(g), None


class RCM4(_RCM):
    number = 4

    def _evaluate(self, X):
        f, g, _h = [None] * 2, [None] * 4, [None] * 0
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        P = 6000
        L = 14
        E = 30e6
        tmax = 13600
        sigmax = 30000
        G = 12e6
        Pc = (
            (4.013 * E * ((x3**2 + x4**6) / 36) ** 0.5)
            / (L**2)
            * (1 - x3 / (2 * L) * (E / (4 * G)) ** (0.5))
        )
        sigma = (6 * P * L) / (x4 * x3**2)
        J = 2 * (np.sqrt(2) * x1 * x2 * (x2**2 / 12 + ((x1 + x3) / 2) ** 2))
        R = np.sqrt(x2**2 / 4 + ((x1 + x3) / 2) ** 2)
        M = P * (L + x2 / 2)
        tho1 = P / (np.sqrt(2) * x1 * x2)
        tho2 = M * R / J
        tho = np.sqrt(tho1**2 + 2 * tho1 * tho2 * x2 / (2 * R) + tho2**2)
        f[0] = 1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (14 + x2)
        f[1] = (4 * P * L**3) / (E * x4 * x3**3)
        g[0] = tho - tmax
        g[1] = sigma - sigmax
        g[2] = x1 - x4
        g[3] = P - Pc
        return np.column_stack(f), np.column_stack(g), None


class RCM5(_RCM):
    number = 5

    def _evaluate(self, X):
        f, g, _h = [None] * 2, [None] * 4, [None] * 0
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        f[0] = 4.9e-5 * (x2**2 - x1**2) * (x4 - 1)
        f[1] = 9.82e6 * (x2**2 - x1**2) / (x3 * x4 * (x2**3 - x1**3))
        g[0] = 20 - (x2 - x1)
        g[1] = x3 / (3.14 * (x2**2 - x1**2)) - 0.4
        g[2] = 2.22e-3 * x3 * (x2**3 - x1**3) / (x2**2 - x1**2) ** 2 - 1
        g[3] = 900 - 2.66e-2 * x3 * x4 * (x2**3 - x1**3) / (x2**2 - x1**2)
        return np.column_stack(f), np.column_stack(g), None


class RCM8(_RCM):
    number = 8

    def _evaluate(self, X):
        f, g, _h = [None] * 3, [None] * 9, [None] * 0
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]
        VMBP = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        VFD = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        f[0] = (
            1.98 + 4.9 * x1 * 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 1e-5 * x6 + 2.73 * x7
        )
        f[1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        f[2] = 0.5 * (VMBP + VFD)
        g[0] = -1 + 1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3
        g[1] = (
            -0.32
            + 0.261
            - 0.0159 * x1 * x2
            - 0.06486 * x1
            - 0.019 * x2 * x7
            + 0.0144 * x2 * x5
            + 0.0154464 * x6
        )
        g[2] = -0.32 + 0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2**2
        g[3] = (
            -0.32
            + 0.214
            + 0.00817 * x5
            - 0.045195 * x1
            - 0.0135168 * x1
            + 0.03099 * x2 * x6
            - 0.018 * x2 * x7
            + 0.007176 * x3
            + 0.023232 * x3
            - 0.00364 * x5 * x6
            - 0.018 * x2**2
        )
        g[4] = -32 + 33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728
        g[5] = -32 + 28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7
        g[6] = -32 + 46.36 - 9.9 * x2 - 4.4505 * x1
        g[7] = f[1] - 4
        g[8] = VMBP - 9.9
        return np.column_stack(f), np.column_stack(g), None


class RCM9(_RCM):
    number = 9

    def _evaluate(self, X):
        f, _g, _h = [None] * 2, [None] * 0, [None] * 0
        F = 10
        E = 2e5
        L = 200
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        f[0] = L * (2 * x1 + np.sqrt(2) * x2 + np.sqrt(2) * x3 + x4)
        f[1] = F * L / E * (2 / x1 + 2 * np.sqrt(2) / x2 - 2 * np.sqrt(2) / x3 + 2 / x4)
        return np.column_stack(f), None, None


class RCM10(_RCM):
    number = 10

    def _evaluate(self, X):
        f, g, h = [None] * 2, [None] * 2, [None] * 0
        x1 = X[:, 0]
        x2 = X[:, 1]
        rho = 0.283
        h = 100
        P = 104
        E = 3e7
        rho0 = 2e4
        f[0] = 2 * rho * h * x2 * np.sqrt(1 + x1**2)
        f[1] = rho * h * (1 + x1**2) ** 1.5 * (1 + x1**4) ** 0.5 / (2 * np.sqrt(2) * E * x1**2 * x2)
        g[0] = P * (1 + x1) * (1 + x1**2) ** 0.5 / (2 * np.sqrt(2) * x1 * x2) - rho0
        g[1] = P * (-x1 + 1) * (1 + x1**2) ** 0.5 / (2 * np.sqrt(2) * x1 * x2) - rho0
        return np.column_stack(f), np.column_stack(g), None


class RCM11(_RCM):
    number = 11

    def _evaluate(self, X):
        f, g, _h = [None] * 5, [None] * 7, [None] * 0
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        f[0] = 106780.37 * (x2 + x3) + 61704.67
        f[1] = 3000 * x1
        f[2] = 305700 * 2289 * x2 / np.power(0.06 * 2289, 0.65)
        f[3] = 250 * 2289 * np.exp(-39.75 * x2 + 9.9 * x3 + 2.74)
        f[4] = 25 * (1.39 / (x1 * x2) + 4940 * x3 - 80)
        g[0] = 1 - (0.00139 / (x1 * x2) + 4.94 * x3 - 0.08)
        g[1] = 1 - (0.000306 / (x1 * x2) + 1.082 * x3 - 0.0986)
        g[2] = 50000 - (12.307 / (x1 * x2) + 49408.24 * x3 + 4051.02)
        g[3] = 16000 - (2.098 / (x1 * x2) + 8046.33 * x3 - 696.71)
        g[4] = 10000 - (2.138 / (x1 * x2) + 7883.39 * x3 - 705.04)
        g[5] = 2000 - (0.417 * x1 * x2 + 1721.26 * x3 - 136.54)
        g[6] = 550 - (0.164 / (x1 * x2) + 631.13 * x3 - 54.48)
        g = [-v for v in g]
        return np.column_stack(f), np.column_stack(g), None


class RCM12(_RCM):
    number = 12

    def _evaluate(self, X):
        f, g, _h = [None] * 2, [None] * 1, [None] * 0
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        P = 600
        L = 200
        E = 2e4
        f[0] = 2 * x2 * x4 + x3 * (x1 - 2 * x4)
        f[1] = (
            P
            * L**3
            / (
                48
                * E
                * (x3 * ((x1 - 2 * x4) ** 3) + 2 * x2 * x4 * (4 * x4 * x4 + 3 * x1 * (x1 - 2 * x4)))
                / 12
            )
        )
        g[0] = (
            -16
            + 180000
            * x1
            / (x3 * ((x1 - 2 * x4) ** 3) + 2 * x2 * x4 * (4 * x4 * x4 + 3 * x1 * (x1 - 2 * x4)))
            + 15000 * x2 / ((x1 - 2 * x4) * x3**3 + 2 * x4 * x2**3)
        )
        return np.column_stack(f), np.column_stack(g), None


class RCM14(_RCM):
    number = 14

    def _evaluate(self, X):
        f, g, _h = [None] * 2, [None] * 8, [None] * 0
        Mf = 3
        Ms = 40
        Iz = 55
        n = 250
        Tmax = 15
        s = 1.5
        delta = 0.5
        Vsrmax = 10
        rho = 0.0000078
        pmax = 1
        mu = 0.6
        Lmax = 30
        delR = 20
        Rsr = 2 / 3 * (X[:, 1] ** 3 - X[:, 0] ** 3) / (X[:, 1] ** 2 * X[:, 0] ** 2)
        Vsr = np.pi * Rsr * n / 30
        A = np.pi * (X[:, 1] ** 2 - X[:, 0] ** 2)
        Prz = X[:, 3] / A
        w = np.pi * n / 30
        Mh = (
            2
            / 3
            * mu
            * X[:, 3]
            * X[:, 4]
            * (X[:, 1] ** 3 - X[:, 0] ** 3)
            / (X[:, 1] ** 2 - X[:, 0] ** 2)
        )
        T = Iz * w / (Mh + Mf)
        f[0] = np.pi * (X[:, 1] ** 2 - X[:, 0] ** 2) * X[:, 2] * (X[:, 4] + 1) * rho
        f[1] = T
        g[0] = -X[:, 1] + X[:, 0] + delR
        g[1] = (X[:, 4] + 1) * (X[:, 2] + delta) - Lmax
        g[2] = Prz - pmax
        g[3] = Prz * Vsr - pmax * Vsrmax
        g[4] = Vsr - Vsrmax
        g[5] = T - Tmax
        g[6] = s * Ms - Mh
        g[7] = -T
        return np.column_stack(f), np.column_stack(g), None


class RCM16(_RCM):
    number = 16

    def _evaluate(self, X):
        f, g, _h = [None] * 2, [None] * 2, [None] * 0
        x1 = X[:, 0]
        x2 = X[:, 1]
        P = 1
        E = 207000000
        Sy = 300000
        delta_max = 0.005
        rho = 7800
        f[0] = 0.25 * rho * np.pi * x2 * x1**2
        f[1] = (64 * P * x2**3) / (3 * E * np.pi * x1**4)
        g[0] = -Sy + (32 * P * x2) / (np.pi * x1**3)
        g[1] = -delta_max + (64 * P * x2**3) / (3 * E * np.pi * x1**4)
        return np.column_stack(f), np.column_stack(g), None


class RCM17(_RCM):
    number = 17

    def _evaluate(self, X):
        f, g, _h = [None] * 3, [None] * 9, [None] * 0
        L = X[:, 0]
        B = X[:, 1]
        D = X[:, 2]
        T = X[:, 3]
        V_k = X[:, 4]
        C_B = X[:, 5]
        a = 4977.06 * C_B**2 - 8105.61 * C_B + 4456.51
        b = -10847.2 * C_B**2 + 12817 * C_B - 6960.32
        F_n = 0.5144 / (9.8065 * L) ** 0.5
        P = ((1.025 * L * B * T * C_B) ** (2 / 3) * V_k**3) / (a + b * F_n)
        W_s = 0.034 * L**1.7 * B**0.6 * D**0.4 * C_B**0.5
        W_o = L**0.8 * B**0.6 * D**0.3 * C_B**0.1
        W_m = 0.17 * P**0.9
        ls = W_s + W_o + W_m
        D_wt = 1.025 * L * B * T * C_B - ls
        F_c = 0.19 * 24 * P / 1000 + 0.2
        D_cwt = D_wt - F_c * ((5000 * V_k) / 24 + 5) - 2 * D_wt**0.5
        R_trp = 350 / ((5000 * V_k) / 24 + 2 * (D_cwt / 8000 + 0.5))
        ac = D_cwt * R_trp
        S_d = 5000 * V_k / 24
        C_c = 0.2 * 1.3 * (2000 * W_s**0.85 + 3500 * W_o + 2400 * P**0.8)
        C_r = 40000 * D_wt**0.3
        C_v = (1.05 * 100 * F_c * S_d + 6.3 * D_wt**0.8) * R_trp
        f[0] = (C_c + C_r + C_v) / ac
        f[1] = ls
        f[2] = -ac
        g[0] = L / B - 6
        g[1] = 15 - L / D
        g[2] = 19 - L / T
        g[3] = 0.45 * D_wt**0.31 - T
        g[4] = 0.7 * D + 0.7 - T
        g[5] = 0.32 - F_n
        g[6] = 0.53 * T + ((0.085 * C_B - 0.002) * B**2) / (T * C_B) - (1 + 0.52 * D) - 0.07 * B
        g[7] = D_wt - 3000
        g[8] = 500000 - D_wt
        g = [-v for v in g]
        return np.column_stack(f), np.column_stack(g), None


class RCM18(_RCM):
    number = 18

    def _evaluate(self, X):
        f, g, _h = [None] * 2, [None] * 3, [None] * 0
        hh = X[:, 0]
        w = X[:, 1]
        t = X[:, 2]
        Ea = 14496.5
        Fa = 234.9
        E = (
            -70973.4
            + 958.656 * w
            + 614.173 * hh
            - 3.827 * w * hh
            + 57.023 * w * t
            + 63.274 * hh * t
            - 3.582 * w**2
            - 1.4842 * hh**2
            - 1890.174 * t**2
        )
        F = (
            111.854
            - 20.210 * w
            + 7.560 * hh
            - 0.025 * w * hh
            + 2.731 * w * t
            - 1.479 * hh * t
            + 0.165 * w**2
        )
        f[0] = Ea / E
        f[1] = F / Fa
        g[0] = (hh - 136) * (146 - hh)
        g[1] = (w - 58) * (66 - w)
        g[2] = (t - 1.4) * (2.2 - t)
        g = [-v for v in g]
        return np.column_stack(f), np.column_stack(g), None


class RCM20(_RCM):
    number = 20

    def _evaluate(self, X):
        f, g, h = [None] * 2, [None] * 7, [None] * 0

        def _log10(v):
            return np.log(v) / np.log(10)

        R = X[:, 0]
        Ro = X[:, 1]
        mu = X[:, 2]
        Q = X[:, 3]
        gamma = 0.0307
        C = 0.5
        n = -3.55
        C1 = 10.04
        Ws = 101000
        Pmax = 1000
        delTmax = 50
        hmin = 0.001
        gg = 386.4
        N = 750
        P = (_log10(_log10(8.122 * 1e6 * mu + 0.8)) - C1) / n
        delT = 2 * (10**P - 560)
        Ef = 9336 * Q * gamma * C * delT
        h = (2 * np.pi * N / 60) ** 2 * 2 * np.pi * mu / Ef * (R**4 / 4 - Ro**4 / 4) - 1e-5
        Po = (6 * mu * Q / (np.pi * h**3)) * np.log(R / Ro)
        W = np.pi * Po / 2 * (R**2 - Ro**2) / (np.log(R / Ro) - 1e-5)
        f[0] = (Q * Po / 0.7 + Ef) / 12
        f[1] = gamma / (gg * Po) * (Q / (2 * np.pi * R * h))
        g[0] = Ws - W
        g[1] = Po - Pmax
        g[2] = delT - delTmax
        g[3] = hmin - h
        g[4] = Ro - R
        g[5] = f[1] - 0.001
        g[6] = W / (np.pi * (R**2 - Ro**2) + 1e-5) - 5000
        return np.column_stack(f), np.column_stack(g), None


class RCM21(_RCM):
    number = 21

    def _evaluate(self, X):
        f, g, _h = [None] * 2, [None] * 4, [None] * 0
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        f[0] = (
            1.3667145844797
            - 0.00904459793976106 * x1
            - 0.0016193573938033 * x2
            - 0.00758531275221425 * x3
            - 0.00440727360327102 * x4
            - 0.00572216860791644 * x5
            - 0.00936039926190721 * x6
            + 2.62510221107328 * 10 ** (-6) * (x1**2)
            + 4.92982681358861 * 10 ** (-7) * (x2**2)
            + 2.25524989067108 * 10 ** (-6) * (x3**2)
            + 1.84605439400301 * 10 ** (-6) * (x4**2)
            + 2.17175358243416 * 10 ** (-6) * (x5**2)
            + 3.90158043948054 * 10 ** (-6) * (x6**2)
            + 4.55276994245781 * 10 ** (-7) * x1 * x2
            - 6.37013576290982 * 10 ** (-7) * x1 * x3
            + 8.26736480446359 * 10 ** (-7) * x1 * x4
            + 5.66352809442276 * 10 ** (-8) * x1 * x5
            - 3.20213897443278 * 10 ** (-7) * x1 * x6
            + 1.18015467772812 * 10 ** (-8) * x2 * x3
            + 9.25820391546515 * 10 ** (-8) * x2 * x4
            - 1.05705364119837 * 10 ** (-7) * x2 * x5
            - 4.74797783014687 * 10 ** (-7) * x2 * x6
            - 5.02319867013788 * 10 ** (-7) * x3 * x4
            + 9.54284258085225 * 10 ** (-7) * x3 * x5
            + 1.80533309229454 * 10 ** (-7) * x3 * x6
            - 1.07938022118477 * 10 ** (-6) * x4 * x5
            - 1.81370642220182 * 10 ** (-7) * x4 * x6
            - 2.24238851688047 * 10 ** (-7) * x5 * x6
        )
        f[1] = (
            -1.19896668942683
            + 3.04107017009774 * x1
            + 1.23535701600191 * x2
            + 2.13882039381528 * x3
            + 2.33495178382303 * x4
            + 2.68632494801975 * x5
            + 3.43918953617606 * x6
            - 7.89144544980703 * 10 ** (-4) * (x1**2)
            - 2.06085185698215 * 10 ** (-4) * (x2**2)
            - 7.15269900037858 * 10 ** (-4) * (x3**2)
            - 7.8449237573837 * 10 ** (-4) * (x4**2)
            - 9.31396896237177 * 10 ** (-4) * (x5**2)
            - 1.40826531972195 * 10 ** (-3) * (x6**2)
            - 1.60434988248392 * 10 ** (-4) * x1 * x2
            + 2.0824655419411 * 10 ** (-4) * x1 * x3
            - 3.0530659653553 * 10 ** (-4) * x1 * x4
            - 8.10145973591615 * 10 ** (-5) * x1 * x5
            + 6.94728759651311 * 10 ** (-5) * x1 * x6
            + 1.18015467772812 * 10 ** (-8) * x2 * x3
            + 9.25820391546515 * 10 ** (-8) * x2 * x4
            - 1.05705364119837 * 10 ** (-7) * x2 * x5
            + 1.69935290196781 * 10 ** (-4) * x2 * x6
            + 2.32421829190088 * 10 ** (-5) * x3 * x4
            - 2.0808624041163476 * 10 ** (-4) * x3 * x5
            + 1.75576341867273 * 10 ** (-5) * x3 * x6
            + 2.68422081654044 * 10 ** (-4) * x4 * x5
            + 4.39852066801981 * 10 ** (-5) * x4 * x6
            + 2.96785446021357 * 10 ** (-5) * x5 * x6
        )
        g[0] = f[0] - 5
        g[1] = -f[0]
        g[2] = f[1] - 28
        g[3] = -f[1]
        return np.column_stack(f), np.column_stack(g), None


class RCM22(_RCM):
    number = 22

    def _evaluate(self, X):
        f, g, h = [None] * 2, [None] * 2, [None] * 4
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]
        x8 = X[:, 7]
        x9 = X[:, 8]
        f[0] = -9 * x1 - 15 * x2 + 6 * x3 + 16 * x4
        f[1] = 10 * (x5 + x6)
        g[0] = x9 * x7 + 2 * x5 - 2.5 * x1
        g[1] = x9 * x8 + 2 * x6 - 1.5 * x2
        h[0] = x7 + x8 - x4 - x3
        h[1] = x1 - x5 - x7
        h[2] = x2 - x6 - x8
        h[3] = x9 * x7 + x9 * x8 - 3 * x3 - x4
        return np.column_stack(f), np.column_stack(g), np.column_stack(h)


class RCM23(_RCM):
    number = 23

    def _evaluate(self, X):
        f, g, h = [None] * 2, [None] * 1, [None] * 4
        k1 = 0.09755988
        k2 = 0.99 * k1
        k3 = 0.0391908
        k4 = 0.9 * k3
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        f[0] = -x4
        f[1] = x5 ** (0.5) + x6 ** (0.5)
        g[0] = f[1] - 4
        h[0] = k1 * x5 * x2 + x1 - 1
        h[1] = k3 * x5 * x3 + x3 + x1 - 1
        h[2] = k2 * x6 * x2 - x1 + x2
        h[3] = k4 * x6 * x4 + x2 - x1 + x4 - x3
        return np.column_stack(f), np.column_stack(g), np.column_stack(h)


class RCM24(_RCM):
    number = 24

    def _evaluate(self, X):
        f, _g, h = [None] * 3, [None] * 0, [None] * 6
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]
        x8 = X[:, 7]
        x9 = X[:, 8]
        f[0] = 35 * x1 ** (0.6) + 35 * x2 ** (0.6)
        f[1] = 200 * x1 * x4 - x3
        f[2] = 200 * x2 * x6 - x5
        h[0] = x3 - 1e4 * (x7 - 100)
        h[1] = x5 - 1e4 * (300 - x7)
        h[2] = x3 - 1e4 * (600 - x8)
        h[3] = x5 - 1e4 * (900 - x9)
        h[4] = (
            x4 * np.log(np.abs(x8 - 100) + 1e-6)
            - x4 * np.log(np.abs(600 - x7) + 1e-6)
            - x8
            + x7
            + 500
        )
        h[5] = x6 * np.log(np.abs(x9 - x7) + 1e-6) - x6 * np.log(600) - x9 + x7 + 600
        return np.column_stack(f), None, np.column_stack(h)


class RCM30(_RCM):
    number = 30


class RCM31(_RCM):
    number = 31


class RCM32(_RCM):
    number = 32


class RCM33(_RCM):
    number = 33


class RCM34(_RCM):
    number = 34


class RCM35(_RCM):
    number = 35


class RCM36(_RCM):
    number = 36


class RCM37(_RCM):
    number = 37


class RCM38(_RCM):
    number = 38


class RCM39(_RCM):
    number = 39


class RCM40(_RCM):
    number = 40


class RCM41(_RCM):
    number = 41


class RCM42(_RCM):
    number = 42


class RCM43(_RCM):
    number = 43


class RCM44(_RCM):
    number = 44


class RCM45(_RCM):
    number = 45


class RCM46(_RCM):
    number = 46


class RCM47(_RCM):
    number = 47


class RCM48(_RCM):
    number = 48


class RCM49(_RCM):
    number = 49


class RCM50(_RCM):
    number = 50
