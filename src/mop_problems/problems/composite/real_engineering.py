"""Continuous engineering benchmarks. MIT-adapted reference formulas.

See THIRD_PARTY_NOTICES.md and the per-problem catalogue for provenance.
CRE returns signed residuals; RE retains the published violation objective.
"""

import numpy as np

from ...core import Problem


class RE21(Problem):
    """RE21; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__(
            "RE21",
            4,
            2,
            [1.0, 1.4142135623730951, 1.4142135623730951, 1.0],
            [3.0, 3.0, 3.0, 3.0],
            n_ieq_constr=0,
        )

    def _evaluate(self, X):
        f = [None] * 2
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        F = 10.0
        E = 2.0 * 100000.0
        L = 200.0
        f[0] = L * (2 * x1 + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
        f[1] = F * L / E * (2.0 / x1 + 2.0 * np.sqrt(2.0) / x2 - 2.0 * np.sqrt(2.0) / x3 + 2.0 / x4)
        return (np.column_stack(f), None, None)


class RE24(Problem):
    """RE24; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__("RE24", 2, 2, [0.5, 0.5], [4.0, 50.0], n_ieq_constr=0)

    def _evaluate(self, X):
        f = [None] * 2
        g = [None] * 4
        x1 = X[:, 0]
        x2 = X[:, 1]
        f[0] = x1 + 120 * x2
        E = 700000
        sigma_b_max = 700
        tau_max = 450
        delta_max = 1.5
        sigma_k = E * x1 * x1 / 100
        sigma_b = 4500 / (x1 * x2)
        tau = 1800 / x2
        delta = 56.2 * 10000 / (E * x1 * x2 * x2)
        g[0] = 1 - sigma_b / sigma_b_max
        g[1] = 1 - tau / tau_max
        g[2] = 1 - delta / delta_max
        g[3] = 1 - sigma_b / sigma_k
        g = [np.maximum(-v, 0) for v in g]
        f[1] = g[0] + g[1] + g[2] + g[3]
        return (np.column_stack(f), None, None)


class RE31(Problem):
    """RE31; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__("RE31", 3, 3, [1e-05, 1e-05, 1.0], [100.0, 100.0, 3.0], n_ieq_constr=0)

    def _evaluate(self, X):
        f = [None] * 3
        g = [None] * 3
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        f[0] = x1 * np.sqrt(16.0 + x3 * x3) + x2 * np.sqrt(1.0 + x3 * x3)
        f[1] = 20.0 * np.sqrt(16.0 + x3 * x3) / (x1 * x3)
        g[0] = 0.1 - f[0]
        g[1] = 100000.0 - f[1]
        g[2] = 100000 - 80.0 * np.sqrt(1.0 + x3 * x3) / (x3 * x2)
        g = [np.maximum(-v, 0) for v in g]
        f[2] = g[0] + g[1] + g[2]
        return (np.column_stack(f), None, None)


class RE32(Problem):
    """RE32; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__(
            "RE32", 4, 3, [0.125, 0.1, 0.1, 0.125], [5.0, 10.0, 10.0, 5.0], n_ieq_constr=0
        )

    def _evaluate(self, X):
        f = [None] * 3
        g = [None] * 4
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        P = 6000
        L = 14
        E = 30 * 1000000.0
        G = 12 * 1000000.0
        tauMax = 13600
        sigmaMax = 30000
        f[0] = 1.10471 * x1 * x1 * x2 + 0.04811 * x3 * x4 * (14.0 + x2)
        f[1] = 4 * P * L * L * L / (E * x4 * x3 * x3 * x3)
        M = P * (L + x2 / 2)
        tmpVar = x2 * x2 / 4.0 + np.power((x1 + x3) / 2.0, 2)
        R = np.sqrt(tmpVar)
        tmpVar = x2 * x2 / 12.0 + np.power((x1 + x3) / 2.0, 2)
        J = 2 * np.sqrt(2) * x1 * x2 * tmpVar
        tauDashDash = M * R / J
        tauDash = P / (np.sqrt(2) * x1 * x2)
        tmpVar = (
            tauDash * tauDash + 2 * tauDash * tauDashDash * x2 / (2 * R) + tauDashDash * tauDashDash
        )
        tau = np.sqrt(tmpVar)
        sigma = 6 * P * L / (x4 * x3 * x3)
        tmpVar = 4.013 * E * np.sqrt(x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4 / 36.0) / (L * L)
        tmpVar2 = x3 / (2 * L) * np.sqrt(E / (4 * G))
        PC = tmpVar * (1 - tmpVar2)
        g[0] = tauMax - tau
        g[1] = sigmaMax - sigma
        g[2] = x4 - x1
        g[3] = PC - P
        g = [np.maximum(-v, 0) for v in g]
        f[2] = g[0] + g[1] + g[2] + g[3]
        return (np.column_stack(f), None, None)


class RE33(Problem):
    """RE33; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__(
            "RE33", 4, 3, [55.0, 75.0, 1000.0, 11.0], [80.0, 110.0, 3000.0, 20.0], n_ieq_constr=0
        )

    def _evaluate(self, X):
        f = [None] * 3
        g = [None] * 4
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        f[0] = 4.9 * 1e-05 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        f[1] = 9.82 * 1000000.0 * (x2 * x2 - x1 * x1) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))
        g[0] = x2 - x1 - 20.0
        g[1] = 0.4 - x3 / (3.14 * (x2 * x2 - x1 * x1))
        g[2] = 1.0 - 2.22 * 0.001 * x3 * (x2 * x2 * x2 - x1 * x1 * x1) / np.power(
            x2 * x2 - x1 * x1, 2
        )
        g[3] = 2.66 * 0.01 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1) / (x2 * x2 - x1 * x1) - 900.0
        g = [np.maximum(-v, 0) for v in g]
        f[2] = g[0] + g[1] + g[2] + g[3]
        return (np.column_stack(f), None, None)


class RE34(Problem):
    """RE34; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__("RE34", 5, 3, [1, 1, 1, 1, 1], [3, 3, 3, 3, 3], n_ieq_constr=0)

    def _evaluate(self, X):
        f = [None] * 3
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        f[0] = (
            1640.2823
            + 2.3573285 * x1
            + 2.3220035 * x2
            + 4.5688768 * x3
            + 7.7213633 * x4
            + 4.4559504 * x5
        )
        f[1] = (
            6.5856
            + 1.15 * x1
            - 1.0427 * x2
            + 0.9738 * x3
            + 0.8364 * x4
            - 0.3695 * x1 * x4
            + 0.0861 * x1 * x5
            + 0.3628 * x2 * x4
            - 0.1106 * x1 * x1
            - 0.3437 * x3 * x3
            + 0.1764 * x4 * x4
        )
        f[2] = (
            -0.0551
            + 0.0181 * x1
            + 0.1024 * x2
            + 0.0421 * x3
            - 0.0073 * x1 * x2
            + 0.024 * x2 * x3
            - 0.0118 * x2 * x4
            - 0.0204 * x3 * x4
            - 0.008 * x3 * x5
            - 0.0241 * x2 * x2
            + 0.0109 * x4 * x4
        )
        return (np.column_stack(f), None, None)


class RE37(Problem):
    """RE37; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__("RE37", 4, 3, [0, 0, 0, 0], [1, 1, 1, 1], n_ieq_constr=0)

    def _evaluate(self, X):
        f = [None] * 3
        xAlpha = X[:, 0]
        xHA = X[:, 1]
        xOA = X[:, 2]
        xOPTT = X[:, 3]
        f[0] = (
            0.692
            + 0.477 * xAlpha
            - 0.687 * xHA
            - 0.08 * xOA
            - 0.065 * xOPTT
            - 0.167 * xAlpha * xAlpha
            - 0.0129 * xHA * xAlpha
            + 0.0796 * xHA * xHA
            - 0.0634 * xOA * xAlpha
            - 0.0257 * xOA * xHA
            + 0.0877 * xOA * xOA
            - 0.0521 * xOPTT * xAlpha
            + 0.00156 * xOPTT * xHA
            + 0.00198 * xOPTT * xOA
            + 0.0184 * xOPTT * xOPTT
        )
        f[1] = (
            0.153
            - 0.322 * xAlpha
            + 0.396 * xHA
            + 0.424 * xOA
            + 0.0226 * xOPTT
            + 0.175 * xAlpha * xAlpha
            + 0.0185 * xHA * xAlpha
            - 0.0701 * xHA * xHA
            - 0.251 * xOA * xAlpha
            + 0.179 * xOA * xHA
            + 0.015 * xOA * xOA
            + 0.0134 * xOPTT * xAlpha
            + 0.0296 * xOPTT * xHA
            + 0.0752 * xOPTT * xOA
            + 0.0192 * xOPTT * xOPTT
        )
        f[2] = (
            0.37
            - 0.205 * xAlpha
            + 0.0307 * xHA
            + 0.108 * xOA
            + 1.019 * xOPTT
            - 0.135 * xAlpha * xAlpha
            + 0.0141 * xHA * xAlpha
            + 0.0998 * xHA * xHA
            + 0.208 * xOA * xAlpha
            - 0.0301 * xOA * xHA
            - 0.226 * xOA * xOA
            + 0.353 * xOPTT * xAlpha
            - 0.0497 * xOPTT * xOA
            - 0.423 * xOPTT * xOPTT
            + 0.202 * xHA * xAlpha * xAlpha
            - 0.281 * xOA * xAlpha * xAlpha
            - 0.342 * xHA * xHA * xAlpha
            - 0.245 * xHA * xHA * xOA
            + 0.281 * xOA * xOA * xHA
            - 0.184 * xOPTT * xOPTT * xAlpha
            - 0.281 * xHA * xAlpha * xOA
        )
        return (np.column_stack(f), None, None)


class RE41(Problem):
    """RE41; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__(
            "RE41",
            7,
            4,
            [0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4],
            [1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2],
            n_ieq_constr=0,
        )

    def _evaluate(self, X):
        f = [None] * 4
        g = [None] * 10
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]
        f[0] = (
            1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 1e-05 * x6 + 2.73 * x7
        )
        f[1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        f[2] = 0.5 * (Vmbp + Vfd)
        g[0] = 1 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
        g[1] = 0.32 - (
            0.261
            - 0.0159 * x1 * x2
            - 0.06486 * x1
            - 0.019 * x2 * x7
            + 0.0144 * x3 * x5
            + 0.0154464 * x6
        )
        g[2] = 0.32 - (
            0.214
            + 0.00817 * x5
            - 0.045195 * x1
            - 0.0135168 * x1
            + 0.03099 * x2 * x6
            - 0.018 * x2 * x7
            + 0.007176 * x3
            + 0.023232 * x3
            - 0.00364 * x5 * x6
            - 0.018 * x2 * x2
        )
        g[3] = 0.32 - (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
        g[4] = 32 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
        g[5] = 32 - (33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
        g[6] = 32 - (46.36 - 9.9 * x2 - 4.4505 * x1)
        g[7] = 4 - f[1]
        g[8] = 9.9 - Vmbp
        g[9] = 15.7 - Vfd
        g = [np.maximum(-v, 0) for v in g]
        f[3] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8] + g[9]
        return (np.column_stack(f), None, None)


class RE42(Problem):
    """RE42; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__(
            "RE42",
            6,
            4,
            [150.0, 20.0, 13.0, 10.0, 14.0, 0.63],
            [274.32, 32.31, 25.0, 11.71, 18.0, 0.75],
            n_ieq_constr=0,
        )

    def _evaluate(self, X):
        f = [None] * 4
        constraintFuncs = [None] * 9
        x_L = X[:, 0]
        x_B = X[:, 1]
        x_D = X[:, 2]
        x_T = X[:, 3]
        x_Vk = X[:, 4]
        x_CB = X[:, 5]
        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g = 9.8065
        Fn = V / np.power(g * x_L, 0.5)
        a = 4977.06 * x_CB * x_CB - 8105.61 * x_CB + 4456.51
        b = -10847.2 * x_CB * x_CB + 12817.0 * x_CB - 6960.32
        power = np.power(displacement, 2.0 / 3.0) * np.power(x_Vk, 3.0) / (a + b * Fn)
        outfit_weight = (
            1.0 * np.power(x_L, 0.8) * np.power(x_B, 0.6) * np.power(x_D, 0.3) * np.power(x_CB, 0.1)
        )
        steel_weight = (
            0.034
            * np.power(x_L, 1.7)
            * np.power(x_B, 0.7)
            * np.power(x_D, 0.4)
            * np.power(x_CB, 0.5)
        )
        machinery_weight = 0.17 * np.power(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight
        ship_cost = 1.3 * (
            2000.0 * np.power(steel_weight, 0.85)
            + 3500.0 * outfit_weight
            + 2400.0 * np.power(power, 0.8)
        )
        capital_costs = 0.2 * ship_cost
        DWT = displacement - light_ship_weight
        running_costs = 40000.0 * np.power(DWT, 0.3)
        round_trip_miles = 5000.0
        sea_days = round_trip_miles / 24.0 * x_Vk
        handling_rate = 8000.0
        daily_consumption = 0.19 * power * 24.0 / 1000.0 + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * np.power(DWT, 0.8)
        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * np.power(DWT, 0.5)
        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * (cargo_DWT / handling_rate + 0.5)
        RTPA = 350.0 / (sea_days + port_days)
        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA
        f[0] = annual_costs / annual_cargo
        f[1] = light_ship_weight
        f[2] = -annual_cargo
        constraintFuncs[0] = x_L / x_B - 6.0
        constraintFuncs[1] = -(x_L / x_D) + 15.0
        constraintFuncs[2] = -(x_L / x_T) + 19.0
        constraintFuncs[3] = 0.45 * np.power(DWT, 0.31) - x_T
        constraintFuncs[4] = 0.7 * x_D + 0.7 - x_T
        constraintFuncs[5] = 500000.0 - DWT
        constraintFuncs[6] = DWT - 3000.0
        constraintFuncs[7] = 0.32 - Fn
        KB = 0.53 * x_T
        BMT = (0.085 * x_CB - 0.002) * x_B * x_B / (x_T * x_CB)
        KG = 1.0 + 0.52 * x_D
        constraintFuncs[8] = KB + BMT - KG - 0.07 * x_B
        constraintFuncs = [np.maximum(-v, 0) for v in constraintFuncs]
        f[3] = (
            constraintFuncs[0]
            + constraintFuncs[1]
            + constraintFuncs[2]
            + constraintFuncs[3]
            + constraintFuncs[4]
            + constraintFuncs[5]
            + constraintFuncs[6]
            + constraintFuncs[7]
            + constraintFuncs[8]
        )
        return (np.column_stack(f), None, None)


class RE61(Problem):
    """RE61; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__("RE61", 3, 6, [0.01, 0.01, 0.01], [0.45, 0.1, 0.1], n_ieq_constr=0)

    def _evaluate(self, X):
        f = [None] * 6
        g = [None] * 7
        f[0] = 106780.37 * (X[:, 1] + X[:, 2]) + 61704.67
        f[1] = 3000 * X[:, 0]
        f[2] = 305700 * 2289 * X[:, 1] / np.power(0.06 * 2289, 0.65)
        f[3] = 250 * 2289 * np.exp(-39.75 * X[:, 1] + 9.9 * X[:, 2] + 2.74)
        f[4] = 25 * (1.39 / (X[:, 0] * X[:, 1]) + 4940 * X[:, 2] - 80)
        g[0] = 1 - (0.00139 / (X[:, 0] * X[:, 1]) + 4.94 * X[:, 2] - 0.08)
        g[1] = 1 - (0.000306 / (X[:, 0] * X[:, 1]) + 1.082 * X[:, 2] - 0.0986)
        g[2] = 50000 - (12.307 / (X[:, 0] * X[:, 1]) + 49408.24 * X[:, 2] + 4051.02)
        g[3] = 16000 - (2.098 / (X[:, 0] * X[:, 1]) + 8046.33 * X[:, 2] - 696.71)
        g[4] = 10000 - (2.138 / (X[:, 0] * X[:, 1]) + 7883.39 * X[:, 2] - 705.04)
        g[5] = 2000 - (0.417 * X[:, 0] * X[:, 1] + 1721.26 * X[:, 2] - 136.54)
        g[6] = 550 - (0.164 / (X[:, 0] * X[:, 1]) + 631.13 * X[:, 2] - 54.48)
        g = [np.maximum(-v, 0) for v in g]
        f[5] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6]
        return (np.column_stack(f), None, None)


class CRE21(Problem):
    """CRE21; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__("CRE21", 3, 2, [1e-05, 1e-05, 1.0], [100.0, 100.0, 3.0], n_ieq_constr=3)

    def _evaluate(self, X):
        f = [None] * 2
        g = [None] * 3
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        f[0] = x1 * np.sqrt(16.0 + x3 * x3) + x2 * np.sqrt(1.0 + x3 * x3)
        f[1] = 20.0 * np.sqrt(16.0 + x3 * x3) / (x1 * x3)
        g[0] = 0.1 - f[0]
        g[1] = 100000.0 - f[1]
        g[2] = 100000 - 80.0 * np.sqrt(1.0 + x3 * x3) / (x3 * x2)
        return (np.column_stack(f), -np.column_stack(g), None)


class CRE22(Problem):
    """CRE22; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__(
            "CRE22", 4, 2, [0.125, 0.1, 0.1, 0.125], [5.0, 10.0, 10.0, 5.0], n_ieq_constr=4
        )

    def _evaluate(self, X):
        f = [None] * 2
        g = [None] * 4
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        P = 6000
        L = 14
        E = 30 * 1000000.0
        G = 12 * 1000000.0
        tauMax = 13600
        sigmaMax = 30000
        f[0] = 1.10471 * x1 * x1 * x2 + 0.04811 * x3 * x4 * (14.0 + x2)
        f[1] = 4 * P * L * L * L / (E * x4 * x3 * x3 * x3)
        M = P * (L + x2 / 2)
        tmpVar = x2 * x2 / 4.0 + np.power((x1 + x3) / 2.0, 2)
        R = np.sqrt(tmpVar)
        tmpVar = x2 * x2 / 12.0 + np.power((x1 + x3) / 2.0, 2)
        J = 2 * np.sqrt(2) * x1 * x2 * tmpVar
        tauDashDash = M * R / J
        tauDash = P / (np.sqrt(2) * x1 * x2)
        tmpVar = (
            tauDash * tauDash + 2 * tauDash * tauDashDash * x2 / (2 * R) + tauDashDash * tauDashDash
        )
        tau = np.sqrt(tmpVar)
        sigma = 6 * P * L / (x4 * x3 * x3)
        tmpVar = 4.013 * E * np.sqrt(x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4 / 36.0) / (L * L)
        tmpVar2 = x3 / (2 * L) * np.sqrt(E / (4 * G))
        PC = tmpVar * (1 - tmpVar2)
        g[0] = tauMax - tau
        g[1] = sigmaMax - sigma
        g[2] = x4 - x1
        g[3] = PC - P
        return (np.column_stack(f), -np.column_stack(g), None)


class CRE23(Problem):
    """CRE23; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__(
            "CRE23", 4, 2, [55.0, 75.0, 1000.0, 11.0], [80.0, 110.0, 3000.0, 20.0], n_ieq_constr=4
        )

    def _evaluate(self, X):
        f = [None] * 2
        g = [None] * 4
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        f[0] = 4.9 * 1e-05 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        f[1] = 9.82 * 1000000.0 * (x2 * x2 - x1 * x1) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))
        g[0] = x2 - x1 - 20.0
        g[1] = 0.4 - x3 / (3.14 * (x2 * x2 - x1 * x1))
        g[2] = 1.0 - 2.22 * 0.001 * x3 * (x2 * x2 * x2 - x1 * x1 * x1) / np.power(
            x2 * x2 - x1 * x1, 2
        )
        g[3] = 2.66 * 0.01 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1) / (x2 * x2 - x1 * x1) - 900.0
        return (np.column_stack(f), -np.column_stack(g), None)


class CRE31(Problem):
    """CRE31; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__(
            "CRE31",
            7,
            3,
            [0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4],
            [1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2],
            n_ieq_constr=10,
        )

    def _evaluate(self, X):
        f = [None] * 3
        g = [None] * 10
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]
        f[0] = (
            1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 1e-05 * x6 + 2.73 * x7
        )
        f[1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        f[2] = 0.5 * (Vmbp + Vfd)
        g[0] = 1 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
        g[1] = 0.32 - (
            0.261
            - 0.0159 * x1 * x2
            - 0.06486 * x1
            - 0.019 * x2 * x7
            + 0.0144 * x3 * x5
            + 0.0154464 * x6
        )
        g[2] = 0.32 - (
            0.214
            + 0.00817 * x5
            - 0.045195 * x1
            - 0.0135168 * x1
            + 0.03099 * x2 * x6
            - 0.018 * x2 * x7
            + 0.007176 * x3
            + 0.023232 * x3
            - 0.00364 * x5 * x6
            - 0.018 * x2 * x2
        )
        g[3] = 0.32 - (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
        g[4] = 32 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
        g[5] = 32 - (33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
        g[6] = 32 - (46.36 - 9.9 * x2 - 4.4505 * x1)
        g[7] = 4 - f[1]
        g[8] = 9.9 - Vmbp
        g[9] = 15.7 - Vfd
        return (np.column_stack(f), -np.column_stack(g), None)


class CRE32(Problem):
    """CRE32; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__(
            "CRE32",
            6,
            3,
            [150.0, 20.0, 13.0, 10.0, 14.0, 0.63],
            [274.32, 32.31, 25.0, 11.71, 18.0, 0.75],
            n_ieq_constr=9,
        )

    def _evaluate(self, X):
        f = [None] * 3
        constraintFuncs = [None] * 9
        x_L = X[:, 0]
        x_B = X[:, 1]
        x_D = X[:, 2]
        x_T = X[:, 3]
        x_Vk = X[:, 4]
        x_CB = X[:, 5]
        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g = 9.8065
        Fn = V / np.power(g * x_L, 0.5)
        a = 4977.06 * x_CB * x_CB - 8105.61 * x_CB + 4456.51
        b = -10847.2 * x_CB * x_CB + 12817.0 * x_CB - 6960.32
        power = np.power(displacement, 2.0 / 3.0) * np.power(x_Vk, 3.0) / (a + b * Fn)
        outfit_weight = (
            1.0 * np.power(x_L, 0.8) * np.power(x_B, 0.6) * np.power(x_D, 0.3) * np.power(x_CB, 0.1)
        )
        steel_weight = (
            0.034
            * np.power(x_L, 1.7)
            * np.power(x_B, 0.7)
            * np.power(x_D, 0.4)
            * np.power(x_CB, 0.5)
        )
        machinery_weight = 0.17 * np.power(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight
        ship_cost = 1.3 * (
            2000.0 * np.power(steel_weight, 0.85)
            + 3500.0 * outfit_weight
            + 2400.0 * np.power(power, 0.8)
        )
        capital_costs = 0.2 * ship_cost
        DWT = displacement - light_ship_weight
        running_costs = 40000.0 * np.power(DWT, 0.3)
        round_trip_miles = 5000.0
        sea_days = round_trip_miles / 24.0 * x_Vk
        handling_rate = 8000.0
        daily_consumption = 0.19 * power * 24.0 / 1000.0 + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * np.power(DWT, 0.8)
        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * np.power(DWT, 0.5)
        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * (cargo_DWT / handling_rate + 0.5)
        RTPA = 350.0 / (sea_days + port_days)
        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA
        f[0] = annual_costs / annual_cargo
        f[1] = light_ship_weight
        f[2] = -annual_cargo
        constraintFuncs[0] = x_L / x_B - 6.0
        constraintFuncs[1] = -(x_L / x_D) + 15.0
        constraintFuncs[2] = -(x_L / x_T) + 19.0
        constraintFuncs[3] = 0.45 * np.power(DWT, 0.31) - x_T
        constraintFuncs[4] = 0.7 * x_D + 0.7 - x_T
        constraintFuncs[5] = 500000.0 - DWT
        constraintFuncs[6] = DWT - 3000.0
        constraintFuncs[7] = 0.32 - Fn
        KB = 0.53 * x_T
        BMT = (0.085 * x_CB - 0.002) * x_B * x_B / (x_T * x_CB)
        KG = 1.0 + 0.52 * x_D
        constraintFuncs[8] = KB + BMT - KG - 0.07 * x_B
        return (np.column_stack(f), -np.column_stack(constraintFuncs), None)


class CRE51(Problem):
    """CRE51; see its catalogue card for the original application."""

    def __init__(self):
        super().__init__("CRE51", 3, 5, [0.01, 0.01, 0.01], [0.45, 0.1, 0.1], n_ieq_constr=7)

    def _evaluate(self, X):
        f = [None] * 5
        g = [None] * 7
        f[0] = 106780.37 * (X[:, 1] + X[:, 2]) + 61704.67
        f[1] = 3000 * X[:, 0]
        f[2] = 305700 * 2289 * X[:, 1] / np.power(0.06 * 2289, 0.65)
        f[3] = 250 * 2289 * np.exp(-39.75 * X[:, 1] + 9.9 * X[:, 2] + 2.74)
        f[4] = 25 * (1.39 / (X[:, 0] * X[:, 1]) + 4940 * X[:, 2] - 80)
        g[0] = 1 - (0.00139 / (X[:, 0] * X[:, 1]) + 4.94 * X[:, 2] - 0.08)
        g[1] = 1 - (0.000306 / (X[:, 0] * X[:, 1]) + 1.082 * X[:, 2] - 0.0986)
        g[2] = 50000 - (12.307 / (X[:, 0] * X[:, 1]) + 49408.24 * X[:, 2] + 4051.02)
        g[3] = 16000 - (2.098 / (X[:, 0] * X[:, 1]) + 8046.33 * X[:, 2] - 696.71)
        g[4] = 10000 - (2.138 / (X[:, 0] * X[:, 1]) + 7883.39 * X[:, 2] - 705.04)
        g[5] = 2000 - (0.417 * X[:, 0] * X[:, 1] + 1721.26 * X[:, 2] - 136.54)
        g[6] = 550 - (0.164 / (X[:, 0] * X[:, 1]) + 631.13 * X[:, 2] - 54.48)
        return (np.column_stack(f), -np.column_stack(g), None)
