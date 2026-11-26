"""Original Eq.25 graph with neutral priors, never the released global fit."""
from copy import deepcopy

INITIAL = [1., 1., 0., 2., 1., 1., 1.]
PARAMETER_NAMES = ["C0", "r0", "r1", "lambda", "A0", "s", "q_k"]
BOUNDS = [(0.01, 20.), (0.2, 5.), (-0.1, 5.), (1., 8.), (0.001, 20.), (0.01, 5.), (0.1, 4.)]
RECIPE = {
    "format": "axion-expression-v1", "parameters": INITIAL,
    "pre": [["kappa", "p5 * vw * max(pilot,tiny)**p6"]],
    "post": [
        ["r", "p1 + p2 * vw"],
        ["z", "max(p0 * finf / fanh * max(tau,tiny)**a_exp,tiny)"],
        ["xi_dm", "max(xi_floor**r + z**r,power_floor)**(1/r)"],
        ["xi", "max(xi_floor,xi_dm*max((1-f_bm)**p3+p4*v_factor*g_bm,sum_floor))"],
    ],
}

def certificate(parameters=None):
    result = deepcopy(RECIPE)
    if parameters is not None:
        result["parameters"] = [float(x) for x in parameters]
    return result
