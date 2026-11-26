"""Frozen independent calibration and the original discrete radius quadrature.

The collision distribution is independent of kappa. Cache cumulative tails;
only floating point summation order differs from summing each masked slice.
"""
import json
import math
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import _geom_compact as geom
import _percolation

DATA = Path(__file__).resolve().parent / "physics_data"


class Physics:
    def __init__(self):
        self.aux = json.loads((DATA / "auxiliary.json").read_text())
        self.gs = json.loads((DATA / "geometry_bank/geom_compact_summary.json").read_text())
        self.vw_grid = np.array(self.gs["vw_values"], dtype=float)
        self.h_ceiling = max(self.gs["H_star_values"])
        self._distributions, self._fallbacks, self._features = {}, {}, {}
        self._load_pilots()

    def _load_pilots(self):
        rows = np.genfromtxt(DATA / "pilot_kappa.csv", delimiter=",", names=True)
        self.pilot_vws = np.array(sorted(set(rows["vw"])), dtype=float)
        self.pilot_hbs = np.array(sorted(set(rows["Hb"])), dtype=float)
        aa = np.full((len(self.pilot_vws), len(self.pilot_hbs)), np.nan)
        bb = np.full_like(aa, np.nan)
        for i, vw in enumerate(self.pilot_vws):
            for j, hb in enumerate(self.pilot_hbs):
                mask = (np.abs(rows["vw"] - vw) < 1e-6) & (np.abs(rows["Hb"] - hb) < 1e-9)
                th, kp = rows["theta0"][mask].astype(float), rows["kappa"][mask].astype(float)
                if len(th) < 6:
                    continue
                order = np.argsort(th)
                xr = np.minimum(np.abs(th[order]) / np.pi, 1 - 1e-12)
                xl = np.log(np.e / np.maximum(1 - xr ** self.aux["kappa_eps"], 1e-12))
                slope, intercept = np.polyfit(np.log(xl), np.log(kp[order]), 1)
                aa[i, j], bb[i, j] = math.exp(intercept), slope
        for arr in (aa, bb):
            for i in range(len(self.pilot_vws)):
                missing = np.isnan(arr[i])
                if missing.any() and (~missing).any():
                    arr[i, missing] = np.interp(self.pilot_hbs[missing], self.pilot_hbs[~missing], arr[i, ~missing])
        self.pilot_a = RegularGridInterpolator((self.pilot_vws, self.pilot_hbs), aa, bounds_error=False, fill_value=None)
        self.pilot_b = RegularGridInterpolator((self.pilot_vws, self.pilot_hbs), bb, bounds_error=False, fill_value=None)

    def features(self, row):
        key = tuple(float(row[k]) for k in ("theta0", "hstar", "beta_over_h", "vw"))
        if key in self._features:
            return dict(self._features[key])
        theta0, hstar, beta_over_h, vw = key
        xr = min(abs(theta0) / math.pi, 1 - 1e-12)
        def log_form(eps, alpha):
            return math.log(math.e / max(1 - xr ** eps, 1e-12)) ** alpha
        tp = float(_percolation.t_perc_RD(hstar, beta_over_h, vw))
        point = [[float(np.clip(vw, self.pilot_vws[0], self.pilot_vws[-1])),
                  float(np.clip(hstar, self.pilot_hbs[0], self.pilot_hbs[-1]))]]
        aa, bb = float(self.pilot_a(point)[0]), float(self.pilot_b(point)[0])
        xl = math.log(math.e / max(1 - xr ** self.aux["kappa_eps"], 1e-12))
        vwg = min(vw, 0.9999)
        result = dict(theta0=abs(theta0), hstar=hstar, beta_over_h=beta_over_h, vw=vw,
                      tp=tp, tau=(2 / 3) * tp,
                      fanh=log_form(self.aux["fanh_eps"], self.aux["fanh_alpha"]),
                      finf=log_form(self.aux["finf_eps"], self.aux["finf_alpha"]),
                      pilot=aa * max(xl, 1e-12) ** bb,
                      gamma_vw=vwg / math.sqrt(max(1 - vwg ** 2, 1e-8)),
                      pi=math.pi, e=math.e, tiny=1e-30, sum_floor=1e-20, power_floor=1e-300, **self.aux)
        self._features[key] = result
        return dict(result)

    def _nearest_vw(self, vw):
        return float(self.vw_grid[int(np.argmin(np.abs(self.vw_grid - vw)))])

    def _support(self, vw, hstar, beta_over_h):
        base_rmax = float(self.gs["rmax_by_vw"][f"{self._nearest_vw(vw):.1f}"])
        base_nr = int(self.gs["N_R"])
        try:
            bg = geom.compute_background_RD(b=beta_over_h, H_star=hstar, v_w=vw,
                P_crit=self.gs["P_crit"], x_min_factor=self.gs["x_min_factor"],
                x_max=self.gs["x_max"], dx=self.gs["dx"])
            dynamic = (vw / (beta_over_h * hstar)) * (float(bg["F_obs"]) - float(bg["F"][0]))
        except Exception:
            return base_rmax, base_nr
        rmax = max(base_rmax, 1.2 * dynamic)
        if rmax <= base_rmax:
            return base_rmax, base_nr
        ratio = max(1., rmax / max(base_rmax, 1e-12))
        nr = min(40000, max(base_nr, int(base_nr * min(ratio, 20.))))
        return float(rmax), int(nr)

    def _distribution(self, vw, hstar, beta_over_h):
        rmax, nr = self._support(vw, hstar, beta_over_h)
        radii = np.linspace(0., rmax, nr, dtype=np.float64)
        x_max = float(self.gs["x_max"])
        for attempt in range(8):
            try:
                pdf, _, _, _, mean, _ = geom.compute_p_Rc_one_loop_with_grid_vectorized(
                    b=beta_over_h, H_star=hstar, Rc_grid=radii, v_w=vw,
                    mphi=self.gs["mphi"], kappa=0., P_crit=self.gs["P_crit"],
                    x_min_factor=self.gs["x_min_factor"], x_max=x_max, dx=self.gs["dx"])
                break
            except RuntimeError as exc:
                if "Transition incomplete on this x-grid" not in str(exc) or attempt == 7:
                    raise
                x_max *= 1.5
        weights = pdf * float(radii[1] - radii[0])
        reciprocal = np.zeros_like(radii)
        reciprocal[1:] = 1 / radii[1:]
        tail_f = np.r_[np.cumsum(weights[::-1])[::-1], 0.]
        tail_g = np.r_[np.cumsum((weights * reciprocal)[::-1])[::-1], 0.]
        return radii, tail_f, tail_g, float(mean)

    def _fallback(self, vw, kappa, beta_over_h):
        nearest = self._nearest_vw(vw)
        if nearest not in self._fallbacks:
            paths = sorted((DATA / "geometry_bank").glob(f"BM_geometry_RD_kappa_*_vw{nearest:.1f}_oneloop.json"))
            payloads = [(float(p.stem.split("_")[4]), json.loads(p.read_text())) for p in paths]
            kappas = np.array(sorted(k for k, _ in payloads))
            hs = np.array(sorted(set.intersection(*[{float(h) for h in pl} for _, pl in payloads])))
            bs = np.array(sorted(set.intersection(*[{float(b) for hp in pl.values() for b in hp} for _, pl in payloads])))
            f = np.full((len(hs), len(kappas), len(bs)), np.nan)
            g = np.full_like(f, np.nan)
            for kap, pl in payloads:
                for hk, hp in pl.items():
                    for bk, cell in hp.items():
                        h, b = float(hk), float(bk)
                        if h in hs and b in bs:
                            at = (int(np.searchsorted(hs, h)), int(np.searchsorted(kappas, kap)), int(np.searchsorted(bs, b)))
                            f[at], g[at] = cell["f_BM"], cell["G_BM"]
            self._fallbacks[nearest] = (hs, kappas, bs,
                RegularGridInterpolator((hs, kappas, bs), f, bounds_error=False, fill_value=None),
                RegularGridInterpolator((hs, kappas, bs), g, bounds_error=False, fill_value=None))
        hs, kappas, bs, ff, gg = self._fallbacks[nearest]
        point = [[max(hs), float(np.clip(kappa, min(kappas), max(kappas))),
                  float(np.clip(beta_over_h, min(bs), max(bs)))]]
        f = float(np.clip(ff(point)[0], 1e-12, 1 - 1e-12))
        g = float(max(gg(point)[0], 0.))
        return f, g, g / max(f, 1e-12)

    def geometry(self, vw, hstar, beta_over_h, kappa):
        if not math.isfinite(kappa) or kappa < 0:
            raise ValueError("kappa must be finite and nonnegative")
        key = (float(vw), min(float(hstar), self.h_ceiling), float(beta_over_h))
        if key not in self._distributions:
            try:
                self._distributions[key] = self._distribution(*key)
            except Exception:
                self._distributions[key] = None
        distribution = self._distributions[key]
        if distribution is None:
            return self._fallback(vw, kappa, beta_over_h)
        radii, tail_f, tail_g, mean = distribution
        at = int(np.searchsorted(radii, kappa / float(self.gs["mphi"]), side="left"))
        return float(np.clip(tail_f[at], 1e-12, 1 - 1e-12)), float(max(tail_g[at], 0.)), mean

    def complete(self, features, kappa):
        f, g, rc = self.geometry(features["vw"], features["hstar"], features["beta_over_h"], kappa)
        ar = 1 + (math.sqrt(3) / 2) * features["hstar"] * rc / max(features["vw"], 1e-9)
        return dict(f_bm=f, g_bm=g, rc_mean=rc, a_ratio=ar,
                    v_factor=features["gamma_vw"] ** self.aux["p_mink"] / ar ** 2)
