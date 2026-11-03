"""Vectorised simulator of a disaster-resilient space-air-ground integrated network (SAGIN).

Layers
------
* Ground : K users (survivors, responders, IoT sensors) clustered around hotspots, uplink traffic
           with Markov-modulated (bursty) Poisson arrivals.
* Air    : U rotary-wing UAV aerial base stations. Each UAV owns N_RB resource blocks; spectrum is
           fully reused across UAVs, so co-channel users of other UAVs interfere.
* Space  : S LEO satellites on independent passes (spherical-Earth geometry) offering Ka-band
           backhaul. A satellite beam's bandwidth is shared (FDMA) by the UAVs that select it.
* Ground backhaul: one damaged terrestrial BS (gateway) with residual capacity and random failures.

Stochastic link failures
------------------------
* access links : probabilistic LoS / NLoS blockage (temporally correlated) + Rician/Rayleigh fading,
                 outage when SINR < gamma_th;
* satellite    : Gilbert-Elliott ON/OFF failures per UAV-satellite link, rain fade (Weibull + AR(1)),
                 shadowed elevation-dependent Rician fading, visibility windows (handovers);
* gateway      : Markov alive/failed process (aftershocks / power loss), finite range.

Per slot every UAV agent chooses (hybrid action)
  move  (3, continuous)     : horizontal velocity + climb rate
  sched (N_RB, discrete)    : which of its N_cand nearest users get its RBs (index >= N_cand: idle RB)
  power (N_RB, continuous)  : uplink transmit power of each scheduled user  (power allocation)
  bh    (1, discrete)       : backhaul: 0 = hold, 1..S = satellite, S+1 = terrestrial gateway
  bhp   (1, continuous)     : UAV backhaul transmit power

Packets live in age-indexed fluid queues (GU queue -> UAV buffer -> core network). A packet older
than ``deadline`` slots is dropped, which defines the packet delivery ratio (PDR); the age of
information (AoI) of each user is tracked at the emergency control centre.

All arrays carry a leading environment axis E so that many episodes run in lock-step.
"""
from __future__ import annotations

import numpy as np

from .config import EnvConfig

R_EARTH = 6371.0e3
K_DBW = -228.6  # Boltzmann constant [dBW/K/Hz]


# ----------------------------------------------------------------------------- helpers
def db2lin(x):
    return np.power(10.0, np.asarray(x, dtype=np.float64) / 10.0)


def lin2db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def fspl_db(d, fc):
    return 20.0 * np.log10(np.maximum(d, 1.0)) + 20.0 * np.log10(fc) - 147.55


def fifo_serve(x, cap):
    """Serve fluid packets oldest-first.

    x   : (..., D) packets per age bucket (index = age in slots, larger = older)
    cap : (...)    service capacity in packets
    returns served amount per bucket, same shape as x.
    """
    y = x[..., ::-1]
    before = np.cumsum(y, axis=-1) - y
    served = np.minimum(np.clip(cap[..., None] - before, 0.0, None), y)
    return served[..., ::-1]


def propulsion_power(v, vz, c: EnvConfig):
    """Rotary-wing propulsion power [W] at horizontal speed v and climb rate vz (Zeng et al. 2019)."""
    blade = c.P0 * (1.0 + 3.0 * v ** 2 / c.U_tip ** 2)
    induced = c.Pi * np.sqrt(np.maximum(np.sqrt(1.0 + v ** 4 / (4.0 * c.v0 ** 4)) - v ** 2 / (2.0 * c.v0 ** 2), 0.0))
    parasite = 0.5 * c.d0 * c.rho_air * c.rotor_s * c.rotor_A * v ** 3
    climb = c.weight_N * np.maximum(vz, 0.0)
    return blade + induced + parasite + climb


def rician_power(rng, k_lin, shape):
    """|h|^2 of a unit-mean Rician channel with K-factor k_lin (broadcastable)."""
    k_lin = np.broadcast_to(k_lin, shape)
    los = np.sqrt(k_lin / (k_lin + 1.0))
    sc = np.sqrt(1.0 / (2.0 * (k_lin + 1.0)))
    re = los + sc * rng.standard_normal(shape)
    im = sc * rng.standard_normal(shape)
    return re * re + im * im


# ----------------------------------------------------------------------------- environment
class SAGINDisasterEnv:
    """Vectorised constrained Markov game. ``E`` environments advance synchronously."""

    N_COST = 3  # outage, PDR, AoI
    COST_NAMES = ("outage", "pdr", "aoi")

    def __init__(self, cfg: EnvConfig | None = None, n_envs: int = 16, seed: int = 0):
        self.cfg = cfg or EnvConfig()
        c = self.cfg
        self.E = n_envs
        self.U, self.K, self.S, self.NC = c.n_uav, c.n_gu, c.n_sat, c.n_rb
        self.D, self.N = c.deadline, c.n_cand
        self.M = self.S + 2  # backhaul options: hold, S satellites, gateway
        self.rng = np.random.default_rng(seed)

        # constants
        self.noise_rb = db2lin(c.noise_psd_dbm + 10 * np.log10(c.b_rb) + c.nf_uav_db - 30.0)  # W
        self.p_gu_max = db2lin(c.p_gu_max_dbm - 30.0)
        self.p_hover = c.P0 + c.Pi
        self.noise_sat_dbw = K_DBW + 10 * np.log10(c.t_sys_k) + 10 * np.log10(c.b_sat)
        self.noise_gw_dbm = c.noise_psd_dbm + 10 * np.log10(c.b_gw) + c.nf_gw_db
        th = np.deg2rad(c.sat_min_elev_deg)
        rr = R_EARTH / (R_EARTH + c.sat_alt)
        self.psi_max = np.arccos(rr * np.cos(th)) - th
        self.dg_max = R_EARTH * self.psi_max
        self.lam_norm = self.K * c.lam_base * c.load_scale  # nominal arrivals per slot
        self.gamma_th = db2lin(c.gamma_th_db)
        self.gamma_th_bh = db2lin(c.gamma_th_bh_db)
        # index helpers
        self.others = np.array([[j for j in range(self.U) if j != i] for i in range(self.U)], dtype=np.int64)
        self.eye_u = np.eye(self.U, dtype=bool)

        # observation / state layouts (consumed by the networks)
        G = c.heat_grid
        self.layout = dict(
            self_dim=13 + self.U + self.M,
            others_dim=(self.U - 1) * (4 + self.M),
            cand_n=self.N, cand_f=9,
            bh_n=self.S + 1, bh_f=8,
            heat_dim=3 * G * G,
            n_rb=self.NC, n_opt=self.M, n_uav=self.U,
        )
        L = self.layout
        self.obs_dim = (L["self_dim"] + L["others_dim"] + L["cand_n"] * L["cand_f"]
                        + L["bh_n"] * L["bh_f"] + L["heat_dim"])
        self.state_layout = dict(gu_n=self.K, gu_f=10,
                                 rest_dim=self.U * (7 + self.M) + self.S * (5 + self.U) + 4 + 3)
        self.state_dim = self.K * 10 + self.state_layout["rest_dim"]

    # ------------------------------------------------------------------ reset
    def reset(self, severity=None, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c, rng, E, U, K, S, D = self.cfg, self.rng, self.E, self.U, self.K, self.S, self.D
        self.t = 0
        if severity is None:
            self.sev = rng.uniform(c.sev_low, c.sev_high, size=E)
        else:
            self.sev = np.full(E, float(severity)) if np.isscalar(severity) else np.asarray(severity, float)

        # hotspots (rejection sampling for separation)
        Cn = c.n_clusters
        centers = np.zeros((E, Cn, 2))
        for e in range(E):
            pts = []
            tries = 0
            while len(pts) < Cn:
                p = rng.uniform(250.0, c.area - 250.0, size=2)
                tries += 1
                if all(np.linalg.norm(p - q) >= c.cluster_min_sep for q in pts) or tries > 500:
                    pts.append(p)
            centers[e] = np.array(pts)
        self.centers = centers
        n_cl = Cn * c.gu_per_cluster
        cl_id = np.repeat(np.arange(Cn), c.gu_per_cluster)
        gu = np.zeros((E, K, 2))
        gu[:, :n_cl] = centers[:, cl_id] + rng.normal(0.0, c.cluster_sigma, size=(E, n_cl, 2))
        gu[:, n_cl:] = rng.uniform(50.0, c.area - 50.0, size=(E, K - n_cl, 2))
        self.gu = np.clip(gu, 0.0, c.area)
        # scattered users form their own (independent) traffic group index Cn
        self.gu_cluster = np.concatenate([cl_id, np.full(K - n_cl, Cn)])
        self.gu_mobile = rng.random((E, K)) < c.mobile_frac
        ang = rng.uniform(0, 2 * np.pi, size=(E, K))
        self.gu_vel = np.stack([np.cos(ang), np.sin(ang)], -1) * c.gu_speed * self.gu_mobile[..., None]
        self.lam0 = c.lam_base * c.load_scale * rng.uniform(1 - c.lam_spread, 1 + c.lam_spread, size=(E, K))
        self.surge = np.zeros((E, Cn + 1), dtype=bool)

        self.uav = np.zeros((E, U, 3))
        if c.init_mode == "depot":
            # UAVs launched together from a depot (emergency operation centre) near the middle
            depot = rng.uniform(0.3 * c.area, 0.7 * c.area, size=(E, 2))
            ang = np.arange(U) * 2 * np.pi / U + rng.uniform(0, 2 * np.pi, size=(E, 1))
            self.uav[..., 0] = depot[:, :1] + c.depot_radius * np.cos(ang)
            self.uav[..., 1] = depot[:, 1:] + c.depot_radius * np.sin(ang)
        else:
            # UAVs already dispatched over the area (random, well separated)
            for e in range(E):
                pts = []
                while len(pts) < U:
                    p = rng.uniform(0.1 * c.area, 0.9 * c.area, size=2)
                    if all(np.linalg.norm(p - q) >= 200.0 for q in pts):
                        pts.append(p)
                self.uav[e, :, :2] = np.array(pts)
        self.uav[..., 2] = c.h_init
        self.p_prop_last = np.full((E, U), self.p_hover)

        # satellites: stratified times of closest approach so that passes overlap the episode
        span = c.T * c.tau
        t_ca = -60.0 + (span + 120.0) * (np.arange(S)[None] + rng.random((E, S))) / S
        self.t_ca = rng.permuted(t_ca, axis=1)
        self.sat_off = rng.uniform(-c.sat_max_offset, c.sat_max_offset, size=(E, S))
        self.ge_good = np.ones((E, U, S), dtype=bool)
        # rain: Weibull-distributed mean zenith attenuation with AR(1) log-fluctuation
        scale = c.rain_scale0_db + c.rain_scale_sev_db * self.sev
        self.rain_mean = scale * rng.weibull(c.rain_shape, size=E)
        self.rain_x = rng.normal(0, c.rain_sigma, size=E)

        # surviving terrestrial gateway
        self.gw = rng.uniform(150.0, c.area - 150.0, size=(E, 2))
        self.gw_alive = np.ones(E, dtype=bool)

        # queues
        self.Q = np.zeros((E, K, D))
        self.B = np.zeros((E, U, K, D))
        self.aoi = np.zeros((E, K))
        self.acc = np.zeros((E, K))
        self.Q[..., 0] = rng.poisson(self.lam0)
        self.los = rng.random((E, U, K)) < 0.5
        r0, d0, th0 = self.access_geometry()
        self.g_last = self.access_pathloss(d0, th0)[3]   # outdated CSI (pilot measurement of last slot)
        self.iot_last = np.zeros((E, U))                 # interference-over-thermal seen last slot [dB]

        # last-slot feedback used in observations
        self.last_bh = np.zeros((E, U), dtype=np.int64)
        self.last_bh_ok = np.ones((E, U))
        self.n_last = np.zeros((E, self.M))
        self.served_last = np.full((E, K), -1, dtype=np.int64)

        self.ep = {k: np.zeros(E) for k in (
            "reward", "delivered", "dropped", "arrived", "att_acc", "out_acc", "att_bh", "out_bh",
            "aoi_sum", "aoi_viol", "e_uav", "e_gu", "collisions", "conflicts", "bh_sat", "bh_gw",
            "bh_hold", "handovers", "cost0", "cost1", "cost2", "alt", "speed", "idle_rb")}
        self.ep_dlv_k = np.zeros((E, K))
        self.ep_arr_k = np.zeros((E, K))
        self.ep_arr_k += self.Q[..., 0]
        self.ep["arrived"] += self.Q[..., 0].sum(-1)
        return self._observe()

    # ------------------------------------------------------------------ geometry / channels
    def sat_geometry(self, t_sec):
        c = self.cfg
        along = c.sat_vg * (t_sec - self.t_ca)
        dg = np.sqrt(self.sat_off ** 2 + along ** 2)
        psi = dg / R_EARTH
        rr = R_EARTH / (R_EARTH + c.sat_alt)
        elev = np.arctan2(np.cos(psi) - rr, np.sin(psi))
        slant = np.sqrt(R_EARTH ** 2 + (R_EARTH + c.sat_alt) ** 2
                        - 2.0 * R_EARTH * (R_EARTH + c.sat_alt) * np.cos(psi))
        vis = elev >= np.deg2rad(c.sat_min_elev_deg)
        a_max = np.sqrt(np.maximum(self.dg_max ** 2 - self.sat_off ** 2, 0.0))
        remain = np.where(vis, self.t_ca + a_max / c.sat_vg - t_sec, 0.0)
        return elev, slant, vis, np.maximum(remain, 0.0)

    def rain_zenith_db(self):
        return self.rain_mean * np.exp(self.rain_x)

    def sat_snr_db(self, elev, slant, p_bh_dbm):
        """Mean SNR [dB] (no small-scale fading) of every UAV-satellite link over the full beam bandwidth.
        elev, slant: (E,S); p_bh_dbm: scalar or (E,U).  returns (E,U,S)."""
        c = self.cfg
        sin_e = np.maximum(np.sin(elev), np.sin(np.deg2rad(10.0)))
        loss = fspl_db(slant, c.fc_sat) + (c.atm_loss_db + self.rain_zenith_db()[:, None]) / sin_e
        p = np.asarray(p_bh_dbm, dtype=np.float64)
        p = p[..., None] if p.ndim == 2 else p
        out = (p - 30.0 + c.g_uav_sat_dbi + c.g_sat_dbi) - loss[:, None, :] - self.noise_sat_dbw
        return np.array(np.broadcast_to(out, (self.E, self.U, self.S)))

    def gw_geometry(self):
        c = self.cfg
        dxy = self.uav[..., :2] - self.gw[:, None, :]
        r = np.linalg.norm(dxy, axis=-1)
        d3 = np.sqrt(r ** 2 + (self.uav[..., 2] - c.gw_height) ** 2)
        return r, d3

    def gw_snr_db(self, d3, p_bh_dbm):
        c = self.cfg
        return np.asarray(p_bh_dbm) + c.g_gw_db - fspl_db(d3, c.fc_gw) - self.noise_gw_dbm

    def access_geometry(self):
        """Horizontal distance, 3D distance, elevation [deg] between every UAV and user: (E,U,K)."""
        dxy = self.uav[:, :, None, :2] - self.gu[:, None, :, :]
        r = np.sqrt((dxy ** 2).sum(-1))
        h = self.uav[:, :, None, 2]
        d = np.sqrt(r ** 2 + h ** 2)
        theta = np.degrees(np.arctan2(h, np.maximum(r, 1e-3)))
        return r, d, theta

    def access_pathloss(self, d, theta):
        c = self.cfg
        p_los = 1.0 / (1.0 + c.los_a * np.exp(-c.los_b * (theta - c.los_a)))
        phi = 90.0 - theta  # angle off the nadir-pointing boresight [deg]
        ant = c.ant_gmax_dbi - np.minimum(12.0 * (phi / c.ant_phi3db_deg) ** 2, c.ant_sla_db)
        fs = fspl_db(d, c.fc_acc) - ant
        pl_los = fs + c.eta_los_db
        pl_nlos = fs + c.eta_nlos_db + c.eta_nlos_sev_db * self.sev[:, None, None]
        gbar = p_los * db2lin(-pl_los) + (1.0 - p_los) * db2lin(-pl_nlos)
        return p_los, pl_los, pl_nlos, gbar

    # ------------------------------------------------------------------ observation
    def _observe(self):
        c, E, U, K, S, D, N, M = self.cfg, self.E, self.U, self.K, self.S, self.D, self.N, self.M
        t_sec = self.t * c.tau
        r, d, theta = self.access_geometry()
        _, _, _, gbar = self.access_pathloss(d, theta)
        self._gbar_obs = gbar
        snr_db = lin2db(self.p_gu_max * gbar / self.noise_rb)  # (E,U,K)
        # best SNR among the *other* UAVs (interference / handover indicator)
        order = np.argsort(-snr_db, axis=1)
        best_u = order[:, 0]
        best = np.take_along_axis(snr_db, order[:, :1], 1)[:, 0]
        second = np.take_along_axis(snr_db, order[:, 1:2], 1)[:, 0]
        other_best = np.where(best_u[:, None, :] == np.arange(U)[None, :, None], second[:, None], best[:, None])

        qlen = self.Q.sum(-1)
        nz = self.Q > 1e-6
        oldest = np.where(nz.any(-1), (D - np.argmax(nz[..., ::-1], axis=-1)) / D, 0.0)  # (E,K)

        # candidates: N nearest users of every UAV
        cand = np.argsort(r, axis=-1)[..., :N]
        self.cand_idx = cand
        take = lambda a: np.take_along_axis(a, cand, axis=-1)  # noqa: E731
        expand = lambda a: np.broadcast_to(a[:, None, :], (E, U, K))  # noqa: E731
        dx = (np.take_along_axis(expand(self.gu[..., 0]), cand, -1) - self.uav[..., 0:1]) / 1000.0
        dy = (np.take_along_axis(expand(self.gu[..., 1]), cand, -1) - self.uav[..., 1:2]) / 1000.0
        sl = expand(self.served_last)
        claimed = np.take_along_axis(sl, cand, -1)
        claimed_other = ((claimed >= 0) & (claimed != np.arange(U)[None, :, None])).astype(np.float64)
        cand_f = np.stack([
            dx, dy,
            (take(snr_db) - 20.0) / 20.0,
            (take(other_best) - take(snr_db)) / 20.0,
            np.minimum(np.take_along_axis(expand(qlen), cand, -1) / 5.0, 4.0),
            np.minimum(np.take_along_axis(expand(self.aoi), cand, -1) / c.aoi_max, 5.0),
            np.take_along_axis(expand(oldest), cand, -1),
            claimed_other,
            (take(lin2db(self.p_gu_max * self.g_last / self.noise_rb)) - 20.0) / 20.0,
        ], axis=-1)  # (E,U,N,9)

        # backhaul options
        elev, slant, vis, remain = self.sat_geometry(t_sec)
        self._sat_vis_obs = vis
        snr_sat = self.sat_snr_db(elev, slant, c.p_bh_max_dbm)  # (E,U,S)
        n_sat_last = self.n_last[:, 1:S + 1]
        on_it = self.last_bh[..., None] == (np.arange(S)[None, None] + 1)
        n_eff = np.maximum(n_sat_last[:, None, :] + (~on_it), 1.0)
        est_rate_sat = (c.b_sat / n_eff) * np.log2(1.0 + db2lin(snr_sat) * n_eff) * vis[:, None, :]
        sat_f = np.stack([
            np.broadcast_to(vis[:, None, :], (E, U, S)).astype(np.float64),
            np.broadcast_to(np.sin(elev)[:, None, :], (E, U, S)),
            snr_sat / 20.0,
            est_rate_sat / 20e6,
            np.broadcast_to(n_sat_last[:, None, :] / U, (E, U, S)),
            self.ge_good.astype(np.float64),
            np.broadcast_to(np.minimum(remain / (c.T * c.tau), 1.0)[:, None, :], (E, U, S)),
            np.zeros((E, U, S)),
        ], axis=-1)
        rg, d3g = self.gw_geometry()
        in_rng = rg <= c.gw_range
        gw_avail = in_rng & self.gw_alive[:, None]
        snr_gw = self.gw_snr_db(d3g, c.p_bh_max_dbm)
        n_g_last = self.n_last[:, S + 1]
        n_eff_g = np.maximum(n_g_last[:, None] + (self.last_bh != S + 1), 1.0)
        est_rate_gw = np.minimum((c.b_gw / n_eff_g) * np.log2(1 + db2lin(snr_gw) * n_eff_g), c.gw_cap / n_eff_g)
        gw_f = np.stack([
            gw_avail.astype(np.float64),
            np.clip(1.0 - rg / c.gw_range, -1.0, 1.0),
            snr_gw / 40.0,
            est_rate_gw * gw_avail / 20e6,
            np.broadcast_to(n_g_last[:, None] / U, (E, U)),
            np.broadcast_to(self.gw_alive[:, None], (E, U)).astype(np.float64),
            np.ones((E, U)),
            np.ones((E, U)),
        ], axis=-1)[:, :, None, :]
        bh_f = np.concatenate([sat_f, gw_f], axis=2)  # (E,U,S+1,8)
        mask = np.ones((E, U, M), dtype=bool)
        mask[..., 1:S + 1] = vis[:, None, :]
        mask[..., S + 1] = gw_avail
        self.bh_mask = mask
        self._gw_avail_obs = gw_avail

        # demand-direction cues (km): urgency-weighted centroid of the candidate users, and of the
        # demand-map cells in this UAV's Voronoi region (UAV positions are shared over UAV-UAV links)
        G = c.heat_grid
        w_c = np.take_along_axis(expand(qlen + self.aoi / c.aoi_max), cand, -1)
        cen_c = np.stack([(w_c * dx).sum(-1), (w_c * dy).sum(-1)], -1) / np.maximum(w_c.sum(-1, keepdims=True), 1e-6)
        cell = (np.clip((self.gu[..., 0] / c.area * G).astype(int), 0, G - 1) * G
                + np.clip((self.gu[..., 1] / c.area * G).astype(int), 0, G - 1))
        oh = np.eye(G * G)[cell]  # (E,K,G*G)
        dem = np.einsum("ek,ekc->ec", qlen + self.aoi / c.aoi_max, oh)            # (E,G*G)
        g1 = (np.arange(G) + 0.5) * c.area / G
        cc = np.stack(np.meshgrid(g1, g1, indexing="ij"), -1).reshape(-1, 2)       # cell centres
        dcell = np.linalg.norm(self.uav[:, :, None, :2] - cc[None, None], axis=-1)  # (E,U,G*G)
        mine = (np.argmin(dcell, axis=1)[:, None, :] == np.arange(U)[None, :, None])
        wv = mine * dem[:, None, :]
        rel_c = (cc[None, None] - self.uav[:, :, None, :2]) / 1000.0
        cen_v = (wv[..., None] * rel_c).sum(2) / np.maximum(wv.sum(-1, keepdims=True), 1e-6)

        # own state
        buf = self.B.sum((2, 3))
        bnz = self.B.sum(2) > 1e-6
        b_old = np.where(bnz.any(-1), (D - np.argmax(bnz[..., ::-1], axis=-1)) / D, 0.0)
        last_oh = np.eye(M)[self.last_bh]
        self_f = np.concatenate([
            np.stack([
                self.uav[..., 0] / c.area, self.uav[..., 1] / c.area,
                (self.uav[..., 2] - c.h_min) / (c.h_max - c.h_min),
                np.minimum(buf / 20.0, 5.0), b_old,
                np.full((E, U), self.t / c.T), self.last_bh_ok,
                self.p_prop_last / self.p_hover - 1.0,
                self.iot_last / 20.0,
                cen_c[..., 0], cen_c[..., 1], cen_v[..., 0], cen_v[..., 1],
            ], -1),
            np.broadcast_to(np.eye(U)[None], (E, U, U)),
            last_oh,
        ], axis=-1)
        rel = self.uav[:, self.others] - self.uav[:, :, None, :]  # (E,U,U-1,3)
        oth_f = np.concatenate([
            rel[..., :2] / 1000.0, rel[..., 2:3] / 100.0,
            np.minimum(buf[:, self.others] / 20.0, 5.0)[..., None],
            last_oh[:, self.others],
        ], -1).reshape(E, U, -1)

        # coarse demand map (broadcast by the emergency control centre)
        cnt = oh.sum(1)
        heat = np.concatenate([
            np.minimum(np.einsum("ek,ekc->ec", qlen, oh) / 10.0, 5.0),
            np.einsum("ek,ekc->ec", self.aoi, oh) / np.maximum(cnt, 1.0) / c.aoi_max,
            cnt / 10.0,
        ], -1)
        heat = np.broadcast_to(heat[:, None, :], (E, U, heat.shape[-1]))

        obs = np.concatenate([self_f, oth_f, cand_f.reshape(E, U, -1), bh_f.reshape(E, U, -1), heat], -1)

        # global state for the centralised critic (may use privileged information)
        lam_now = self.lam0 * np.where(self.surge[:, self.gu_cluster], c.surge_factor, 1.0)
        gu_s = np.stack([
            self.gu[..., 0] / c.area, self.gu[..., 1] / c.area,
            np.minimum(qlen / 5.0, 4.0), np.minimum(self.aoi / c.aoi_max, 5.0), oldest,
            lam_now / c.lam_base, (snr_db.max(1) - 20.0) / 20.0,
            (self.served_last >= 0).astype(np.float64),
            self.surge[:, self.gu_cluster].astype(np.float64),
            (lin2db(self.p_gu_max * self.g_last.max(1) / self.noise_rb) - 20.0) / 20.0,
        ], -1).reshape(E, -1)
        uav_s = np.concatenate([self_f[..., :3], self_f[..., 3:5], self_f[..., 6:8], last_oh], -1).reshape(E, -1)
        sat_s = np.concatenate([
            np.stack([vis.astype(np.float64), np.sin(elev), snr_sat.mean(1) / 20.0,
                      np.minimum(remain / (c.T * c.tau), 1.0), n_sat_last / U], -1),
            self.ge_good.transpose(0, 2, 1).astype(np.float64),
        ], -1).reshape(E, -1)
        misc = np.stack([self.gw[:, 0] / c.area, self.gw[:, 1] / c.area, self.gw_alive.astype(np.float64),
                         n_g_last / U, np.full(E, self.t / c.T), self.sev, self.rain_zenith_db() / 5.0], -1)
        state = np.concatenate([gu_s, uav_s, sat_s, misc], -1)
        return dict(obs=obs.astype(np.float32), mask=mask, state=state.astype(np.float32))

    # ------------------------------------------------------------------ step
    def step(self, act):
        c, rng, E, U, K, S, D, NC, M = self.cfg, self.rng, self.E, self.U, self.K, self.S, self.D, self.NC, self.M
        ar_e = np.arange(E)[:, None, None]
        ar_u = np.arange(U)[None, :, None]

        # ---------------- 1. UAV kinematics with a collision-avoidance safety layer
        mv = np.clip(np.asarray(act["move"], dtype=np.float64), -1.0, 1.0)
        vxy = mv[..., :2] * c.v_max
        sp = np.linalg.norm(vxy, axis=-1, keepdims=True)
        vxy = vxy * np.minimum(1.0, c.v_max / np.maximum(sp, 1e-9))
        new = self.uav.copy()
        new[..., :2] = np.clip(new[..., :2] + vxy * c.tau, 0.0, c.area)
        new[..., 2] = np.clip(new[..., 2] + mv[..., 2] * c.vz_max * c.tau, c.h_min, c.h_max)
        coll_u = np.zeros((E, U))
        for _ in range(2):
            dist = np.linalg.norm(new[:, :, None, :] - new[:, None, :, :], axis=-1) + self.eye_u * 1e9
            bad = (dist < c.d_min).any(-1)
            coll_u += bad
            new = np.where(bad[..., None], self.uav, new)
        n_coll = coll_u.sum(-1)
        vel = (new - self.uav) / c.tau
        p_prop = propulsion_power(np.linalg.norm(vel[..., :2], axis=-1), vel[..., 2], c)
        self.uav = new
        self.p_prop_last = p_prop

        # ---------------- 2. ground mobility and traffic regime
        mob = self.gu_mobile[..., None]
        turn = rng.normal(0, 0.3, size=(E, K))
        cs, sn = np.cos(turn), np.sin(turn)
        vx, vy = self.gu_vel[..., 0], self.gu_vel[..., 1]
        self.gu_vel = np.stack([cs * vx - sn * vy, sn * vx + cs * vy], -1)
        pos = self.gu + self.gu_vel * c.tau * mob
        out_lo, out_hi = pos < 0, pos > c.area
        self.gu_vel = np.where(out_lo | out_hi, -self.gu_vel, self.gu_vel)
        self.gu = np.clip(pos, 0.0, c.area)
        p_up = c.p_surge0 + c.p_surge_sev * self.sev[:, None]
        flip = rng.random(self.surge.shape)
        self.surge = np.where(self.surge, flip >= c.p_calm, flip < p_up)

        # ---------------- 3. link-state evolution (stochastic failures)
        t_sec = self.t * c.tau
        elev, slant, vis, _ = self.sat_geometry(t_sec)
        sin_e = np.maximum(np.sin(elev), np.sin(np.deg2rad(5.0)))
        p_gb = (c.ge_pgb0 + c.ge_pgb_sev * self.sev)[:, None, None] * np.clip(np.sin(np.pi / 4) / sin_e, 0.5, 2.0)[:, None, :]
        u_ge = rng.random((E, U, S))
        self.ge_good = np.where(self.ge_good, u_ge >= p_gb, u_ge < c.ge_pbg)
        p_fail = c.gw_pfail0 + c.gw_pfail_sev * self.sev
        u_gw = rng.random(E)
        self.gw_alive = np.where(self.gw_alive, u_gw >= p_fail, u_gw < c.gw_prepair)
        self.rain_x = c.rain_rho * self.rain_x + np.sqrt(1 - c.rain_rho ** 2) * c.rain_sigma * rng.standard_normal(E)

        # ---------------- 4. access channel realisation
        r, d, theta = self.access_geometry()
        p_los, pl_los, pl_nlos, gbar = self.access_pathloss(d, theta)
        resample = rng.random((E, U, K)) >= c.los_persist
        self.los = np.where(resample, rng.random((E, U, K)) < p_los, self.los)
        fad = np.where(self.los, rician_power(rng, db2lin(c.rician_k_db), (E, U, K)), rng.exponential(1.0, (E, U, K)))
        g = np.where(self.los, db2lin(-pl_los), db2lin(-pl_nlos)) * fad  # (E,U,K)

        # ---------------- 5. user association: resolve claims of several UAVs on one user
        sched = np.asarray(act["sched"], dtype=np.int64)
        idle = sched >= self.N                                    # RB left idle (no transmission)
        gsel = np.take_along_axis(self.cand_idx, np.minimum(sched, self.N - 1), axis=-1)  # (E,U,NC) user ids
        has_data = self.Q.sum(-1) > 1e-6
        claim = np.zeros((E, U, K), dtype=bool)
        claim[ar_e, ar_u, gsel] = ~idle
        win = np.argmax(np.where(claim, gbar, -1.0), axis=1)  # (E,K) strongest claimant
        win_at = np.take_along_axis(win, gsel.reshape(E, -1), 1).reshape(E, U, NC)
        is_win = (win_at == np.arange(U)[None, :, None]) & ~idle
        n_conf = (~is_win & ~idle).sum((1, 2)).astype(np.float64)
        data_at = np.take_along_axis(has_data, gsel.reshape(E, -1), 1).reshape(E, U, NC)
        valid = is_win & data_at

        # ---------------- 6. uplink SINR with co-channel interference from other UAV cells
        pa = np.clip(np.asarray(act["power"], dtype=np.float64), -1.0, 1.0)
        p_dbm = c.p_gu_min_dbm + 0.5 * (pa + 1.0) * (c.p_gu_max_dbm - c.p_gu_min_dbm)
        p_w = db2lin(p_dbm - 30.0) * valid
        idx = np.broadcast_to(gsel.reshape(E, 1, U * NC), (E, U, U * NC))
        Gm = np.take_along_axis(g, idx, axis=2).reshape(E, U, U, NC)  # [e, rx uav, tx uav, rb]
        rx = Gm * p_w[:, None, :, :]
        sig = np.einsum("eiic->eic", rx)
        interf = rx.sum(2) - sig
        sinr = sig / (interf + self.noise_rb)
        self.g_last = g
        n_val = valid.sum(-1)
        self.iot_last = np.where(n_val > 0, lin2db(1.0 + (interf * valid).sum(-1) / np.maximum(n_val, 1) / self.noise_rb), 0.0)
        out_acc = valid & (sinr < self.gamma_th)
        succ = valid & ~out_acc
        rate = np.where(succ, c.b_rb * np.log2(1.0 + sinr), 0.0)
        cap_pk = rate * c.tau / c.pkt_bits
        oh = (gsel[..., None] == np.arange(K)) & ~idle[..., None]  # (E,U,NC,K)
        cap_gu = np.einsum("euck,euc->ek", oh, cap_pk)
        e_gu = p_w.sum((1, 2)) * c.tau

        served = fifo_serve(self.Q, cap_gu)  # (E,K,D)
        self.Q -= served
        srv_oh = (win[:, None, :] == np.arange(U)[None, :, None]).astype(np.float64)  # (E,U,K)
        self.B += srv_oh[..., None] * served[:, None]
        ok_user = np.einsum("euck,euc->ek", oh, succ.astype(np.float64)) > 0
        self.served_last = np.where(ok_user, win, -1)

        # ---------------- 7. backhaul: satellite / gateway selection with shared capacity
        bh = np.asarray(act["bh"], dtype=np.int64)
        feas = np.take_along_axis(self.bh_mask, bh[..., None], -1)[..., 0]
        bh = np.where(feas, bh, 0)
        bpa = np.clip(np.asarray(act.get("bhp", np.ones((E, U))), dtype=np.float64), -1.0, 1.0)
        p_bh_dbm = c.p_bh_min_dbm + 0.5 * (bpa + 1.0) * (c.p_bh_max_dbm - c.p_bh_min_dbm)
        buf_tot = self.B.sum((2, 3))
        attempt = (bh > 0) & (buf_tot > 1e-6)
        opt_oh = (bh[..., None] == np.arange(M)) & attempt[..., None]  # (E,U,M)
        n_opt = opt_oh.sum(1).astype(np.float64)  # (E,M)

        snr_sat = db2lin(self.sat_snr_db(elev, slant, p_bh_dbm))  # (E,U,S) mean SNR, full beam
        k_sat = db2lin(c.sat_k_db0 + c.sat_k_db1 * np.sin(np.maximum(elev, 0.0)))[:, None, :]
        snr_sat = snr_sat * rician_power(rng, k_sat, (E, U, S))
        share = np.maximum(n_opt[:, None, 1:S + 1], 1.0)
        snr_sh = snr_sat * share
        ok_sat = self.ge_good & vis[:, None, :] & (snr_sh >= self.gamma_th_bh)
        rate_sat = np.where(ok_sat, (c.b_sat / share) * np.log2(1.0 + snr_sh), 0.0)

        rg, d3g = self.gw_geometry()
        snr_gw = db2lin(self.gw_snr_db(d3g, p_bh_dbm)) * rician_power(rng, 10.0, (E, U))
        ng = np.maximum(n_opt[:, None, S + 1], 1.0)
        ok_gw = self.gw_alive[:, None] & (rg <= c.gw_range) & (snr_gw * ng >= self.gamma_th_bh)
        rate_gw = np.where(ok_gw, np.minimum((c.b_gw / ng) * np.log2(1.0 + snr_gw * ng), c.gw_cap / ng), 0.0)

        rate_all = np.concatenate([np.zeros((E, U, 1)), rate_sat, rate_gw[..., None]], -1)
        bh_rate = np.take_along_axis(rate_all, bh[..., None], -1)[..., 0] * attempt
        out_bh = attempt & (bh_rate <= 0.0)
        e_bh = (db2lin(p_bh_dbm - 30.0) * attempt).sum(-1) * c.tau
        cap_bh = bh_rate * c.tau / c.pkt_bits
        h_age = self.B.sum(2)  # (E,U,D)
        s_age = fifo_serve(h_age, cap_bh)
        frac = np.where(h_age > 1e-12, s_age / np.maximum(h_age, 1e-12), 0.0)
        dlv_ukd = self.B * frac[:, :, None, :]
        self.B -= dlv_ukd
        dlv_kd = dlv_ukd.sum(1)  # (E,K,D)
        dlv_k = dlv_kd.sum(-1)

        # ---------------- 8. age of information at the control centre
        self.aoi += 1.0
        self.acc += dlv_k
        anyd = dlv_kd > 1e-9
        young = np.argmax(anyd, axis=-1).astype(np.float64)
        reset = anyd.any(-1) & (self.acc >= 1.0 - 1e-9)
        self.aoi = np.where(reset, np.minimum(self.aoi, young + 1.0), self.aoi)
        self.acc = np.where(reset, self.acc - np.floor(self.acc + 1e-9), self.acc)
        self.aoi = np.minimum(self.aoi, c.aoi_cap)

        # ---------------- 9. ageing, deadline drops, new arrivals
        self._drop_q = self.Q[..., D - 1].sum(-1)
        self._drop_b = self.B[..., D - 1].sum((1, 2))
        drop_k = self.Q[..., D - 1] + self.B[..., D - 1].sum(1)
        self.Q[..., 1:] = self.Q[..., :-1].copy()
        self.Q[..., 0] = 0.0
        self.B[..., 1:] = self.B[..., :-1].copy()
        self.B[..., 0] = 0.0
        lam = self.lam0 * np.where(self.surge[:, self.gu_cluster], c.surge_factor, 1.0)
        arr = rng.poisson(lam * c.tau).astype(np.float64)
        self.Q[..., 0] = arr

        # ---------------- 10. reward (energy-aware goodput) and reliability costs
        n_dlv, n_drop, n_arr = dlv_k.sum(-1), drop_k.sum(-1), arr.sum(-1)
        n_att = valid.sum((1, 2)) + attempt.sum(-1)
        n_out = out_acc.sum((1, 2)) + out_bh.sum(-1)
        mean_aoi = self.aoi.mean(-1)
        e_uav_n = (p_prop / self.p_hover).mean(-1)
        e_bh_n = e_bh / (U * db2lin(c.p_bh_max_dbm - 30.0) * c.tau)
        e_gu_n = e_gu / (U * NC * self.p_gu_max * c.tau)
        # per-UAV decomposition: team reward = mean over UAVs
        dlv_u = dlv_ukd.sum((2, 3))
        r_agent = (U * dlv_u / self.lam_norm - c.w_uav_energy * p_prop / self.p_hover
                   - c.w_bh_energy * db2lin(p_bh_dbm - 30.0) * attempt / db2lin(c.p_bh_max_dbm - 30.0)
                   - c.w_gu_energy * p_w.sum(-1) / (NC * self.p_gu_max) - c.w_collision * coll_u)
        reward = r_agent.mean(-1)
        m = c.cost_margin  # optional safety margin (training targets tighter than the nominal ones)
        eps, rho, a_th = c.eps_out * (1 - m), c.pdr_min + m * (1 - c.pdr_min), c.aoi_max * (1 - m)
        cost = np.stack([
            (n_out - eps * n_att) / (eps * U * (NC + 1)),
            (rho * n_drop - (1 - rho) * n_dlv) / ((1 - rho) * self.lam_norm),
            (mean_aoi - a_th) / c.aoi_max,
        ], -1)

        # ---------------- 11. bookkeeping
        ho = ((bh != self.last_bh) & (bh > 0) & (self.last_bh > 0)).sum(-1)
        self.last_bh_ok = np.where(attempt, (~out_bh).astype(np.float64), np.where(bh > 0, 1.0, 0.5))
        self.last_bh = bh
        self.n_last = n_opt
        ep = self.ep
        ep["reward"] += reward
        ep["delivered"] += n_dlv
        ep["dropped"] += n_drop
        ep["arrived"] += n_arr
        ep["att_acc"] += valid.sum((1, 2))
        ep["out_acc"] += out_acc.sum((1, 2))
        ep["att_bh"] += attempt.sum(-1)
        ep["out_bh"] += out_bh.sum(-1)
        ep["aoi_sum"] += mean_aoi
        ep["aoi_viol"] += (self.aoi > c.aoi_max).mean(-1)
        ep["e_uav"] += p_prop.sum(-1) * c.tau
        ep["e_gu"] += e_gu
        ep["collisions"] += n_coll
        ep["conflicts"] += n_conf
        ep["bh_sat"] += ((bh >= 1) & (bh <= S)).sum(-1)
        ep["bh_gw"] += (bh == S + 1).sum(-1)
        ep["bh_hold"] += (bh == 0).sum(-1)
        ep["handovers"] += ho
        ep["alt"] += self.uav[..., 2].mean(-1)
        ep["speed"] += np.linalg.norm(vel[..., :2], axis=-1).mean(-1)
        ep["idle_rb"] += idle.mean((1, 2))
        for i in range(3):
            ep[f"cost{i}"] += cost[:, i]
        self.ep_dlv_k += dlv_k
        self.ep_arr_k += arr

        self.t += 1
        done = self.t >= c.T
        info = dict(n_dlv=n_dlv, n_drop=n_drop, n_out=n_out, n_att=n_att, mean_aoi=mean_aoi,
                    bh=bh, bh_rate=bh_rate, out_bh=out_bh, succ=succ, gsel=gsel, p_prop=p_prop,
                    valid=valid, sig=sig, interf=interf, drop_q=self._drop_q, drop_b=self._drop_b,
                    reward_agents=r_agent)
        return self._observe(), reward, cost, done, info

    # ------------------------------------------------------------------ episode summary
    def episode_metrics(self):
        c, ep, T = self.cfg, self.ep, max(self.t, 1)
        done = ep["delivered"] + ep["dropped"]
        pdr_k = self.ep_dlv_k / np.maximum(self.ep_arr_k, 1e-9)
        jain = pdr_k.sum(-1) ** 2 / (self.K * (pdr_k ** 2).sum(-1) + 1e-12)
        return dict(
            reward=ep["reward"],
            goodput_mbps=ep["delivered"] * c.pkt_bits / (T * c.tau) / 1e6,
            pdr=ep["delivered"] / np.maximum(done, 1e-9),
            outage=(ep["out_acc"] + ep["out_bh"]) / np.maximum(ep["att_acc"] + ep["att_bh"], 1e-9),
            outage_acc=ep["out_acc"] / np.maximum(ep["att_acc"], 1e-9),
            outage_bh=ep["out_bh"] / np.maximum(ep["att_bh"], 1e-9),
            aoi=ep["aoi_sum"] / T,
            aoi_viol=ep["aoi_viol"] / T,
            e_uav_kj=ep["e_uav"] / 1e3 / self.U,
            e_gu_j=ep["e_gu"] / self.K,
            collisions=ep["collisions"],
            conflicts=ep["conflicts"] / T,
            frac_sat=ep["bh_sat"] / (T * self.U),
            frac_gw=ep["bh_gw"] / (T * self.U),
            frac_hold=ep["bh_hold"] / (T * self.U),
            handovers=ep["handovers"],
            jain=jain,
            alt_m=ep["alt"] / T, speed_ms=ep["speed"] / T, idle_rb=ep["idle_rb"] / T,
            severity=self.sev.copy(),
            cost_outage=ep["cost0"] / T, cost_pdr=ep["cost1"] / T, cost_aoi=ep["cost2"] / T,
        )
