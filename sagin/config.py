"""System parameters of the disaster-resilient space-air-ground integrated network (SAGIN).

All quantities are in SI units unless the name says otherwise (``_db``, ``_dbm``, ``_deg``).
One time slot lasts ``tau`` seconds; an episode has ``T`` slots.
"""
from dataclasses import dataclass, asdict, field


@dataclass
class EnvConfig:
    # ------------------------------------------------------------------ topology
    area: float = 2000.0            # side of the square disaster area [m]
    n_uav: int = 4                  # UAV aerial base stations (agents)
    n_clusters: int = 4             # user hotspots (shelters / collapsed blocks)
    gu_per_cluster: int = 8
    n_scattered: int = 8            # isolated users outside hotspots
    cluster_sigma: float = 120.0    # spread of a hotspot [m]
    cluster_min_sep: float = 450.0  # min distance between hotspot centres [m]
    mobile_frac: float = 0.25       # fraction of users that move (responders)
    gu_speed: float = 1.5           # mean speed of moving users [m/s]
    T: int = 200                    # slots per episode
    tau: float = 1.0                # slot length [s]

    # ------------------------------------------------------------------ UAVs
    h_min: float = 60.0
    h_max: float = 300.0
    h_init: float = 100.0
    v_max: float = 25.0             # max horizontal speed [m/s]
    vz_max: float = 5.0             # max vertical speed [m/s]
    d_min: float = 30.0             # collision-avoidance distance [m]
    init_mode: str = "random"       # "random": UAVs already dispatched; "depot": launched together
    depot_radius: float = 40.0      # UAVs launched around a depot (emergency operation centre)
    # rotary-wing propulsion model (Zeng et al., TWC 2019)
    P0: float = 79.86
    Pi: float = 88.63
    U_tip: float = 120.0
    v0: float = 4.03
    d0: float = 0.6
    rho_air: float = 1.225
    rotor_s: float = 0.05
    rotor_A: float = 0.503
    weight_N: float = 20.0          # used for climbing power  W * v_z

    # ------------------------------------------------------------------ access link (GU -> UAV, uplink)
    fc_acc: float = 2.0e9
    n_rb: int = 4                   # resource blocks per UAV (full frequency reuse across UAVs)
    b_rb: float = 0.5e6             # bandwidth of one RB [Hz]
    p_gu_max_dbm: float = 23.0
    p_gu_min_dbm: float = 0.0
    noise_psd_dbm: float = -174.0
    nf_uav_db: float = 7.0
    # probabilistic LoS (Al-Hourani, urban) + excess losses
    los_a: float = 9.61
    los_b: float = 0.16
    eta_los_db: float = 1.0
    eta_nlos_db: float = 20.0
    eta_nlos_sev_db: float = 6.0    # extra NLoS loss from debris at full severity
    los_persist: float = 0.8        # temporal correlation of the blockage state
    rician_k_db: float = 10.0       # small-scale fading on LoS access links (NLoS: Rayleigh)
    # down-tilted directional UAV receive antenna (3GPP-style parabolic pattern around nadir)
    ant_gmax_dbi: float = 6.0
    ant_phi3db_deg: float = 60.0
    ant_sla_db: float = 20.0
    gamma_th_db: float = 5.0        # SINR outage threshold of the access link
    n_cand: int = 10                # candidate users a UAV can schedule (its nearest users)

    # ------------------------------------------------------------------ traffic (time-varying, bursty)
    pkt_bits: float = 1.0e6         # status/data packet size [bit]
    lam_base: float = 0.4           # mean packet arrival rate per user [pkt/slot]
    lam_spread: float = 0.5         # per-user rate ~ lam_base * U(1-spread, 1+spread)
    surge_factor: float = 3.0       # traffic multiplier during an emergency surge
    p_surge0: float = 0.01          # P(normal -> surge) per slot at zero severity
    p_surge_sev: float = 0.03       # ... additional at full severity
    p_calm: float = 0.05            # P(surge -> normal)
    deadline: int = 10              # packets older than this many slots are dropped
    aoi_cap: float = 100.0

    # ------------------------------------------------------------------ LEO satellites (backhaul)
    n_sat: int = 3
    sat_alt: float = 550.0e3
    sat_vg: float = 7.0e3           # ground-track speed [m/s]
    sat_min_elev_deg: float = 25.0
    sat_max_offset: float = 900.0e3 # max cross-track offset of a pass [m]
    fc_sat: float = 20.0e9          # Ka band
    b_sat: float = 3.0e6            # bandwidth of a satellite beam, shared by its UAVs (FDMA)
    p_bh_max_dbm: float = 37.0      # UAV backhaul transmit power range
    p_bh_min_dbm: float = 25.0
    g_uav_sat_dbi: float = 20.0
    g_sat_dbi: float = 32.0
    t_sys_k: float = 500.0
    atm_loss_db: float = 0.5        # zenith atmospheric loss (scaled by 1/sin(elev))
    rain_scale0_db: float = 0.5     # Weibull scale of the zenith rain attenuation, zero severity
    rain_scale_sev_db: float = 2.5  # ... additional at full severity
    rain_shape: float = 1.5
    rain_rho: float = 0.97          # AR(1) correlation of the rain process
    rain_sigma: float = 0.4
    sat_k_db0: float = 5.0          # Rician K of the satellite link: K = k0 + k1*sin(elev)
    sat_k_db1: float = 10.0
    # Gilbert-Elliott ON/OFF link failures (blockage, mis-pointing, handover failure)
    ge_pgb0: float = 0.005          # P(good -> failed) at zero severity
    ge_pgb_sev: float = 0.06        # ... additional at full severity
    ge_pbg: float = 0.25            # P(failed -> good)
    gamma_th_bh_db: float = 3.0     # SNR outage threshold of the backhaul

    # ------------------------------------------------------------------ surviving terrestrial gateway (damaged BS)
    gw_range: float = 700.0         # horizontal range of the UAV-BS backhaul [m]
    gw_cap: float = 25.0e6          # residual capacity of the damaged BS fronthaul [bit/s]
    fc_gw: float = 3.5e9
    b_gw: float = 10.0e6
    g_gw_db: float = 15.0           # total antenna gain of the UAV-BS link
    gw_height: float = 25.0
    nf_gw_db: float = 5.0
    gw_pfail0: float = 0.002        # P(alive -> failed) at zero severity (aftershocks, power loss)
    gw_pfail_sev: float = 0.02
    gw_prepair: float = 0.03

    # ------------------------------------------------------------------ disaster severity (domain randomisation)
    sev_low: float = 0.0
    sev_high: float = 1.0
    load_scale: float = 1.0         # multiplies lam_base (used for load sweeps)

    # ------------------------------------------------------------------ objective (reward) weights
    w_uav_energy: float = 1.0       # propulsion power / hover power
    w_bh_energy: float = 0.1        # backhaul tx power / max
    w_gu_energy: float = 0.3        # user tx energy / max
    w_collision: float = 0.2        # safety-layer interventions

    # ------------------------------------------------------------------ reliability constraints
    eps_out: float = 0.10           # outage probability  <= eps_out
    pdr_min: float = 0.80           # packet delivery ratio >= pdr_min
    aoi_max: float = 12.0           # average AoI [slots] <= aoi_max
    cost_margin: float = 0.0        # safety margin m used ONLY in the training costs: eps*(1-m),
                                    # pdr_min + m*(1-pdr_min), aoi_max*(1-m); evaluation uses the nominal targets

    # ------------------------------------------------------------------ observation
    heat_grid: int = 4              # coarse demand map broadcast by the emergency control centre

    @property
    def n_gu(self) -> int:
        return self.n_clusters * self.gu_per_cluster + self.n_scattered

    def to_dict(self):
        return asdict(self)


@dataclass
class AlgoConfig:
    algo: str = "mappo_l"           # mappo_l | mappo | mappo_pen | ippo_l
    n_envs: int = 16
    n_iters: int = 400
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    n_minibatch: int = 4
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    ent_coef: float = 0.003
    max_grad_norm: float = 0.5
    hidden: int = 128
    critic_hidden: int = 256
    init_logstd: float = -1.0       # initial log-std of the Gaussian action heads
    reward_mix: float = 0.0         # alpha: r_i = alpha * r_i(own UAV) + (1-alpha) * r_team
    init_actor: str = ""            # optional behaviour-cloned actor to start from (marl/pretrain.py)
    critic_warmup: int = 0          # iterations with a frozen actor (critic fitted to the initial policy)
    resume: str = ""                # continue a run: restores actor, critic, value-norm, multipliers, counters
    # Lagrangian (PI-controlled projected dual ascent)
    lambda_lr: float = 0.05         # integral gain K_I
    lambda_kp: float = 0.5          # proportional gain K_P on the EMA of the violation
    lambda_init: float = 0.0
    lambda_max: float = 20.0
    lambda_signal_clip: float = 1.0 # bound on the per-iteration constraint signal used by the dual step
    penalty_coef: float = 1.0       # fixed multiplier for mappo_pen
    seed: int = 0
    torch_threads: int = 1
    log_every: int = 1
    val_every: int = 10             # deterministic evaluation on fixed held-out scenarios
    val_seed: int = 4242
    save_every: int = 50
    out_dir: str = "results/run"

    def to_dict(self):
        return asdict(self)
