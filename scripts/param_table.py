"""Print the simulation / training parameter tables (Markdown) straight from the config dataclasses,
so the report can never drift from the code:   python scripts/param_table.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sagin.config import AlgoConfig, EnvConfig  # noqa: E402


def tables(c: EnvConfig = None, a: AlgoConfig = None) -> str:
    c, a = c or EnvConfig(), a or AlgoConfig()
    env_rows = [
        ("Area, horizon, slot", f"{c.area / 1e3:.0f} km × {c.area / 1e3:.0f} km; T = {c.T} slots of τ = {c.tau:.0f} s"),
        ("UAVs / users", f"U = {c.n_uav}; K = {c.n_gu} ({c.n_clusters} hotspots × {c.gu_per_cluster} users, σ_c = {c.cluster_sigma:.0f} m, "
                         f"+ {c.n_scattered} scattered; {int(c.mobile_frac * 100)} % mobile at {c.gu_speed} m/s)"),
        ("UAV kinematics", f"h ∈ [{c.h_min:.0f}, {c.h_max:.0f}] m, v_max = {c.v_max:.0f} m/s, v_z,max = {c.vz_max:.0f} m/s, d_min = {c.d_min:.0f} m"),
        ("Propulsion model", f"P0 = {c.P0} W, Pi = {c.Pi} W, U_tip = {c.U_tip:.0f} m/s, v0 = {c.v0} m/s, d0 = {c.d0}, ρ = {c.rho_air} kg/m³, "
                             f"s = {c.rotor_s}, A = {c.rotor_A} m², W = {c.weight_N:.0f} N"),
        ("Access spectrum", f"{c.fc_acc / 1e9:.0f} GHz; N_RB = {c.n_rb} RBs × {c.b_rb / 1e6:.1f} MHz per UAV, full reuse; "
                            f"N0 = {c.noise_psd_dbm:.0f} dBm/Hz, NF = {c.nf_uav_db:.0f} dB"),
        ("User transmit power", f"[{c.p_gu_min_dbm:.0f}, {c.p_gu_max_dbm:.0f}] dBm per scheduled RB"),
        ("Air-to-ground channel", f"P_LoS: a = {c.los_a}, b = {c.los_b}; η_LoS = {c.eta_los_db:.0f} dB, η_NLoS = {c.eta_nlos_db:.0f} + "
                                  f"{c.eta_nlos_sev_db:.0f}σ dB; Rician K = {c.rician_k_db:.0f} dB (LoS) / Rayleigh (NLoS); "
                                  f"LoS-state persistence ρ_l = {c.los_persist}"),
        ("UAV antenna", f"G_max = {c.ant_gmax_dbi:.0f} dBi, φ_3dB = {c.ant_phi3db_deg:.0f}°, SLA = {c.ant_sla_db:.0f} dB, nadir-pointing"),
        ("Access outage threshold", f"γ_th = {c.gamma_th_db:.0f} dB"),
        ("Traffic", f"L_p = {c.pkt_bits / 1e6:.0f} Mbit; λ_k⁰ = {c.lam_base} × U[{1 - c.lam_spread:.1f}, {1 + c.lam_spread:.1f}] pkt/slot; "
                    f"surge ×{c.surge_factor:.0f} with P(on) = {c.p_surge0} + {c.p_surge_sev}σ, P(off) = {c.p_calm}; deadline D = {c.deadline} slots"),
        ("LEO passes", f"S = {c.n_sat}, H = {c.sat_alt / 1e3:.0f} km, ground speed {c.sat_vg / 1e3:.0f} km/s, |c_s| ≤ {c.sat_max_offset / 1e3:.0f} km, "
                       f"θ_min = {c.sat_min_elev_deg:.0f}°"),
        ("Satellite link", f"{c.fc_sat / 1e9:.0f} GHz (Ka), B_s = {c.b_sat / 1e6:.0f} MHz per beam (FDMA-shared), P_bh ∈ [{c.p_bh_min_dbm:.0f}, "
                           f"{c.p_bh_max_dbm:.0f}] dBm, G_UAV = {c.g_uav_sat_dbi:.0f} dBi, G_sat = {c.g_sat_dbi:.0f} dBi, T_sys = {c.t_sys_k:.0f} K, "
                           f"K(θ) = {c.sat_k_db0:.0f} + {c.sat_k_db1:.0f} sin θ dB, γ_th^bh = {c.gamma_th_bh_db:.0f} dB"),
        ("Rain, atmosphere", f"A_z ~ Weibull(k = {c.rain_shape}, scale {c.rain_scale0_db} + {c.rain_scale_sev_db}σ dB) × log-normal AR(1) "
                             f"(ρ = {c.rain_rho}, σ = {c.rain_sigma}); {c.atm_loss_db} dB at zenith; both scaled by 1/sin θ"),
        ("Link failures", f"Gilbert–Elliott: P(up→fail) = ({c.ge_pgb0} + {c.ge_pgb_sev}σ)·clip(sin45°/sin θ, 0.5, 2), P(fail→up) = {c.ge_pbg}"),
        ("Damaged BS (gateway)", f"range {c.gw_range:.0f} m, residual capacity {c.gw_cap / 1e6:.0f} Mbit/s, {c.b_gw / 1e6:.0f} MHz at "
                                 f"{c.fc_gw / 1e9} GHz; P(fail) = {c.gw_pfail0} + {c.gw_pfail_sev}σ, P(repair) = {c.gw_prepair}"),
        ("Disaster severity", f"σ ~ U[{c.sev_low:.0f}, {c.sev_high:.0f}] per training episode; fixed σ in the sweeps"),
        ("Reward weights", f"w_U = {c.w_uav_energy}, w_B = {c.w_bh_energy}, w_G = {c.w_gu_energy}, w_col = {c.w_collision}"),
        ("Reliability targets (evaluation)", f"P_out ≤ {c.eps_out:.2f}, PDR ≥ {c.pdr_min:.2f}, mean AoI ≤ {c.aoi_max:.0f} slots"),
        ("Training targets (safety margin m = 0.1)", f"P_out ≤ {c.eps_out * 0.9:.2f}, PDR ≥ {c.pdr_min + 0.1 * (1 - c.pdr_min):.2f}, "
                                                     f"mean AoI ≤ {c.aoi_max * 0.9:.1f} slots"),
    ]
    algo_rows = [
        ("Rollout", f"{a.n_envs} parallel episodes × {c.T} slots = {a.n_envs * c.T} environment steps per iteration"),
        ("Discount, GAE", f"γ = {a.gamma}, λ_GAE = {a.gae_lambda}"),
        ("PPO", f"clip {a.clip}, {a.epochs} epochs × {a.n_minibatch} minibatches, entropy coefficient {a.ent_coef}, "
                f"gradient-norm clip {a.max_grad_norm}"),
        ("Optimiser", f"Adam, actor lr {a.lr_actor:g}, critic lr {a.lr_critic:g}"),
        ("Actor", f"trunk 2 × {a.hidden} ReLU (LayerNorm input), candidate encoder 2 × 64, pointer heads 64, "
                  f"initial log-std {a.init_logstd}"),
        ("Critic", f"DeepSets user encoder 2 × 64 (mean + max pooling) + 2 × {a.critic_hidden}; U reward heads + 3 cost heads; "
                   "value normalisation, Huber loss"),
        ("Dual update", f"PI control: K_I = {a.lambda_lr}, K_P = {a.lambda_kp}, signal clipped to ±{a.lambda_signal_clip}, "
                        f"λ ∈ [0, {a.lambda_max:.0f}], λ₀ = {a.lambda_init}"),
        ("Warm start", "behaviour cloning of Greedy-Track on 192 episodes (153,600 agent samples), 2000 Adam steps, batch 2048; "
                       f"critic warm-up 5 iterations"),
        ("Validation", f"every {a.val_every} iterations, 16 fixed held-out episodes, deterministic policy"),
    ]
    out = ["| Parameter | Value |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in env_rows]
    out += ["", "| Training hyper-parameter | Value |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in algo_rows]
    return "\n".join(out)


if __name__ == "__main__":
    print(tables())
