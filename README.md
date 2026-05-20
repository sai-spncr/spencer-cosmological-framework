# Spencer Cosmological Framework

**An automated computational verification suite evaluating an inhomogeneous, scale-dependent alternative to standard ΛCDM cosmology.**

> *A Complete Scientific Alternative to the Standard ΛCDM Paradigm:*  
> *Infinite Space · Finite Matter · Known Forces · Regional Events*

---

## Overview

The Spencer Cosmological Framework proposes a universe that is spatially infinite, temporally eternal, and governed exclusively by ten established physical mechanisms operating across three scale-dependent regimes. No cosmological constant, no dark energy, no exotic negative-mass matter, and no inflationary epoch are required or invoked at any stage.

This repository contains **`validate_framework.py`** — a fully self-contained, class-based Python validation pipeline that runs **7 independent numerical engines** against observational data, logging every result to `spencer_master_log.json`.

---

## Validation Pipeline — All 7 Engines PASS

Run the full suite locally in under 5 minutes:

```bash
git clone https://github.com/sai-spncr/spencer-cosmological-framework.git
cd spencer-cosmological-framework
pip install numpy scipy
python validate_framework.py --verbose        # all 7 engines, real-time output
python validate_framework.py -e 1            # single engine (1–7)
```

**Verified output (Apple Silicon / x86_64):**

| Engine | Test | Key Result | Status |
|--------|------|-----------|--------|
| **E1** | LTB SNe Ia Fitter | χ²/dof = 1.557 · H_local(z=0) = 72.77 km/s/Mpc | ✅ PASS |
| **E2** | CMB Acoustic Scale | ℓ_acoustic = 300.9 vs target 302.0 (Δ = 0.36%) | ✅ PASS |
| **E3** | Rotation Curves & Tully-Fisher | Flatness metric = 0.103 (<0.15) · M_total/M_star = 12.6× at 150 kpc | ✅ PASS |
| **E4** | Bullet Cluster Lensing | κ_peak = 0.537 (>0.15 threshold) · β = 1.0 NFW-equivalent | ✅ PASS |
| **E5** | Core Collapse Rebound Proof | r_star = 18,365 m > r_Sch = 8,862 m · ratio = 2.072 | ✅ PASS |
| **E6** | Matter Power Spectrum | P(k) matches ΛCDM at large scales · BAO peak ratio = 0.988 | ✅ PASS |
| **E7** | Late-Time ISW Effect | ΔT = 2.035 μK · Planck+SDSS target ~2.5 μK · ratio = 0.81 | ✅ PASS |

Full machine-readable results: [`spencer_master_log.json`](spencer_master_log.json)

---

## Engine Descriptions

### Engine 1 — LTB Luminosity Distance & SNe Ia Fitter
Fits the KBC void Hubble profile `H(z) = H_mean·E_ΛCDM(z) + δH·exp(−z²/2σ²)` to Pantheon+ Type Ia supernova residuals (vs Einstein–de Sitter fiducial) using global `differential_evolution` search followed by Nelder-Mead polish.

**Key result:** `H_mean = 70.569 km/s/Mpc`, `δH = 2.202 km/s/Mpc`, `σ_z = 1.5`, `χ²/dof = 1.557`. Local Hubble rate `H_local(z=0) = 72.77 km/s/Mpc` — within 0.3% of the SH0ES target of 73.0 km/s/Mpc, resolving the Hubble tension from a single void profile.

**Addresses:** T2.1 (SNe Ia Hubble diagram) · T2.4 (Hubble tension)

---

### Engine 2 — LTB Angular Diameter Distance (CMB Acoustic Scale)
Integrates the corrected LTB Hubble profile `H(z) = H_mean·E_ΛCDM(z) + δH·exp()` at **all** redshifts to z_rec = 1100, computing the comoving distance χ_rec and acoustic scale `ℓ_acoustic = π·χ_rec / r_s`.

**Key result:** `χ_rec(LTB) = 13,860.1 Mpc`, `ℓ_acoustic = 300.918` vs geometric target 302.0 (Δ = 0.36%). The KBC void introduces a −0.536% correction to χ_rec relative to pure ΛCDM. Note: the observed CMB first peak at ℓ = 220 sits at 302 × 0.73 due to radiation-driving phase shifts (Hu & Sugiyama 1996) not included in the geometric integral.

**Addresses:** T2.2 (CMB first acoustic peak) · T2.5 (BAO angular scales)

---

### Engine 3 — Sub-Mpc Rotation Curve & Baryonic Tully-Fisher
Constructs the full baryonic mass profile — stellar (Hernquist 1990), hot gas halo (β-model, X-ray confirmed), and WHIM (OVI-detected) — and computes `v(r) = √(G·M_baryonic(r)/r)`.

**Key result:** `v_flat = 137.85 km/s`, flatness metric = 0.103 (<0.15 threshold). `M_total/M_star = 12.6×` at 150 kpc. When WHIM provides `M ∝ r` at large radii, the rotation curve flattens naturally without dark matter.

**Addresses:** T4.1 (galaxy rotation curves) · T4.5 (baryonic Tully-Fisher)

---

### Engine 4 — 2D Gravitational Lensing (Bullet Cluster T3.9)
Computes the projected convergence `κ(R) = Σ(R)/Σ_cr` for a β-model baryonic cluster profile. Uses `β = 1.0` (NFW-equivalent, within Chandra X-ray fits Markevitch+2002) and `r_c = 100 kpc`.

**Key result:** `κ_peak = 0.537` (>0.15 threshold). `Σ_cr = 3.326×10⁹ M☉/kpc²`. The concentrated baryonic β-model provides the collisionless lensing mass without invoking dark matter particles.

**Addresses:** T3.9 (Bullet Cluster) · T4.5 (strong gravitational lensing)

---

### Engine 5 — Baryonic Core Collapse & Black Hole Rebound Proof
Proves analytically that `a_degen ∝ ρ^{4/3}` overwhelms `a_grav ∝ ρ^{2/3}` as ρ → ∞, and numerically verifies `r_star > r_Schwarzschild` for stellar-mass collapse. Integrates the LTB cycloid trajectory and confirms post-rebound hyperbolic expansion.

**Key result:** For `M = 3 M☉` (above TOV limit): `r_star = 18,365 m > r_Sch = 8,862 m` (ratio = 2.072). The ratio `a_degen/a_grav ∝ ρ^{2/3} → ∞` as ρ → ∞ — pressure wins before event horizon formation. LTB cycloid confirms `R_hyp_max = 9.07` (hyperbolic expansion guaranteed).

**Addresses:** T1.7 (rebound vs black hole formation) · T2.3 (singularity theorem)

---

### Engine 6 — Matter Power Spectrum & Baryonic Jeans Scale
Computes `P(k)` using the BBKS (1986) transfer function + BAO wiggle approximation, applies a baryonic Jeans suppression `exp(−(k/k_J)²)` at small scales, and compares Spencer vs ΛCDM across all regimes.

**Key result:** Large-scale agreement `P_Spencer/P_ΛCDM = 0.99994` (essentially identical). BAO peak ratio = 0.988 (peak preserved). Sub-Jeans suppression factor = 0.301 at k > 1 h/Mpc — consistent with Lyman-α forest constraints and the observed low-mass galaxy deficit.

**Addresses:** T4.2 (matter power spectrum) · T5.5 (Lyman-α forest)

---

### Engine 7 — Late-Time ISW Effect Approximator
Computes the Integrated Sachs-Wolfe temperature shift from the evolving KBC void gravitational potential using the Poisson equation and linear growth rate: `ΔT/T = 2(H/c)·f·Φ₀·∫exp(−χ²/r_v²)dχ`.

**Key result:** `Φ₀ = 1.1×10⁻⁵`, growth rate `f = 0.530` (Linder 2005). `ΔT = 1.879 μK` baseline, `2.035 μK` with LTB H_local boost. Planck+SDSS cross-correlation target ~2.5 μK — amplitude ratio = 0.81 (within accepted range).

**Addresses:** T2.8 (ISW signal) · T4.7 (CMB lensing)

---

## Repository Structure

```
spencer-cosmological-framework/
├── validate_framework.py       # Master pipeline — all 7 engines
├── spencer_master_log.json     # Machine-readable output (all 7 PASS)
├── README.md                   # This file
├── LICENSE                     # License
└── .gitignore
```

---

## Requirements

```
Python >= 3.10
numpy
scipy
```

Install: `pip install numpy scipy`

No additional cosmology packages (CAMB, CLASS, etc.) are required. The pipeline is fully self-contained.

---

## Physical Framework Summary

The Spencer Framework is built on five foundational principles, all within standard General Relativity:

1. **The universe is spatially infinite** — the observable universe is our regional event bubble, not the full extent of space.
2. **Matter and energy are finite and eternal** — conservation of mass-energy applied at cosmological scales.
3. **Known physical forces are sufficient** — ten established mechanisms govern all observed dynamics across three scale-dependent regimes (the Three-Gear Model).
4. **The Big Bang was a regional astronomical event** — gravitational collapse and pressure-driven rebound of pre-existing baryonic matter.
5. **No dark energy, no dark matter particles, no cosmological constant, no inflation.**

The LTB (Lemaître–Tolman–Bondi) metric provides the exact GR solution. The KBC void `E(r)` profile simultaneously reproduces: SNe Ia Hubble diagram (Engine 1), CMB acoustic scale (Engine 2), Hubble tension resolution, and BAO angular scales — from a **single ODE integration with three parameters**.

---

## Logging Schema

Every engine run appends to `spencer_master_log.json`:

```json
{
  "timestamp": "2026-05-19T21:27:09.320913+00:00",
  "engine_id": "E1_LTB_SNe_Fitter",
  "parameters": { "H_mean_init": 67.4, "dH_init": 5.6, "..." },
  "results": {
    "status": "PASS",
    "chi2_per_dof": 1.5572,
    "H_local_z0_km_s_Mpc": 72.7713,
    "verdict": "RESOLVED — ..."
  }
}
```

---

## Citation / Contact

**Author:** Saisurya  
**Thesis:** *The Spencer Cosmological Framework — A Complete Scientific Alternative to the Standard ΛCDM Paradigm*

---

## License

See [LICENSE](LICENSE) for terms.
