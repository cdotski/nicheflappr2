# =====================================================================
# wing_power.jl
#
# Top-level driver that ties together
#   - the afpt port (`AFPT`)  → flight power, V_mp, V_mr, strokeplane
#     optimisation, kinematic optima (kf, φ, T/L, amplitude)
#   - wing geometry         (`WingPlates`)
#   - flapping kinematics   (`WingKinematics`)
#   - wing-element heat balance (`WingHeatBalance`)
#
# Pipeline (`Q_for_mass`):
#   m  ──► AFPT.AfptBird (allometric)
#       ──► AFPT.compute_flapping_power(strokeplane = :opt) at V_mr
#            ↳ V_mp, V_mr, optimal strokeplane, kf, φ, T/L, amplitude
#       ──► WingPlates.build_wing_for_mass
#       ──► WingKinematics.build_kinematics_for_mass
#            with afpt-optimised strokeplane + afpt-optimised amplitude
#       ──► WingHeatBalance.compute_wingbeat_heatbalance
#       ╰► bundled into `BirdScaling`
#
# This module no longer carries a duplicated bird/drag/power model:
# all flight aerodynamics now come from `AFPT` (the in-house port of
# Klein Heerenbrink et al. 2015 / Pennycuick 2008).  The previous
# `:simple` afpt-lite drag decomposition has been removed.
# =====================================================================

include("afpt.jl")
include("wing_heatbalance_2.0.jl")

module WingPower

using ..WingPlates
using ..WingKinematics
using ..WingHeatBalance
using ..WingHeatBalance: PlateConvectionRegime, LaminarPlate, TurbulentPlate, MixedPlate
using ..AFPT
using ..AFPT: AfptBird, build_afpt_bird, compute_flapping_power,
              find_minimum_power_speed, find_maximum_range_speed,
              compute_available_power, compute_flight_performance,
              wing_span_allometry, wing_area_allometry,
              compute_body_frontal_area, estimate_frequency,
              estimate_basal_metabolic_rate
using Unitful
using Printf


# ── ISA defaults (used only when the caller doesn't override) ───────

const G_MS2     = 9.81
const RHO_ISA   = 1.225
const ISA_KVISC = 1.46e-5


# =====================================================================
# BirdScaling — bundle of all structs produced for one bird at one env
# =====================================================================

"""
    BirdScaling

Container for the full pipeline output.  The flight-power state at
the maximum-range speed (`result_mr`) is the NamedTuple returned by
`AFPT.compute_flapping_power`, and contains the strokeplane-optimised
solution (`strokeplane`, `amplitude`, `kf`, `ToverL`, individual power
components, etc.).
"""
struct BirdScaling{WD<:WingDiscretization,
                   K<:FlappingKinematics,
                   A<:AirProperties,
                   WT<:WingTemperatures,
                   M<:Microclimate,
                   R<:NamedTuple,
                   WBH<:WingbeatHeatBalance}
    bird::AfptBird
    wing_disc::WD
    kinematics::K
    air::A
    temperatures::WT
    microclimate::M
    V_mr::Float64                 # m/s
    V_mp::Float64                 # m/s
    P_avail::Float64              # W  (available aerobic muscle power)
    result_mr::R                  # AFPT.compute_flapping_power(...) at V_mr
    heatbalance::WBH
end


# ── Accessors that read from the embedded structs ───────────────────

"Full wingspan [m]."
wingspan(b::BirdScaling) = 2 * b.wing_disc.wing.wing_length

"Total planform area of one modelled (half-) wing [m²]."
wing_planform_area(b::BirdScaling) = b.wing_disc.total_dorsal_area

"Total active surface area [m²] of one modelled wing (dorsal + ventral)."
wing_surface_area(b::BirdScaling) =
    b.wing_disc.total_dorsal_area + b.wing_disc.total_ventral_area

"Mean chord of the modelled wing [m]."
mean_chord(b::BirdScaling) = b.wing_disc.wing.root_chord

"Wingbeat frequency [Hz]."
frequency(b::BirdScaling) = b.kinematics.frequency

"Optimal (afpt) strokeplane angle [°] used by the kinematics."
strokeplane_deg(b::BirdScaling) = b.result_mr.strokeplane

"Mechanical power at V_mr [W]."
P_mech(b::BirdScaling) = b.result_mr.power_total

"Chemical power at V_mr [W]."
P_chem(b::BirdScaling) = b.result_mr.power_chem

Q_mean(b::BirdScaling) = b.heatbalance.Q_conv_mean
Q_max(b::BirdScaling)  = maximum(b.heatbalance.Q_conv_series)
Q_min(b::BirdScaling)  = minimum(b.heatbalance.Q_conv_series)

Q_conv_mean(b::BirdScaling)   = b.heatbalance.Q_conv_mean
Q_solar_mean(b::BirdScaling)  = b.heatbalance.Q_solar_mean
Q_lw_in_mean(b::BirdScaling)  = b.heatbalance.Q_lw_in_mean
Q_lw_out_mean(b::BirdScaling) = b.heatbalance.Q_lw_out_mean
Q_lw_net_mean(b::BirdScaling) = b.heatbalance.Q_lw_net_mean
Q_net_mean(b::BirdScaling)    = b.heatbalance.Q_net_mean

"Heat-flux density q = Q_conv_mean / (dorsal + ventral area)."
q_density(b::BirdScaling) = uconvert(u"W/m^2", Q_mean(b) / wing_surface_area(b))

"Driving temperature difference T_wing − T_air [K] (using element 1)."
ΔT(b::BirdScaling) =
    uconvert(u"K", b.temperatures.T_dorsal[1]) - uconvert(u"K", b.air.T_air)


# =====================================================================
# Q_for_mass — full pipeline
# =====================================================================

"""
    Q_for_mass(m_kg; …) → BirdScaling

Run the full pipeline:

1.  Build an `AFPT.AfptBird` from `m_kg` using afpt-canonical
    allometric span / area / frontal area / frequency / BMR
    (overridable per-field).
2.  Solve V_mp, V_mr via `AFPT.find_*_power_speed`, then evaluate
    `AFPT.compute_flapping_power(bird, V_mr; strokeplane = :opt)`.
    The returned NamedTuple holds the strokeplane-optimised solution
    along with `amplitude`, `kf`, `T/L`, power components, etc.
3.  Build the wing geometry with `WingPlates.build_wing_for_mass`.
4.  Build the flapping kinematics with the afpt-optimised strokeplane
    (unless the caller passes `stroke_plane_deg`) and afpt-optimised
    amplitude (unless the caller passes a different `amp`).
5.  Assign wing temperatures.
6.  Run `WingHeatBalance.compute_wingbeat_heatbalance` with `V_forward
    = V_mr` so the convective load reflects the realised element-wise
    airspeeds during flight at the maximum-range speed.

# Environment / temperatures
- `T_air`, `T_wing`        : ambient and wing surface temperatures (Unitful)
- `altitude`               : altitude (Unitful length); used if `P` is `nothing`
- `P`                      : ambient pressure (Unitful); overrides `altitude`
- `microclimate`           : pre-built `Microclimate`; overrides individual env kwargs
- `temperatures_builder`   : `(wing_disc) -> WingTemperatures` for a non-uniform profile

# Microclimate kwargs (only used if `microclimate === nothing`)
- `sky_temperature`, `ground_temperature`, `relative_humidity`,
  `wind_speed`, `zenith_angle`, `global_radiation`, `diffuse_fraction`, `shade`

# Radiative optical properties
- `absorptivities`, `emissivities`, `view_factors` : HeatExchange struct overrides

# Discretisation
- `n_elements`             : span-wise plate elements (default 10)
- `n_steps`                : time steps per wingbeat (default 40)
- `disc_method`            : "uniform" or "nonuniform" span-wise spacing
- `thickness_factor`       : wing thickness as fraction of mean chord
- `dorsal_active`, `ventral_active` : surface activity flags

# Kinematics
- `amp`                    : stroke half-amplitude — degrees, an `AmpScaling`,
                             or `nothing` (default) to use the
                             afpt-optimised amplitude
                             (`AfptOptAmplitude(result_mr.amplitude)`).
- `stroke_plane_deg`       : stroke-plane tilt [°].  `nothing` (default)
                             means "use the afpt-optimised value
                             (`result_mr.strokeplane`)".  Pass a number
                             to override.
- `freq_method`            : a `FreqScaling` instance (default
                             `StrouhalFreq()` — uses V_mr internally).

# Bird allometry / aero overrides (forwarded to `build_afpt_bird`)
- `type`                   : `:passerine`, `:seabird`, `:bat`, `:other`
- `wing_span`, `wing_area`, `body_frontal_area`
- `wingbeat_frequency`, `basal_metabolic_rate`
- `k_profile_lift`, `CD_body`
- `conversion_efficiency`, `respiration_factor`
- `muscleFraction`, `coef_activeStrain`, `coef_isometricStress`
- `massEmpty`, `massFat`, `massLoad`

# Power-speed search bracket
- `V_lo`, `V_hi`           : bracket passed to the afpt solvers

# Convection model
- `convection_model`       : `PlateConvectionRegime` (default `MixedPlate()`)
                             or `nothing` for HeatExchange's package default.

# Strokeplane / climb (forwarded to AFPT)
- `strokeplane`            : `:opt` (default) or a fixed angle [°]
- `climbAngle`             : level flight = 0 (default)
- `maxPowerAero`           : optional cap on available muscle power [W]
"""
function Q_for_mass(m_kg::Real;
                    # ── environment ─────────────────────────────────
                    T_air::Quantity        = 20.0u"°C",
                    T_wing::Quantity       = 23.0u"°C",
                    altitude::Quantity     = 0.0u"m",
                    P                      = nothing,
                    temperatures_builder   = nothing,
                    # ── microclimate ────────────────────────────────
                    microclimate                  = nothing,
                    sky_temperature::Quantity     = 0.0u"°C",
                    ground_temperature::Quantity  = 25.0u"°C",
                    relative_humidity::Real       = 0.5,
                    wind_speed::Quantity          = 1.0u"m/s",
                    zenith_angle::Quantity        = 30.0u"°",
                    global_radiation::Quantity    = 800.0u"W/m^2",
                    diffuse_fraction::Real        = 0.15,
                    shade::Real                   = 0.0,
                    # ── radiation optics ────────────────────────────
                    absorptivities = WingHeatBalance.default_absorptivities(),
                    emissivities   = WingHeatBalance.default_emissivities(),
                    view_factors   = WingHeatBalance.default_view_factors(),
                    # ── discretisation ──────────────────────────────
                    n_elements::Int        = 10,
                    n_steps::Int           = 40,
                    disc_method::String    = "uniform",
                    thickness_factor::Real = 0.05,
                    dorsal_active::Bool    = true,
                    ventral_active::Bool   = true,
                    # ── kinematics ──────────────────────────────────
                    amp                          = nothing,   # default → AfptOptAmplitude
                    stroke_plane_deg             = nothing,   # default → result_mr.strokeplane
                    freq_method::FreqScaling     = StrouhalFreq(),
                    # ── bird allometry / aero overrides ─────────────
                    type::Symbol                = :other,
                    wing_span                   = nothing,
                    wing_area                   = nothing,
                    body_frontal_area           = nothing,
                    k_profile_lift::Real        = 0.03,
                    CD_body::Real               = 0.2,
                    wingbeat_frequency          = nothing,
                    basal_metabolic_rate        = nothing,
                    conversion_efficiency::Real = 0.23,
                    respiration_factor::Real    = 1.1,
                    muscleFraction::Real        = 0.17,
                    coef_activeStrain::Real     = 0.26,
                    coef_isometricStress::Real  = 400e3,
                    massEmpty                   = nothing,
                    massFat::Real               = 0.0,
                    massLoad::Real              = 0.0,
                    # ── power-speed search ──────────────────────────
                    V_lo::Real             = 2.0,
                    V_hi::Real             = 50.0,
                    # ── convection model ────────────────────────────
                    convection_model::Union{Nothing,PlateConvectionRegime} = MixedPlate(),
                    # ── afpt strokeplane / climb ────────────────────
                    strokeplane            = :opt,
                    climbAngle::Real       = 0.0,
                    maxPowerAero           = nothing)

    # ── Microclimate / air ──────────────────────────────────────────
    micro = microclimate === nothing ?
            Microclimate(; air_temperature      = T_air,
                           sky_temperature      = sky_temperature,
                           ground_temperature   = ground_temperature,
                           relative_humidity    = relative_humidity,
                           wind_speed           = wind_speed,
                           atmospheric_pressure =
                               P === nothing ?
                                   WingHeatBalance.atmospheric_pressure(altitude) : P,
                           zenith_angle         = zenith_angle,
                           global_radiation     = global_radiation,
                           diffuse_fraction     = diffuse_fraction,
                           shade                = shade,
                           altitude             = altitude) :
            microclimate
    air = air_properties(T_air; altitude = altitude, P = micro.atmospheric_pressure)
    ρ   = ustrip(u"kg/m^3", air.ρ)
    ν   = ustrip(u"m^2/s",  air.ν)

    # ── AFPT bird (allometric defaults, override only if requested) ─
    b_used  = wing_span         === nothing ? wing_span_allometry(m_kg)                    : float(wing_span)
    S_used  = wing_area         === nothing ? wing_area_allometry(m_kg)                    : float(wing_area)
    Sb_used = body_frontal_area === nothing ? compute_body_frontal_area(m_kg, type) : float(body_frontal_area)
    f_used  = wingbeat_frequency   === nothing ? estimate_frequency(m_kg, b_used, S_used; ρ = ρ) : float(wingbeat_frequency)
    bmr_used = basal_metabolic_rate === nothing ? estimate_basal_metabolic_rate(m_kg, type) : float(basal_metabolic_rate)
    mE_used  = massEmpty === nothing ? m_kg : float(massEmpty)

    bird = build_afpt_bird(m_kg, b_used;
                           wingArea           = S_used,
                           type               = type,
                           massEmpty          = mE_used,
                           massFat            = massFat,
                           massLoad           = massLoad,
                           bodyFrontalArea    = Sb_used,
                           wingbeatFrequency  = f_used,
                           basalMetabolicRate = bmr_used,
                           muscleFraction     = muscleFraction,
                           coef_profileDragLiftFactor = k_profile_lift,
                           coef_bodyDragCoefficient   = CD_body,
                           coef_conversionEfficiency  = conversion_efficiency,
                           coef_respirationFactor     = respiration_factor,
                           coef_activeStrain          = coef_activeStrain,
                           coef_isometricStress       = coef_isometricStress)

    # ── Flight speeds and power at V_mr ─────────────────────────────
    V_mp = find_minimum_power_speed(bird; V_lo = V_lo, V_hi = V_hi,
                                    ρ = ρ, ν = ν,
                                    strokeplane = strokeplane,
                                    climbAngle  = climbAngle)
    V_mr = find_maximum_range_speed(bird; V_lo = V_lo, V_hi = V_hi,
                                    ρ = ρ, ν = ν,
                                    strokeplane = strokeplane,
                                    climbAngle  = climbAngle)
    result_mr = compute_flapping_power(bird, V_mr;
                                       ρ = ρ, ν = ν,
                                       strokeplane = strokeplane,
                                       climbAngle  = climbAngle)
    P_avail = compute_available_power(bird; maxPowerAero = maxPowerAero)

    # ── Kinematics ──────────────────────────────────────────────────
    # Default to afpt-optimised strokeplane and amplitude unless the
    # caller has provided their own.  This is the single place where
    # the previously-disconnected afpt optimisation is plumbed back
    # into the kinematics pipeline.
    sp_used  = stroke_plane_deg === nothing ? float(result_mr.strokeplane) : float(stroke_plane_deg)
    amp_used = amp === nothing ? AfptOptAmplitude(float(result_mr.amplitude)) : amp

    kin = build_kinematics_for_mass(m_kg, freq_method;
              amp = amp_used, stroke_plane_deg = sp_used,
              V_forward_ms = V_mr)

    # ── Wing discretisation + temperatures + heat balance ──────────
    wd = build_wing_for_mass(m_kg;
              n_elements      = n_elements,
              method          = disc_method,
              thickness_factor = thickness_factor,
              dorsal_active   = dorsal_active,
              ventral_active  = ventral_active)

    wt = temperatures_builder === nothing ?
            uniform_temperature(wd, T_wing) :
            temperatures_builder(wd)

    wbh = compute_wingbeat_heatbalance(kin, wd, wt, micro;
              n_steps        = n_steps,
              V_forward      = V_mr * u"m/s",
              absorptivities = absorptivities,
              emissivities   = emissivities,
              view_factors   = view_factors,
              convection_model = convection_model)

    return BirdScaling(bird, wd, kin, air, wt, micro,
                       Float64(V_mr), Float64(V_mp), Float64(P_avail),
                       result_mr, wbh)
end


"""
    summarize(b::BirdScaling) → NamedTuple

Flat scalar view (SI base units, plain Float64) for plotting.
"""
function summarize(b::BirdScaling)
    A_wing = ustrip(u"m^2", wing_surface_area(b))
    return (
        m_kg      = b.bird.massTotal,
        b_m       = ustrip(u"m",  wingspan(b)),
        S_m2      = ustrip(u"m^2", wing_planform_area(b)),
        chord_m   = ustrip(u"m",  mean_chord(b)),
        f_Hz      = ustrip(u"Hz", frequency(b)),
        V_mr      = b.V_mr,
        V_mp      = b.V_mp,
        P_mech_W  = b.result_mr.power_total,
        P_chem_W  = b.result_mr.power_chem,
        P_avail_W = b.P_avail,
        sp_deg    = b.result_mr.strokeplane,
        amp_deg   = b.result_mr.amplitude,
        kf        = b.result_mr.kf,
        ToverL    = b.result_mr.ToverL,
        Q_mean_W  = ustrip(u"W",  Q_mean(b)),
        Q_max_W   = ustrip(u"W",  Q_max(b)),
        Q_min_W   = ustrip(u"W",  Q_min(b)),
        q_per_m2  = ustrip(u"W",  Q_mean(b)) / A_wing,
        ΔT_K      = ustrip(u"K",  ΔT(b)),
    )
end


export BirdScaling,
       Q_for_mass, summarize,
       PlateConvectionRegime, LaminarPlate, TurbulentPlate, MixedPlate,
       wingspan, wing_planform_area, wing_surface_area, mean_chord,
       frequency, strokeplane_deg, P_mech, P_chem,
       Q_mean, Q_max, Q_min, q_density, ΔT,
       Q_conv_mean, Q_solar_mean, Q_lw_in_mean, Q_lw_out_mean,
       Q_lw_net_mean, Q_net_mean


end # module WingPower
