# =====================================================================
# wing_power.jl
#
# Bird struct + flight-power calculations following the structure of
# the `afpt` R package (Klein Heerenbrink et al. 2015, derived from
# Pennycuick 2008).  The full flapping-correction lookup table
# (`FLAPPINGMODELCOEFFS` / `fD.ind`, `fP.ind`, … in afpt) requires
# fitted coefficients distributed as `sysdata.rda`; the simplified
# power-curve here uses the same drag decomposition (induced + two
# profile + parasite) without that correction layer, which is what
# afpt itself uses for default Vmp / Vmr bracketing
# (see `internalFunctions.R::.simplifiedPerformance`).
#
# Pipeline:
#   m  ──► allometric `Bird`
#       ──► wing geometry + kinematics + temperatures + air
#       ──► flapping_power, V_mp, V_mr
#       ──► convective heat loss
#       ╰► bundled into `BirdScaling`
# =====================================================================

include("wing_convection_2.0.jl")

module WingPower

using ..WingPlates
using ..WingKinematics
using ..WingConvection
using Unitful
using Printf


# ── Constants ───────────────────────────────────────────────────────

const G_MS2     = 9.81
const RHO_ISA   = 1.225
const ISA_VISC  = 1.81e-5      # dynamic viscosity at 15 °C [Pa·s]
const ISA_KVISC = 1.46e-5      # kinematic viscosity at 15 °C [m²/s]


# =====================================================================
# Bird struct (afpt-style)
# =====================================================================

"""
    Bird

A bird/bat individual.  Mirrors the field naming of the `afpt` R
package so its formulae transfer with minimal change.

# Required physical state
- `mass_total`           : total in-flight mass [kg]
- `wing_span`            : full wingspan b [m]
- `wing_area`            : total wing area S [m²] (both wings)
- `body_frontal_area`    : projected frontal body area Sb [m²]

# Aerodynamic coefficients (afpt defaults)
- `k_induced`            : induced-drag factor `coef.inducedDragFactor`   (= 1.2)
- `k_profile_lift`       : lift-dep profile-drag factor `coef.profileDragLiftFactor` (= 0.03)
- `CD_body`              : body drag coefficient `coef.bodyDragCoefficient` (= 0.2)

# Energetics
- `wingbeat_frequency`   : characteristic Hz (Pennycuick / afpt allometry)
- `basal_metabolic_rate` : BMR [W]
- `conversion_efficiency`: mechanical→chemical efficiency η (= 0.23)
- `respiration_factor`   : R (= 1.1)

`type` is `:passerine`, `:seabird`, `:bat`, or `:other`.
"""
@kwdef struct Bird
    mass_total::Float64                          # kg
    wing_span::Float64                           # m
    wing_area::Float64                           # m²
    body_frontal_area::Float64                   # m²
    type::Symbol = :other

    k_induced::Float64                = 1.2
    k_profile_lift::Float64           = 0.03
    CD_body::Float64                  = 0.2

    wingbeat_frequency::Float64       = NaN      # Hz
    basal_metabolic_rate::Float64     = NaN      # W
    conversion_efficiency::Float64    = 0.23
    respiration_factor::Float64       = 1.1
end


# ── afpt allometries ────────────────────────────────────────────────

"Body frontal area [m²]; afpt `computeBodyFrontalArea`."
function body_frontal_area_afpt(m_kg::Real; type::Symbol = :other)
    return type == :passerine ? 0.0129 * m_kg^0.614 :
                                 0.00813 * m_kg^0.666
end

"Characteristic wingbeat frequency [Hz] (afpt `.estimateFrequency`)."
function estimate_frequency_afpt(m_kg::Real, b::Real, S::Real; ρ::Real = RHO_ISA)
    return m_kg^(3/8) * sqrt(G_MS2) * b^(-23/24) * S^(-1/3) * ρ^(-3/8)
end

"Basal metabolic rate [W]; afpt `.estimateBasalMetabolicRate`."
function basal_metabolic_rate_afpt(m_kg::Real; type::Symbol = :other)
    if type == :passerine
        return 6.25 * m_kg^0.724
    elseif type == :seabird
        return 5.43 * m_kg^0.72                  # Ellis & Gabrielsen 2002
    elseif type == :bat
        return 3.14 * m_kg^0.744                 # Speakman & Thomas 2003
    else
        return 3.79 * m_kg^0.723
    end
end


"""
    build_bird_from_mass(m_kg; type = :other, …) → Bird

Construct a `Bird` populated entirely from allometric relationships
(Pennycuick 2008 wing geometry + afpt frontal area / frequency / BMR).

Any allometric default can be overridden by passing the field name as a
keyword.
"""
function build_bird_from_mass(m_kg::Real;
                              type::Symbol             = :other,
                              wing_span::Real          = wing_span_m(m_kg),
                              wing_area::Real          = wing_area_m2(m_kg),
                              body_frontal_area::Real  = body_frontal_area_afpt(m_kg, type = type),
                              ρ::Real                  = RHO_ISA,
                              k_induced::Real          = 1.2,
                              k_profile_lift::Real     = 0.03,
                              CD_body::Real            = 0.2,
                              wingbeat_frequency::Real = estimate_frequency_afpt(m_kg, wing_span, wing_area; ρ = ρ),
                              basal_metabolic_rate::Real = basal_metabolic_rate_afpt(m_kg, type = type),
                              conversion_efficiency::Real = 0.23,
                              respiration_factor::Real    = 1.1)
    return Bird(
        mass_total            = m_kg,
        wing_span             = wing_span,
        wing_area             = wing_area,
        body_frontal_area     = body_frontal_area,
        type                  = type,
        k_induced             = k_induced,
        k_profile_lift        = k_profile_lift,
        CD_body               = CD_body,
        wingbeat_frequency    = wingbeat_frequency,
        basal_metabolic_rate  = basal_metabolic_rate,
        conversion_efficiency = conversion_efficiency,
        respiration_factor    = respiration_factor,
    )
end


# =====================================================================
# Drag forces  (afpt `dragForces.R`)
# =====================================================================

"Reynolds number used by afpt for the wing chord: Re = V · (S/b) / ν."
compute_reynolds(V::Real, length::Real, ν::Real = ISA_KVISC) = V * length / ν

"Laminar flat-plate friction-drag coefficient (afpt `calcCDf.lam`)."
CDf_laminar(Re::Real) = 2.66 / sqrt(Re)

"Induced drag [N]: D_i = k · L² / (½·ρ·V²·π·b²)."
induced_drag(L::Real, V::Real, b::Real; ρ::Real = RHO_ISA, k::Real = 1) =
    k * L^2 / (0.5 * ρ * V^2 * π * b^2)

"Zero-lift profile drag [N]: D_p0 = ½·ρ·V²·CDpro0·S."
profile_drag0(V::Real, S::Real, CDpro0::Real; ρ::Real = RHO_ISA) =
    0.5 * ρ * V^2 * CDpro0 * S

"Lift-dependent profile drag [N]: D_p2 = k_p · L² / (½·ρ·V²·S)."
profile_drag2(L::Real, V::Real, S::Real; ρ::Real = RHO_ISA, k::Real = 0.03) =
    k * L^2 / (0.5 * ρ * V^2 * S)

"Parasite (body) drag [N]: D_par = ½·ρ·V²·CDb·Sb."
parasite_drag(V::Real, Sb::Real, CD_body::Real; ρ::Real = RHO_ISA) =
    0.5 * ρ * V^2 * CD_body * Sb


# =====================================================================
# Flapping power  (simplified afpt model)
# =====================================================================

"""
    FlightPower

Decomposed mechanical / chemical flight power at one airspeed.
"""
@kwdef struct FlightPower
    speed::Float64                  # airspeed [m/s]
    P_induced::Float64              # induced power [W]
    P_profile0::Float64             # zero-lift profile power [W]
    P_profile2::Float64             # lift-dep profile power [W]
    P_parasite::Float64             # body parasite power [W]
    P_mech::Float64                 # total mechanical power [W]
    P_chem::Float64                 # chemical power [W]
    Re::Float64                     # Reynolds number used for CDpro0
    CDpro0::Float64                 # zero-lift profile drag coefficient
end


"""
    mech2chem(P_mech, bird) → Float64

afpt `mech2chem`:  (P_mech / η + BMR) · R.
"""
mech2chem(P_mech::Real, bird::Bird) =
    (P_mech / bird.conversion_efficiency + bird.basal_metabolic_rate) * bird.respiration_factor

"""
    chem2mech(P_chem, bird) → Float64
"""
chem2mech(P_chem::Real, bird::Bird) =
    (P_chem / bird.respiration_factor - bird.basal_metabolic_rate) * bird.conversion_efficiency


"""
    flapping_power(bird, V; ρ, ν, CDpro0 = nothing, lift = nothing) → FlightPower

Total mechanical flight power [W] following the afpt drag decomposition
(without flapping-correction factors kD/kP, which require afpt's
internal coefficient table).

`CDpro0` defaults to the laminar flat-plate correlation
`2.66/√Re` evaluated at the mean-chord Reynolds number.  `lift`
defaults to weight (m·g, level flight).
"""
function flapping_power(bird::Bird, V::Real;
                        ρ::Real         = RHO_ISA,
                        ν::Real         = ISA_KVISC,
                        CDpro0          = nothing,
                        lift::Real      = bird.mass_total * G_MS2)
    b  = bird.wing_span
    S  = bird.wing_area
    Sb = bird.body_frontal_area

    Re_chord = compute_reynolds(V, S / b, ν)
    CDp0     = CDpro0 === nothing ? CDf_laminar(Re_chord) : CDpro0

    D_i  = induced_drag(lift, V, b;        ρ = ρ, k = bird.k_induced)
    D_p0 = profile_drag0(V, S, CDp0;       ρ = ρ)
    D_p2 = profile_drag2(lift, V, S;       ρ = ρ, k = bird.k_profile_lift)
    D_pa = parasite_drag(V, Sb, bird.CD_body; ρ = ρ)

    P_i, P_p0, P_p2, P_pa = D_i*V, D_p0*V, D_p2*V, D_pa*V
    P_mech = P_i + P_p0 + P_p2 + P_pa

    return FlightPower(
        speed      = V,
        P_induced  = P_i,
        P_profile0 = P_p0,
        P_profile2 = P_p2,
        P_parasite = P_pa,
        P_mech     = P_mech,
        P_chem     = mech2chem(P_mech, bird),
        Re         = Re_chord,
        CDpro0     = CDp0,
    )
end


# =====================================================================
# Characteristic flight speeds
# =====================================================================

"""
    minimum_power_speed(bird; ρ, ν, V_lo = 2, V_hi = 40, n = 4001) → Float64

V_mp [m/s]: airspeed at which `flapping_power` is minimised.
"""
function minimum_power_speed(bird::Bird; ρ::Real = RHO_ISA, ν::Real = ISA_KVISC,
                             V_lo::Real = 2.0, V_hi::Real = 40.0, n::Int = 4001)
    Vs = range(V_lo, V_hi; length = n)
    Ps = [flapping_power(bird, V; ρ = ρ, ν = ν).P_mech for V in Vs]
    return Vs[argmin(Ps)]
end

"""
    maximum_range_speed(bird; ρ, ν, V_lo, V_hi, n) → Float64

V_mr [m/s]: airspeed at which `P_mech / V` (cost-of-transport) is
minimised — i.e. the tangent from the origin to the power curve.
"""
function maximum_range_speed(bird::Bird; ρ::Real = RHO_ISA, ν::Real = ISA_KVISC,
                             V_lo::Real = 2.0, V_hi::Real = 40.0, n::Int = 4001)
    Vs   = range(V_lo, V_hi; length = n)
    cost = [flapping_power(bird, V; ρ = ρ, ν = ν).P_mech / V for V in Vs]
    return Vs[argmin(cost)]
end


# =====================================================================
# Full pipeline: BirdScaling bundle
# =====================================================================

"""
    BirdScaling

Bundles all of the structs produced for one bird at one environment.
"""
struct BirdScaling{WD<:WingDiscretization,
                   K<:FlappingKinematics,
                   A<:AirProperties,
                   WT<:WingTemperatures,
                   WBC<:WingbeatConvection}
    bird::Bird
    wing_disc::WD
    kinematics::K
    air::A
    temperatures::WT
    V_mr::Float64                # m/s
    V_mp::Float64                # m/s
    power_mr::FlightPower
    convection::WBC
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

Q_mean(b::BirdScaling) = b.convection.Q_mean
Q_max(b::BirdScaling)  = b.convection.Q_max
Q_min(b::BirdScaling)  = b.convection.Q_min

"Heat-flux density q = Q_mean / (dorsal + ventral area)."
q_density(b::BirdScaling) = uconvert(u"W/m^2", Q_mean(b) / wing_surface_area(b))

"Driving temperature difference T_wing − T_air [K] (using element 1)."
ΔT(b::BirdScaling) =
    uconvert(u"K", b.temperatures.T_dorsal[1]) - uconvert(u"K", b.air.T_air)


"""
    Q_for_mass(m_kg; …) → BirdScaling

Run the full pipeline:
1. Bird ← `build_bird_from_mass`
2. Wing ← `build_wing_for_mass`
3. Air ← `air_properties` (altitude- or pressure-based)
4. V_mp, V_mr ← afpt-simplified `flapping_power`
5. Kinematics ← `build_kinematics_for_mass` (V_mr available for Strouhal)
6. Temperatures ← `uniform_temperature` (override via `temperatures_builder`)
7. Convection ← `compute_wingbeat_convection`

# Environment / temperatures
- `T_air`, `T_wing`        : ambient and wing surface temperatures (Unitful)
- `altitude`               : altitude (Unitful length); used if `P` is `nothing`
- `P`                      : ambient pressure (Unitful); overrides `altitude`
- `temperatures_builder`   : `(wing_disc) -> WingTemperatures` for a non-uniform profile

# Discretisation
- `n_elements`             : span-wise plate elements (default 10)
- `n_steps`                : time steps per wingbeat (default 40)
- `disc_method`            : "uniform" or "nonuniform" span-wise spacing
- `thickness_factor`       : wing thickness as fraction of mean chord
- `dorsal_active`, `ventral_active` : surface activity flags

# Kinematics
- `amp`                    : stroke half-amplitude — degrees or an `AmpScaling`
- `stroke_plane_deg`       : stroke-plane tilt [°]
- `freq_method`            : a `FreqScaling` instance (default `Greenewalt1975()`)

# Bird allometry / aerodynamics — overrides for `build_bird_from_mass`
- `type`                   : `:passerine`, `:seabird`, `:bat`, `:other`
- `wing_span`              : override allometric span [m]
- `wing_area`              : override allometric area [m²]
- `body_frontal_area`      : override afpt body frontal area [m²]
- `k_induced`              : induced-drag factor (afpt default 1.2)
- `k_profile_lift`         : lift-dependent profile-drag factor (afpt default 0.03)
- `CD_body`                : body drag coefficient (afpt default 0.2; Pennycuick 0.10)
- `wingbeat_frequency`     : override frequency used inside the bird struct [Hz]
- `basal_metabolic_rate`   : override BMR [W]
- `conversion_efficiency`  : mechanical→chemical efficiency η (default 0.23)
- `respiration_factor`     : respiration factor R (default 1.1)

# Power-speed search bracket (passed to `minimum_power_speed` / `maximum_range_speed`)
- `V_lo`, `V_hi`, `V_n`    : search range and grid resolution
"""
function Q_for_mass(m_kg::Real;
                    # ── environment ─────────────────────────────────
                    T_air::Quantity        = 20.0u"°C",
                    T_wing::Quantity       = 25.0u"°C",
                    altitude::Quantity     = 0.0u"m",
                    P                      = nothing,
                    temperatures_builder   = nothing,
                    # ── discretisation ──────────────────────────────
                    n_elements::Int        = 10,
                    n_steps::Int           = 40,
                    disc_method::String    = "uniform",
                    thickness_factor::Real = 0.05,
                    dorsal_active::Bool    = true,
                    ventral_active::Bool   = true,
                    # ── kinematics ──────────────────────────────────
                    amp                    = 60.0,
                    stroke_plane_deg::Real = 80.0,
                    freq_method::FreqScaling = Greenewalt1975(),
                    # ── bird allometry / aero overrides ─────────────
                    type::Symbol                = :other,
                    wing_span                   = nothing,
                    wing_area                   = nothing,
                    body_frontal_area           = nothing,
                    k_induced::Real             = 1.2,
                    k_profile_lift::Real        = 0.03,
                    CD_body::Real               = 0.2,
                    wingbeat_frequency          = nothing,
                    basal_metabolic_rate        = nothing,
                    conversion_efficiency::Real = 0.23,
                    respiration_factor::Real    = 1.1,
                    # ── power-speed search ──────────────────────────
                    V_lo::Real             = 2.0,
                    V_hi::Real             = 40.0,
                    V_n::Int               = 4001)
    air = air_properties(T_air; altitude = altitude, P = P)
    ρ   = ustrip(u"kg/m^3", air.ρ)
    ν   = ustrip(u"m^2/s",  air.ν)

    bird_kwargs = Dict{Symbol,Any}(:type => type, :ρ => ρ,
                                   :k_induced => k_induced,
                                   :k_profile_lift => k_profile_lift,
                                   :CD_body => CD_body,
                                   :conversion_efficiency => conversion_efficiency,
                                   :respiration_factor => respiration_factor)
    wing_span         === nothing || (bird_kwargs[:wing_span]            = wing_span)
    wing_area         === nothing || (bird_kwargs[:wing_area]            = wing_area)
    body_frontal_area === nothing || (bird_kwargs[:body_frontal_area]    = body_frontal_area)
    wingbeat_frequency   === nothing || (bird_kwargs[:wingbeat_frequency]   = wingbeat_frequency)
    basal_metabolic_rate === nothing || (bird_kwargs[:basal_metabolic_rate] = basal_metabolic_rate)

    bird = build_bird_from_mass(m_kg; bird_kwargs...)

    V_mp = minimum_power_speed(bird; ρ = ρ, ν = ν, V_lo = V_lo, V_hi = V_hi, n = V_n)
    V_mr = maximum_range_speed(bird; ρ = ρ, ν = ν, V_lo = V_lo, V_hi = V_hi, n = V_n)
    P_mr = flapping_power(bird, V_mr; ρ = ρ, ν = ν)

    kin = build_kinematics_for_mass(m_kg, freq_method;
              amp = amp, stroke_plane_deg = stroke_plane_deg,
              V_forward_ms = V_mr)

    wd = build_wing_for_mass(m_kg;
              n_elements      = n_elements,
              method          = disc_method,
              thickness_factor = thickness_factor,
              dorsal_active   = dorsal_active,
              ventral_active  = ventral_active)

    wt = temperatures_builder === nothing ?
            uniform_temperature(wd, T_wing) :
            temperatures_builder(wd)

    wbc = compute_wingbeat_convection(kin, wd, wt, air;
              n_steps = n_steps, V_forward = V_mr * u"m/s")

    return BirdScaling(bird, wd, kin, air, wt,
                       Float64(V_mr), Float64(V_mp), P_mr, wbc)
end


"""
    summarize(b::BirdScaling) → NamedTuple

Flat scalar view (SI base units, plain Float64) for plotting.
"""
function summarize(b::BirdScaling)
    A_wing = ustrip(u"m^2", wing_surface_area(b))
    return (
        m_kg     = b.bird.mass_total,
        b_m      = ustrip(u"m",  wingspan(b)),
        S_m2     = ustrip(u"m^2", wing_planform_area(b)),
        chord_m  = ustrip(u"m",  mean_chord(b)),
        f_Hz     = ustrip(u"Hz", frequency(b)),
        V_mr     = b.V_mr,
        V_mp     = b.V_mp,
        P_mech_W = b.power_mr.P_mech,
        P_chem_W = b.power_mr.P_chem,
        Q_mean_W = ustrip(u"W",  Q_mean(b)),
        Q_max_W  = ustrip(u"W",  Q_max(b)),
        Q_min_W  = ustrip(u"W",  Q_min(b)),
        q_per_m2 = ustrip(u"W",  Q_mean(b)) / A_wing,
        ΔT_K     = ustrip(u"K",  ΔT(b)),
        amp_deg  = rad2deg(b.kinematics.amplitude),
    )
end


export Bird, FlightPower, BirdScaling,
       body_frontal_area_afpt, estimate_frequency_afpt, basal_metabolic_rate_afpt,
       build_bird_from_mass,
       compute_reynolds, CDf_laminar,
       induced_drag, profile_drag0, profile_drag2, parasite_drag,
       flapping_power, mech2chem, chem2mech,
       minimum_power_speed, maximum_range_speed,
       Q_for_mass, summarize,
       wingspan, wing_planform_area, wing_surface_area, mean_chord,
       frequency, Q_mean, Q_max, Q_min, q_density, ΔT


end # module WingPower
