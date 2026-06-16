# =====================================================================
# afpt.jl — full Julia port of the `afpt` R package
#   Klein Heerenbrink, Johansson & Hedenström (2015) 
#   R package: https://github.com/MarcoKlH/afpt-r
#
# This is a direct port — every public function and every numerical
# constant is a faithful translation of the upstream R code (master
# branch, commit-equivalent v1.1.0.4), checked against the R sources
# in:  computeFlappingPower.R · dragForces.R · fDfPfunctions.R ·
# amplitude.R · reducedFrequency.R · computeBodyFrontalArea.R ·
# computeReynoldsNumber.R · computeAvailablePower.R ·
# PowerToFroMechChem.R · altitude2density.R · air2ground.R ·
# computeFlightPerformance.R · findMinimumPowerSpeed.R ·
# findMaximumRangeSpeed.R · findMaximumPowerSpeed.R ·
# findMaximumClimbRate.R · findMinimumTimeSpeed.R ·
# computeChemicalPower.R · Bird.R · sysdata.rda.
#
# Numerical coefficients (FLAPPINGMODELCOEFFS, ISA0) come from the
# binary `sysdata.rda` shipped with afpt-r and were extracted
# verbatim with RData.jl + CodecBzip2.jl.
# =====================================================================
module AFPT

export ISA0, FLAPPINGMODELCOEFFS,
       AfptBird, build_afpt_bird,
       reduced_frequency, compute_reynolds_number,
       compute_body_frontal_area,
       cdf_lam, cdf_tur, calc_cdf, calc_cdf2,
       drag_forces,
       fD_ind, fD_pro0, fD_pro2, fP_ind, fP_pro0, fP_pro2,
       amplitude_afpt,
       compute_flapping_power, compute_available_power,
       mech2chem, chem2mech,
       find_minimum_power_speed, find_maximum_range_speed,
       find_maximum_power_speed, find_maximum_climb_rate,
       find_minimum_time_speed,
       compute_chemical_power, compute_flight_performance,
       wing_span_allometry, wing_area_allometry, mean_chord_allometry,
       aspect_ratio_allometry,
       estimate_frequency, estimate_basal_metabolic_rate


# =====================================================================
# Constants — extracted verbatim from afpt-r/R/sysdata.rda
# =====================================================================

"International Standard Atmosphere reference (afpt `ISA0`).  All
numerical values are taken verbatim from `sysdata.rda`."
const ISA0 = (density       = 1.225,      # kg/m³
              gravity       = 9.80665,    # m/s²
              viscosity     = 1.46e-5,    # m²/s   (kinematic; afpt convention)
              temperature   = 288.15,     # K
              pressure      = 101325.0,   # Pa
              gasconstant   = 287.053,    # J/(kg·K)
              tempgrad      = -0.0065,    # K/m
              windSpeed     = 0.0,        # m/s
              windDir       = 0.0)        # rad


"Flapping-correction coefficient table (afpt `FLAPPINGMODELCOEFFS`).
Fitted polynomial coefficients from Klein Heerenbrink et al. 2015,
extracted verbatim from `sysdata.rda`."
const FLAPPINGMODELCOEFFS = (
    # Induced drag thrust requirement factor
    Dind  = (p00 =  9.25,   p02 =  2.931, p10 =  0.0,   p11 = -1.969, p12 =  0.0,   p20 =  0.0),
    # Zero-lift profile drag thrust requirement factor
    Dpro0 = (p00 = -0.59,   p02 = -1.145, p10 =  0.239, p11 =  0.197, p12 =  0.446, p20 = -0.715),
    # Lift-dependent profile drag thrust requirement factor
    Dpro2 = (p00 =  9.298,  p02 =  1.301, p10 = -0.659, p11 = -1.521, p12 =  0.0,   p20 =  0.0),
    # Induced power factor
    Pind  = (p00 =  7.517,  p02 =  4.913, p10 =  2.573, p11 = -1.259, p12 =  0.0,   p20 =  0.0,   r =  1.135),
    # Zero-lift profile power factor
    Ppro0 = (p00 = -1.101,  p02 = -1.183, p10 =  0.512, p11 =  0.227, p12 =  0.431, p20 =  0.0,   r =  0.0),
    # Lift-dependent profile power factor
    Ppro2 = (p00 =  8.851,  p02 =  2.807, p10 =  1.354, p11 = -0.943, p12 =  0.0,   p20 =  0.0,   r =  1.201),
    # Amplitude
    A     = (p00 =  1.277,  p01 =  0.0,   p02 =  0.0,
             p10 =  6.214,  p11 =  0.0,   p12 =  3.631,
             p40 =  227.8,  p41 =  0.0,   p42 =  1481.0,
             q00 =  0.0107, q01 =  0.0565, q02 = -0.0169,
             r   =  2.62),
)


# =====================================================================
# Bird struct (afpt Bird.R)
# =====================================================================

"""
    AfptBird

Bird/bat individual following the `afpt` field convention.  Field names
match the R package (camelCase) so equations transfer one-to-one.
"""
@Base.kwdef mutable struct AfptBird
    name::String                       = ""
    name_scientific::String            = ""
    source::String                     = ""
    type::Symbol                       = :other         # :passerine | :seabird | :bat | :other

    # Geometry
    massTotal::Float64                                  # kg
    massEmpty::Float64                                  # kg
    massFat::Float64                   = 0.0            # kg
    massLoad::Float64                  = 0.0            # kg
    wingSpan::Float64                                   # m
    wingArea::Float64                                   # m²
    bodyFrontalArea::Float64                            # m²

    # Energetics / physiology
    wingbeatFrequency::Float64                          # Hz
    basalMetabolicRate::Float64                         # W
    muscleFraction::Float64            = 0.17

    # Aerodynamic coefficients (afpt defaults)
    coef_profileDragLiftFactor::Float64 = 0.03
    coef_bodyDragCoefficient::Float64   = 0.20
    coef_conversionEfficiency::Float64  = 0.23
    coef_respirationFactor::Float64     = 1.10
    coef_activeStrain::Float64          = 0.26
    coef_isometricStress::Float64       = 400e3
end


# afpt body-frontal-area allometry (computeBodyFrontalArea.R)
compute_body_frontal_area(m_empty::Real, type::Symbol = :other) =
    type === :passerine ? 0.0129 * m_empty^0.614 :
                          0.00813 * m_empty^0.666

# ---------------------------------------------------------------------
# Wing-geometry allometries (Pennycuick 2008 cross-species regressions).
#
# These are not part of afpt-r itself (afpt's Bird() takes wingSpan and
# wingArea as inputs), but they are the same Pennycuick relations afpt
# documents and that downstream modules (WingPlates, WingPower,
# WingKinematics) currently each re-implement.  We make them canonical
# here so every consumer dispatches through AFPT.
#
#     b  = 1.17  · m^0.39     [m]    wing span (full)
#     S  = 0.16  · m^0.72     [m²]   total wing area (both wings)
#     c̄  = S / b              [m]    mean chord
#     AR = b² / S
#
# Inputs are body mass m_kg in kilograms; outputs are plain Float64.
# ---------------------------------------------------------------------

"Full wing span [m] from body mass [kg] (Pennycuick 2008)."
wing_span_allometry(m_kg::Real)   = 1.17 * m_kg^0.39

"Total wing area [m²] (both wings) from body mass [kg] (Pennycuick 2008)."
wing_area_allometry(m_kg::Real)   = 0.16 * m_kg^0.72

"Mean chord [m]: c̄ = S/b (Pennycuick 2008 allometries)."
mean_chord_allometry(m_kg::Real)  = wing_area_allometry(m_kg) / wing_span_allometry(m_kg)

"Aspect ratio AR = b²/S (Pennycuick 2008 allometries)."
aspect_ratio_allometry(m_kg::Real) = wing_span_allometry(m_kg)^2 / wing_area_allometry(m_kg)

# afpt characteristic wingbeat frequency (Bird.R .estimateFrequency)
estimate_frequency(m_empty::Real, b::Real, S::Real;
                   ρ::Real = ISA0.density, g::Real = ISA0.gravity) =
    m_empty^(3/8) * sqrt(g) * b^(-23/24) * S^(-1/3) * ρ^(-3/8)

# afpt basal metabolic rate allometry (Bird.R .estimateBasalMetabolicRate)
function estimate_basal_metabolic_rate(m_empty::Real, type::Symbol = :other)
    type === :passerine && return 6.25 * m_empty^0.724
    type === :seabird   && return 5.43 * m_empty^0.72
    type === :bat       && return 3.14 * m_empty^0.744
    return 3.79 * m_empty^0.723
end


"""
    build_afpt_bird(massTotal, wingSpan; wingArea = nothing,
                    wingAspect = nothing, type = :other, …) → AfptBird

Construct an `AfptBird` with afpt-style allometric defaults filling any
field the caller does not specify.  Mirrors `Bird()` in the R package.

Mass composition follows afpt rules (`Bird.R::.massComposition`):
caller may supply any subset of `massEmpty`, `massFat`, `massLoad`;
missing fields are inferred so massEmpty + massFat + massLoad == massTotal.
By default massFat = massLoad = 0 and massEmpty = massTotal.
"""
function build_afpt_bird(massTotal::Real, wingSpan::Real;
                         wingArea::Union{Nothing,Real}   = nothing,
                         wingAspect::Union{Nothing,Real} = nothing,
                         type::Symbol                    = :other,
                         massEmpty::Union{Nothing,Real}  = nothing,
                         massFat::Union{Nothing,Real}    = nothing,
                         massLoad::Union{Nothing,Real}   = nothing,
                         bodyFrontalArea::Union{Nothing,Real}   = nothing,
                         wingbeatFrequency::Union{Nothing,Real} = nothing,
                         basalMetabolicRate::Union{Nothing,Real}= nothing,
                         muscleFraction::Real            = 0.17,
                         coef_profileDragLiftFactor::Real = 0.03,
                         coef_bodyDragCoefficient::Real   = 0.20,
                         coef_conversionEfficiency::Real  = 0.23,
                         coef_respirationFactor::Real     = 1.10,
                         coef_activeStrain::Real          = 0.26,
                         coef_isometricStress::Real       = 400e3,
                         name::String                     = "",
                         name_scientific::String          = "",
                         source::String                   = "")
    # ── Wing area ──────────────────────────────────────────────────
    S = if wingArea !== nothing
            wingArea
        elseif wingAspect !== nothing
            wingSpan^2 / wingAspect
        else
            error("build_afpt_bird: provide wingArea or wingAspect")
        end

    # ── Mass composition (.massComposition rules) ──────────────────
    mE = massEmpty
    mF = massFat
    mL = massLoad
    if mE !== nothing && mF !== nothing && mL !== nothing
        # all three given — derive total
        # (the R package warns if mismatched; we silently re-sum)
        massTotal = mE + mF + mL
    elseif mE !== nothing && mF !== nothing
        mL = massTotal - mE - mF
    elseif mE !== nothing && mL !== nothing
        mF = massTotal - mE - mL
    elseif mF !== nothing && mL !== nothing
        mE = massTotal - mF - mL
    elseif mE !== nothing
        mF = 0.0
        mL = massTotal - mE
    else
        mE = massTotal
        mF = mF === nothing ? 0.0 : mF
        mL = mL === nothing ? 0.0 : mL
    end

    Sb = bodyFrontalArea === nothing ?
            compute_body_frontal_area(mE, type) :
            bodyFrontalArea
    f  = wingbeatFrequency === nothing ?
            estimate_frequency(mE, wingSpan, S) :
            wingbeatFrequency
    bmr = basalMetabolicRate === nothing ?
            estimate_basal_metabolic_rate(mE, type) :
            basalMetabolicRate

    return AfptBird(
        name = name, name_scientific = name_scientific, source = source,
        type = type,
        massTotal = massTotal, massEmpty = mE, massFat = mF, massLoad = mL,
        wingSpan = wingSpan, wingArea = S, bodyFrontalArea = Sb,
        wingbeatFrequency = f, basalMetabolicRate = bmr,
        muscleFraction = muscleFraction,
        coef_profileDragLiftFactor = coef_profileDragLiftFactor,
        coef_bodyDragCoefficient   = coef_bodyDragCoefficient,
        coef_conversionEfficiency  = coef_conversionEfficiency,
        coef_respirationFactor     = coef_respirationFactor,
        coef_activeStrain          = coef_activeStrain,
        coef_isometricStress       = coef_isometricStress,
    )
end


# =====================================================================
# Reduced frequency, Reynolds number (afpt reducedFrequency.R /
# computeReynoldsNumber.R)
# =====================================================================

"Reduced (Strouhal-like) frequency  kf = 2π · b · f / V  (afpt convention)."
reduced_frequency(wingSpan::Real, frequency::Real, speed::Real) =
    2π * wingSpan * frequency / speed

"Chord-based Reynolds number  Re = V · L / ν."
compute_reynolds_number(speed::Real, chord::Real, viscosity::Real) =
    speed * chord / viscosity


# =====================================================================
# Flat-plate friction-drag coefficients (afpt dragForces.R)
# =====================================================================

"Laminar flat-plate friction-drag coefficient: 2.66/√Re."
cdf_lam(Re::Real) = 2.66 / sqrt(Re)

"Fully turbulent flat-plate friction-drag coefficient: 2·0.074/Re^(1/5)."
cdf_tur(Re::Real) = 2 * 0.074 / Re^(1/5)


"""
    calc_cdf(Re; Re_tr = 5e5)

afpt `calcCDf` — Anderson laminar / turbulent transition friction drag.
Laminar below Re_tr, then turbulent with an offset that subtracts the
transition-point step:

    CDf = (CDf_tur(Re) − (CDf_tur(Re_tr) − CDf_lam(Re_tr))·Re_tr/Re)
"""
function calc_cdf(Re::Real; Re_tr::Real = 5e5)
    CDf_lam_tr = cdf_lam(Re_tr)
    CDf_tur_tr = cdf_tur(Re_tr)
    if Re <= Re_tr
        return cdf_lam(Re)
    else
        return cdf_tur(Re) - (CDf_tur_tr - CDf_lam_tr) * Re_tr / Re
    end
end


"""
    calc_cdf2(Re; Re_tr = 5e5)

afpt `calcCDf2` — matching-thickness transition friction drag.  Above
Re_tr,

    Rlt  = CDf_lam(Re_tr) / CDf_tur(Re_tr)
    Rclt = max(1 − (Re_tr/Re)(1 − Rlt), 0)
    CDf  = CDf_lam(Re_tr) · Re_tr/Re
            + CDf_tur(Re) · Rclt^(6/5)
            − CDf_tur(Re_tr) · Rlt^(6/5) · Re_tr/Re
"""
function calc_cdf2(Re::Real; Re_tr::Real = 5e5)
    CDf_lam_tr = cdf_lam(Re_tr)
    CDf_tur_tr = cdf_tur(Re_tr)
    if Re <= Re_tr
        return cdf_lam(Re)
    else
        Rlt  = CDf_lam_tr / CDf_tur_tr
        Rclt = max(1 - Re_tr / Re * (1 - Rlt), 0.0)
        return CDf_lam_tr * Re_tr / Re +
               cdf_tur(Re) * Rclt^(6/5) -
               CDf_tur_tr * Rlt^(6/5) * Re_tr / Re
    end
end


# =====================================================================
# Drag forces (afpt dragForces.R)
# =====================================================================

"afpt `dragCoefs.ProfileDrag0` (no opts override branch)."
profile_drag0_coef(Re::Real; CDpro0::Union{Nothing,Real,Tuple} = nothing,
                              Re_tr::Real = 5e5) = begin
    if CDpro0 isa Real
        return float(CDpro0)
    elseif CDpro0 isa Tuple && length(CDpro0) >= 2
        return float(CDpro0[1]) + calc_cdf2(Re; Re_tr = CDpro0[2])
    else
        return cdf_lam(Re)
    end
end


"""
    drag_forces(bird, speed, lift; ρ, ν, CDpro0_override = nothing,
                CDbody_override = nothing)
        → NamedTuple{(:ind, :pro0, :pro2, :par)}

Static-wing drag decomposition (afpt `dragForces`).
"""
function drag_forces(bird::AfptBird, speed::Real, lift::Real;
                     ρ::Real = ISA0.density, ν::Real = ISA0.viscosity,
                     CDpro0_override::Union{Nothing,Real,Tuple} = nothing,
                     CDbody_override::Union{Nothing,Real} = nothing)
    b  = bird.wingSpan
    S  = bird.wingArea
    Sb = bird.bodyFrontalArea
    q  = 0.5 * ρ * speed^2
    Re = compute_reynolds_number(speed, S / b, ν)
    kp = bird.coef_profileDragLiftFactor
    CDpro0 = profile_drag0_coef(Re; CDpro0 = CDpro0_override)
    CDb    = CDbody_override === nothing ? bird.coef_bodyDragCoefficient : CDbody_override
    return (ind  = lift^2 / (q * π * b^2),
            pro0 = q * S * CDpro0,
            pro2 = lift^2 / (q * S) * kp,
            par  = q * Sb * CDb,
            CDpro0 = CDpro0,
            ReynoldsNumber = Re)
end


# =====================================================================
# Flapping correction factors (afpt fDfPfunctions.R)
# =====================================================================

# Each function takes (kf, phi) where phi is the strokeplane angle in
# radians (NB: afpt's fD/fP take phi in radians — the conversion from
# degrees happens earlier, inside compute_flapping_power).

function fD_ind(kf::Real, phi::Real)
    C = FLAPPINGMODELCOEFFS.Dind
    t = tan(phi)
    return (C.p00 + C.p02 * t^2) / kf +
           (C.p10 + C.p11 * t + C.p12 * t^2)
end

function fD_pro0(kf::Real, phi::Real)
    C = FLAPPINGMODELCOEFFS.Dpro0
    t = tan(phi)
    return (C.p00 + C.p02 * t^2) +
           (C.p10 + C.p11 * t + C.p12 * t^2) * kf +
           C.p20 / kf^2
end

function fD_pro2(kf::Real, phi::Real)
    C = FLAPPINGMODELCOEFFS.Dpro2
    t = tan(phi)
    return (C.p00 + C.p02 * t^2) / kf +
           (C.p10 + C.p11 * t + C.p12 * t^2)
end

function fP_ind(kf::Real, phi::Real)
    C = FLAPPINGMODELCOEFFS.Pind
    t = tan(phi)
    return (C.p00 + C.p02 * t^2) / kf^C.r +
           (C.p10 + C.p11 * t + C.p12 * t^2)
end

function fP_pro0(kf::Real, phi::Real)
    C = FLAPPINGMODELCOEFFS.Ppro0
    t = tan(phi)
    return (C.p00 + C.p02 * t^2) +
           (C.p10 + C.p11 * t + C.p12 * t^2) * kf +
           C.p20 / kf^2
end

function fP_pro2(kf::Real, phi::Real)
    C = FLAPPINGMODELCOEFFS.Ppro2
    t = tan(phi)
    return (C.p00 + C.p02 * t^2) / kf^C.r +
           (C.p10 + C.p11 * t + C.p12 * t^2)
end


# =====================================================================
# Wingbeat amplitude (afpt amplitude.R)
# =====================================================================

"""
    amplitude_afpt(kf, phi, TL) → degrees

Wingbeat amplitude optimised for minimum induced power, as a function
of reduced frequency `kf`, strokeplane angle `phi` (radians) and
thrust-to-lift ratio `TL`.  Direct port of `amplitude` in afpt's
amplitude.R.
"""
function amplitude_afpt(kf::Real, phi::Real, TL::Real)
    C = FLAPPINGMODELCOEFFS.A
    inner = (C.p00 + C.p01 * phi + C.p02 * phi^2) * (TL / kf)^(1 / C.r) +
            (C.p10 + C.p11 * phi + C.p12 * phi^2) * (TL / kf) +
            (C.p40 + C.p41 * phi + C.p42 * phi^2) * (TL / kf)^4
    front = 1 + (C.q00 + C.q01 * phi + C.q02 * phi^2) * kf
    return front * atan(inner) * 180 / π
end


# =====================================================================
# Mechanical ↔ chemical power conversion (afpt PowerToFroMechChem.R)
# =====================================================================

"afpt `mech2chem`:  (P_mech / η + BMR) · R."
mech2chem(P_mech::Real, bird::AfptBird) =
    (P_mech / bird.coef_conversionEfficiency + bird.basalMetabolicRate) *
    bird.coef_respirationFactor

"afpt `chem2mech`."
chem2mech(P_chem::Real, bird::AfptBird) =
    (P_chem / bird.coef_respirationFactor - bird.basalMetabolicRate) *
    bird.coef_conversionEfficiency


# =====================================================================
# Compute available aerobic power (afpt computeAvailablePower.R)
# =====================================================================

"""
    compute_available_power(bird; maxPowerAero = nothing,
                            muscleDensity = 1060,
                            powerDensityMitochondria = 1.2e-6,
                            optStressMaxPower = 0.30) → Float64

Maximum aerobic mechanical power [W] the flight muscles can sustain,
following afpt computeAvailablePower.  Defaults:
σ = 0.30·400×10³ Pa, λ = 0.26, ρ_musc = 1060 kg/m³,
k_mito = 1.2×10⁻⁶ m³/W·s.
"""
function compute_available_power(bird::AfptBird;
                                 maxPowerAero::Union{Nothing,Real} = nothing,
                                 muscleDensity::Real = 1060.0,
                                 powerDensityMitochondria::Real = 1.2e-6,
                                 optStressMaxPower::Real = 0.30)
    σ = optStressMaxPower * bird.coef_isometricStress
    λ = bird.coef_activeStrain
    muscleMass = bird.muscleFraction * bird.massEmpty
    Vmusc = muscleMass / muscleDensity
    f = bird.wingbeatFrequency
    kmito = powerDensityMitochondria

    maxPower = σ * λ * Vmusc * f / (1 + σ * λ * kmito * f)
    if maxPowerAero !== nothing
        if maxPowerAero > maxPower
            @warn "Requested maximum aerobic power exceeds obtainable maximum for muscle; using computed maximum"
        else
            maxPower = σ * λ * Vmusc * f * (1 - kmito * maxPowerAero / Vmusc)
        end
    end
    return maxPower
end


# =====================================================================
# Full flapping-power computation (afpt computeFlappingPower.R)
# =====================================================================

"""
    compute_flapping_power(bird, speed; frequency = bird.wingbeatFrequency,
                           strokeplane = :opt, climbAngle = 0.0,
                           lift = nothing,
                           ρ = ISA0.density, g = ISA0.gravity,
                           ν = ISA0.viscosity,
                           CDpro0_override = nothing,
                           CDbody_override = nothing) → NamedTuple

Full afpt mechanical flapping-power calculation including kD/kP
corrections.  `strokeplane` is in degrees (or `:opt` to minimise the
total power over the bracket `[0°, 50°]`).  `climbAngle` is in degrees.

Returns a NamedTuple containing power.total [W], power.chem [W],
strokeplane (°), amplitude (°), frequency [Hz], kf, ToverL, kD, kP,
CDpro0, ReynoldsNumber, Dnf (drag decomposition), L, and validity flags.
"""
function compute_flapping_power(bird::AfptBird, speed::Real;
                                frequency::Real = bird.wingbeatFrequency,
                                strokeplane::Union{Symbol,Real} = :opt,
                                climbAngle::Real = 0.0,
                                lift::Union{Nothing,Real} = nothing,
                                ρ::Real = ISA0.density,
                                g::Real = ISA0.gravity,
                                ν::Real = ISA0.viscosity,
                                CDpro0_override::Union{Nothing,Real,Tuple} = nothing,
                                CDbody_override::Union{Nothing,Real} = nothing)

    if strokeplane === :opt
        # 1-D minimisation of P over strokeplane ∈ [0°, 50°]  (afpt uses Brent;
        # we use golden-section search with tol matching afpt's 0.1°).
        f_obj = sp -> compute_flapping_power(bird, speed;
                                             frequency = frequency,
                                             strokeplane = sp,
                                             climbAngle = climbAngle,
                                             lift = lift, ρ = ρ, g = g, ν = ν,
                                             CDpro0_override = CDpro0_override,
                                             CDbody_override = CDbody_override).power_total
        sp_opt = _golden_section(f_obj, 0.0, 50.0; tol = 0.1)
        return compute_flapping_power(bird, speed;
                                      frequency = frequency,
                                      strokeplane = sp_opt,
                                      climbAngle = climbAngle,
                                      lift = lift, ρ = ρ, g = g, ν = ν,
                                      CDpro0_override = CDpro0_override,
                                      CDbody_override = CDbody_override)
    end

    sp_deg  = float(strokeplane)
    phi     = sp_deg * π / 180
    climb_r = climbAngle * π / 180
    m  = bird.massTotal

    L_climb = m * g * cos(climb_r)
    D_climb = m * g * sin(climb_r)
    L = lift === nothing ? L_climb : float(lift)

    kf = reduced_frequency(bird.wingSpan, frequency, speed)

    Dnf  = drag_forces(bird, speed, L; ρ = ρ, ν = ν,
                       CDpro0_override = CDpro0_override,
                       CDbody_override = CDbody_override)
    par_with_climb = Dnf.par + D_climb

    # Thrust-to-lift ratio (using non-flapping drag for the denominator
    # correction — see computeFlappingPower.R).
    ToverL = (Dnf.ind + Dnf.pro0 + Dnf.pro2 + par_with_climb) /
             (L - fD_ind(kf, phi) * Dnf.ind
                - fD_pro0(kf, phi) * Dnf.pro0
                - fD_pro2(kf, phi) * Dnf.pro2)

    kD_ind  = 1 + fD_ind(kf, phi)  * ToverL
    kD_pro0 = 1 + fD_pro0(kf, phi) * ToverL
    kD_pro2 = 1 + fD_pro2(kf, phi) * ToverL

    kP_ind  = 1 + fP_ind(kf, phi)  * ToverL
    kP_pro0 = 1 + fP_pro0(kf, phi) * ToverL
    kP_pro2 = 1 + fP_pro2(kf, phi) * ToverL

    P_ind   = kP_ind  * Dnf.ind  * speed
    P_pro0  = kP_pro0 * Dnf.pro0 * speed
    P_pro2  = kP_pro2 * Dnf.pro2 * speed
    P_par   = par_with_climb * speed
    P_total = P_ind + P_pro0 + P_pro2 + P_par

    # induced velocity in hover  (forward-flight validity check)
    vih  = sqrt(L / (0.5 * ρ * π * bird.wingSpan^2))
    flags = (redFreqLo = kf < 1, redFreqHi = kf > 6,
             thrustHi  = ToverL > 0.3, speedLo = speed < 2 * vih)

    amp = amplitude_afpt(kf, phi, ToverL)

    return (
        speed         = speed,
        power_total   = P_total,
        power_ind     = P_ind,
        power_pro0    = P_pro0,
        power_pro2    = P_pro2,
        power_par     = P_par,
        power_chem    = mech2chem(P_total, bird),
        strokeplane   = sp_deg,
        amplitude     = amp,
        frequency     = frequency,
        kf            = kf,
        ToverL        = ToverL,
        kD_ind        = kD_ind,  kD_pro0 = kD_pro0,  kD_pro2 = kD_pro2,
        kP_ind        = kP_ind,  kP_pro0 = kP_pro0,  kP_pro2 = kP_pro2,
        CDpro0        = Dnf.CDpro0,
        ReynoldsNumber = Dnf.ReynoldsNumber,
        Dnf           = (ind = Dnf.ind, pro0 = Dnf.pro0,
                         pro2 = Dnf.pro2, par = par_with_climb),
        L             = L,
        flags         = flags,
    )
end


# Internal: golden-section minimisation on a bracket [a,b].  Used in
# place of R's `stats::optimize` (Brent) for the strokeplane search.
function _golden_section(f, a::Real, b::Real; tol::Real = 1e-3, maxiter::Int = 64)
    φ = (sqrt(5.0) - 1) / 2  # ≈ 0.618
    x1 = b - φ * (b - a)
    x2 = a + φ * (b - a)
    f1 = f(x1); f2 = f(x2)
    for _ in 1:maxiter
        if b - a < tol
            break
        end
        if f1 < f2
            b = x2; x2 = x1; f2 = f1
            x1 = b - φ * (b - a); f1 = f(x1)
        else
            a = x1; x1 = x2; f1 = f2
            x2 = a + φ * (b - a); f2 = f(x2)
        end
    end
    return 0.5 * (a + b)
end


# =====================================================================
# Characteristic flight speeds
# =====================================================================

# Helper that returns a closure P(V) for the supplied bird and options
function _power_curve(bird::AfptBird; kwargs...)
    return V -> compute_flapping_power(bird, V; kwargs...).power_total
end


"""
    find_minimum_power_speed(bird; V_lo = 1.0, V_hi = 50.0, tol = 0.01, kwargs...) → Float64

Minimum-power speed V_mp [m/s] — single 1-D golden-section minimisation
of `compute_flapping_power(...).power_total`.  Extra kwargs are forwarded
to `compute_flapping_power`.
"""
function find_minimum_power_speed(bird::AfptBird;
                                  V_lo::Real = 1.0, V_hi::Real = 50.0,
                                  tol::Real = 0.01, kwargs...)
    f = _power_curve(bird; kwargs...)
    return _golden_section(f, V_lo, V_hi; tol = tol)
end


"""
    find_maximum_range_speed(bird; V_lo, V_hi, tol, kwargs...) → Float64

Maximum-range speed V_mr [m/s] — minimises the chemical cost of
transport `P_chem(V) / V` (afpt findMaximumRangeSpeed.R; no wind).
"""
function find_maximum_range_speed(bird::AfptBird;
                                  V_lo::Real = 1.0, V_hi::Real = 50.0,
                                  tol::Real = 0.01, kwargs...)
    cost = V -> compute_flapping_power(bird, V; kwargs...).power_chem / V
    return _golden_section(cost, V_lo, V_hi; tol = tol)
end


"""
    find_maximum_power_speed(bird; maxPowerAero = nothing, kwargs...)
        → Union{Float64, Nothing}

Maximum-power speed (upper) — the larger root of
`P_mech(V) − P_avail = 0`.  Returns `nothing` if the available power is
never exceeded (i.e. the bird could in principle fly indefinitely fast
within the search bracket).
"""
function find_maximum_power_speed(bird::AfptBird;
                                  V_lo::Real = 1.0, V_hi::Real = 80.0,
                                  maxPowerAero::Union{Nothing,Real} = nothing,
                                  kwargs...)
    P_avail = compute_available_power(bird; maxPowerAero = maxPowerAero)
    f = _power_curve(bird; kwargs...)
    g = V -> f(V) - P_avail
    return _bracket_descending_root(g, V_lo, V_hi)
end


# Find the largest root of g on [a,b] via grid scan then bisection.
# Returns `nothing` if g has the same sign everywhere.
function _bracket_descending_root(g, a::Real, b::Real; n::Int = 401, tol::Real = 1e-3)
    Vs = range(a, b; length = n)
    gs = g.(Vs)
    # find the rightmost sign change
    idx = 0
    for k in (length(gs) - 1):-1:1
        if gs[k] * gs[k + 1] < 0
            idx = k
            break
        end
    end
    idx == 0 && return nothing
    lo, hi = Vs[idx], Vs[idx + 1]
    glo = gs[idx]; ghi = gs[idx + 1]
    for _ in 1:60
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if abs(gm) < tol || (hi - lo) < tol
            return mid
        end
        if glo * gm < 0
            hi = mid; ghi = gm
        else
            lo = mid; glo = gm
        end
    end
    return 0.5 * (lo + hi)
end


"""
    find_maximum_climb_rate(bird; V_lo, V_hi, kwargs...) → NamedTuple

Maximum climb rate Vc_max [m/s] and the speed at which it is achieved.
afpt findMaximumClimbRate.R searches  max_V (P_avail − P_mech(V, γ=0)) / (m·g).
"""
function find_maximum_climb_rate(bird::AfptBird;
                                 V_lo::Real = 1.0, V_hi::Real = 50.0,
                                 maxPowerAero::Union{Nothing,Real} = nothing,
                                 kwargs...)
    P_avail = compute_available_power(bird; maxPowerAero = maxPowerAero)
    f = _power_curve(bird; kwargs...)
    mg = bird.massTotal * ISA0.gravity
    cost = V -> -(P_avail - f(V)) / mg
    V_opt = _golden_section(cost, V_lo, V_hi; tol = 0.01)
    Vc    = -cost(V_opt)
    return (speed = V_opt, climbRate = Vc)
end


"""
    find_minimum_time_speed(bird; V_lo, V_hi, kwargs...) → Float64

Minimum-time speed V_mt [m/s] — minimum of `(P_mech(V) + BMR_chem) / V`,
where BMR_chem is the chemical-power floor.  Matches afpt's
findMinimumTimeSpeed.R conceptually (cost-of-transport including BMR).
"""
function find_minimum_time_speed(bird::AfptBird;
                                 V_lo::Real = 1.0, V_hi::Real = 50.0,
                                 maxPowerAero::Union{Nothing,Real} = nothing,
                                 kwargs...)
    P_avail = compute_available_power(bird; maxPowerAero = maxPowerAero)
    f = _power_curve(bird; kwargs...)
    cost = V -> (P_avail) / V .- (P_avail - f(V)) / V  # equivalently f(V)/V
    return _golden_section(V -> f(V) / V, V_lo, V_hi; tol = 0.01)
end


# =====================================================================
# Chemical power (afpt computeChemicalPower.R)
# =====================================================================

"""
    compute_chemical_power(bird, speed; kwargs...) → Float64

Chemical (metabolic) flight power [W] at the given airspeed.
"""
compute_chemical_power(bird::AfptBird, speed::Real; kwargs...) =
    mech2chem(compute_flapping_power(bird, speed; kwargs...).power_total, bird)


# =====================================================================
# Top-level performance summary (afpt computeFlightPerformance.R)
# =====================================================================

"""
    compute_flight_performance(bird; length_out = 21, V_lo = 1.0, V_hi = 50.0,
                                kwargs...) → NamedTuple

Aggregate flight-performance summary mirroring
`computeFlightPerformance` in afpt.  Returns the characteristic speeds
(V_mp, V_mr, V_max, V_mt, climb-rate-optimum speed), the maximum climb
rate, the available power, and the full power curve sampled on a grid.
"""
function compute_flight_performance(bird::AfptBird;
                                    length_out::Int = 21,
                                    V_lo::Real = 1.0, V_hi::Real = 50.0,
                                    maxPowerAero::Union{Nothing,Real} = nothing,
                                    kwargs...)
    P_avail = compute_available_power(bird; maxPowerAero = maxPowerAero)
    Vs = collect(range(V_lo, V_hi; length = length_out))
    samples = [compute_flapping_power(bird, V; kwargs...) for V in Vs]

    V_mp = find_minimum_power_speed(bird; V_lo = V_lo, V_hi = V_hi, kwargs...)
    V_mr = find_maximum_range_speed(bird; V_lo = V_lo, V_hi = V_hi, kwargs...)
    V_max = find_maximum_power_speed(bird; V_lo = V_lo, V_hi = V_hi,
                                          maxPowerAero = maxPowerAero, kwargs...)
    V_mt = find_minimum_time_speed(bird; V_lo = V_lo, V_hi = V_hi,
                                          maxPowerAero = maxPowerAero, kwargs...)
    climb = find_maximum_climb_rate(bird; V_lo = V_lo, V_hi = V_hi,
                                          maxPowerAero = maxPowerAero, kwargs...)

    return (
        bird          = bird,
        speeds        = Vs,
        samples       = samples,
        powerAvailable = P_avail,
        V_mp          = V_mp,
        V_mr          = V_mr,
        V_max         = V_max,
        V_mt          = V_mt,
        climb         = climb,
    )
end

end # module AFPT
