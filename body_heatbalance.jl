# =====================================================================
# body_heatbalance.jl
#
# Whole-body (trunk + insulation) heat-balance model for nicheflappr2,
# rewritten on the modern BiophysicalEcology stack:
#
#   • BiophysicalGeometry  — Ellipsoid shape + CompositeInsulation(Fur, Fat)
#   • HeatExchange         — InsulationParameters / FibreProperties,
#                            HeatExchangeTraits, Organism, solve_metabolic_rate
#   • BiophysicalBehaviour — OrganismTraits + ThermoregulationLimits +
#                            thermoregulate(...) (rule-based sequential
#                            controller: piloerect → uncurl → vasodilate
#                            → hyperthermia → pant → sweat).
#
# Design (Option A, approved 2026-06):
#   `run_body_thermoregulation` mirrors the BiophysicalBehaviour
#   "budgerigar" example call exactly — it builds an `Organism` + an
#   `environment = (; environment_vars, environment_pars)` NamedTuple
#   and dispatches to `BiophysicalBehaviour.thermoregulate`, returning
#   the standard `endotherm_out` NamedTuple with fields
#       thermoregulation, morphology, energy_fluxes, mass_fluxes
#   plus the `organism` and `environment` it was solved at, so callers
#   (most importantly `whole_animal_heatbalance.jl`) can re-use them.
#
#   Wing fluxes are NOT added here.  They will be injected at "step 0"
#   inside the custom whole-animal loop, before any thermoregulatory
#   behaviour kicks in.  See whole_animal_heatbalance.jl.
#
# Allometric scaling (kept identical to the previous version):
#     axis_ratio (a/b = c=b) : log-linear 1.1 (5 g) → 1.6 (15 kg)
#     feather depth          : d_f = 0.01737 · m^0.33   [m]
#     flesh density          : 1000 kg/m³
#     fibre diameter         : 30 μm
#     fibre density          : 5e8 /m²
#     fat fraction           : 5 %
#     insulation reflectance : 0.248 dorsal / 0.351 ventral
#     body emissivity        : 0.99
#     ground/sky emissivities: 1.0 / 1.0
#     view factors           : 0.5 sky / 0.5 ground
#     ground albedo          : 0.8
#     core temperature       : 312.15 K (39 °C)
#     metabolic rate         : McKechnieWolf() (avian basal allometry),
#                              Q_basal evaluated at the build mass
#
# Module-loading note: this file expects `environment.jl` (the
# `FlightEnvironment` module) and `afpt.jl` (`AFPT`) to have been
# `include`d already — that is what `wing_power.jl` does before
# including the body module.
# =====================================================================

module BodyHeatBalance

# ── Stdlib & sister modules ──────────────────────────────────────────
using Unitful
using ..FlightEnvironment
using ..FlightEnvironment: Microclimate
using ..AFPT
using ..AFPT: build_afpt_bird, wing_span_allometry, wing_area_allometry,
              find_maximum_range_speed

# ── BiophysicalEcology stack ─────────────────────────────────────────
# BiophysicalGeometry: shape + insulation primitives that go into a
# `Body`.  The new API renames the insulation types:
#   FibrousLayer → Fur        (kept on the dorsal/ventral fibre props)
#   FatLayer     → Fat        (fat layer wraps the flesh)
using BiophysicalGeometry: Body, Ellipsoid, CompositeInsulation, Fur, Fat,
                            Intermediate

# HeatExchange: parameter structs + Organism + metabolic-rate equations.
# `Fur`/`Fat` go into `Body(...)`; the dorsal/ventral *fibre properties*
# that drive radiation/conduction live in `InsulationParameters` (the
# HE traits struct) using `FibreProperties`.
import HeatExchange
using HeatExchange: Organism, HeatExchangeTraits, SolveMetabolicRateOptions,
                    InsulationParameters, FibreProperties,
                    ExternalConductionParameters, InternalConductionParameters,
                    RadiationParameters, ConvectionParameters,
                    EvaporationParameters, HydraulicParameters,
                    RespirationParameters, MetabolismParameters,
                    EnvironmentalVars, EnvironmentalPars,
                    McKechnieWolf, Kleiber

# BiophysicalBehaviour: thermal strategy + behaviour traits +
# thermoregulation control loop (CoreFirst etc.).
using BiophysicalBehaviour: BehavioralTraits, OrganismTraits, Endotherm,
                             RuleBasedSequentialControl, CoreFirst,
                             CoreAndPantingFirst, CorePantingSweatingFirst,
                             ThermoregulationLimits, InsulationLimits,
                             SteppedParameter, PantingLimits, Diurnal,
                             thermoregulate, thermoregulation

# FluidProperties: gas-mixture struct used by EnvironmentalPars.
using FluidProperties: GasFractions


# =====================================================================
# Allometric scaling laws & constants
# =====================================================================

"""
    axis_ratio_allometry(m_kg) → Float64

Body axis ratio a/b (with c = b) for a bird of body mass `m_kg` [kg].
Log-linear between 1.1 at 5 g and 1.6 at 15 kg — captures the trend
from near-spherical small passerines toward more elongated trunks of
larger birds while staying close to the budgie reference value (≈1.1)
across most of the small-bird range.
"""
function axis_ratio_allometry(m_kg::Real)
    m_lo, m_hi = 0.005, 15.0
    r_lo, r_hi = 1.1, 1.6
    x = (log(m_kg) - log(m_lo)) / (log(m_hi) - log(m_lo))
    return r_lo + (r_hi - r_lo) * clamp(x, 0.0, 1.0)
end

"""
    feather_depth_allometry(m_kg) → Float64

Feather (insulation) depth [m] as a function of body mass [kg].
Anchored to the budgie value (5.7 mm at 34 g) with the canonical
avian allometric exponent 0.33:  d_f = 0.01737 · m^0.33.
Yields ≈ 3.1 mm at 5 g and ≈ 42 mm at 15 kg.
"""
feather_depth_allometry(m_kg::Real) = 0.01737 * m_kg^0.33

# ── Physical / morphological constants ──────────────────────────────
# All sourced from the BiophysicalBehaviour budgerigar example unless
# otherwise noted in the header docstring.
const ρ_FLESH         = 1000.0u"kg/m^3"
const K_FLESH         = 0.9u"W/m/K"       # baseline (pre-vasodilation)
const K_FLESH_MAX     = 2.8u"W/m/K"       # at full vasodilation
const K_FLESH_STEP    = 0.1u"W/m/K"
const FIBRE_DIAMETER  = 30.0e-6u"m"
const FIBRE_LENGTH    = 23.9e-3u"m"
const FIBRE_DENSITY   = 5.0e7u"1/m^2"  # 5×10^7 /m²  (budgerigar ref: 5000e+04)
const FIBRE_CONDUCT   = 0.209u"W/m/K"
const FAT_FRACTION    = 0.05
const FAT_DENSITY     = 901.0u"kg/m^3"
const K_FAT           = 0.23u"W/m/K"
const REFL_DORSAL     = 0.248
const REFL_VENTRAL    = 0.351
const EMISS_BODY      = 0.99
const EMISS_GROUND    = 1.0
const EMISS_SKY       = 1.0
const VF_SKY          = 0.5
const VF_GROUND       = 0.5
const VF_BUSH         = 0.0
const VENTRAL_FRAC    = 0.5
const GROUND_ALBEDO   = 0.8
const SKIN_WETNESS    = 0.005
const SKIN_WETNESS_MAX  = 0.05    # realistic cap (budgerigar ref: 0.05)
const SKIN_WETNESS_STEP = 0.0025
const T_CORE_DEFAULT  = 311.15u"K"               # 38 °C  (budgerigar setpoint)
const T_CORE_MAX      = T_CORE_DEFAULT + 5.0u"K" # 43 °C hyperthermia ceiling
const T_CORE_STEP     = 0.1u"K"
const PANT_MAX        = 15.0
const PANT_STEP       = 0.01
const PANT_MULT       = 1.0


# =====================================================================
# BirdBody — geometry + insulation bundle
# =====================================================================

"""
    BirdBody

Lightweight container that holds the BG `Body` plus the allometric
inputs used to build it.  Useful for re-using a body across many
microclimates without rebuilding it (and for inspection / plotting of
the geometry independently of the heat balance).
"""
struct BirdBody{B<:Body}
    m_kg::Float64
    axis_ratio::Float64
    feather_depth::typeof(0.0u"m")
    shape::Ellipsoid
    fur::Fur
    fat::Fat
    body::B
end

"""
    build_body_for_mass(m_kg; axis_ratio, feather_depth_m, ρ_flesh,
                              fibre_diameter, fibre_density,
                              fat_fraction, fat_density) → BirdBody

Build an ellipsoidal `BirdBody` for body mass `m_kg` [kg] using the
approved allometric defaults.  Any default may be overridden per call.

NOTE: `Fur` here is the *geometric* insulation layer that wraps the
ellipsoid in BG; the *radiative / conductive* fibre properties that
drive insulation heat flow live separately inside the HE
`InsulationParameters` struct (see `build_organism`).  The BG `Fur`
needs only depth/diameter/density to compute outer-surface area; the
HE `FibreProperties` carries the full optical+thermal description.
"""
function build_body_for_mass(m_kg::Real;
                             axis_ratio::Real      = axis_ratio_allometry(m_kg),
                             feather_depth_m::Real = feather_depth_allometry(m_kg),
                             ρ_flesh               = ρ_FLESH,
                             fibre_diameter        = FIBRE_DIAMETER,
                             fibre_density         = FIBRE_DENSITY,
                             fat_fraction::Real    = FAT_FRACTION,
                             fat_density           = FAT_DENSITY)
    m  = float(m_kg) * u"kg"
    fd = float(feather_depth_m) * u"m"
    # Ellipsoid takes (mass, density, axis_ratio_b, axis_ratio_c) where
    # b and c are the two minor axes (we set them equal — prolate).
    shape = Ellipsoid(m, ρ_flesh, float(axis_ratio), float(axis_ratio))
    # `Fat` and `Fur` are the BG geometric layers (FatLayer/FibrousLayer
    # were renamed in the new API).
    fat   = Fat(float(fat_fraction), fat_density)
    fur   = Fur(fd, fibre_diameter, fibre_density)
    body  = Body(shape, CompositeInsulation(fur, fat))
    return BirdBody(float(m_kg), float(axis_ratio), fd, shape, fur, fat, body)
end


# =====================================================================
# Environment builder
# =====================================================================

"""
    build_environment(micro; V_air, ground_albedo, ϵ_ground, ϵ_sky,
                              elevation, fluid, convection_enhancement)
        → NamedTuple

Build the `(; environment_vars, environment_pars)` NamedTuple that the
HeatExchange/BiophysicalBehaviour endotherm solver expects.  Pulls
state (T_air, T_sky, T_ground, RH, wind, P_atmos, zenith, global
radiation, etc.) straight out of the supplied `Microclimate`.

`V_air` overrides `micro.wind_speed` — pass it to mimic an in-flight
bird (e.g. `V_air = V_mr`).
"""
function build_environment(micro::Microclimate;
                           V_air                  = nothing,
                           ground_albedo::Real    = GROUND_ALBEDO,
                           ϵ_ground::Real         = EMISS_GROUND,
                           ϵ_sky::Real            = EMISS_SKY,
                           elevation              = 0.0u"m",
                           fluid::Int             = 0,                # 0 = air
                           convection_enhancement::Real = 1.0)
    # ── Variables (per-instant state) ─────────────────────────────
    # `Microclimate` already stores RH in [0,1] and global radiation
    # in W/m^2, so just forward.
    V_used = V_air === nothing ? micro.wind_speed : V_air
    ev = EnvironmentalVars(;
        T_air            = uconvert(u"K", micro.air_temperature),
        T_air_reference  = uconvert(u"K", micro.air_temperature),
        T_sky            = uconvert(u"K", micro.sky_temperature),
        T_ground         = uconvert(u"K", micro.ground_temperature),
        T_substrate      = uconvert(u"K", micro.ground_temperature),
        T_bush           = uconvert(u"K", micro.air_temperature),
        T_vegetation     = uconvert(u"K", micro.air_temperature),
        rh               = float(micro.relative_humidity),
        wind_speed       = uconvert(u"m/s", V_used),
        P_atmos          = uconvert(u"Pa",  micro.atmospheric_pressure),
        zenith_angle     = uconvert(u"°",   micro.zenith_angle),
        k_substrate      = 2.79u"W/m/K",                  # rock/soil default
        global_radiation = uconvert(u"W/m^2", micro.global_radiation),
        diffuse_fraction = float(micro.diffuse_fraction),
        shade            = float(micro.shade),
    )

    # ── Parameters (per-environment configuration) ────────────────
    ep = EnvironmentalPars(;
        α_ground               = float(ground_albedo),
        ϵ_ground               = float(ϵ_ground),
        ϵ_sky                  = float(ϵ_sky),
        elevation              = uconvert(u"m", elevation),
        fluid                  = fluid,
        gasfrac                = micro.gas_fractions,
        convection_enhancement = float(convection_enhancement),
    )

    return (; environment_vars = ev, environment_pars = ep)
end


# =====================================================================
# HeatExchangeTraits builder
# =====================================================================

"""
    build_heat_exchange_traits(bb; T_core, Q_basal, metabolic_model,
                                    conduction_fraction, refl_dorsal,
                                    refl_ventral, emiss_body, F_sky,
                                    F_ground, F_bush, ventral_fraction,
                                    solar_orientation, skin_wetness, ...)
        → HeatExchangeTraits

Translate the nicheflappr2 allometric defaults into the new HE struct
API.  The `InsulationParameters` is the *radiative+thermal* description
of the insulation (dorsal/ventral FibreProperties) — this is
independent of the BG `Fur` geometric layer baked into `bb.body`.
"""
function build_heat_exchange_traits(bb::BirdBody;
                                    T_core              = T_CORE_DEFAULT,
                                    Q_basal             = nothing,
                                    metabolic_model     = McKechnieWolf(),
                                    q10::Real           = 1.0,   # 1.0 = no Q10 scaling in normal range (budgerigar ref)
                                    conduction_fraction::Real = 0.0,
                                    refl_dorsal::Real   = REFL_DORSAL,
                                    refl_ventral::Real  = REFL_VENTRAL,
                                    emiss_body::Real    = EMISS_BODY,
                                    F_sky::Real         = VF_SKY,
                                    F_ground::Real      = VF_GROUND,
                                    F_bush::Real        = VF_BUSH,
                                    ventral_fraction::Real = VENTRAL_FRAC,
                                    solar_orientation        = Intermediate(),
                                    skin_wetness::Real       = SKIN_WETNESS,
                                    insulation_wetness::Real = 0.0,
                                    eye_fraction::Real       = 0.0,
                                    bare_skin_fraction::Real = 0.0,
                                    fO2_extract::Real        = 0.25,  # budgerigar ref: 0.25
                                    rq::Real                 = 0.80,
                                    Δ_breath                 = 5.0u"K",  # exhaled 5 K above inhaled (budgerigar ref)
                                    rh_exit::Real            = 1.0,
                                    pant_current::Real       = 1.0,
                                    k_flesh                  = K_FLESH,
                                    respire::Bool            = true,
                                    simulsol_tolerance       = 1e-3u"K",
                                    resp_tolerance::Real     = 1e-5)

    # ── Insulation — same depth/fibres dorsal & ventral by default ──
    fp_dorsal = FibreProperties(;
        diameter     = FIBRE_DIAMETER,
        length       = FIBRE_LENGTH,
        density      = FIBRE_DENSITY,
        depth        = bb.feather_depth,
        reflectance  = float(refl_dorsal),
        conductivity = FIBRE_CONDUCT,
    )
    fp_ventral = FibreProperties(;
        diameter     = FIBRE_DIAMETER,
        length       = FIBRE_LENGTH,
        density      = FIBRE_DENSITY,
        depth        = bb.feather_depth,
        reflectance  = float(refl_ventral),
        conductivity = FIBRE_CONDUCT,
    )
    insulation_pars = InsulationParameters(;
        dorsal                  = fp_dorsal,
        ventral                 = fp_ventral,
        depth_compressed        = bb.feather_depth,     # uncompressed by default
        longwave_depth_fraction = 1.0,
    )

    # ── Conduction — external (ground contact) + internal (fat) ───
    cond_ex = ExternalConductionParameters(; conduction_fraction = float(conduction_fraction))
    cond_in = InternalConductionParameters(; fat_fraction = FAT_FRACTION,
                                             k_flesh      = k_flesh,
                                             k_fat        = K_FAT,
                                             ρ_fat        = FAT_DENSITY)

    # ── Radiation ─────────────────────────────────────────────────
    rad = RadiationParameters(;
        α_body_dorsal     = 1 - float(refl_dorsal),
        α_body_ventral    = 1 - float(refl_ventral),
        ϵ_body_dorsal     = float(emiss_body),
        ϵ_body_ventral    = float(emiss_body),
        F_sky             = float(F_sky),
        F_ground          = float(F_ground),
        F_bush            = float(F_bush),
        ventral_fraction  = float(ventral_fraction),
        solar_orientation = solar_orientation,
    )

    # ── Convection — defaults (forced+free combined inside HE) ───
    conv = ConvectionParameters()

    # ── Evaporation / hydraulics / respiration ────────────────────
    evap = EvaporationParameters(; skin_wetness        = float(skin_wetness),
                                    insulation_wetness = float(insulation_wetness),
                                    eye_fraction       = float(eye_fraction),
                                    bare_skin_fraction = float(bare_skin_fraction),
                                    insulation_fraction = 1.0)
    hyd  = HydraulicParameters()
    resp = RespirationParameters(; fO2_extract = float(fO2_extract),
                                   pant        = float(pant_current),
                                   rq          = float(rq),
                                   Δ_breath    = Δ_breath,
                                   rh_exit     = float(rh_exit))

    # ── Metabolism: McKechnieWolf basal allometry by default ──────
    Q_b = if Q_basal === nothing
        HeatExchange.metabolic_rate(metabolic_model, bb.m_kg * u"kg", uconvert(u"K", T_core))
    else
        isa(Q_basal, Quantity) ? uconvert(u"W", Q_basal) : Q_basal * u"W"
    end
    metab = MetabolismParameters(; T_core       = uconvert(u"K", T_core),
                                    Q_metabolism = uconvert(u"W", Q_b),
                                    q10          = float(q10),
                                    model        = metabolic_model)

    # ── Solver options ────────────────────────────────────────────
    opts = SolveMetabolicRateOptions(; respire            = respire,
                                       simulsol_tolerance = simulsol_tolerance,
                                       resp_tolerance     = float(resp_tolerance))

    # HeatExchangeTraits is a positional struct (no kwargs accepted);
    # canonical order: shape, insulation, cond_ext, cond_int, radiation,
    # convection, evaporation, hydraulic, respiration, metabolism, options.
    return HeatExchangeTraits(bb.shape, insulation_pars, cond_ex, cond_in,
                              rad, conv, evap, hyd, resp, metab, opts)
end


# =====================================================================
# ThermoregulationLimits builder
# =====================================================================

"""
    build_thermoregulation_limits(bb, phys; thermoregulation_mode, ...)

Build a `ThermoregulationLimits` whose **current** values match the
physiology traits `phys` (so the loop starts from the build-time
state) and whose **max/step** values define the bang-bang ranges for
the six sequential effectors:

    1. piloerection   (insulation depth)
    2. uncurl         (shape_b)
    3. vasodilation   (k_flesh)
    4. hyperthermia   (T_core)
    5. panting        (pant)
    6. sweating       (skin_wetness)

`Q_minimum_ref` defaults to the basal metabolic rate from `phys`.
"""
function build_thermoregulation_limits(bb::BirdBody,
                                       phys::HeatExchangeTraits;
                                       thermoregulation_mode = CorePantingSweatingFirst(),
                                       tolerance::Real       = 0.005,
                                       max_iterations::Int   = 1000,
                                       Q_minimum_ref         = nothing,
                                       insulation_depth_max  = bb.feather_depth,
                                       insulation_step::Real = 0.0,
                                       shape_b_max::Real     = 5.0,  # allow uncurling (budgerigar ref: 5.0)
                                       shape_b_step::Real    = 0.1,
                                       k_flesh_max           = K_FLESH_MAX,
                                       k_flesh_step          = K_FLESH_STEP,
                                       T_core_max            = T_CORE_MAX,
                                       T_core_step           = T_CORE_STEP,
                                       pant_max::Real        = PANT_MAX,
                                       pant_step::Real       = PANT_STEP,
                                       pant_cost             = 0.0u"W",
                                       pant_multiplier::Real = PANT_MULT,
                                       skin_wetness_max::Real  = SKIN_WETNESS_MAX,
                                       skin_wetness_step::Real = SKIN_WETNESS_STEP)
    Q_ref = if Q_minimum_ref === nothing
        phys.metabolism_pars.Q_metabolism
    else
        isa(Q_minimum_ref, Quantity) ? uconvert(u"W", Q_minimum_ref) : Q_minimum_ref * u"W"
    end
    T_core_ref = phys.metabolism_pars.T_core

    ThermoregulationLimits(;
        control = RuleBasedSequentialControl(;
            mode           = thermoregulation_mode,
            tolerance      = float(tolerance),
            max_iterations = max_iterations,
        ),
        Q_minimum_ref = uconvert(u"W", Q_ref),
        insulation = InsulationLimits(;
            dorsal = SteppedParameter(;
                current   = phys.insulation_pars.dorsal.depth,
                reference = phys.insulation_pars.dorsal.depth,
                max       = uconvert(u"m", insulation_depth_max),
                step      = float(insulation_step),
            ),
            ventral = SteppedParameter(;
                current   = phys.insulation_pars.ventral.depth,
                reference = phys.insulation_pars.ventral.depth,
                max       = uconvert(u"m", insulation_depth_max),
                step      = float(insulation_step),
            ),
        ),
        shape_b = SteppedParameter(;
            current = bb.axis_ratio,
            max     = float(shape_b_max),
            step    = float(shape_b_step),
        ),
        k_flesh = SteppedParameter(;
            current = phys.conduction_pars_internal.k_flesh,
            max     = uconvert(u"W/m/K", k_flesh_max),
            step    = uconvert(u"W/m/K", k_flesh_step),
        ),
        T_core = SteppedParameter(;
            current   = T_core_ref,
            reference = T_core_ref,
            max       = uconvert(u"K", T_core_max),
            step      = uconvert(u"K", T_core_step),
        ),
        panting = PantingLimits(;
            pant = SteppedParameter(;
                current = phys.respiration_pars.pant,
                max     = float(pant_max),
                step    = float(pant_step),
            ),
            cost       = uconvert(u"W", pant_cost),
            multiplier = float(pant_multiplier),
            T_core_ref = T_core_ref,
        ),
        skin_wetness = SteppedParameter(;
            current = phys.evaporation_pars.skin_wetness,
            max     = float(skin_wetness_max),
            step    = float(skin_wetness_step),
        ),
    )
end


# =====================================================================
# Organism assembly
# =====================================================================

"""
    build_organism(bb; T_core, Q_basal, thermoregulation_mode, ...) → Organism

Build the physiology traits + behaviour traits + assemble an
`Organism(body, organism_traits)`.  Kwargs accepted by either
`build_heat_exchange_traits` or `build_thermoregulation_limits` are
forwarded — see those for the full list.
"""
function build_organism(bb::BirdBody;
                        # Physiology kwargs
                        T_core              = T_CORE_DEFAULT,
                        Q_basal             = nothing,
                        metabolic_model     = McKechnieWolf(),
                        # Behaviour kwargs
                        thermoregulation_mode = CorePantingSweatingFirst(),
                        tolerance::Real       = 0.005,
                        max_iterations::Int   = 1000,
                        Q_minimum_ref         = nothing,
                        # Activity
                        activity              = Diurnal(),
                        # Kwargs forwarded to build_thermoregulation_limits
                        # (e.g. pant_step, pant_multiplier, pant_max, shape_b_max, shape_b_step)
                        thermoregulation_kwargs::NamedTuple = NamedTuple(),
                        # Bucket for any other physiology kwarg → build_heat_exchange_traits
                        kwargs...)
    phys   = build_heat_exchange_traits(bb;
                                        T_core          = T_core,
                                        Q_basal         = Q_basal,
                                        metabolic_model = metabolic_model,
                                        kwargs...)
    limits = build_thermoregulation_limits(bb, phys;
                                           thermoregulation_mode = thermoregulation_mode,
                                           tolerance      = tolerance,
                                           max_iterations = max_iterations,
                                           Q_minimum_ref  = Q_minimum_ref,
                                           thermoregulation_kwargs...)
    behav  = BehavioralTraits(; thermoregulation = limits, activity = activity)
    return Organism(bb.body, OrganismTraits(Endotherm(), phys, behav))
end


# =====================================================================
# Top-level driver — mirrors the budgerigar example
# =====================================================================

"""
    run_body_thermoregulation(bb, micro;
                              V_air, T_skin_init, T_insulation_init,
                              Q_gen_init, q10_hot, environment_kwargs,
                              organism_kwargs)
        → (; organism, environment, endotherm_out)

Run BB's `thermoregulate` on a pre-built `BirdBody` under the given
`Microclimate`.  Returns a NamedTuple matching the shape of the
budgerigar demo, so downstream code can do
`out.endotherm_out.energy_fluxes.Q_gen`,
`out.endotherm_out.thermoregulation.T_skin`, etc.

This is the "fluxes + thermoregulation" entry point.  The whole-animal
module wraps a custom version of this loop that injects wing fluxes
at step 0 before any behaviours are deployed.

**Automatic Q10:**  When `T_air ≥ T_core_max` (hyperthermic range) and the
caller has not supplied `q10` inside `organism_kwargs`, the metabolic Q10 is
automatically set to `q10_hot` (default 2.0).  Below the threshold q10 = 1.0.
Supply `organism_kwargs = (q10 = custom,)` to override on any individual run.

Initial-condition defaults:
  T_skin       = T_core − 5 K
  T_insulation = T_air + 0.5·(T_core − T_air)
  Q_gen        = 0 W   (the solver re-derives it on the first call)
"""
function run_body_thermoregulation(bb::BirdBody, micro::Microclimate;
                                   V_air                  = nothing,
                                   T_skin_init            = nothing,
                                   T_insulation_init      = nothing,
                                   Q_gen_init             = 0.0u"W",
                                   q10_hot::Real          = 2.0,
                                   environment_kwargs::NamedTuple = NamedTuple(),
                                   organism_kwargs::NamedTuple    = NamedTuple())
    # ── Auto Q10: ramp to q10_hot once T_air enters the hyperthermic range ─
    if !haskey(organism_kwargs, :q10)
        T_air_K = uconvert(u"K", micro.air_temperature)
        auto_q10 = T_air_K >= T_CORE_MAX ? float(q10_hot) : 1.0
        organism_kwargs = merge(organism_kwargs, (; q10 = auto_q10))
    end
    organism    = build_organism(bb; organism_kwargs...)
    environment = build_environment(micro; V_air = V_air, environment_kwargs...)

    # ── Initial-condition guesses ─────────────────────────────────
    T_core = organism.traits.physiology.metabolism_pars.T_core
    T_air  = environment.environment_vars.T_air
    T_s    = T_skin_init       === nothing ? T_core - 5.0u"K" : uconvert(u"K", T_skin_init)
    T_i    = T_insulation_init === nothing ?
                T_air + 0.5 * (T_core - T_air) :
                uconvert(u"K", T_insulation_init)
    Q_g    = isa(Q_gen_init, Quantity) ? uconvert(u"W", Q_gen_init) : Q_gen_init * u"W"

    endotherm_out = thermoregulate(organism, environment, Q_g, T_s, T_i)
    return (; organism = organism, environment = environment,
              endotherm_out = endotherm_out)
end

"""
    run_body_thermoregulation_for_mass(m_kg, micro;
                                       V_air = nothing, bird_type = :other,
                                       build_kwargs = NamedTuple(),
                                       kwargs...) → NamedTuple

One-call helper: build the allometric `BirdBody` for `m_kg`, pick
`V_air = V_mr(m_kg)` from AFPT (when `V_air === nothing`), and run
the body thermoregulation.  Returns the same NamedTuple as
`run_body_thermoregulation`, plus the underlying `BirdBody` as `bird`.
"""
function run_body_thermoregulation_for_mass(m_kg::Real, micro::Microclimate;
                                            V_air        = nothing,
                                            bird_type::Symbol = :other,
                                            build_kwargs::NamedTuple = NamedTuple(),
                                            kwargs...)
    bb = build_body_for_mass(m_kg; build_kwargs...)
    V  = V_air === nothing ? afpt_v_mr(m_kg; type = bird_type) * u"m/s" : V_air
    r  = run_body_thermoregulation(bb, micro; V_air = V, kwargs...)
    return (; bird = bb, r...)
end


# =====================================================================
# AFPT convenience
# =====================================================================

"""
    afpt_v_mr(m_kg; type = :other) → Float64 [m/s]

Maximum-range airspeed for a canonical-allometry AFPT bird at body
mass `m_kg`.  Used as the default forced-convection wind speed for an
in-flight body.
"""
function afpt_v_mr(m_kg::Real; type::Symbol = :other)
    bird = build_afpt_bird(m_kg, wing_span_allometry(m_kg);
                           wingArea = wing_area_allometry(m_kg),
                           type     = type)
    return find_maximum_range_speed(bird)
end


# =====================================================================
# Exports
# =====================================================================

export BirdBody,
       build_body_for_mass,
       build_environment, build_heat_exchange_traits,
       build_thermoregulation_limits, build_organism,
       run_body_thermoregulation, run_body_thermoregulation_for_mass,
       afpt_v_mr,
       axis_ratio_allometry, feather_depth_allometry,
       # Re-exported BB controller types for caller convenience
       CoreFirst, CoreAndPantingFirst, CorePantingSweatingFirst,
       # Constants
       ρ_FLESH, FIBRE_DIAMETER, FIBRE_DENSITY,
       REFL_DORSAL, REFL_VENTRAL,
       EMISS_BODY, EMISS_GROUND, EMISS_SKY,
       VF_SKY, VF_GROUND, T_CORE_DEFAULT

end # module BodyHeatBalance
