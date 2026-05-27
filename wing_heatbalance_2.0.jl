# =====================================================================
# wing_heatbalance_2.0.jl
#
# Full wing heat balance built on the BiophysicalEcology suite:
#   • BiophysicalGeometry.jl  — flat-plate geometry per wing strip
#   • HeatExchange.jl         — convection + radiation_in / radiation_out / solar
#   • FluidProperties.jl      — dry-air properties, atmospheric_pressure
#
# NO heat-transfer correlations are defined in this file — every
# convective and radiative flux comes from the upstream packages.
# We only:
#   1. Build a `Plate` `Body` matching each `WingElement`,
#   2. Provide a `Microclimate` container of environmental variables,
#   3. Call the package functions for each element / surface / instant,
#   4. Aggregate over the wingbeat.
#
# Temperature-assignment helpers (`WingTemperatures`, `uniform_temperature`,
# etc.) are kept here unchanged from `wing_convection_2.0.jl`; they are
# inputs to the heat balance, not heat-transfer formulas.
# =====================================================================

include("environment.jl")
include("wing_kinematics_2.0.jl")
include("convection_regimes.jl")

module WingHeatBalance

using ..WingPlates
using ..WingKinematics
using ..FlightEnvironment
using ..FlightEnvironment: Microclimate, microclimate_at_altitude,
                           microclimate_from_microresult
using ..ConvectionRegimes
using ..ConvectionRegimes: PlateConvectionRegime, LaminarPlate, TurbulentPlate,
                           MixedPlate, nusselt_plate, reynolds_number, convection_h
using Unitful

using BiophysicalGeometry
using BiophysicalGeometry: Body, Plate, Naked, total_area, silhouette_area
using HeatExchange: convection, radiation_in, radiation_out, solar,
                    Absorptivities, Emissivities, ViewFactors,
                    SolarConditions, EnvironmentTemperatures,
                    DorsalVentral, Air,
                    ScaledDimension, VolumeCubeRoot
using FluidProperties: dry_air_properties, atmospheric_pressure, GasFractions


# =====================================================================
# Surface-temperature assignment (carried over from wing_convection)
# =====================================================================

"""
    WingTemperatures

Dorsal and ventral element surface temperatures for a discretised wing.
"""
@kwdef struct WingTemperatures{L<:Unitful.Length, T<:Number}
    element_ids::Vector{Int}
    span_positions::Vector{L}
    T_dorsal::Vector{T}
    T_ventral::Vector{T}
end

to_K(T::Quantity) = uconvert(u"K", T)
to_K(T::Real)     = T * u"K"
to_C(T_K::Quantity) = uconvert(u"°C", T_K)

_unpack(wd::WingDiscretization) =
    ([e.element_id    for e in wd.elements],
     [e.span_position for e in wd.elements])

uniform_temperature(wd::WingDiscretization, T) = begin
    ids, spans = _unpack(wd); n = length(ids)
    WingTemperatures(ids, spans, fill(T, n), fill(T, n))
end

function linear_gradient(wd::WingDiscretization, T_root, T_tip)
    ids, spans = _unpack(wd); L = wd.wing.wing_length
    T_root_K = to_K(T_root); T_tip_K = to_K(T_tip)
    temps_K = [T_root_K + (T_tip_K - T_root_K) * (s / L) for s in spans]
    temps   = [to_C(t) for t in temps_K]
    WingTemperatures(ids, spans, temps, copy(temps))
end

function exponential_decay(wd::WingDiscretization, T_root, T_tip; decay_rate = 3.0)
    ids, spans = _unpack(wd); L = wd.wing.wing_length
    T_root_K = to_K(T_root); T_tip_K = to_K(T_tip)
    ΔT_K = T_root_K - T_tip_K
    temps_K = [T_tip_K + ΔT_K * exp(-decay_rate * s / L) for s in spans]
    temps   = [to_C(t) for t in temps_K]
    WingTemperatures(ids, spans, temps, copy(temps))
end

function custom_per_element(wd::WingDiscretization, temps::Vector)
    ids, spans = _unpack(wd)
    length(temps) == length(ids) ||
        error("Expected $(length(ids)) temperatures, got $(length(temps))")
    WingTemperatures(ids, spans, copy(temps), copy(temps))
end

function from_function(wd::WingDiscretization, f::Function)
    ids, spans = _unpack(wd)
    temps_raw = [f(s) for s in spans]
    temps = [isa(t, Quantity) && unit(t) == u"K" ? to_C(t) : t for t in temps_raw]
    WingTemperatures(ids, spans, temps, copy(temps))
end

dorsal_ventral_split(wd::WingDiscretization, T_dorsal, T_ventral)::WingTemperatures = begin
    ids, spans = _unpack(wd); n = length(ids)
    WingTemperatures(ids, spans, fill(T_dorsal, n), fill(T_ventral, n))
end

dorsal_ventral_split(wd::WingDiscretization, f_d::Function, f_v::Function)::WingTemperatures = begin
    ids, spans = _unpack(wd)
    WingTemperatures(ids, spans, [f_d(s) for s in spans], [f_v(s) for s in spans])
end


# =====================================================================
# Microclimate container
# =====================================================================
#
# `Microclimate` and `microclimate_at_altitude` are defined in
# `environment.jl` (module `FlightEnvironment`) so they can be shared
# with a future body heat-balance module.  They are re-exported below
# for backward compatibility.


# =====================================================================
# Air properties (lightweight wrapper around FluidProperties)
# =====================================================================

"""
    AirProperties

Thermophysical properties of dry air at one (T, P) point.
Wraps the values returned by `FluidProperties.dry_air_properties`.
"""
@kwdef struct AirProperties{T<:Quantity}
    T_air::T
    P::Quantity
    altitude::Quantity
    ρ::Quantity
    μ::Quantity
    ν::Quantity
    k_air::Quantity
    Pr::Float64
end

function air_properties(T_air; altitude = 0.0u"m", P = nothing,
                        gas_fractions::GasFractions = GasFractions())
    T_K    = isa(T_air, Quantity) ? uconvert(u"K", T_air) : T_air * u"K"
    P_used = P === nothing ? atmospheric_pressure(altitude) : P
    fp     = dry_air_properties(T_K, P_used; gas_fractions = gas_fractions)
    cp_air = 1005.8u"J/(kg*K)"
    Pr_val = ustrip(u"NoUnits",
                    fp.dynamic_viscosity * cp_air / fp.thermal_conductivity)
    return AirProperties(
        T_air    = T_air,
        P        = uconvert(u"Pa", P_used),
        altitude = uconvert(u"m", altitude),
        ρ        = uconvert(u"kg/m^3",  fp.density),
        μ        = uconvert(u"Pa*s",    fp.dynamic_viscosity),
        ν        = uconvert(u"m^2/s",   fp.kinematic_viscosity),
        k_air    = uconvert(u"W/(m*K)", fp.thermal_conductivity),
        Pr       = Float64(Pr_val),
    )
end


# =====================================================================
# Wing-element → Plate Body
# =====================================================================

"""
    element_to_body(elem; tissue_density = 1000 kg/m³) → BiophysicalGeometry.Body

Build a `Body(Plate(...), Naked())` whose `Geometry.length` exactly
matches the wing element:

    length_skin = chord
    width_skin  = span_length         (strip span-wise width)
    height_skin = thickness

so that `total_area`, `silhouette_area` and the convection
characteristic-dimension dispatch all match the strip.
"""
function element_to_body(elem::WingElement; tissue_density = 1000.0u"kg/m^3")
    L = elem.chord_length     # streamwise (long) dimension
    W = elem.span_length      # span-wise width of the strip

    # Use a negligibly thin height so that BiophysicalGeometry.total_area ≈ 2·L·W
    # (dorsal + ventral faces only).  The real `elem.thickness` is physically
    # meaningful for structural mass, but for radiation HeatExchange.solar /
    # radiation_in / radiation_out multiply `total_area` by the view factors.
    # A true plate has edge area ≈ 0, so H → 0 is the correct limit.
    # With thickness_factor = 0.05, H ≈ 0.05·L and the four edge panels add
    # ~40 % to total_area, inflating Q_solar and Q_lw proportionally.
    H = min(W, L) * 1e-4      # ≪ L and W; edges negligible

    V = L * W * H
    b = L / W
    c = L / H
    m = tissue_density * V
    plate = Plate(m, tissue_density, b, c)
    return Body(plate, Naked())
end


# =====================================================================
# Heat-balance results
# =====================================================================

"""
    ElementHeatBalance

Per-element heat flows at one instant, every value from HeatExchange.jl.
"""
@kwdef struct ElementHeatBalance
    element_id::Int
    chord::Quantity
    span_position::Quantity
    airspeed::Quantity

    # Convection (dorsal + ventral handled separately)
    Q_conv_dorsal::Quantity
    Q_conv_ventral::Quantity
    Q_conv::Quantity
    h_conv::Quantity        # combined heat-transfer coefficient (avg surface)

    # Solar (positive = absorbed)
    Q_solar::Quantity
    Q_solar_direct::Quantity
    Q_solar_sky::Quantity
    Q_solar_ground::Quantity

    # Longwave (positive in = absorbed; positive out = emitted)
    Q_lw_in::Quantity
    Q_lw_out::Quantity
    Q_lw_net::Quantity      # = Q_lw_in − Q_lw_out

    # Energy balance summary (positive = heat gained by wing)
    Q_net::Quantity         # = Q_solar + Q_lw_net − Q_conv
end


"""
    WingHeatSnapshot

Whole-wing instantaneous heat balance summed from the element results.
"""
@kwdef struct WingHeatSnapshot
    t::Quantity
    elements::Vector{ElementHeatBalance}
    Q_conv::Quantity
    Q_solar::Quantity
    Q_lw_in::Quantity
    Q_lw_out::Quantity
    Q_lw_net::Quantity
    Q_net::Quantity
end


"""
    WingbeatHeatBalance

Cycle-averaged heat balance plus the per-step series.
"""
@kwdef struct WingbeatHeatBalance{TT<:Unitful.Time}
    period::TT
    n_steps::Int
    snapshots::Vector{WingHeatSnapshot}
    times::Vector{TT}

    # Time series and means
    Q_conv_series::Vector{<:Quantity};   Q_conv_mean::Quantity
    Q_solar_series::Vector{<:Quantity};  Q_solar_mean::Quantity
    Q_lw_in_series::Vector{<:Quantity};  Q_lw_in_mean::Quantity
    Q_lw_out_series::Vector{<:Quantity}; Q_lw_out_mean::Quantity
    Q_lw_net_series::Vector{<:Quantity}; Q_lw_net_mean::Quantity
    Q_net_series::Vector{<:Quantity};    Q_net_mean::Quantity
end


# =====================================================================
# Per-element heat balance (delegates to HeatExchange)
# =====================================================================

# ─────────────────────────────────────────────────────────────────
# Default radiative properties — taken from the budgerigar example
# in BiophysicalBehaviour.jl so we are consistent with the published
# BiophysicalEcology defaults.  Sources:
#
#   • α_body_dorsal / α_body_ventral come from the budgerigar's
#     `insulation_reflectance_*` fields in examples/budgerigar.jl
#     (α = 1 − reflectance):
#         insulation_reflectance_dorsal  = 0.248  →  α_d = 0.752
#         insulation_reflectance_ventral = 0.351  →  α_v = 0.649
#     https://github.com/BiophysicalEcology/BiophysicalBehaviour.jl/
#         blob/main/examples/budgerigar.jl
#
#   • ϵ_body_dorsal, ϵ_body_ventral, F_sky, F_ground come from
#     `example_radiation_pars` (the defaults the budgie example uses
#     via `radiation_pars = example_radiation_pars()`):
#         ϵ_body_dorsal  = 0.99,  ϵ_body_ventral = 0.99
#         F_sky          = 0.5,   F_ground       = 0.5
#     https://github.com/BiophysicalEcology/BiophysicalBehaviour.jl/
#         blob/main/src/endotherm/example_variables_and_parameters.jl
#
#   • α_ground, ϵ_ground, ϵ_sky come from `example_environment_pars`:
#         α_ground = 0.8   (ground SOLAR ABSORPTIVITY, so the ground
#                          REFLECTS 1 − 0.8 = 0.20 of incoming solar —
#                          the grass-like value).  Passed straight
#                          into HeatExchange's `Absorptivities.ground`.
#         ϵ_ground = 1.0,  ϵ_sky = 1.0
#     (same file as above).
# ─────────────────────────────────────────────────────────────────
const _DEFAULT_BODY_ABS_DORSAL  = 0.752   # = 1 − 0.248 (budgie dorsal reflectance)
const _DEFAULT_BODY_ABS_VENTRAL = 0.649   # = 1 − 0.351 (budgie ventral reflectance)
const _DEFAULT_BODY_EMISS       = 0.99    # ϵ_body_dorsal = ϵ_body_ventral in example_radiation_pars
const _DEFAULT_GROUND_ALBEDO    = 0.8     # ground SOLAR ABSORPTIVITY (NOT reflectance);
                                          # matches example_environment_pars α_ground = 0.8
const _DEFAULT_GROUND_EMISS     = 1.0     # example_environment_pars ϵ_ground = 1.0
const _DEFAULT_SKY_EMISS        = 1.0     # example_environment_pars ϵ_sky    = 1.0
const _DEFAULT_SKY_VF           = 0.5     # example_radiation_pars F_sky      = 0.5
const _DEFAULT_GROUND_VF        = 0.5     # example_radiation_pars F_ground   = 0.5


"""
    element_heat_balance(elem, body, V_air, T_dorsal, T_ventral, micro;
                         absorptivities, emissivities, view_factors,
                         characteristic_dim_formula, conduction_fraction,
                         silhouette_area_override) → ElementHeatBalance

All fluxes come directly from `HeatExchange.{convection, radiation_in,
radiation_out, solar}`.  Dorsal and ventral surfaces are handled
separately so the dorsal/ventral temperature contrast carried in
`WingTemperatures` is preserved.

`silhouette_area_override` lets the caller substitute a flat-plate
silhouette of `length×width × cos(zenith)` (the natural choice for a
wing held horizontal) rather than the package's orientation-averaged
default.  `nothing` falls back to `BiophysicalGeometry.silhouette_area`.
"""
function element_heat_balance(elem::WingElement,
                              body,
                              V_air,
                              T_dorsal,
                              T_ventral,
                              micro::Microclimate;
                              absorptivities::Absorptivities,
                              emissivities::Emissivities,
                              view_factors::ViewFactors,
                              characteristic_dim_formula = ScaledDimension(:length_skin),
                              conduction_fraction::Real = 0.0,
                              silhouette_area_override = nothing,
                              convection_model::Union{Nothing,PlateConvectionRegime} = nothing,
                              air::Union{Nothing,AirProperties} = nothing)::ElementHeatBalance

    T_d_K = uconvert(u"K", isa(T_dorsal,  Quantity) ? T_dorsal  : T_dorsal  * u"°C")
    T_v_K = uconvert(u"K", isa(T_ventral, Quantity) ? T_ventral : T_ventral * u"°C")
    T_air = uconvert(u"K", micro.air_temperature)
    A_d   = elem.has_dorsal  ? elem.dorsal_area  : 0.0u"m^2"
    A_v   = elem.has_ventral ? elem.ventral_area : 0.0u"m^2"

    # ── Convection ───────────────────────────────────────────────────
    # Two code paths:
    #   (a) convection_model === nothing  → HeatExchange.convection
    #                                       (default turbulent flat plate;
    #                                        package default)
    #   (b) convection_model isa PlateConvectionRegime → explicit
    #       flat-plate correlation using local chord, air.ν, air.Pr,
    #       air.k_air with Re = V_air · chord / ν.
    Q_conv_d = 0.0u"W"; Q_conv_v = 0.0u"W"; h_avg = 0.0u"W/(m^2*K)"
    if convection_model isa PlateConvectionRegime
        # Need an `AirProperties` block; build it from the microclimate
        # if the caller did not provide one.  This is the slow path
        # (re-derives gas properties per element); the snapshot driver
        # passes a shared `air` argument to avoid the cost.
        air_used = air === nothing ?
                        air_properties(micro.air_temperature;
                                       P = micro.atmospheric_pressure,
                                       gas_fractions = micro.gas_fractions) :
                        air
        L  = elem.chord_length
        Re = reynolds_number(V_air, L, air_used.ν)
        h  = convection_h(convection_model, Re, air_used.Pr, air_used.k_air, L)
        n_faces = 0
        if A_d > 0u"m^2"
            Q_conv_d = uconvert(u"W", h * A_d * (T_d_K - T_air))
            h_avg   += h
            n_faces += 1
        end
        if A_v > 0u"m^2"
            Q_conv_v = uconvert(u"W", h * A_v * (T_v_K - T_air))
            h_avg   += h
            n_faces += 1
        end
        if n_faces == 2
            h_avg /= 2
        end
    else
        # HeatExchange.convection (package default — turbulent flat plate)
        if A_d > 0u"m^2"
            cd = convection(; body = body, area = A_d,
                              air_temperature       = T_air,
                              surface_temperature   = T_d_K,
                              wind_speed            = V_air,
                              atmospheric_pressure  = micro.atmospheric_pressure,
                              fluid                 = Air(),
                              gas_fractions         = micro.gas_fractions,
                              characteristic_dimension_formula = characteristic_dim_formula)
            Q_conv_d = uconvert(u"W", cd.convection_flow)
            h_avg   += cd.heat_transfer_coefficient.combined
        end
        if A_v > 0u"m^2"
            cv = convection(; body = body, area = A_v,
                              air_temperature       = T_air,
                              surface_temperature   = T_v_K,
                              wind_speed            = V_air,
                              atmospheric_pressure  = micro.atmospheric_pressure,
                              fluid                 = Air(),
                              gas_fractions         = micro.gas_fractions,
                              characteristic_dimension_formula = characteristic_dim_formula)
            Q_conv_v = uconvert(u"W", cv.convection_flow)
            h_avg   += cv.heat_transfer_coefficient.combined
        end
        if A_d > 0u"m^2" && A_v > 0u"m^2"
            h_avg /= 2
        end
    end

    # ── Radiation: build the env structs HeatExchange expects ──────
    env_T = EnvironmentTemperatures(T_air,
                                    uconvert(u"K", micro.sky_temperature),
                                    uconvert(u"K", micro.ground_temperature),
                                    T_air, T_air, T_air)

    # Solar — use the explicit-silhouette form so we can orient the
    # wing as a flat plate with its dorsal face up.
    sil = silhouette_area_override === nothing ?
            silhouette_area(body, micro.zenith_angle) :
            silhouette_area_override
    cond_area = total_area(body) * conduction_fraction
    sc = SolarConditions(zenith_angle     = uconvert(u"°", micro.zenith_angle),
                         global_radiation = uconvert(u"W/m^2", micro.global_radiation),
                         diffuse_fraction = float(micro.diffuse_fraction),
                         shade            = float(micro.shade))
    sol = solar(body, absorptivities, view_factors, sc, sil, cond_area)
    Q_sol = uconvert(u"W", sol.solar_flow)

    # Longwave in / out
    rin  = radiation_in(body, view_factors, emissivities, env_T;
                        conduction_fraction = conduction_fraction)
    rout = radiation_out(body, view_factors, emissivities,
                         conduction_fraction, T_d_K, T_v_K)
    Q_lw_in  = uconvert(u"W", rin.longwave_flow_in)
    Q_lw_out = uconvert(u"W", rout.longwave_flow_out)
    Q_lw_net = Q_lw_in - Q_lw_out

    Q_conv  = Q_conv_d + Q_conv_v
    Q_net   = Q_sol + Q_lw_net - Q_conv

    return ElementHeatBalance(
        element_id      = elem.element_id,
        chord           = elem.chord_length,
        span_position   = elem.span_position,
        airspeed        = uconvert(u"m/s", V_air),
        Q_conv_dorsal   = Q_conv_d,
        Q_conv_ventral  = Q_conv_v,
        Q_conv          = Q_conv,
        h_conv          = uconvert(u"W/(m^2*K)", h_avg),
        Q_solar         = Q_sol,
        Q_solar_direct  = uconvert(u"W", sol.solar_direct_flow),
        Q_solar_sky     = uconvert(u"W", sol.solar_sky_flow),
        Q_solar_ground  = uconvert(u"W", sol.solar_substrate_flow),
        Q_lw_in         = Q_lw_in,
        Q_lw_out        = Q_lw_out,
        Q_lw_net        = Q_lw_net,
        Q_net           = Q_net,
    )
end


# =====================================================================
# Defaults
# =====================================================================

"""
    default_absorptivities(; α_d, α_v, α_ground) → HeatExchange.Absorptivities
"""
default_absorptivities(; α_d = _DEFAULT_BODY_ABS_DORSAL,
                          α_v = _DEFAULT_BODY_ABS_VENTRAL,
                          α_ground = _DEFAULT_GROUND_ALBEDO) =
    Absorptivities(; body = DorsalVentral(α_d, α_v), ground = α_ground)

"""
    default_emissivities(; ε_d, ε_v, ε_ground, ε_sky) → HeatExchange.Emissivities
"""
default_emissivities(; ε_d = _DEFAULT_BODY_EMISS,
                        ε_v = _DEFAULT_BODY_EMISS,
                        ε_ground = _DEFAULT_GROUND_EMISS,
                        ε_sky    = _DEFAULT_SKY_EMISS) =
    Emissivities(; body = DorsalVentral(ε_d, ε_v),
                   ground = ε_ground, sky = ε_sky)

"""
    default_view_factors(; F_sky, F_ground) → HeatExchange.ViewFactors
"""
default_view_factors(; F_sky = _DEFAULT_SKY_VF, F_ground = _DEFAULT_GROUND_VF) =
    ViewFactors(F_sky, F_ground, 0.0, 0.0)


# =====================================================================
# Wing-level snapshot and wingbeat cycle
# =====================================================================

"""
    wing_bodies(wd; tissue_density) → Vector{Body}

Pre-build one `BiophysicalGeometry.Body(Plate, Naked)` per wing element.
"""
wing_bodies(wd::WingDiscretization; tissue_density = 1000.0u"kg/m^3") =
    [element_to_body(e; tissue_density = tissue_density) for e in wd.elements]


"""
    compute_heatbalance_snapshot(wd, bodies, ev, wt, micro;
                                 absorptivities, emissivities, view_factors, …)
        → WingHeatSnapshot
"""
function compute_heatbalance_snapshot(wd::WingDiscretization,
                                      bodies::Vector,
                                      ev::ElementVelocities,
                                      wt::WingTemperatures,
                                      micro::Microclimate;
                                      absorptivities::Absorptivities = default_absorptivities(),
                                      emissivities::Emissivities    = default_emissivities(),
                                      view_factors::ViewFactors     = default_view_factors(),
                                      characteristic_dim_formula    = ScaledDimension(:length_skin),
                                      conduction_fraction::Real     = 0.0,
                                      use_flat_plate_silhouette::Bool = true,
                                      convection_model::Union{Nothing,PlateConvectionRegime} = nothing,
                                      air::Union{Nothing,AirProperties} = nothing)
    n = length(wd.elements)
    elems = Vector{ElementHeatBalance}(undef, n)
    Q_conv = 0.0u"W"; Q_sol = 0.0u"W"
    Q_lw_in = 0.0u"W"; Q_lw_out = 0.0u"W"

    cosθ = max(cos(uconvert(u"rad", micro.zenith_angle)), 0.0)

    # If using the analytic regime, compute the per-snapshot air
    # properties once (they depend on micro only).
    air_used = air
    if convection_model isa PlateConvectionRegime && air_used === nothing
        air_used = air_properties(micro.air_temperature;
                                  P = micro.atmospheric_pressure,
                                  gas_fractions = micro.gas_fractions)
    end

    for i in 1:n
        elem = wd.elements[i]
        body = bodies[i]
        sil_override = if use_flat_plate_silhouette
            elem.dorsal_area * cosθ
        else
            nothing
        end

        elems[i] = element_heat_balance(
            elem, body, ev.realised_airspeed[i],
            wt.T_dorsal[i], wt.T_ventral[i], micro;
            absorptivities = absorptivities,
            emissivities   = emissivities,
            view_factors   = view_factors,
            characteristic_dim_formula = characteristic_dim_formula,
            conduction_fraction        = conduction_fraction,
            silhouette_area_override   = sil_override,
            convection_model           = convection_model,
            air                        = air_used,
        )

        Q_conv  += elems[i].Q_conv
        Q_sol   += elems[i].Q_solar
        Q_lw_in += elems[i].Q_lw_in
        Q_lw_out += elems[i].Q_lw_out
    end

    Q_lw_net = Q_lw_in - Q_lw_out
    Q_net    = Q_sol + Q_lw_net - Q_conv
    return WingHeatSnapshot(t = ev.t, elements = elems,
                            Q_conv = Q_conv, Q_solar = Q_sol,
                            Q_lw_in = Q_lw_in, Q_lw_out = Q_lw_out,
                            Q_lw_net = Q_lw_net, Q_net = Q_net)
end


"""
    compute_wingbeat_heatbalance(kin, wd, wt, micro; …) → WingbeatHeatBalance
"""
function compute_wingbeat_heatbalance(kin::FlappingKinematics,
                                      wd::WingDiscretization,
                                      wt::WingTemperatures,
                                      micro::Microclimate;
                                      n_steps::Int = 40,
                                      # Bird's forward flight speed (free-stream
                                      # airspeed of the oncoming flow).  This is
                                      # NOT the ambient `micro.wind_speed`; it is
                                      # the speed at which the bird is travelling
                                      # through the air (e.g. V_mr from the power
                                      # model).  Default 0 = hovering, matching
                                      # `compute_element_velocities`.
                                      V_forward                  = 0.0u"m/s",
                                      tissue_density             = 1000.0u"kg/m^3",
                                      absorptivities::Absorptivities = default_absorptivities(),
                                      emissivities::Emissivities    = default_emissivities(),
                                      view_factors::ViewFactors     = default_view_factors(),
                                      characteristic_dim_formula    = ScaledDimension(:length_skin),
                                      conduction_fraction::Real     = 0.0,
                                      use_flat_plate_silhouette::Bool = true,
                                      convection_model::Union{Nothing,PlateConvectionRegime} = nothing)
    bodies = wing_bodies(wd; tissue_density = tissue_density)

    freq_Hz = isa(kin.frequency, Quantity) ? ustrip(u"Hz", kin.frequency) : kin.frequency
    period  = 1.0u"s" / freq_Hz
    times   = [(i - 1) * period / n_steps for i in 1:n_steps]

    # Pre-compute the (micro-only) air properties once when an analytic
    # convection regime is requested, so we don't re-derive gas mixture
    # properties for every element/snapshot.
    air_shared = if convection_model isa PlateConvectionRegime
        air_properties(micro.air_temperature;
                       P = micro.atmospheric_pressure,
                       gas_fractions = micro.gas_fractions)
    else
        nothing
    end

    snaps = Vector{WingHeatSnapshot}(undef, n_steps)
    for (j, t) in enumerate(times)
        ev = compute_element_velocities(kin, wd, t, V_forward = V_forward)
        snaps[j] = compute_heatbalance_snapshot(
            wd, bodies, ev, wt, micro;
            absorptivities = absorptivities,
            emissivities   = emissivities,
            view_factors   = view_factors,
            characteristic_dim_formula = characteristic_dim_formula,
            conduction_fraction        = conduction_fraction,
            use_flat_plate_silhouette  = use_flat_plate_silhouette,
            convection_model           = convection_model,
            air                        = air_shared,
        )
    end

    Q_conv_s   = [s.Q_conv   for s in snaps]
    Q_solar_s  = [s.Q_solar  for s in snaps]
    Q_lw_in_s  = [s.Q_lw_in  for s in snaps]
    Q_lw_out_s = [s.Q_lw_out for s in snaps]
    Q_lw_net_s = [s.Q_lw_net for s in snaps]
    Q_net_s    = [s.Q_net    for s in snaps]

    return WingbeatHeatBalance(
        period          = period,
        n_steps         = n_steps,
        snapshots       = snaps,
        times           = times,
        Q_conv_series   = Q_conv_s,   Q_conv_mean   = sum(Q_conv_s)/n_steps,
        Q_solar_series  = Q_solar_s,  Q_solar_mean  = sum(Q_solar_s)/n_steps,
        Q_lw_in_series  = Q_lw_in_s,  Q_lw_in_mean  = sum(Q_lw_in_s)/n_steps,
        Q_lw_out_series = Q_lw_out_s, Q_lw_out_mean = sum(Q_lw_out_s)/n_steps,
        Q_lw_net_series = Q_lw_net_s, Q_lw_net_mean = sum(Q_lw_net_s)/n_steps,
        Q_net_series    = Q_net_s,    Q_net_mean    = sum(Q_net_s)/n_steps,
    )
end


# =====================================================================
# Whole-bird helpers (two wings)
# =====================================================================

"""
    both_wings(Q) → 2*Q

Convenience: results are computed for a single (half-) wing; multiply
by 2 for a paired-wing total.
"""
both_wings(Q) = 2 * Q


# =====================================================================
# Convection-only path (merged in from former WingConvection module)
# ─────────────────────────────────────────────────────────────────────
# These structs / drivers translate the realised wing-element airspeeds
# (from `WingKinematics.compute_element_velocities`) into convective
# heat fluxes — *without* the radiation/solar pathways used by the
# full heat-balance.  Useful when you only need the convective loss
# at one instant or averaged over the wingbeat, e.g. for an isolated
# convection diagnostic plot.
#
# The full radiative + convective balance lives above
# (`compute_heatbalance_snapshot` / `compute_wingbeat_heatbalance`).
# =====================================================================

"""
    ElementConvection

Convective heat-transfer state for one plate element at one instant
(no radiation).
"""
@kwdef struct ElementConvection{L<:Unitful.Length, V<:Unitful.Velocity, P<:Quantity}
    element_id::Int
    span_position::L
    chord::L
    airspeed::V             # realised airspeed = sqrt((v·cosβ + V_fwd)² + (v·sinβ)²)
    Re::Float64
    Nu::Float64
    h::Quantity             # heat-transfer coefficient [W/(m²·K)]
    ΔT_dorsal::P            # T_d − T_air [K]
    ΔT_ventral::P
    Q_dorsal::Quantity      # convective flux out of dorsal face [W]
    Q_ventral::Quantity
    Q_total::Quantity       # dorsal + ventral
    dorsal_area::Quantity
    ventral_area::Quantity
end

"""
    ConvectionSnapshot

Whole-wing convection at a single instant.
"""
@kwdef struct ConvectionSnapshot{T<:Quantity}
    t::Quantity
    elements::Vector{<:ElementConvection}
    Q_dorsal_total::T
    Q_ventral_total::T
    Q_total::T
end

"""
    WingbeatConvection

Cycle-resolved convection plus summary statistics.
"""
@kwdef struct WingbeatConvection{TT<:Unitful.Time, TQ<:Unitful.Power}
    period::TT
    n_steps::Int
    snapshots::Vector{<:ConvectionSnapshot}
    times::Vector{TT}
    Q_timeseries::Vector{TQ}
    Q_mean::TQ
    Q_max::TQ
    Q_min::TQ
    Q_dorsal_mean::TQ
    Q_ventral_mean::TQ
    element_Q_mean::Vector{TQ}
end

"""
    heat_transfer_coeff(Nu, k_air, L) → Quantity [W/(m²·K)]

Compute h = Nu · k_air / L.  Backward-compat wrapper kept from the
former `WingConvection` module; equivalent to
`ConvectionRegimes.convection_h` once you know the Nu directly.
"""
function heat_transfer_coeff(Nu::Real, k_air, L)
    k_num = isa(k_air, Quantity) ? ustrip(u"W/(m*K)", k_air) : k_air
    L_num = isa(L,     Quantity) ? ustrip(u"m",       L)     : L
    return (Nu * k_num / L_num) * u"W/(m^2*K)"
end

"""
    compute_convection_snapshot(wd, ev, wt, air;
                                regime = LaminarPlate()) → ConvectionSnapshot

Per-element convective heat transfer for one instant in the wingbeat.
Uses each element's chord as the characteristic length and the
realised airspeed (combined flapping + forward, through the stroke
plane) stored in `ev.realised_airspeed`.
"""
function compute_convection_snapshot(wd::WingDiscretization,
                                     ev::ElementVelocities,
                                     wt::WingTemperatures,
                                     air::AirProperties;
                                     regime::PlateConvectionRegime = LaminarPlate())
    n = length(wd.elements)
    elems = Vector{ElementConvection}(undef, n)
    Q_d_total = 0.0u"W"; Q_v_total = 0.0u"W"

    T_air_C = isa(air.T_air, Quantity) ? uconvert(u"°C", air.T_air) : air.T_air * u"°C"

    for i in 1:n
        elem  = wd.elements[i]
        V     = ev.realised_airspeed[i]
        chord = elem.chord_length

        Re = reynolds_number(V, chord, air.ν)
        Nu = nusselt_plate(regime, Re, air.Pr)
        h  = heat_transfer_coeff(Nu, air.k_air, chord)

        T_d_C = isa(wt.T_dorsal[i],  Quantity) ? uconvert(u"°C", wt.T_dorsal[i])  : wt.T_dorsal[i]  * u"°C"
        T_v_C = isa(wt.T_ventral[i], Quantity) ? uconvert(u"°C", wt.T_ventral[i]) : wt.T_ventral[i] * u"°C"
        # Subtract in °C to avoid affine-temperature arithmetic;
        # the difference itself has units of K.
        ΔT_d = ustrip(u"°C", T_d_C) - ustrip(u"°C", T_air_C)
        ΔT_v = ustrip(u"°C", T_v_C) - ustrip(u"°C", T_air_C)

        A_d = elem.has_dorsal  ? elem.dorsal_area  : 0.0u"m^2"
        A_v = elem.has_ventral ? elem.ventral_area : 0.0u"m^2"

        Q_d = uconvert(u"W", h * A_d * ΔT_d * u"K")
        Q_v = uconvert(u"W", h * A_v * ΔT_v * u"K")

        elems[i] = ElementConvection(
            element_id    = elem.element_id,
            span_position = elem.span_position,
            chord         = chord,
            airspeed      = V,
            Re            = Re,
            Nu            = Nu,
            h             = h,
            ΔT_dorsal     = ΔT_d * u"K",
            ΔT_ventral    = ΔT_v * u"K",
            Q_dorsal      = Q_d,
            Q_ventral     = Q_v,
            Q_total       = Q_d + Q_v,
            dorsal_area   = A_d,
            ventral_area  = A_v,
        )

        Q_d_total = uconvert(u"W", Q_d_total + Q_d)
        Q_v_total = uconvert(u"W", Q_v_total + Q_v)
    end

    return ConvectionSnapshot(
        t               = ev.t,
        elements        = elems,
        Q_dorsal_total  = Q_d_total,
        Q_ventral_total = Q_v_total,
        Q_total         = Q_d_total + Q_v_total,
    )
end

"""
    compute_wingbeat_convection(kin, wd, wt, air;
                                n_steps = 50, V_forward = 0u"m/s",
                                regime = LaminarPlate()) → WingbeatConvection

Cycle-averaged convective heat loss.  Steps through `n_steps` instants
across one wingbeat period, computes the per-element velocities (with
the supplied forward airspeed), and aggregates per-element / per-
surface fluxes.
"""
function compute_wingbeat_convection(kin::FlappingKinematics,
                                     wd::WingDiscretization,
                                     wt::WingTemperatures,
                                     air::AirProperties;
                                     n_steps::Int = 50,
                                     V_forward = 0.0u"m/s",
                                     regime::PlateConvectionRegime = LaminarPlate())
    freq_Hz = isa(kin.frequency, Quantity) ? ustrip(u"Hz", kin.frequency) : kin.frequency
    period  = 1.0u"s" / freq_Hz
    times   = [(i - 1) * period / n_steps for i in 1:n_steps]

    snapshots    = Vector{ConvectionSnapshot}(undef, n_steps)
    Q_timeseries = Vector{typeof(1.0u"W")}(undef, n_steps)

    for (j, t) in enumerate(times)
        ev = compute_element_velocities(kin, wd, t, V_forward = V_forward)
        snapshots[j]    = compute_convection_snapshot(wd, ev, wt, air; regime = regime)
        Q_timeseries[j] = snapshots[j].Q_total
    end

    n_elem = length(wd.elements)
    element_Q_mean = Vector{typeof(1.0u"W")}(undef, n_elem)
    for i in 1:n_elem
        element_Q_mean[i] = sum(snapshots[j].elements[i].Q_total for j in 1:n_steps) / n_steps
    end

    Q_dorsal_mean  = sum(s.Q_dorsal_total  for s in snapshots) / n_steps
    Q_ventral_mean = sum(s.Q_ventral_total for s in snapshots) / n_steps

    return WingbeatConvection(
        period         = period,
        n_steps        = n_steps,
        snapshots      = snapshots,
        times          = times,
        Q_timeseries   = Q_timeseries,
        Q_mean         = sum(Q_timeseries) / n_steps,
        Q_max          = maximum(Q_timeseries),
        Q_min          = minimum(Q_timeseries),
        Q_dorsal_mean  = Q_dorsal_mean,
        Q_ventral_mean = Q_ventral_mean,
        element_Q_mean = element_Q_mean,
    )
end


export WingTemperatures,
       AirProperties, air_properties,
       PlateConvectionRegime, LaminarPlate, TurbulentPlate, MixedPlate,
       Microclimate, microclimate_at_altitude, microclimate_from_microresult,
       ElementHeatBalance, WingHeatSnapshot, WingbeatHeatBalance,
       ElementConvection, ConvectionSnapshot, WingbeatConvection,
       uniform_temperature, linear_gradient, exponential_decay,
       custom_per_element, from_function, dorsal_ventral_split,
       to_K, to_C,
       element_to_body, wing_bodies,
       element_heat_balance,
       compute_heatbalance_snapshot, compute_wingbeat_heatbalance,
       compute_convection_snapshot, compute_wingbeat_convection,
       heat_transfer_coeff,
       default_absorptivities, default_emissivities, default_view_factors,
       both_wings


end # module WingHeatBalance
