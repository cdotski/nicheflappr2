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

module WingHeatBalance

using ..WingPlates
using ..WingKinematics
using ..FlightEnvironment
using ..FlightEnvironment: Microclimate, microclimate_at_altitude,
                           microclimate_from_microresult
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
    H = elem.thickness        # plate thickness

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

const _DEFAULT_GROUND_ALBEDO   = 0.2
const _DEFAULT_GROUND_EMISS    = 0.95
const _DEFAULT_SKY_EMISS       = 0.95
const _DEFAULT_BODY_ABS        = 0.85
const _DEFAULT_BODY_EMISS      = 0.95
const _DEFAULT_SKY_VF          = 0.5
const _DEFAULT_GROUND_VF       = 0.5


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
                              silhouette_area_override = nothing)::ElementHeatBalance

    T_d_K = uconvert(u"K", isa(T_dorsal,  Quantity) ? T_dorsal  : T_dorsal  * u"°C")
    T_v_K = uconvert(u"K", isa(T_ventral, Quantity) ? T_ventral : T_ventral * u"°C")
    T_air = uconvert(u"K", micro.air_temperature)
    A_d   = elem.has_dorsal  ? elem.dorsal_area  : 0.0u"m^2"
    A_v   = elem.has_ventral ? elem.ventral_area : 0.0u"m^2"

    # ── Convection (call HeatExchange separately per surface) ──────
    Q_conv_d = 0.0u"W"; Q_conv_v = 0.0u"W"; h_avg = 0.0u"W/(m^2*K)"
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
default_absorptivities(; α_d = _DEFAULT_BODY_ABS,
                          α_v = _DEFAULT_BODY_ABS,
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
                                      use_flat_plate_silhouette::Bool = true)
    n = length(wd.elements)
    elems = Vector{ElementHeatBalance}(undef, n)
    Q_conv = 0.0u"W"; Q_sol = 0.0u"W"
    Q_lw_in = 0.0u"W"; Q_lw_out = 0.0u"W"

    cosθ = max(cos(uconvert(u"rad", micro.zenith_angle)), 0.0)

    for i in 1:n
        elem = wd.elements[i]
        body = bodies[i]
        sil_override = if use_flat_plate_silhouette
            # dorsal face area × cos(zenith) — a horizontal wing seen from the sun
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
                                      V_forward                  = micro.wind_speed,
                                      tissue_density             = 1000.0u"kg/m^3",
                                      absorptivities::Absorptivities = default_absorptivities(),
                                      emissivities::Emissivities    = default_emissivities(),
                                      view_factors::ViewFactors     = default_view_factors(),
                                      characteristic_dim_formula    = ScaledDimension(:length_skin),
                                      conduction_fraction::Real     = 0.0,
                                      use_flat_plate_silhouette::Bool = true)
    bodies = wing_bodies(wd; tissue_density = tissue_density)

    freq_Hz = isa(kin.frequency, Quantity) ? ustrip(u"Hz", kin.frequency) : kin.frequency
    period  = 1.0u"s" / freq_Hz
    times   = [(i - 1) * period / n_steps for i in 1:n_steps]

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


export WingTemperatures,
       AirProperties, air_properties,
       Microclimate, microclimate_at_altitude, microclimate_from_microresult,
       ElementHeatBalance, WingHeatSnapshot, WingbeatHeatBalance,
       uniform_temperature, linear_gradient, exponential_decay,
       custom_per_element, from_function, dorsal_ventral_split,
       to_K, to_C,
       element_to_body, wing_bodies,
       element_heat_balance,
       compute_heatbalance_snapshot, compute_wingbeat_heatbalance,
       default_absorptivities, default_emissivities, default_view_factors,
       both_wings


end # module WingHeatBalance
