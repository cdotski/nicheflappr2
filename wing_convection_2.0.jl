# =====================================================================
# wing_convection_2.0.jl
#
# Wing surface-temperature assignment, air properties (with altitude
# support), and convective heat-transfer calculations.
#
# Air properties are taken from `FluidProperties.dry_air_properties`
# (BiophysicalEcology suite) when available; a Sutherland-law fallback
# keeps the module usable stand-alone.  Altitude is supported through
# `FluidProperties.atmospheric_pressure`.
#
# Heat-transfer model (per element):
#   Re = V · c / ν
#   Nu = 0.664 · Re^0.5 · Pr^(1/3)        (laminar flat-plate)
#   h  = Nu · k_air / c
#   Q  = h · A · (T_surface − T_air)
# Dorsal and ventral surfaces are treated independently.
# =====================================================================

include("wing_kinematics_2.0.jl")

module WingConvection

using ..WingPlates
using ..WingKinematics
using Unitful

# Optional dependency: FluidProperties.jl (BiophysicalEcology)
const _HAS_FLUIDPROPS = try
    @eval using FluidProperties: dry_air_properties as _fp_dry_air_properties,
                                  atmospheric_pressure as _fp_atmospheric_pressure
    true
catch
    false
end


# =====================================================================
# Surface-temperature assignment
# =====================================================================

"""
    WingTemperatures

Temperatures assigned to every element of a discretised wing.
Dorsal and ventral surfaces are stored independently to allow e.g.
asymmetric solar heating.
"""
@kwdef struct WingTemperatures{L<:Unitful.Length, T<:Number}
    element_ids::Vector{Int}
    span_positions::Vector{L}
    T_dorsal::Vector{T}
    T_ventral::Vector{T}
end


# Convenience converters (avoid affine °C arithmetic)
to_K(T::Quantity) = uconvert(u"K", T)
to_K(T::Real)     = T * u"K"
to_C(T_K::Quantity) = uconvert(u"°C", T_K)


_unpack(wd::WingDiscretization) =
    ([e.element_id    for e in wd.elements],
     [e.span_position for e in wd.elements])


"""
    uniform_temperature(wd, T)

Assign the same temperature to every element on both surfaces.
"""
function uniform_temperature(wd::WingDiscretization, T)
    ids, spans = _unpack(wd)
    n = length(ids)
    return WingTemperatures(ids, spans, fill(T, n), fill(T, n))
end


"""
    linear_gradient(wd, T_root, T_tip)

Linear interpolation from `T_root` at the wing root to `T_tip` at the tip.
"""
function linear_gradient(wd::WingDiscretization, T_root, T_tip)
    ids, spans = _unpack(wd)
    L = wd.wing.wing_length
    T_root_K = to_K(T_root); T_tip_K = to_K(T_tip)
    temps_K = [T_root_K + (T_tip_K - T_root_K) * (s / L) for s in spans]
    temps   = [to_C(t) for t in temps_K]
    return WingTemperatures(ids, spans, temps, copy(temps))
end


"""
    exponential_decay(wd, T_root, T_tip; decay_rate = 3.0)

Exponential decay from `T_root` to `T_tip` along the span:
    T(s) = T_tip + (T_root − T_tip) · exp(−k·s/L)
"""
function exponential_decay(wd::WingDiscretization, T_root, T_tip; decay_rate = 3.0)
    ids, spans = _unpack(wd)
    L = wd.wing.wing_length
    T_root_K = to_K(T_root); T_tip_K = to_K(T_tip)
    ΔT_K = T_root_K - T_tip_K
    temps_K = [T_tip_K + ΔT_K * exp(-decay_rate * s / L) for s in spans]
    temps   = [to_C(t) for t in temps_K]
    return WingTemperatures(ids, spans, temps, copy(temps))
end


"""
    custom_per_element(wd, temps)

Directly specify a temperature vector (one per element, both surfaces).
"""
function custom_per_element(wd::WingDiscretization, temps::Vector)
    ids, spans = _unpack(wd)
    length(temps) == length(ids) ||
        error("Expected $(length(ids)) temperatures, got $(length(temps))")
    return WingTemperatures(ids, spans, copy(temps), copy(temps))
end


"""
    from_function(wd, f)

Compute element temperatures from a user-supplied function `f(span_pos)`.
"""
function from_function(wd::WingDiscretization, f::Function)
    ids, spans = _unpack(wd)
    temps_raw = [f(s) for s in spans]
    temps = [isa(t, Quantity) && unit(t) == u"K" ? to_C(t) : t for t in temps_raw]
    return WingTemperatures(ids, spans, temps, copy(temps))
end


"""
    dorsal_ventral_split(wd, T_dorsal, T_ventral)

Uniform but separate dorsal and ventral temperatures.
"""
function dorsal_ventral_split(wd::WingDiscretization, T_dorsal, T_ventral)::WingTemperatures
    ids, spans = _unpack(wd)
    n = length(ids)
    return WingTemperatures(ids, spans, fill(T_dorsal, n), fill(T_ventral, n))
end


"""
    dorsal_ventral_split(wd, f_dorsal::Function, f_ventral::Function)

Functional form: separate per-surface temperature profiles.
"""
function dorsal_ventral_split(wd::WingDiscretization,
                              f_dorsal::Function,
                              f_ventral::Function)::WingTemperatures
    ids, spans = _unpack(wd)
    return WingTemperatures(ids, spans,
                            [f_dorsal(s)  for s in spans],
                            [f_ventral(s) for s in spans])
end


# =====================================================================
# Air properties (uses FluidProperties.jl when available)
# =====================================================================

"""
    AirProperties

Thermophysical properties of dry air at one temperature/pressure point.
"""
@kwdef struct AirProperties{T<:Quantity}
    T_air::T
    P::Quantity                 # ambient pressure  [Pa]
    altitude::Quantity          # geometric altitude [m]
    ρ::Quantity                 # density [kg/m³]
    μ::Quantity                 # dynamic viscosity [Pa·s]
    ν::Quantity                 # kinematic viscosity [m²/s]
    k_air::Quantity             # thermal conductivity [W/(m·K)]
    Pr::Float64                 # Prandtl number [-]
end


"""
    pressure_at_altitude(altitude; P0 = 101325 Pa, T_sea = 288 K, …) → Quantity

Barometric pressure at a given altitude.  Delegates to
`FluidProperties.atmospheric_pressure` when available, otherwise uses
the ISA tropospheric approximation locally.
"""
function pressure_at_altitude(altitude;
                              P0::Quantity        = 101325.0u"Pa",
                              T_sea::Quantity     = 288.0u"K",
                              lapse::Quantity     = -0.0065u"K/m",
                              M::Quantity         = 0.0289644u"kg/mol")
    if _HAS_FLUIDPROPS
        return _fp_atmospheric_pressure(altitude;
            L_ref = lapse, T_ref = T_sea, M = M)
    end
    g = 9.80665u"m/s^2"
    R = 8.31446u"J/(mol*K)"
    return P0 * (1 + (lapse / T_sea) * altitude) ^ ((-g * M) / (R * lapse))
end


"""
    air_properties(T_air; altitude = 0 m, P = nothing) → AirProperties

Thermophysical properties of dry air at temperature `T_air` (°C or K).

Pressure is determined by:
- `P` if given (a Unitful pressure),
- otherwise `pressure_at_altitude(altitude)`.

Uses `FluidProperties.dry_air_properties` when available; falls back to
Sutherland's law otherwise.
"""
function air_properties(T_air;
                        altitude = 0.0u"m",
                        P = nothing)
    T_K = isa(T_air, Quantity) ? uconvert(u"K", T_air) : T_air * u"K"
    P_used = P === nothing ? pressure_at_altitude(altitude) : P

    if _HAS_FLUIDPROPS
        fp = _fp_dry_air_properties(T_K, P_used)
        cp_air = 1005.8u"J/(kg*K)"
        Pr_val = ustrip(u"NoUnits",
                        fp.dynamic_viscosity * cp_air / fp.thermal_conductivity)
        return AirProperties(
            T_air    = T_air,
            P        = uconvert(u"Pa", P_used),
            altitude = uconvert(u"m", altitude),
            ρ        = uconvert(u"kg/m^3", fp.density),
            μ        = uconvert(u"Pa*s",   fp.dynamic_viscosity),
            ν        = uconvert(u"m^2/s",  fp.kinematic_viscosity),
            k_air    = uconvert(u"W/(m*K)", fp.thermal_conductivity),
            Pr       = Float64(Pr_val),
        )
    end

    # ── Sutherland fallback ────────────────────────────────────────
    T_K_num = ustrip(u"K", T_K)
    P_num   = ustrip(u"Pa", P_used)
    R_spec  = 287.05
    ρ = (P_num / (R_spec * T_K_num)) * u"kg/m^3"
    μ_ref = 1.716e-5; T_ref = 273.15; S = 110.4
    μ = μ_ref * ((T_ref + S) / (T_K_num + S)) * (T_K_num / T_ref)^1.5 * u"Pa*s"
    ν = uconvert(u"m^2/s", μ / ρ)
    T_C = T_K_num - 273.15
    k_air = (0.02414 + 7.56e-5 * T_C) * u"W/(m*K)"
    Pr = clamp(0.7 + 4.8e-4 * (20.0 - T_C), 0.69, 0.74)

    return AirProperties(
        T_air = T_air, P = uconvert(u"Pa", P_used),
        altitude = uconvert(u"m", altitude),
        ρ = ρ, μ = μ, ν = ν, k_air = k_air, Pr = Pr,
    )
end


# =====================================================================
# Convection per element / per snapshot / per cycle
# =====================================================================

"""
    ElementConvection

Convective heat-transfer state for one plate element at one instant.
"""
@kwdef struct ElementConvection{L<:Unitful.Length, V<:Unitful.Velocity, P<:Quantity}
    element_id::Int
    span_position::L
    chord::L
    airspeed::V
    Re::Float64
    Nu::Float64
    h::Quantity
    ΔT_dorsal::P
    ΔT_ventral::P
    Q_dorsal::Quantity
    Q_ventral::Quantity
    Q_total::Quantity
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


export WingTemperatures, AirProperties,
       ElementConvection, ConvectionSnapshot, WingbeatConvection,
       uniform_temperature, linear_gradient, exponential_decay,
       custom_per_element, from_function, dorsal_ventral_split,
       to_K, to_C,
       air_properties, pressure_at_altitude,
       reynolds_number, nusselt_laminar, heat_transfer_coeff,
       compute_convection_snapshot, compute_wingbeat_convection


# ── Convection primitives ───────────────────────────────────────────

function reynolds_number(V, L, ν)::Float64
    V_num = isa(V, Quantity) ? ustrip(u"m/s",   V) : V
    L_num = isa(L, Quantity) ? ustrip(u"m",     L) : L
    ν_num = isa(ν, Quantity) ? ustrip(u"m^2/s", ν) : ν
    return V_num * L_num / ν_num
end

function nusselt_laminar(Re::Float64, Pr::Float64)::Float64
    Re ≤ 0.0 && return 0.0
    return 0.664 * sqrt(Re) * Pr^(1/3)
end

function heat_transfer_coeff(Nu::Float64, k_air, L)
    k_num = isa(k_air, Quantity) ? ustrip(u"W/(m*K)", k_air) : k_air
    L_num = isa(L,     Quantity) ? ustrip(u"m",       L)     : L
    return (Nu * k_num / L_num) * u"W/(m^2*K)"
end


"""
    compute_convection_snapshot(wing_disc, ev, wt, air) → ConvectionSnapshot
"""
function compute_convection_snapshot(wing_disc::WingDiscretization,
                                     ev::ElementVelocities,
                                     wt::WingTemperatures,
                                     air)::ConvectionSnapshot
    n = length(wing_disc.elements)
    elems = Vector{ElementConvection}(undef, n)
    Q_d_total = 0.0u"W"
    Q_v_total = 0.0u"W"

    T_air_C = isa(air.T_air, Quantity) ? uconvert(u"°C", air.T_air) : air.T_air * u"°C"

    for i in 1:n
        elem  = wing_disc.elements[i]
        V     = ev.realised_airspeed[i]
        chord = elem.chord_length

        Re = reynolds_number(V, chord, air.ν)
        Nu = nusselt_laminar(Re, air.Pr)
        h  = heat_transfer_coeff(Nu, air.k_air, chord)

        T_d_C = isa(wt.T_dorsal[i],  Quantity) ? uconvert(u"°C", wt.T_dorsal[i])  : wt.T_dorsal[i]  * u"°C"
        T_v_C = isa(wt.T_ventral[i], Quantity) ? uconvert(u"°C", wt.T_ventral[i]) : wt.T_ventral[i] * u"°C"
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
    compute_wingbeat_convection(kin, wing_disc, wt, air;
                                n_steps = 50, V_forward = 0u"m/s") → WingbeatConvection
"""
function compute_wingbeat_convection(kin::FlappingKinematics,
                                     wing_disc::WingDiscretization,
                                     wt::WingTemperatures,
                                     air;
                                     n_steps::Int = 50,
                                     V_forward = 0.0u"m/s")
    freq_Hz = isa(kin.frequency, Quantity) ? ustrip(u"Hz", kin.frequency) : kin.frequency
    period  = 1.0u"s" / freq_Hz
    times   = [(i - 1) * period / n_steps for i in 1:n_steps]

    snapshots    = Vector{ConvectionSnapshot}(undef, n_steps)
    Q_timeseries = Vector{typeof(1.0u"W")}(undef, n_steps)

    for (j, t) in enumerate(times)
        ev = compute_element_velocities(kin, wing_disc, t, V_forward = V_forward)
        snapshots[j]    = compute_convection_snapshot(wing_disc, ev, wt, air)
        Q_timeseries[j] = snapshots[j].Q_total
    end

    n_elem = length(wing_disc.elements)
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


end # module WingConvection
