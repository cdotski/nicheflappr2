# =====================================================================
# environment.jl
#
# Shared single-instant environmental state for the wing (and, later,
# whole-body) heat balance.  Designed to be compatible with the
# time-series outputs of BiophysicalEcology/Microclimate.jl: each
# `Microclimate` here is conceptually one timestep of a `MicroResult`.
#
# This module defines ONLY the environment container, the altitude
# helper, and an adapter that extracts a single timestep from a
# Microclimate.jl `MicroResult`.  It contains no animal-surface
# properties — those (absorptivities, emissivities, view factors) live
# with the heat balance that consumes them, because they are
# anatomical, not environmental.
# =====================================================================

module FlightEnvironment

using Unitful
using FluidProperties: atmospheric_pressure, GasFractions


# =====================================================================
# Microclimate container (single instant)
# =====================================================================

"""
    Microclimate(; air_temperature, sky_temperature, ground_temperature,
                   relative_humidity, wind_speed, atmospheric_pressure,
                   zenith_angle, global_radiation, diffuse_fraction, shade,
                   altitude, gas_fractions)

A flat container of the environmental variables consumed by
`HeatExchange.convection`, `HeatExchange.radiation_in`,
`HeatExchange.radiation_out` and `HeatExchange.solar`.

Conceptually a single timestep of a `Microclimate.jl` `MicroResult`
(use `microclimate_from_microresult` to extract one).  Field names
follow the `HeatExchange.jl` convention, not the `MicroResult`
convention; see the adapter for the field-name mapping.

Defaults represent a clear-sky temperate day at sea level with the sun
roughly overhead.
"""
@kwdef struct Microclimate{TA,TS,TG,RH,WS,PA,ZA,GR,FD,SH,AL,GF}
    air_temperature::TA       = 20.0u"°C"
    sky_temperature::TS       = 0.0u"°C"           # effective sky temperature
    ground_temperature::TG    = 25.0u"°C"
    relative_humidity::RH     = 0.5
    wind_speed::WS            = 1.0u"m/s"
    atmospheric_pressure::PA  = 101325.0u"Pa"
    zenith_angle::ZA          = 30.0u"°"
    global_radiation::GR      = 800.0u"W/m^2"
    diffuse_fraction::FD      = 0.15
    shade::SH                 = 0.0                # fraction in shade [0,1]
    altitude::AL              = 0.0u"m"
    gas_fractions::GF         = GasFractions()
end

"""
    microclimate_at_altitude(; altitude, …)

Convenience constructor that fills `atmospheric_pressure` from
`FluidProperties.atmospheric_pressure(altitude)`.
"""
function microclimate_at_altitude(; altitude::Quantity = 0.0u"m", kwargs...)
    P = atmospheric_pressure(altitude)
    return Microclimate(; altitude = altitude, atmospheric_pressure = P, kwargs...)
end


# =====================================================================
# Adapter: Microclimate.jl MicroResult → Microclimate (one timestep)
# =====================================================================

"""
    microclimate_from_microresult(mr, step::Int; kwargs...) → Microclimate

Extract a single timestep from a `Microclimate.jl` `MicroResult` (or
any object exposing the same field names) into a `Microclimate`
instance suitable for the wing/body heat balance.

Field-name mapping (`MicroResult` → `Microclimate`):

| `MicroResult.…`            | `Microclimate.…`        |
|----------------------------|-------------------------|
| `reference_temperature`    | `air_temperature`       |
| `reference_wind_speed`     | `wind_speed`            |
| `reference_humidity`       | `relative_humidity`     |
| `pressure`                 | `atmospheric_pressure`  |
| `sky_temperature`          | `sky_temperature`       |
| `global_radiation`         | `global_radiation`      |
| `diffuse_fraction`         | `diffuse_fraction`      |
| `soil_temperature[step,1]` | `ground_temperature`    |
| `solar_radiation.zenith[step]` (if present) | `zenith_angle` |

Any `kwarg` overrides the value pulled from `mr`; use this for fields
that `MicroResult` does not store (e.g. `shade`, `altitude`,
`gas_fractions`) or to substitute a different `zenith_angle` source.
"""
function microclimate_from_microresult(mr, step::Int; kwargs...)
    overrides = Dict{Symbol,Any}(kwargs)

    # Scalar time series
    get_field(name, default = nothing) = begin
        if haskey(overrides, name)
            overrides[name]
        elseif hasproperty(mr, name)
            getproperty(mr, name)[step]
        else
            default
        end
    end

    # Soil surface temperature: first (top) node of soil_temperature matrix.
    ground_T = if haskey(overrides, :ground_temperature)
        overrides[:ground_temperature]
    elseif hasproperty(mr, :soil_temperature)
        mr.soil_temperature[step, 1]
    else
        25.0u"°C"
    end

    # Zenith angle: lives inside the `solar_radiation` NamedTuple in
    # MicroResult.  Be lenient about exact field name.
    zenith = if haskey(overrides, :zenith_angle)
        overrides[:zenith_angle]
    elseif hasproperty(mr, :solar_radiation)
        sr = mr.solar_radiation
        z_field = hasproperty(sr, :zenith)       ? sr.zenith       :
                  hasproperty(sr, :zenith_angle) ? sr.zenith_angle :
                  nothing
        z_field === nothing ? 30.0u"°" : z_field[step]
    else
        30.0u"°"
    end

    return Microclimate(
        air_temperature      = get_field(:reference_temperature, 20.0u"°C"),
        sky_temperature      = get_field(:sky_temperature,        0.0u"°C"),
        ground_temperature   = ground_T,
        relative_humidity    = get_field(:reference_humidity,     0.5),
        wind_speed           = get_field(:reference_wind_speed,   1.0u"m/s"),
        atmospheric_pressure = get_field(:pressure,               101325.0u"Pa"),
        zenith_angle         = zenith,
        global_radiation     = get_field(:global_radiation,       800.0u"W/m^2"),
        diffuse_fraction     = get_field(:diffuse_fraction,       0.15),
        shade                = get(overrides, :shade,             0.0),
        altitude             = get(overrides, :altitude,          0.0u"m"),
        gas_fractions        = get(overrides, :gas_fractions,     GasFractions()),
    )
end


export Microclimate, microclimate_at_altitude, microclimate_from_microresult


end # module FlightEnvironment
