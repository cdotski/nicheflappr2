# =====================================================================
# wing_kinematics_2.0.jl
#
# Flapping-wing kinematics and per-element velocities.
#
# Provides:
#   1. By-hand construction via `FlappingKinematics`
#   2. Allometric "scaled" construction via `build_kinematics_for_mass`
#      with multiple frequency and amplitude scaling laws including
#      Strouhal-based and Pennycuick / afpt formulations.
#   3. Per-element velocity and airspeed calculations.
#
# The flight speed needed by some scalings (e.g. Strouhal) is taken
# in as an argument so that this module does not depend on flight
# power calculations — those live in `wing_power.jl`.
# =====================================================================

include("wing_geometry_2.0.jl")

module WingKinematics

using ..WingPlates
using Unitful


# ── Constants ───────────────────────────────────────────────────────

const G_MS2     = 9.81           # gravitational acceleration [m/s²]
const RHO_ISA   = 1.225          # ISA sea-level air density [kg/m³]


# ── Kinematics struct ───────────────────────────────────────────────

"""
    FlappingKinematics

Parameters describing a single wing's sinusoidal flapping motion:

    θ(t) = Φ · sin(2πft),     ω(t) = Φ · 2πf · cos(2πft)

where Φ is the stroke half-amplitude (full sweep = 2Φ) in radians and
the stroke plane is tilted `stroke_plane_angle` radians from horizontal.
"""
@kwdef struct FlappingKinematics{F<:Unitful.Frequency}
    frequency::F                  # [Hz]
    amplitude::Float64            # stroke half-amplitude [rad]
    stroke_plane_angle::Float64   # stroke plane angle from horizontal [rad]
end


"""
    ElementVelocities

Per-element velocities at one instant.
"""
@kwdef struct ElementVelocities{T<:Unitful.Time, AngV<:Number,
                                L<:Unitful.Length, V<:Number}
    t::T
    theta::Float64
    omega::AngV
    element_ids::Vector{Int}
    span_positions::Vector{L}
    speeds::Vector{V}
    v_horizontal::Vector{V}
    v_vertical::Vector{V}
    v_forward::V
    realised_airspeed::Vector{V}
end


# ── Frequency scaling methods (multiple dispatch) ───────────────────

abstract type FreqScaling end

"Greenewalt (1975) best-fit allometry: f = 3.87 · m^(-1/3)"
struct Greenewalt1975 <: FreqScaling end

"""
    Pennycuick2008MinPower(ρ_kgm3)

Pennycuick (2008) minimum-power-speed wingbeat-frequency formula
(also used internally by `afpt`'s `.estimateFrequency`):

    f = m^(3/8) · g^(1/2) · b^(-23/24) · S^(-1/3) · ρ^(-3/8)
"""
struct Pennycuick2008MinPower <: FreqScaling
    ρ_kgm3::Float64
end
Pennycuick2008MinPower() = Pennycuick2008MinPower(RHO_ISA)

"Bullen (2012) bat allometry: f = 3.06 · m^(-0.264)"
struct Bulleen2012Bats <: FreqScaling end

"Rayner (1988) bird allometry: f = 3.98 · m^(-0.27)"
struct Rayner1988Birds <: FreqScaling end

"""
    StrouhalFreq(St, ρ_kgm3, V_forward_ms = NaN)

Strouhal-number-based wingbeat frequency:

    f = St · U / (b · sin(θ(b))),   θ(b) = 33.5 · b^(-0.24)  [°]

If `V_forward_ms` is finite it is used directly as U; otherwise the
caller must pre-compute it (e.g. via `WingPower.max_range_speed`) and
pass it through `wingbeat_hz(m, ::StrouhalFreq; V_forward_ms=...)`.
"""
struct StrouhalFreq <: FreqScaling
    St::Float64
    ρ_kgm3::Float64
    V_forward_ms::Float64
end
StrouhalFreq()    = StrouhalFreq(0.21, RHO_ISA, NaN)
StrouhalFreq(St)  = StrouhalFreq(St,   RHO_ISA, NaN)
StrouhalFreq(St, ρ) = StrouhalFreq(St, ρ, NaN)

wingbeat_hz(m_kg, ::Greenewalt1975)  = 3.87 * m_kg^(-1/3)
wingbeat_hz(m_kg, ::Bulleen2012Bats) = 3.06 * m_kg^(-0.264)
wingbeat_hz(m_kg, ::Rayner1988Birds) = 3.98 * m_kg^(-0.27)

function wingbeat_hz(m_kg, method::Pennycuick2008MinPower)
    b = wing_span_m(m_kg)
    S = wing_area_m2(m_kg)
    return m_kg^(3/8) * sqrt(G_MS2) * b^(-23/24) * S^(-1/3) * method.ρ_kgm3^(-3/8)
end

"""
    wingbeat_hz(m_kg, method::StrouhalFreq; V_forward_ms = method.V_forward_ms)

Strouhal-based wingbeat frequency.  `V_forward_ms` must be supplied via
either the keyword argument or `method.V_forward_ms`.
"""
function wingbeat_hz(m_kg, method::StrouhalFreq; V_forward_ms = method.V_forward_ms)
    isnan(V_forward_ms) && error(
        "StrouhalFreq requires a forward airspeed. " *
        "Pass V_forward_ms or use the keyword form of build_kinematics_for_mass.")
    b = wing_span_m(m_kg)
    A = b * sin(deg2rad(33.5 * b^(-0.24)))
    return method.St * V_forward_ms / A
end


# ── Amplitude scaling methods ───────────────────────────────────────

abstract type AmpScaling end

"Hold stroke half-amplitude constant at `deg` degrees."
struct FixedAmplitude <: AmpScaling
    deg::Float64
end

"""
    WingspanScaling()

`log₁₀(angle°) = 1.83 − 0.24 · log₁₀(b)`  ⇒  angle = 10^(1.83 − 0.24·log₁₀ b).
"""
struct WingspanScaling <: AmpScaling end

"""
    StrouhalAmplitude()

θ(b) = 33.5 · b^(-0.24)  [°] — same wingspan-based amplitude used inside
`StrouhalFreq`.
"""
struct StrouhalAmplitude <: AmpScaling end

amplitude_deg(m_kg::Real, a::FixedAmplitude)    = a.deg
amplitude_deg(m_kg::Real, ::WingspanScaling)    =
    10^(1.83 - 0.24 * log10(wing_span_m(m_kg)))
amplitude_deg(m_kg::Real, ::StrouhalAmplitude)  =
    33.5 * wing_span_m(m_kg)^(-0.24)
amplitude_deg(m_kg::Real, deg::Real)            = Float64(deg)


export FlappingKinematics, ElementVelocities,
       angular_position, angular_velocity, element_speed,
       compute_element_velocities, peak_element_speed,
       mean_element_speed, realised_airspeed, reduced_frequency,
       FreqScaling, Greenewalt1975, Pennycuick2008MinPower,
       Bulleen2012Bats, Rayner1988Birds, StrouhalFreq,
       AmpScaling, FixedAmplitude, WingspanScaling, StrouhalAmplitude,
       wingbeat_hz, amplitude_deg,
       build_kinematics_for_mass


# ═════════════════════════════════════════════════════════════════════
# Angular kinematics
# ═════════════════════════════════════════════════════════════════════

"Wing angular position θ(t) = Φ · sin(2πft)."
angular_position(kin::FlappingKinematics, t) =
    kin.amplitude * sin(2π * kin.frequency * t)

"Wing angular velocity ω(t) = Φ · 2πf · cos(2πft)."
angular_velocity(kin::FlappingKinematics, t) =
    kin.amplitude * 2π * kin.frequency * cos(2π * kin.frequency * t)

"Linear speed of an element at span position r: v = r · |ω|."
element_speed(kin::FlappingKinematics, span_pos, t) =
    span_pos * abs(angular_velocity(kin, t))


# ═════════════════════════════════════════════════════════════════════
# Per-element velocities at one instant
# ═════════════════════════════════════════════════════════════════════

"""
    compute_element_velocities(kin, wing_disc, t; V_forward = 0u"m/s")

Per-element flapping + forward-flight velocities at time `t`.
Returns an `ElementVelocities` struct.
"""
function compute_element_velocities(kin::FlappingKinematics,
                                    wing_disc::WingDiscretization,
                                    t;
                                    V_forward = 0.0u"m/s")
    θ  = angular_position(kin, t)
    ω  = uconvert(u"s^-1", angular_velocity(kin, t))
    β  = kin.stroke_plane_angle
    Vf = uconvert(u"m/s", V_forward)

    n = length(wing_disc.elements)
    L = typeof(wing_disc.elements[1].span_position)

    ids    = Vector{Int}(undef, n)
    spans  = Vector{L}(undef, n)
    speeds = Vector{typeof(Vf)}(undef, n)
    v_h    = Vector{typeof(Vf)}(undef, n)
    v_v    = Vector{typeof(Vf)}(undef, n)
    v_real = Vector{typeof(Vf)}(undef, n)

    for (i, elem) in enumerate(wing_disc.elements)
        ids[i]   = elem.element_id
        spans[i] = elem.span_position
        v        = uconvert(u"m/s", elem.span_position * abs(ω))
        speeds[i] = v
        v_h[i]   = uconvert(u"m/s", v * cos(β))
        v_v[i]   = uconvert(u"m/s", v * sin(β))
        v_real[i] = uconvert(u"m/s", sqrt((v_h[i] + Vf)^2 + v_v[i]^2))
    end

    return ElementVelocities(t, θ, ω, ids, spans, speeds, v_h, v_v, Vf, v_real)
end


"""
    realised_airspeed(kin, span_pos, t; V_forward = 0u"m/s")

Magnitude of the velocity vector at one span position, combining
flapping and forward components through the stroke plane angle.
"""
function realised_airspeed(kin::FlappingKinematics, span_pos, t;
                           V_forward = 0.0u"m/s")
    v  = element_speed(kin, span_pos, t)
    β  = kin.stroke_plane_angle
    vh = v * cos(β) + V_forward
    vv = v * sin(β)
    return sqrt(vh^2 + vv^2)
end


"""
    peak_element_speed(kin, span_pos; V_forward = 0u"m/s")

Maximum realised airspeed at this span position over the cycle.
"""
function peak_element_speed(kin::FlappingKinematics, span_pos; V_forward = 0.0u"m/s")
    v_max = uconvert(u"m/s", span_pos * kin.amplitude * 2π * kin.frequency)
    Vf    = uconvert(u"m/s", V_forward)
    β     = kin.stroke_plane_angle
    return sqrt((v_max * cos(β) + Vf)^2 + (v_max * sin(β))^2)
end


"""
    mean_element_speed(kin, span_pos; V_forward = 0u"m/s", n_steps = 500)

Cycle-averaged realised airspeed (numerical integration).
"""
function mean_element_speed(kin::FlappingKinematics, span_pos;
                            V_forward = 0.0u"m/s", n_steps::Int = 500)
    T     = uconvert(u"s", 1.0 / kin.frequency)
    Vf    = uconvert(u"m/s", V_forward)
    β     = kin.stroke_plane_angle
    total = 0.0u"m/s"
    for i in 1:n_steps
        t  = (i - 0.5) * T / n_steps
        v  = uconvert(u"m/s", element_speed(kin, span_pos, t))
        total += sqrt((v * cos(β) + Vf)^2 + (v * sin(β))^2)
    end
    return total / n_steps
end


"""
    reduced_frequency(b, f, U) → Float64

Reduced (Strouhal-style) frequency used by afpt:
    k_f = 2π · f · b / U
where `b` is wingspan and `U` is flight speed.
"""
function reduced_frequency(b, f, U)
    b_m = isa(b, Quantity) ? ustrip(u"m",   b) : b
    f_h = isa(f, Quantity) ? ustrip(u"Hz",  f) : f
    U_m = isa(U, Quantity) ? ustrip(u"m/s", U) : U
    return 2π * f_h * b_m / U_m
end


# ═════════════════════════════════════════════════════════════════════
# Allometric / scaled construction
# ═════════════════════════════════════════════════════════════════════

"""
    build_kinematics_for_mass(m_kg, freq_method = StrouhalFreq();
                              amp = StrouhalAmplitude(),
                              stroke_plane_deg = 80.0,
                              V_forward_ms = NaN) → FlappingKinematics

Build a `FlappingKinematics` struct from body mass [kg].

# Arguments
- `m_kg`              : body mass [kg]
- `freq_method`       : a `FreqScaling` instance (default `StrouhalFreq()`).
                        For `StrouhalFreq`, supply `V_forward_ms` so the
                        forward airspeed is known.
- `amp`               : stroke half-amplitude — `Real` [°] or an
                        `AmpScaling` instance (default `StrouhalAmplitude()`).
- `stroke_plane_deg`  : stroke-plane angle from horizontal [°].
- `V_forward_ms`      : forward airspeed [m/s] required by `StrouhalFreq`.
"""
function build_kinematics_for_mass(m_kg::Real,
                                   freq_method::FreqScaling = StrouhalFreq();
                                   amp = StrouhalAmplitude(),
                                   stroke_plane_deg::Real = 80.0,
                                   V_forward_ms::Real = NaN)
    f_hz = if freq_method isa StrouhalFreq
        Vf = isnan(V_forward_ms) ? freq_method.V_forward_ms : V_forward_ms
        wingbeat_hz(m_kg, freq_method; V_forward_ms = Vf)
    else
        wingbeat_hz(m_kg, freq_method)
    end
    return FlappingKinematics(
        frequency          = f_hz * u"Hz",
        amplitude          = deg2rad(amplitude_deg(m_kg, amp)),
        stroke_plane_angle = deg2rad(stroke_plane_deg),
    )
end


end # module WingKinematics
