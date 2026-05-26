# =====================================================================
# wing_geometry_2.0.jl
#
# Wing planform geometry and span-wise discretization into plate
# elements.  Provides both:
#
#   1. By-hand construction via the `WingGeometry` struct, and
#   2. Allometric "scaled" construction from body mass via
#      `build_wing_for_mass`.
#
# All lengths are Unitful quantities (SI).  Module name is kept as
# `WingPlates` for backwards compatibility with downstream modules.
# =====================================================================

module WingPlates

using Printf
using Unitful
using ..AFPT: wing_span_allometry, wing_area_allometry,
              mean_chord_allometry, aspect_ratio_allometry,
              compute_body_frontal_area

# ── Allometry constants ─────────────────────────────────────────────
#
# Pennycuick (2008) cross-species regressions:
#   b  = 1.17 · m^0.39   [m]   wing span (full)
#   S  = 0.16 · m^0.72   [m²]  total wing area (both wings)
#   Sb = 0.00813 · m^0.666 [m²] body frontal area (non-passerine)
#   Sb = 0.0129  · m^0.614 [m²] body frontal area (passerine)
#
# All formulas take body mass m in kilograms.
# ─────────────────────────────────────────────────────────────────────


# ── Geometry struct (manual construction) ───────────────────────────

"""
    WingGeometry

Planform geometry of one wing.  Lengths must be Unitful quantities
(e.g. `0.15u"m"`).
"""
@kwdef struct WingGeometry{L<:Unitful.Length}
    wing_length::L                # half-span (root → tip) [m]
    root_chord::L                 # chord at root [m]
    tip_chord::L                  # chord at tip [m]
    thickness::L                  # characteristic thickness [m]
    dorsal_active::Bool = true    # does dorsal surface exchange heat?
    ventral_active::Bool = true   # does ventral surface exchange heat?
end


"""
    Discretization

Parameters controlling the span-wise plate discretization.
`method` may be `"uniform"` or `"nonuniform"`.
"""
@kwdef struct Discretization
    n_elements::Int
    method::String = "uniform"
end


"""
    WingElement

A single span-wise plate element.
"""
@kwdef struct WingElement{L<:Unitful.Length, A<:Unitful.Area}
    element_id::Int
    span_position::L
    span_length::L
    chord_length::L
    thickness::L

    left_edge::L
    right_edge::L
    plate_size::L

    dorsal_area::A
    ventral_area::A

    has_dorsal::Bool
    has_ventral::Bool
end


"""
    WingDiscretization

Result of discretising a `WingGeometry` into plate elements.
"""
@kwdef struct WingDiscretization{L<:Unitful.Length, A<:Unitful.Area}
    wing::WingGeometry{L}
    discretization::Discretization
    elements::Vector{WingElement{L,A}}
    total_dorsal_area::A
    total_ventral_area::A
end


export WingGeometry,
       Discretization,
       WingElement,
       WingDiscretization,
       discretize_wing,
       chord_at_span,
       get_plate_dimensions,
       wing_span_m,
       wing_area_m2,
       mean_chord_m,
       aspect_ratio,
       body_frontal_m2,
       build_wing_for_mass


# ═════════════════════════════════════════════════════════════════════
# Core geometry functions
# ═════════════════════════════════════════════════════════════════════

"""
    chord_at_span(wing, s)

Linearly-interpolated chord at span-wise position `s` (Unitful length).
"""
function chord_at_span(wing::WingGeometry, s)
    return wing.root_chord + (wing.tip_chord - wing.root_chord) * (s / wing.wing_length)
end


"""
    get_plate_dimensions(wing, span_pos, span_length) → NamedTuple

Returns `(chord, dorsal_area, ventral_area)` for a plate at
`span_pos` with span-wise width `span_length`.
"""
function get_plate_dimensions(wing::WingGeometry, span_pos, span_length)
    chord = chord_at_span(wing, span_pos)
    dorsal_area  = chord * span_length
    ventral_area = chord * span_length
    return (chord = chord, dorsal_area = dorsal_area, ventral_area = ventral_area)
end


"""
    discretize_wing(wing, disc) → WingDiscretization

Build span-wise plate elements for a `WingGeometry`.
"""
function discretize_wing(wing::WingGeometry{L}, disc::Discretization) where {L<:Unitful.Length}
    n = disc.n_elements
    span_length = wing.wing_length / n

    span_positions = if disc.method == "uniform"
        [(i - 0.5) * span_length for i = 1:n]
    elseif disc.method == "nonuniform"
        s_norm = [(i - 0.5) / n for i = 1:n]
        wing.wing_length .* (s_norm .^ 1.5)
    else
        error("Unknown discretization method: $(disc.method)")
    end

    A = typeof(chord_at_span(wing, span_positions[1]) * span_length)
    elements = Vector{WingElement{L,A}}(undef, n)

    for i = 1:n
        s_pos = span_positions[i]
        left  = (i - 1) * span_length
        right = i * span_length
        dims  = get_plate_dimensions(wing, s_pos, span_length)
        elements[i] = WingElement(
            element_id    = i,
            span_position = s_pos,
            span_length   = span_length,
            chord_length  = dims.chord,
            thickness     = wing.thickness,
            left_edge     = left,
            right_edge    = right,
            plate_size    = span_length,
            dorsal_area   = dims.dorsal_area,
            ventral_area  = dims.ventral_area,
            has_dorsal    = wing.dorsal_active,
            has_ventral   = wing.ventral_active,
        )
    end

    zero_area     = zero(elements[1].dorsal_area)
    total_dorsal  = mapreduce(e -> e.has_dorsal  ? e.dorsal_area  : zero_area, +, elements)
    total_ventral = mapreduce(e -> e.has_ventral ? e.ventral_area : zero_area, +, elements)

    return WingDiscretization(
        wing               = wing,
        discretization     = disc,
        elements           = elements,
        total_dorsal_area  = total_dorsal,
        total_ventral_area = total_ventral,
    )
end


# ═════════════════════════════════════════════════════════════════════
# Allometric scaling helpers
# ═════════════════════════════════════════════════════════════════════
#
# These are thin wrappers around the canonical formulas in `AFPT`
# (see `afpt.jl` → `wing_span_allometry`, `wing_area_allometry`,
# `mean_chord_allometry`, `aspect_ratio_allometry`,
# `compute_body_frontal_area`).  Keeping the public names here for
# back-compatibility — the numerical formulas live in one place.
#
# All inputs are plain Float64 in kg; outputs are plain Float64 too,
# so allometry is independent of Unitful (units are reattached at the
# call site where needed).
# ─────────────────────────────────────────────────────────────────────

"Full wing span [m] from body mass [kg] (Pennycuick 2008).  Wrapper for `AFPT.wing_span_allometry`."
wing_span_m(m_kg::Real)     = wing_span_allometry(m_kg)

"Total wing area [m²] (both wings) from body mass [kg] (Pennycuick 2008).  Wrapper for `AFPT.wing_area_allometry`."
wing_area_m2(m_kg::Real)    = wing_area_allometry(m_kg)

"Mean chord [m]: c̄ = S/b.  Wrapper for `AFPT.mean_chord_allometry`."
mean_chord_m(m_kg::Real)    = mean_chord_allometry(m_kg)

"Aspect ratio AR = b²/S.  Wrapper for `AFPT.aspect_ratio_allometry`."
aspect_ratio(m_kg::Real)    = aspect_ratio_allometry(m_kg)

"""
    body_frontal_m2(m_kg; type = :other) → Float64

Body frontal area [m²] for a bird of body mass `m_kg` [kg].
Wrapper for `AFPT.compute_body_frontal_area` (allometry from
afpt-r `computeBodyFrontalArea.R`):

  * `:passerine` → 0.0129 · m^0.614
  * otherwise    → 0.00813 · m^0.666
"""
body_frontal_m2(m_kg::Real; type::Symbol = :other) =
    compute_body_frontal_area(m_kg, type)


"""
    build_wing_for_mass(m_kg; n_elements=10, method="uniform",
                              thickness_factor=0.05,
                              dorsal_active=true, ventral_active=true) → WingDiscretization

Construct a `WingDiscretization` for a single wing using Pennycuick's
allometric scaling.  Assumes a rectangular planform with root chord =
tip chord = mean chord.  The struct represents **one** wing of half-span
`b/2` — multiply heat-loss / power outputs by 2 for the full pair.

`thickness_factor` sets `t ≈ thickness_factor · mean_chord`, with a
small floor (0.5 mm) for very small birds.
"""
function build_wing_for_mass(m_kg::Real;
                             n_elements::Int = 10,
                             method::String = "uniform",
                             thickness_factor::Real = 0.05,
                             dorsal_active::Bool = true,
                             ventral_active::Bool = true)
    b  = wing_span_m(m_kg)
    c̄  = mean_chord_m(m_kg)
    th = max(thickness_factor * c̄, 5e-4)
    wing = WingGeometry(
        wing_length    = (b/2) * u"m",
        root_chord     = c̄ * u"m",
        tip_chord      = c̄ * u"m",
        thickness      = th * u"m",
        dorsal_active  = dorsal_active,
        ventral_active = ventral_active,
    )
    return discretize_wing(wing, Discretization(n_elements = n_elements, method = method))
end


end # module WingPlates
