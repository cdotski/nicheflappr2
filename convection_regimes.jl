# =====================================================================
# convection_regimes.jl
#
# Lightweight shared module defining the three plate-convection
# correlations used by both wing_convection_2.0.jl (snapshot-only
# module) and wing_heatbalance_2.0.jl (full heat balance).
#
# Three regimes are exposed:
#   LaminarPlate()      — Pohlhausen:  Nu = 0.664·Re^0.5·Pr^(1/3)
#   TurbulentPlate()    — fully turb.: Nu = 0.037·Re^0.8·Pr^(1/3)
#   MixedPlate(Re_c=5e5) — laminar below Re_c, then
#                          Nu = (0.037·Re^0.8 − A)·Pr^(1/3)
#                         with  A = 0.037·Re_c^0.8 − 0.664·Re_c^0.5
#                         (continuous at the transition).
#
# `convection_h(regime, Re, Pr, k_air, L) → Quantity [W/(m²·K)]`
# returns the heat-transfer coefficient h = Nu·k_air/L.
# =====================================================================
module ConvectionRegimes

using Unitful

export PlateConvectionRegime, LaminarPlate, TurbulentPlate, MixedPlate,
       nusselt_plate, nusselt_laminar, nusselt_turbulent, nusselt_mixed,
       convection_h, reynolds_number

abstract type PlateConvectionRegime end

struct LaminarPlate   <: PlateConvectionRegime end
struct TurbulentPlate <: PlateConvectionRegime end
struct MixedPlate     <: PlateConvectionRegime
    Re_c::Float64
end
MixedPlate() = MixedPlate(5.0e5)

function reynolds_number(V, L, ν)::Float64
    V_num = isa(V, Quantity) ? ustrip(u"m/s",   V) : V
    L_num = isa(L, Quantity) ? ustrip(u"m",     L) : L
    ν_num = isa(ν, Quantity) ? ustrip(u"m^2/s", ν) : ν
    return V_num * L_num / ν_num
end

function nusselt_laminar(Re::Real, Pr::Real)::Float64
    Re ≤ 0.0 && return 0.0
    return 0.664 * sqrt(Re) * Pr^(1/3)
end

function nusselt_turbulent(Re::Real, Pr::Real)::Float64
    Re ≤ 0.0 && return 0.0
    return 0.037 * Re^0.8 * Pr^(1/3)
end

function nusselt_mixed(Re::Real, Pr::Real; Re_c::Real = 5.0e5)::Float64
    Re ≤ 0.0 && return 0.0
    if Re ≤ Re_c
        return nusselt_laminar(Re, Pr)
    else
        A = 0.037 * Re_c^0.8 - 0.664 * sqrt(Re_c)
        return (0.037 * Re^0.8 - A) * Pr^(1/3)
    end
end

nusselt_plate(::LaminarPlate,   Re::Real, Pr::Real) = nusselt_laminar(Re, Pr)
nusselt_plate(::TurbulentPlate, Re::Real, Pr::Real) = nusselt_turbulent(Re, Pr)
nusselt_plate(r::MixedPlate,    Re::Real, Pr::Real) = nusselt_mixed(Re, Pr; Re_c = r.Re_c)

function convection_h(regime::PlateConvectionRegime, Re::Real, Pr::Real, k_air, L)
    k_num = isa(k_air, Quantity) ? ustrip(u"W/(m*K)", k_air) : k_air
    L_num = isa(L,     Quantity) ? ustrip(u"m",       L)     : L
    Nu = nusselt_plate(regime, Re, Pr)
    return (Nu * k_num / L_num) * u"W/(m^2*K)"
end

end # module ConvectionRegimes
