# =====================================================================
# whole_animal_heatbalance.jl
#
# Couples the body thermoregulation loop (BiophysicalBehaviour's
# `thermoregulate(::Endotherm, ::RuleBasedSequentialControl, ...)`) to
# the wing-element heat balance from `wing_heatbalance_2.0.jl`.
#
# Pipeline ("step 0 → behaviours"):
#
#   STEP 0  Compute the wingbeat-averaged net heat loss from BOTH wings
#           ONCE, before any body behaviours are engaged.  Wing
#           temperature is set deterministically to `T_wing = T_air + 2 K`
#           and forward airspeed defaults to V_mr (the AFPT maximum-
#           range speed for the bird's mass).  This produces a fixed
#           offset `Q_wing_loss` [W] = −2·`Q_net_mean`, i.e. the extra
#           heat dumped to the environment by the two wings per second.
#
#   STEPS 1–6  The classic BB sequential controller runs unchanged
#           (piloerect → uncurl → vasodilate → hyperthermia → pant →
#           sweat) EXCEPT that the convergence check becomes
#               (Q_gen + Q_wing_loss) < Q_minimum · (1 − tolerance)
#           — i.e. the wing heat sink offsets the metabolic-rate
#           shortfall.  Panting / hyperthermia math (which adjusts
#           `Q_minimum`) still uses the bare Q_gen / Q_minimum values,
#           per the design decision recorded in the project plan.
#
# Return value mirrors the budgerigar example but with an extra
# `wing_fluxes` field (the full `WingbeatHeatBalance`) and `Q_wing_loss`:
#       (; thermoregulation, morphology, energy_fluxes, mass_fluxes,
#          wing_fluxes, Q_wing_loss, organism)
# =====================================================================

module WholeAnimalHeatBalance

using Unitful
using Setfield: @set

# Sister modules — included earlier by wing_power.jl
using ..AFPT: build_afpt_bird, wing_span_allometry, wing_area_allometry,
              find_maximum_range_speed
using ..FlightEnvironment: Microclimate
using ..WingPlates: build_wing_for_mass, WingDiscretization
using ..WingKinematics: FlappingKinematics, build_kinematics_for_mass
using ..WingHeatBalance: WingTemperatures, WingbeatHeatBalance,
                         uniform_temperature, compute_wingbeat_heatbalance,
                         default_absorptivities, default_emissivities,
                         default_view_factors
using ..BodyHeatBalance: BirdBody, build_body_for_mass,
                         build_environment, build_organism, afpt_v_mr

# BiophysicalBehaviour primitives & helpers
import BiophysicalBehaviour as BB
using BiophysicalBehaviour: Organism, Endotherm, RuleBasedSequentialControl,
                            thermoregulation, control_strategy,
                            piloerect, uncurl, vasodilate, hyperthermia,
                            pant, sweat
using HeatExchange: solve_metabolic_rate


# =====================================================================
# Step 0 — wing-Q precomputation
# =====================================================================

"""
    compute_wing_loss(m_kg, micro; V_forward, T_wing_offset, n_elements,
                                   n_steps, stroke_plane_deg, kin)
        → (; Q_wing_loss, wing_fluxes, kinematics, geometry, temperatures)

Build a fresh wing geometry+kinematics for body mass `m_kg`, set the
wing surface temperature to `T_air + T_wing_offset` (default 2 K), and
compute the wingbeat-averaged heat balance with
`compute_wingbeat_heatbalance`.

`Q_wing_loss` = −2 · `wing_fluxes.Q_net_mean`  →  positive when the
two wings act as a net heat sink (T_wing > T_environment).  This is
the value injected into the body's metabolic-rate convergence check.

If `kin` is provided, the supplied kinematics override the allometric
default — useful for diagnostics.
"""
function compute_wing_loss(m_kg::Real, micro::Microclimate;
                           V_forward                  = nothing,
                           T_wing_offset              = 2.0u"K",
                           n_elements::Int            = 10,
                           n_steps::Int               = 40,
                           stroke_plane_deg::Real     = 80.0,
                           kin::Union{Nothing,FlappingKinematics} = nothing)
    # 1. forward airspeed (default = V_mr from AFPT)
    V_fwd = V_forward === nothing ? afpt_v_mr(m_kg) * u"m/s" : V_forward
    Vf_ms = ustrip(u"m/s", uconvert(u"m/s", V_fwd))

    # 2. wing geometry + kinematics (allometric)
    wd = build_wing_for_mass(m_kg; n_elements = n_elements)
    kk = kin === nothing ?
            build_kinematics_for_mass(m_kg;
                                       stroke_plane_deg = stroke_plane_deg,
                                       V_forward_ms     = Vf_ms) :
            kin

    # 3. uniform wing temperature = T_air + offset
    T_air_K = uconvert(u"K", micro.air_temperature)
    T_wing  = T_air_K + uconvert(u"K", T_wing_offset) - 0.0u"K"   # ensure Quantity
    wt = uniform_temperature(wd, T_wing)

    # 4. wingbeat heat balance (single half-wing)
    wbhb = compute_wingbeat_heatbalance(kk, wd, wt, micro;
                                        n_steps   = n_steps,
                                        V_forward = V_fwd)

    # 5. Both wings, sign flipped:  Q_wing_loss > 0  ⇔  body loses heat
    Q_wing_loss = -2 * wbhb.Q_net_mean

    return (; Q_wing_loss   = uconvert(u"W", Q_wing_loss),
              wing_fluxes   = wbhb,
              kinematics    = kk,
              geometry      = wd,
              temperatures  = wt)
end


# =====================================================================
# Step 1–6 — modified BB rule-based sequential controller
#
# Ported verbatim from BiophysicalBehaviour/src/endotherm/homeothermy.jl
# (function `thermoregulate(::Endotherm, ::RuleBasedSequentialControl,
#  …)`) — with the SINGLE modification that the outer `while` test
# compares `(Q_gen + Q_wing_loss)` against the minimum instead of bare
# `Q_gen`.  All panting/hyperthermia/sweating math is left untouched so
# `Q_minimum` updates retain the original semantics.
# =====================================================================

"""
    thermoregulate_with_wings(organism, environment, Q_gen, T_skin,
                              T_insulation, Q_wing_loss) → endotherm_out

Run the rule-based sequential controller on `organism`, treating
`Q_wing_loss` (a constant power dissipated by the wings) as additional
heat-loss capacity in the convergence test only.
"""
function thermoregulate_with_wings(organism::Organism,
                                   environment::NamedTuple,
                                   Q_gen,
                                   T_skin,
                                   T_insulation,
                                   Q_wing_loss)
    # ── Extract limits (mutated locally as behaviours fire) ──────────
    limits        = thermoregulation(organism)
    endotherm_out = nothing

    (; mode, tolerance, max_iterations) = limits.control

    insulation_limits   = limits.insulation
    shape_b_limits      = limits.shape_b
    k_flesh_limits      = limits.k_flesh
    T_core_limits       = limits.T_core
    panting_limits      = limits.panting
    skin_wetness_limits = limits.skin_wetness

    # ── Optional piloerect-start (verbatim from BB) ──────────────────
    if insulation_limits.dorsal.step > 0.0 &&
       (insulation_limits.dorsal.current + insulation_limits.ventral.current) > 0u"mm"
        insulation_limits = @set insulation_limits.dorsal.current  = insulation_limits.dorsal.max
        insulation_limits = @set insulation_limits.ventral.current = insulation_limits.ventral.max
        zero_step_limits = @set insulation_limits.dorsal.step = 0.0
        zero_step_limits = @set zero_step_limits.ventral.step = 0.0
        insulation_limits, organism = piloerect(organism, zero_step_limits)
        insulation_limits = @set insulation_limits.dorsal.step  = limits.insulation.dorsal.step
        insulation_limits = @set insulation_limits.ventral.step = limits.insulation.ventral.step
    end

    # ── Initial solve ────────────────────────────────────────────────
    endotherm_out = solve_metabolic_rate(organism, environment, T_skin, T_insulation)
    T_skin       = endotherm_out.thermoregulation.T_skin
    T_insulation = endotherm_out.thermoregulation.T_insulation
    Q_gen        = endotherm_out.energy_fluxes.Q_gen

    Q_minimum = limits.Q_minimum_ref
    iteration = 0

    # ── Main loop ─ MODIFIED comparison includes wing heat loss ──────
    while (Q_gen + Q_wing_loss) < Q_minimum * (1 - tolerance)
        iteration += 1
        if iteration > max_iterations
            @warn "max_iterations exceeded in whole-animal thermoregulation"
            return endotherm_out
        end

        # 1. Piloerection
        if (insulation_limits.dorsal.current > insulation_limits.dorsal.reference) &&
           (insulation_limits.ventral.current > insulation_limits.ventral.reference)
            insulation_limits, organism = piloerect(organism, insulation_limits)

        # 2. Uncurl
        elseif shape_b_limits.current < shape_b_limits.max
            shape_b_limits, organism = uncurl(organism, shape_b_limits)

        # 3. Vasodilate
        elseif k_flesh_limits.current < k_flesh_limits.max
            k_flesh_limits, organism = vasodilate(organism, k_flesh_limits)

        # 4. Hyperthermia (+ optional parallel pant/sweat)
        elseif T_core_limits.current < T_core_limits.max
            T_core_limits, Q_minimum, organism = hyperthermia(
                organism, T_core_limits, panting_limits.cost
            )
            if BB.simultaneous_pant(mode) &&
               panting_limits.pant.current < panting_limits.pant.max
                panting_limits, Q_minimum, organism = pant(organism, panting_limits)
            end
            if BB.simultaneous_sweat(mode)
                if (skin_wetness_limits.current > skin_wetness_limits.max) ||
                   (skin_wetness_limits.step <= 0)
                    @warn "All thermoregulatory options exhausted"
                    return endotherm_out
                end
                skin_wetness_limits, organism = sweat(organism, skin_wetness_limits)
            end

        # 5. Pant (+ optional parallel sweat)
        elseif panting_limits.pant.current < panting_limits.pant.max
            panting_limits, Q_minimum, organism = pant(organism, panting_limits)
            if BB.simultaneous_sweat(mode)
                if (skin_wetness_limits.current > skin_wetness_limits.max) ||
                   (skin_wetness_limits.step <= 0)
                    @warn "All thermoregulatory options exhausted"
                    return endotherm_out
                end
                skin_wetness_limits, organism = sweat(organism, skin_wetness_limits)
            end

        # 6. Sweat
        else
            if (skin_wetness_limits.current > skin_wetness_limits.max) ||
               (skin_wetness_limits.step <= 0)
                return endotherm_out
            end
            skin_wetness_limits, organism = sweat(organism, skin_wetness_limits)
        end

        endotherm_out = solve_metabolic_rate(organism, environment, T_skin, T_insulation)
        T_skin       = endotherm_out.thermoregulation.T_skin
        T_insulation = endotherm_out.thermoregulation.T_insulation
        Q_gen        = endotherm_out.energy_fluxes.Q_gen
    end

    return endotherm_out
end


# =====================================================================
# Top-level driver — mirrors run_body_thermoregulation
# =====================================================================

"""
    run_whole_animal(bb, micro;
                     V_air,                # forward airspeed (body convection AND wing forward speed)
                     T_wing_offset,        # K above T_air, default 2 K
                     wing_kwargs,          # NamedTuple forwarded to compute_wing_loss
                     environment_kwargs,
                     organism_kwargs,
                     T_skin_init, T_insulation_init, Q_gen_init)
        → (; thermoregulation, morphology, energy_fluxes, mass_fluxes,
            wing_fluxes, Q_wing_loss, organism, environment)

End-to-end whole-animal thermoregulation:
  • build Organism from `bb`
  • build environment from `micro` (at airspeed `V_air`)
  • compute wing heat loss at the same `V_air` and `T_wing = T_air + 2 K`
  • run modified BB rule-based loop

`V_air` defaults to V_mr (AFPT maximum-range speed for `bb.m_kg`).
"""
function run_whole_animal(bb::BirdBody, micro::Microclimate;
                          V_air                  = nothing,
                          T_wing_offset          = 2.0u"K",
                          wing_kwargs::NamedTuple        = NamedTuple(),
                          environment_kwargs::NamedTuple = NamedTuple(),
                          organism_kwargs::NamedTuple    = NamedTuple(),
                          T_skin_init             = nothing,
                          T_insulation_init       = nothing,
                          Q_gen_init              = 0.0u"W")

    # Default forward airspeed = V_mr; reused for both body forced
    # convection and wing forward flight.
    V = V_air === nothing ? afpt_v_mr(bb.m_kg) * u"m/s" : V_air

    # Build organism + environment exactly as run_body_thermoregulation
    organism    = build_organism(bb; organism_kwargs...)
    environment = build_environment(micro; V_air = V, environment_kwargs...)

    # Step 0: wing heat loss
    wing = compute_wing_loss(bb.m_kg, micro;
                             V_forward     = V,
                             T_wing_offset = T_wing_offset,
                             wing_kwargs...)

    # Initial conditions (match budgerigar example defaults)
    T_core = organism.traits.physiology.metabolism_pars.T_core
    T_air  = environment.environment_vars.T_air
    T_s    = T_skin_init       === nothing ? T_core - 5.0u"K" : uconvert(u"K", T_skin_init)
    T_i    = T_insulation_init === nothing ?
                T_air + 0.5 * (T_core - T_air) :
                uconvert(u"K", T_insulation_init)
    Q_g    = isa(Q_gen_init, Quantity) ? uconvert(u"W", Q_gen_init) : Q_gen_init * u"W"

    endotherm_out = thermoregulate_with_wings(organism, environment,
                                              Q_g, T_s, T_i, wing.Q_wing_loss)

    return (; thermoregulation = endotherm_out.thermoregulation,
              morphology       = endotherm_out.morphology,
              energy_fluxes    = endotherm_out.energy_fluxes,
              mass_fluxes      = endotherm_out.mass_fluxes,
              wing_fluxes      = wing.wing_fluxes,
              Q_wing_loss      = wing.Q_wing_loss,
              organism         = organism,
              environment      = environment)
end

"""
    run_whole_animal_for_mass(m_kg, micro; build_kwargs, kwargs...)

Convenience wrapper that allocates the `BirdBody` from `m_kg` using
`build_body_for_mass(m_kg; build_kwargs...)` and then calls
`run_whole_animal`.  All other kwargs are forwarded.
"""
function run_whole_animal_for_mass(m_kg::Real, micro::Microclimate;
                                   build_kwargs::NamedTuple = NamedTuple(),
                                   kwargs...)
    bb = build_body_for_mass(m_kg; build_kwargs...)
    return (; bird = bb, run_whole_animal(bb, micro; kwargs...)...)
end


# =====================================================================
# Exports
# =====================================================================

export compute_wing_loss,
       thermoregulate_with_wings,
       run_whole_animal,
       run_whole_animal_for_mass

end # module WholeAnimalHeatBalance
