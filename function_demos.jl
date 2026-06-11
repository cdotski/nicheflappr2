# =====================================================================
# function_demos.jl
#
# Consolidated, runnable demonstrations for the streamlined modules:
#
#   • wing_geometry_2.0.jl   →  module WingPlates
#   • wing_kinematics_2.0.jl →  module WingKinematics
#   • wing_heatbalance_2.0.jl →  module WingHeatBalance
#   • wing_power.jl          →  module WingPower
#
# Each demo is wrapped in its own `let … end` block and is independent
# of the others, so individual sections can be evaluated on their own
# in the REPL.  Plotting blocks degrade gracefully if `Plots` is not
# installed.
# =====================================================================

include("wing_power.jl")

using .WingPlates, .WingKinematics, .WingHeatBalance, .WingPower
using .BodyHeatBalance
using .FlightEnvironment: Microclimate, microclimate_at_altitude
using .AFPT: build_afpt_bird, compute_flight_performance,
             compute_flapping_power, find_maximum_range_speed,
             find_minimum_power_speed,
             wing_span_allometry, wing_area_allometry
using Unitful
using Printf

"""
    quick_afpt_bird(m_kg; type = :other) → AfptBird

Convenience helper used by the demos: build an `AfptBird` with the
canonical afpt-style allometric span / area for body mass `m_kg`.
"""
quick_afpt_bird(m_kg::Real; type::Symbol = :other) =
    build_afpt_bird(m_kg, wing_span_allometry(m_kg);
                    wingArea = wing_area_allometry(m_kg), type = type)

const _PLOTS_AVAILABLE = try
    @eval using Plots
    true
catch
    @info "Plots.jl not available — plot sections will be skipped."
    false
end


# ─────────────────────────────────────────────────────────────────────
# 1. Geometry demo
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 1. Wing geometry =========================================")

    # By-hand wing
    wing = WingGeometry(
        wing_length = 0.20u"m",
        root_chord  = 0.06u"m",
        tip_chord   = 0.03u"m",
        thickness   = 0.002u"m",
    )
    disc = discretize_wing(wing, Discretization(n_elements = 8))
    @printf "Manual wing: b/2 = %s, S_dorsal = %s\n" wing.wing_length disc.total_dorsal_area

    # Allometric wing for a 0.05 kg passerine
    wd = build_wing_for_mass(0.05; n_elements = 10)
    @printf "Allometric 50 g bird: b/2 = %s, mean chord = %s\n" wd.wing.wing_length wd.wing.root_chord
end


# ─────────────────────────────────────────────────────────────────────
# 2. Kinematics demo (strouhal based)
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 2. Kinematics + element velocities (strouhal based) ====")

    m_kg = 0.05
    wd   = build_wing_for_mass(m_kg; n_elements = 10)

    # Strouhal frequency needs a forward airspeed — get V_mr from a
    # quick AFPT bird build, then either pass V_forward_ms as a kwarg
    # or bake it into the StrouhalFreq struct.
    bird  = quick_afpt_bird(m_kg)
    V_mr  = find_maximum_range_speed(bird)        # [m/s]

    kin  = build_kinematics_for_mass(m_kg, StrouhalFreq();
                                     amp = StrouhalAmplitude(), stroke_plane_deg = 80.0,
                                     V_forward_ms = V_mr)

    @printf "V_mr = %.2f m/s,  f = %.2f Hz, amplitude = %.1f°\n" V_mr ustrip(u"Hz", kin.frequency) rad2deg(kin.amplitude)

    ev = compute_element_velocities(kin, wd, 0.01u"s"; V_forward = V_mr * u"m/s")
    @printf "v_tip = %s (realised)\n" ev.realised_airspeed[end]

    if _PLOTS_AVAILABLE
        ts = range(0u"s", uconvert(u"s", 1/kin.frequency); length = 200)
        vs = [ustrip(u"m/s", realised_airspeed(kin, wd.elements[end].span_position, t;
                                               V_forward = V_mr * u"m/s")) for t in ts]
        display(plot(ustrip.(u"s", ts), vs;
                     xlabel = "t [s]", ylabel = "tip speed [m/s]",
                     title  = "Tip airspeed over wingbeat"))
    end

    # Alternative: bake U into the StrouhalFreq struct itself
    str = StrouhalFreq(0.21, 1.225, V_mr)
    @printf "Strouhal-based f = %.2f Hz (at U = %.2f m/s)\n" wingbeat_hz(m_kg, str) V_mr
end


# ─────────────────────────────────────────────────────────────────────
# 2.1 Kinematics demo — AFPT-based (Pennycuick frequency + optimised
#     strokeplane + afpt-consistent amplitude).  Mirrors Demo 2 so the
#     two parameterisations can be compared side-by-side.
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 2.1 Kinematics (AFPT-based) =============================")

    m_kg = 0.05
    wd   = build_wing_for_mass(m_kg; n_elements = 10)

    # Build the canonical afpt bird and solve its V_mr + flapping-power
    # optimum.  The returned NamedTuple carries the optimised
    # strokeplane angle and the kinematically-consistent amplitude that
    # belong with afpt's `compute_flapping_power` minimum.
    bird = quick_afpt_bird(m_kg)
    V_mr = find_maximum_range_speed(bird)
    res  = compute_flapping_power(bird, V_mr; strokeplane = :opt)

    # AFPT-driven kinematics — using new defaults (all computed inside the function):
    #   • frequency  → Pennycuick2008MinPower (afpt's own formula)
    #   • amplitude  → afpt power-minimised (computed from V_forward_ms)
    #   • stroke plane → afpt optimal, converted from afpt's from-vertical
    #                    convention to our from-horizontal: 90° − φ_afpt
    kin = build_kinematics_for_mass(m_kg; V_forward_ms = V_mr)

    @printf "V_mr = %.2f m/s,  f = %.2f Hz, amplitude = %.1f°, φ_opt(from horiz) = %.1f°\n" V_mr ustrip(u"Hz", kin.frequency) rad2deg(kin.amplitude) rad2deg(kin.stroke_plane_angle)
    @printf "kf   = %.3f,  T/L = %.3f\n" res.kf res.ToverL

    ev = compute_element_velocities(kin, wd, 0.01u"s"; V_forward = V_mr * u"m/s")
    @printf "v_tip = %s (realised)\n" ev.realised_airspeed[end]

    if _PLOTS_AVAILABLE
        ts = range(0u"s", uconvert(u"s", 1/kin.frequency); length = 200)
        vs = [ustrip(u"m/s", realised_airspeed(kin, wd.elements[end].span_position, t;
                                               V_forward = V_mr * u"m/s")) for t in ts]
        display(plot(ustrip.(u"s", ts), vs;
                     xlabel = "t [s]", ylabel = "tip speed [m/s]",
                     title  = "Tip airspeed over wingbeat (AFPT kinematics)"))
    end
end


# ─────────────────────────────────────────────────────────────────────
# 3. Temperature-assignment methods
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 3. Wing-temperature assignment ==========================")

    wd = build_wing_for_mass(0.05; n_elements = 8)

    t1 = uniform_temperature(wd, 25.0u"°C")
    t2 = linear_gradient(wd, 32.0u"°C", 24.0u"°C")
    t3 = exponential_decay(wd, 32.0u"°C", 24.0u"°C"; decay_rate = 3.0)
    t4 = dorsal_ventral_split(wd, 30.0u"°C", 27.0u"°C")
    t5 = from_function(wd, s -> 32.0u"°C" - 8.0u"K" * (s / wd.wing.wing_length))

    for (name, t) in [("uniform",t1), ("linear",t2), ("exp",t3),
                       ("d/v split",t4), ("functional",t5)]
        @printf "%-12s  T_root = %s, T_tip = %s\n" name t.T_dorsal[1] t.T_dorsal[end]
    end
end


# ─────────────────────────────────────────────────────────────────────
# 4. Air properties at altitude
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 4. Air properties at altitude ===========================")
    for h in (0.0u"m", 1000.0u"m", 3000.0u"m", 5000.0u"m")
        air = air_properties(15.0u"°C"; altitude = h)
        @printf "h = %5s  →  ρ = %s, ν = %s, k = %s\n" h air.ρ air.ν air.k_air
    end
end


# ─────────────────────────────────────────────────────────────────────
# 5. Full heat balance over one wingbeat (single microclimate)
#
# Same configuration as demo 5.3 (100 g bird, AFPT-driven kinematics —
# Pennycuick freq + afpt-optimised amplitude + afpt-optimised stroke-
# plane — MixedPlate convection, T_wing = T_air + 1.5 K, evaluated at
# h = 30 m above ground), but driven through `Q_for_mass` for ONE
# microclimate so the full per-wingbeat snapshot stack
# (`b.heat.snapshots`, `b.heat.*_series`) is available for plotting.
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 5. Wing heat balance (single microclimate) ==============")

    m_kg   = 1.00
    h_fly  = 30.0u"m"
    T_wing = 25.0u"°C"     # absolute wing surface temperature

    micro = microclimate_at_altitude(altitude         = h_fly,
                                     air_temperature    = 23.0u"°C",
                                     ground_temperature = 25.0u"°C",
                                     sky_temperature    =  5.0u"°C",
                                     zenith_angle       = 10.0u"°",
                                     global_radiation   = 500.0u"W/m^2",
                                     diffuse_fraction   = 0.20,
                                     relative_humidity  = 0.50)

    T_air = micro.air_temperature
    ΔT_wa = uconvert(u"K", uconvert(u"K", T_wing) - uconvert(u"K", T_air))

    # AFPT-driven kinematics (amp + φ left as nothing ⇒ Q_for_mass uses
    # the afpt-optimised values).  Convection = MixedPlate (ours).
    b = Q_for_mass(m_kg;
                   T_air        = T_air,
                   T_wing       = T_wing,
                   altitude     = h_fly,
                   microclimate = micro,
                   n_elements   = 10, n_steps = 40,
                   freq_method      = Pennycuick2008MinPower(),
                   convection_model = MixedPlate())

    wbh    = b.heatbalance # WingbeatHeatBalance with per-step series
    s      = summarize(b)
    A_one  = ustrip(u"m^2",
                    b.wing_disc.total_dorsal_area +
                    b.wing_disc.total_ventral_area)

    @printf "m = %.0f g,  V_mr = %.2f m/s,  f = %.2f Hz,  amp = %.1f°,  φ = %.1f°\n" (m_kg*1000) s.V_mr s.f_Hz s.amp_deg s.sp_deg
    @printf "h = %.1f m,  P = %.0f Pa,  T_air = %.1f °C,  T_wing = %.1f °C  (ΔT = %.1f K)\n" ustrip(u"m", h_fly) ustrip(u"Pa", micro.atmospheric_pressure) ustrip(u"°C", T_air) ustrip(u"°C", T_wing) ustrip(u"K", ΔT_wa)
    @printf "A_one-wing(d+v) = %.4f m²\n\n" A_one

    pathways = ("Q_conv", "Q_solar", "Q_lw_in", "Q_lw_out", "Q_lw_net", "Q_net")
    vals     = (ustrip(u"W", wbh.Q_conv_mean),
                ustrip(u"W", wbh.Q_solar_mean),
                ustrip(u"W", wbh.Q_lw_in_mean),
                ustrip(u"W", wbh.Q_lw_out_mean),
                ustrip(u"W", wbh.Q_lw_net_mean),
                ustrip(u"W", wbh.Q_net_mean))

    @printf "   %-10s %12s %14s\n" "pathway" "Q [W]" "q [W/m²]"
    println("   " * "─"^40)
    for (lbl, q) in zip(pathways, vals)
        @printf "   %-10s %+12.5f %+14.4f\n" lbl q (q / A_one)
    end
    @printf "\n   (positive Q_net ⇒ net heating; both wings  Q_net = %+10.5f W)\n" (2 * ustrip(u"W", wbh.Q_net_mean))

    if _PLOTS_AVAILABLE
        ts = ustrip.(u"s", wbh.times)
        display(plot(ts, ustrip.(u"W", wbh.Q_conv_series);
                     label = "Q_conv", xlabel = "t [s]", ylabel = "Q [W]",
                     title  = "One-wing instantaneous heat flows"))
        plot!(ts, ustrip.(u"W", wbh.Q_solar_series);  label = "Q_solar")
        plot!(ts, ustrip.(u"W", wbh.Q_lw_net_series); label = "Q_lw,net")
        plot!(ts, ustrip.(u"W", wbh.Q_net_series);    label = "Q_net", lw = 2)
    end
end


# ─────────────────────────────────────────────────────────────────────
# 5.1 Convection-only comparison: AFPT kinematics (new defaults) vs Strouhal
#
# Minimal environment — T_air is fixed 3 °C below the wing.  Everything
# else (radiation, microclimate) is omitted so the only heat-loss
# pathway is forced convection.  Both kinematics parameterisations are
# fed the same wing geometry, the same forward airspeed (V_mr from
# AFPT) and the same wing temperature; the only differences are the
# frequency, amplitude and stroke-plane angle.
#
# AFPT row uses the new build_kinematics_for_mass defaults: Pennycuick
# frequency, afpt-optimal amplitude and stroke plane (converted from
# afpt's from-vertical convention to our from-horizontal: φ_horiz = 90°−φ_afpt).
# Strouhal row uses the old explicit parameters for direct comparison.
# φ [°] column is from-horizontal throughout.
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 5.1 Convection-only: AFPT vs Strouhal kinematics ========")

    m_kg     = 1
    T_wing   = 25.0u"°C"
    ΔT       = 2.0u"K"                      # air is 2 °C cooler than the wing
    T_air    = T_wing - ΔT

    # Shared inputs ----------------------------------------------------
    wd   = build_wing_for_mass(m_kg; n_elements = 10)
    wt   = uniform_temperature(wd, T_wing)
    air  = air_properties(T_air)

    bird = quick_afpt_bird(m_kg)
    V_mr = find_maximum_range_speed(bird)
    res  = compute_flapping_power(bird, V_mr; strokeplane = :opt)

    # AFPT kinematics: new defaults — Pennycuick2008MinPower freq +
    # afpt-optimal amp and stroke plane (φ convention converted inside)
    kin_afpt = build_kinematics_for_mass(m_kg; V_forward_ms = V_mr)

    # Strouhal kinematics with the previously-assumed φ = 80°
    kin_str  = build_kinematics_for_mass(m_kg, StrouhalFreq();
                                         amp              = StrouhalAmplitude(),
                                         stroke_plane_deg = 80.0,
                                         V_forward_ms     = V_mr)

    # Cycle-averaged convection on each kinematics set ----------------
    conv_afpt = compute_wingbeat_convection(kin_afpt, wd, wt, air;
                                            n_steps = 40, V_forward = V_mr * u"m/s")
    conv_str  = compute_wingbeat_convection(kin_str,  wd, wt, air;
                                            n_steps = 40, V_forward = V_mr * u"m/s")

    # Strip units once for the table
    Q_afpt = ustrip(u"W", conv_afpt.Q_mean)
    Q_str  = ustrip(u"W", conv_str.Q_mean)
    f_afpt = ustrip(u"Hz", kin_afpt.frequency)
    f_str  = ustrip(u"Hz", kin_str.frequency)
    A_afpt = rad2deg(kin_afpt.amplitude)
    A_str  = rad2deg(kin_str.amplitude)

    @printf "%-12s %8s %8s %8s %10s %10s %10s\n" "kinematics" "f [Hz]" "amp[°]" "φ [°]" "Q̄_conv[W]" "Q_max[W]" "Q_min[W]"
    println("─"^72)
    @printf "%-12s %8.2f %8.1f %8.1f %10.4f %10.4f %10.4f\n" "AFPT"     f_afpt A_afpt rad2deg(kin_afpt.stroke_plane_angle) Q_afpt ustrip(u"W", conv_afpt.Q_max) ustrip(u"W", conv_afpt.Q_min)
    @printf "%-12s %8.2f %8.1f %8.1f %10.4f %10.4f %10.4f\n" "Strouhal" f_str  A_str  rad2deg(kin_str.stroke_plane_angle)  Q_str  ustrip(u"W", conv_str.Q_max)  ustrip(u"W", conv_str.Q_min)
    println("─"^72)
    @printf "Δ (AFPT − Strouhal) = %+.4f W  (%+.1f %% relative to Strouhal)\n" (Q_afpt - Q_str) 100*(Q_afpt - Q_str)/Q_str
    @printf "Same V_forward = %.2f m/s,  ΔT = %.1f K,  one wing only.\n" V_mr ustrip(u"K", ΔT)

    if _PLOTS_AVAILABLE
        ts_a = ustrip.(u"s", conv_afpt.times)
        ts_s = ustrip.(u"s", conv_str.times)
        display(plot(ts_a, ustrip.(u"W", conv_afpt.Q_timeseries);
                     label = "AFPT",     xlabel = "t [s]", ylabel = "Q_conv [W]",
                     title  = "Convective loss over one wingbeat"))
        plot!(ts_s, ustrip.(u"W", conv_str.Q_timeseries);  label = "Strouhal")
    end
end



# ─────────────────────────────────────────────────────────────────────
# 6. Bird power curve + Vmp / Vmr
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 6. Flight-power curve ===================================")

    bird = quick_afpt_bird(0.05; type = :passerine)
    @printf "Built bird: m = %.3f kg, b = %.3f m, S = %.4f m², BMR = %.3f W\n" bird.massTotal bird.wingSpan bird.wingArea bird.basalMetabolicRate

    V_mp = find_minimum_power_speed(bird)
    V_mr = find_maximum_range_speed(bird)
    @printf "V_mp = %.2f m/s,  V_mr = %.2f m/s\n" V_mp V_mr

    if _PLOTS_AVAILABLE
        Vs      = range(0.5, 30.0; length = 200)
        samples = [compute_flapping_power(bird, V) for V in Vs]

        # afpt's flapping-power model is only validated in the forward-
        # flight regime where the reduced frequency kf = π·b·f/V stays
        # below ~6 and the airspeed exceeds ~2·v_ih (induced velocity
        # in hover).  Below ~5 m/s for a 50 g passerine both flags trip
        # and the kP_* / kD_* polynomial corrections are extrapolated
        # well outside their fit range, producing a non-smooth bump.
        # Mask those points so only the validated J-curve is drawn.
        invalid(s) = s.flags.redFreqHi || s.flags.speedLo
        Ps_valid   = [invalid(s) ? NaN : s.power_total for s in samples]
        Ps_inv     = [invalid(s) ? s.power_total : NaN for s in samples]

        plt = plot(collect(Vs), Ps_valid;
                   label  = "valid (kf ≤ 6, V ≥ 2·v_ih)",
                   xlabel = "V [m/s]", ylabel = "P_mech [W]",
                   title  = "Mechanical flight power (afpt)", lw = 2)
        plot!(plt, collect(Vs), Ps_inv;
              label = "out of afpt validity", ls = :dash, color = :grey)
        vline!(plt, [V_mp]; ls = :dot, color = :black, label = "V_mp")
        vline!(plt, [V_mr]; ls = :dot, color = :red,   label = "V_mr")
        display(plt)
    end
end


# ─────────────────────────────────────────────────────────────────────
# 7. Full pipeline: Q_for_mass scaling sweep
#    Sweeps the three plate-convection regimes side-by-side so the
#    laminar / turbulent / mixed q[W/m²] behaviour can be compared.
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 7. Q_for_mass sweep =====================================")

    demo_birds = [
        ("Hummingbird (4 g)",   0.004),
        ("Sparrow (30 g)",      0.030),
        ("Pigeon (300 g)",      0.300),
        ("Crow (500 g)",        0.500),
        ("Owl (1.5 kg)",        1.5),
        ("Goose (5 kg)",        5.0),
        ("Duck (8 kg)",         8.0),
        ("Swan (15 kg)",       15.0),
    ]

    modes = [(:laminar,   "Laminar (Pohlhausen, Nu = 0.664·√Re·Pr^(1/3))",            LaminarPlate()),
             (:turbulent, "Turbulent (Nu = 0.037·Re^0.8·Pr^(1/3))",                   TurbulentPlate()),
             (:mixed,     "Mixed laminar/turbulent (transition at Re_c = 5·10^5)",    MixedPlate())]

    for (label, header, mode) in modes
        println("\n--- $header ---")
        @printf "%-22s %-7s %-7s %-9s %-9s\n" "Bird" "m[kg]" "V_mr" "Q[W]" "q[W/m²]"
        println("─"^60)
        for (nm, m) in demo_birds
            # `amp` and `stroke_plane_deg` are omitted so the afpt-
            # optimised values propagate from compute_flapping_power.
            b = Q_for_mass(m;
                           T_air  = 20.0u"°C",
                           T_wing = 22.0u"°C",
                           altitude = 0.0u"m",
                           n_elements = 10,
                           n_steps    = 40,
                           convection_model = mode)
            s = summarize(b)
            @printf "%-22s %-7.4f %-7.2f %-9.4f %-9.2f\n" nm s.m_kg s.V_mr s.Q_mean_W s.q_per_m2
        end
    end
end


# ─────────────────────────────────────────────────────────────────────
# 7b. AFPT flight-performance summary (full Klein Heerenbrink port)
#     V_mp, V_mr, V_max, V_mt and maximum climb-rate for each demo bird.
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 7b. AFPT flight performance =============================")

    @printf "%-22s %-7s %-7s %-7s %-7s %-9s\n" "Bird" "V_mp" "V_mr" "V_max" "V_mt" "climb[m/s]"
    println("─"^72)
    for (nm, m) in [("Sparrow (30 g)", 0.030),
                    ("Pigeon (300 g)", 0.300),
                    ("Crow (500 g)",   0.500),
                    ("Owl (1.5 kg)",   1.5),
                    ("Goose (5 kg)",   5.0),
                    ("Swan (15 kg)",  15.0)]
        # Allometric span b ≈ 0.96·m^(1/3), area S ≈ 0.16·m^(2/3)
        # (Pennycuick / Greenewalt-style power laws).
        span = 0.96 * m^(1/3)
        area = 0.16 * m^(2/3)
        ab = build_afpt_bird(m, span; wingArea = area, type = :other)
        perf = compute_flight_performance(ab; length_out = 21,
                                          V_lo = 2.0, V_hi = 40.0)
        Vmax_str = perf.V_max === nothing ? "  —  " :
                   string(round(perf.V_max, digits = 2))
        @printf "%-22s %-7.2f %-7.2f %-7s %-7.2f %-9.3f\n" nm perf.V_mp perf.V_mr Vmax_str perf.V_mt perf.climb.climbRate
    end
end



# ─────────────────────────────────────────────────────────────────────
# 9. Wireframe of element airspeed over the wingbeat cycle
#    (specify a wing directly, or scale one from mass)
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 9. Element-airspeed wireframe ===========================")

    have_makie = try
        @eval using GLMakie
        true
    catch
        @info "GLMakie not available — wireframe skipped."
        false
    end
    have_makie || return

    GLMakie.activate!()

    # ---- choose one: scale from mass, or build a wing by hand --------
    use_scaled = true   # flip to false to use the by-hand wing below
    m_kg       = 0.05
    n_elements = 16
    n_steps    = 60

    if use_scaled
        wd    = build_wing_for_mass(m_kg; n_elements = n_elements)
        bird  = quick_afpt_bird(m_kg)
        V_mr  = find_maximum_range_speed(bird)
        kin   = build_kinematics_for_mass(m_kg, V_forward_ms = V_mr)
    else
        wing = WingGeometry(wing_length = 0.20u"m",
                            root_chord  = 0.06u"m",
                            tip_chord   = 0.03u"m",
                            thickness   = 0.002u"m")
        wd   = discretize_wing(wing, Discretization(n_elements = n_elements))
        V_mr = 8.0
        kin  = build_kinematics_for_mass(0.05, V_forward_ms = V_mr)
    end

    # ---- sample v_realised on the (element × time) grid --------------
    T_period = uconvert(u"s", 1 / kin.frequency)
    times    = collect(range(0u"s", T_period; length = n_steps))
    elem_ids = collect(1:n_elements)
    V        = Matrix{Float64}(undef, n_elements, n_steps)
    for (j, t) in enumerate(times)
        ev = compute_element_velocities(kin, wd, t; V_forward = V_mr * u"m/s")
        V[:, j] .= ustrip.(u"m/s", ev.realised_airspeed)
    end
    ts_s = ustrip.(u"s", times)

    fig = Figure(size = (900, 650))
    Label(fig[0, 1],
          "Realised element airspeed — m = $(round(m_kg*1000, digits=1)) g, " *
          "V_mr = $(round(V_mr, digits=2)) m/s, f = " *
          "$(round(ustrip(u"Hz", kin.frequency), digits=2)) Hz";
          fontsize = 15, halign = :center)
    ax = Axis3(fig[1, 1];
               xlabel = "element id",
               ylabel = "t [s]",
               zlabel = "v_realised [m/s]",
               azimuth = 1.2π, elevation = π/8)
    GLMakie.wireframe!(ax, elem_ids, ts_s, V; color = :steelblue)
    display(fig)
end


# ─────────────────────────────────────────────────────────────────────
# 10. Interactive wireframe — slide body mass, everything else scales
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 10. Interactive element-airspeed wireframe ==============")

    have_makie = try
        @eval using GLMakie
        true
    catch
        @info "GLMakie not available — interactive wireframe skipped."
        false
    end
    have_makie || return

    GLMakie.activate!()

    n_elements = 16
    n_steps    = 60
    elem_ids   = collect(1:n_elements)

    fig = Figure(size = (1000, 720))
    Label(fig[0, 1:2],
          "Element airspeed vs wingbeat phase — slide body mass";
          fontsize = 16, halign = :center)

    sg = SliderGrid(fig[2, 1:2],
        (label = "mass [g]", range = 5:5:15000, startvalue = 50),
        tellheight = false,
    )
    s_m = sg.sliders[1]

    # Surface (element × time) re-computed live from mass
    grid = lift(s_m.value) do mg
        m_kg = mg / 1000
        wd   = build_wing_for_mass(m_kg; n_elements = n_elements)
        bird = quick_afpt_bird(m_kg)
        V_mr = find_maximum_range_speed(bird)
        kin  = build_kinematics_for_mass(m_kg, V_forward_ms = V_mr)
        T_period = uconvert(u"s", 1 / kin.frequency)
        times    = collect(range(0u"s", T_period; length = n_steps))
        V        = Matrix{Float64}(undef, n_elements, n_steps)
        for (j, t) in enumerate(times)
            ev = compute_element_velocities(kin, wd, t; V_forward = V_mr * u"m/s")
            V[:, j] .= ustrip.(u"m/s", ev.realised_airspeed)
        end
        (ts = ustrip.(u"s", times), V = V, V_mr = V_mr,
         f_Hz = ustrip(u"Hz", kin.frequency),
         b2_m = ustrip(u"m", wd.wing.wing_length))
    end

    ts_obs = lift(g -> g.ts, grid)
    V_obs  = lift(g -> g.V,  grid)

    info = lift(grid) do g
        "V_mr = $(round(g.V_mr, digits=2)) m/s   |   " *
        "f = $(round(g.f_Hz, digits=2)) Hz   |   " *
        "b/2 = $(round(g.b2_m, digits=3)) m"
    end
    Label(fig[3, 1:2], info; tellwidth = false)

    ax = Axis3(fig[1, 1:2];
               xlabel = "element id",
               ylabel = "t [s]",
               zlabel = "v_realised [m/s]",
               azimuth = 1.2π, elevation = π/8)
    GLMakie.wireframe!(ax, elem_ids, ts_obs, V_obs; color = :steelblue)

    # Fixed z-range so the wireframe shape is visually comparable across masses.
    # y-axis (period) still fits to data since period shrinks with mass.
    GLMakie.zlims!(ax, 5, 30)
    on(grid) do g
        GLMakie.ylims!(ax, 0, maximum(g.ts))
    end
    notify(grid)

    display(fig)
end


# ─────────────────────────────────────────────────────────────────────
# 11. Mass sweep — wing cooling power scaling across bird sizes
#     Demonstrates how relative wing cooling power (W/m²) changes with body size.
#     Reports mass, maximum range speed, kinematics, total cooling power (W),
#     and area-normalized cooling power (W/m²) in a summary table.
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 11. Wing cooling power — mass sweep ======================")

    # Define a range of bird masses (grams)
    mass_g_list = [4, 10, 30, 50, 100, 300, 500, 1000, 1500, 5000, 8000, 15000]
    
    # Fixed environmental conditions for fair comparison
    T_air   = 20.0u"°C"
    T_wing  = 22.0u"°C"
    altitude = 30.0u"m"
    
    # Table header
    @printf "%-10s %-10s %-8s %-8s %-8s %-12s %-12s\n" "mass[g]" "V_mr[m/s]" "f[Hz]" "amp[°]" "φ[°]" "Q_conv[W]" "q[W/m²]"
    println("─"^80)
    
    for mass_g in mass_g_list
        m_kg = mass_g / 1000
        
        # Compute wing heat balance for this mass
        b = Q_for_mass(m_kg;
                       T_air        = T_air,
                       T_wing       = T_wing,
                       altitude     = altitude,
                       n_elements   = 10,
                       n_steps      = 40,
                       freq_method      = Pennycuick2008MinPower(),

                       convection_model = MixedPlate())
        
        # Extract summary data (s.Q_mean_W = Q_conv_mean; s.q_per_m2 = Q_conv_mean / A_wing)
        s = summarize(b)
        
        @printf "%-10.1f %-10.2f %-8.2f %-8.1f %-8.1f %-12.5f %-12.2f\n" mass_g s.V_mr s.f_Hz s.amp_deg s.sp_deg s.Q_mean_W s.q_per_m2
    end
    
    println()
    T_air_C = ustrip(u"°C", T_air)
    T_wing_C = ustrip(u"°C", T_wing)
    ΔT_K = ustrip(u"K", T_wing - T_air)
    alt_m = ustrip(u"m", altitude)
    
    @printf "Conditions: T_air = %.1f °C, T_wing = %.1f °C (ΔT = %.1f K),\n" T_air_C T_wing_C ΔT_K
    @printf "             altitude = %.1f m, mixed plate convection\n" alt_m
end


# ─────────────────────────────────────────────────────────────────────
# 12. Body thermoregulation — single 100 g bird in one microclimate
# ─────────────────────────────────────────────────────────────────────
# Uses the new BiophysicalBehaviour pipeline (`run_body_thermoregulation`):
# builds an `Organism` (HeatExchangeTraits + BehavioralTraits) and runs
# the rule-based sequential controller (piloerect → uncurl → vasodilate
# → hyperthermia → pant → sweat) to find heat balance.
# ─────────────────────────────────────────────────────────────────────
let
    using .WholeAnimalHeatBalance: run_whole_animal
    println("\n=== 12. Body thermoregulation — single 100 g bird ============")

    micro = microclimate_at_altitude(
        altitude           = 30.0u"m",
        air_temperature    = 15.0u"°C",
        ground_temperature = 25.0u"°C",
        sky_temperature    = 5.0u"°C",
        zenith_angle       = 10.0u"°",
        global_radiation   = 500.0u"W/m^2",
        diffuse_fraction   = 0.20,
        relative_humidity  = 0.50,
    )

    # Use V_air = 0.1 m/s as a "perched" reference (no flight)
    r  = run_body_thermoregulation_for_mass(0.100, micro; V_air = 0.1u"m/s")
    bb = r.bird
    tr = r.endotherm_out.thermoregulation
    ef = r.endotherm_out.energy_fluxes
    mf = r.endotherm_out.mass_fluxes
    mo = r.endotherm_out.morphology

    @printf "  mass            = %6.1f g\n"      1000 * bb.m_kg
    @printf "  axis ratio a/b  = %6.3f\n"        bb.axis_ratio
    @printf "  feather depth   = %6.2f mm\n"     1000 * ustrip(u"m", bb.feather_depth)
    @printf "  area_total      = %6.4f m²\n"     ustrip(u"m^2", mo.area_total)
    println("  ─── temperatures [°C] ───────────────────────")
    @printf "   T_core         = %6.2f\n" ustrip(u"°C", tr.T_core)
    @printf "   T_skin         = %6.2f\n" ustrip(u"°C", tr.T_skin)
    @printf "   T_insulation   = %6.2f\n" ustrip(u"°C", tr.T_insulation)
    @printf "   T_lung         = %6.2f\n" ustrip(u"°C", tr.T_lung)
    println("  ─── energy fluxes [W] ──────────────────────")
    @printf "   Q_gen          = %+8.4f\n" ustrip(u"W", ef.Q_gen)
    @printf "   Q_solar        = %+8.4f\n" ustrip(u"W", ef.Q_solar)
    @printf "   Q_lw_in        = %+8.4f\n" ustrip(u"W", ef.Q_longwave_in)
    @printf "   Q_lw_out       = %+8.4f\n" ustrip(u"W", ef.Q_longwave_out)
    @printf "   Q_conv         = %+8.4f\n" ustrip(u"W", ef.Q_convection)
    @printf "   Q_evap         = %+8.4f\n" ustrip(u"W", ef.Q_evaporation)
    @printf "   balance        = %+8.4f\n" ustrip(u"W", ef.balance)
    println("  ─── thermoreg / behaviour state ────────────")
    @printf "   pant level     = %6.3f (1 = resting)\n" tr.pant
    @printf "   skin_wetness   = %6.4f\n" tr.skin_wetness
    @printf "   k_flesh        = %6.3f W/m/K\n" ustrip(u"W/m/K", tr.k_flesh)
    println("  ─── mass fluxes ────────────────────────────")
    @printf "   ventilation    = %8.4f L/h\n" 3600 * 1000 * ustrip(u"m^3/s", mf.V_air)
    @printf "   evap water     = %8.4f mg/s\n" 1e6 * ustrip(u"kg/s", mf.m_evap)
end


# ─────────────────────────────────────────────────────────────────────
# 13. Budgerigar-style 4-panel reproduction across air temperature
# ─────────────────────────────────────────────────────────────────────
# Sweeps T_air over the bird's tolerable range and plots:
#   (a) metabolic rate (Q_gen)
#   (b) evaporative water loss (total, respiratory, and cutaneous)
#   (c) body temperatures (T_core / T_skin / T_insulation)
#   (d) panting multiplier and minute ventilation
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 13. Budgie-style 4-panel: 34 g bird vs T_air =============")

    # ── Humidity profile matching Weathers (1976) lab conditions ─────────
    # The experiment used a fixed absolute humidity equivalent to 30 % RH
    # at 40 °C.  Below 30 °C a floor of 15 % is applied.  Implemented via
    # the Magnus saturation-vapour-pressure formula so we can compute the
    # corresponding RH at every air temperature without extra dependencies.
    _e_sat(T_C) = 611.2 * exp(17.67 * T_C / (T_C + 243.5))   # [Pa]
    _e_ref      = 0.30 * _e_sat(40.0)                          # fixed vapour pressure
    _rh_profile(T_C) = T_C < 30.0 ? 0.15 : min(_e_ref / _e_sat(T_C), 1.0)

    # T_core_max threshold for Q10 activation (budgerigar reference: 43 °C)
    # (handled automatically by run_body_thermoregulation — no manual q10 needed)

    # Metabolic-chamber conditions matching Weathers (1976): no solar radiation,
    # chamber walls and floor at air temperature (isothermal radiative environment).
    base_micro(T_air_C; rh = 0.30) = microclimate_at_altitude(
        altitude           = 30.0u"m",
        air_temperature    = T_air_C * u"°C",
        ground_temperature = T_air_C * u"°C",   # chamber floor at T_air
        sky_temperature    = T_air_C * u"°C",   # chamber walls at T_air
        zenith_angle       = 20.0u"°",
        global_radiation   = 0.0u"W/m^2",       # no solar in lab
        diffuse_fraction   = 0,
        relative_humidity  = rh,
        shade              = 0,
    )

    T_air_C = collect(range(0.0, 50.0; length = 23))
    bb      = build_body_for_mass(0.034)    # canonical budgie 0.034 kg

    n        = length(T_air_C)
    Q_gen    = Vector{Float64}(undef, n)
    m_evap   = Vector{Float64}(undef, n)
    m_resp = Vector{Float64}(undef, n)
    m_sweat   = Vector{Float64}(undef, n)
    T_core   = Vector{Float64}(undef, n)
    T_skin   = Vector{Float64}(undef, n)
    T_insul  = Vector{Float64}(undef, n)
    vent     = Vector{Float64}(undef, n)

    for (i, Ta) in enumerate(T_air_C)
        # Apply temperature-dependent humidity to match reference lab conditions
        rh  = _rh_profile(Ta)
        r  = run_body_thermoregulation(bb, base_micro(Ta; rh = rh); V_air = 0.1u"m/s")
        ef = r.endotherm_out.energy_fluxes
        mf = r.endotherm_out.mass_fluxes
        tr = r.endotherm_out.thermoregulation
        Q_gen[i]   = ustrip(u"W",   ef.Q_gen)
        m_evap[i]  = 3.6e6 * ustrip(u"kg/s", mf.m_evap)             # g/hr
        m_resp[i] = 3.6e6 * ustrip(u"kg/s", mf.m_resp)  # g/hr
        m_sweat[i]   = 3.6e6 * ustrip(u"kg/s", mf.m_sweat)    # g/hr
        T_core[i]  = ustrip(u"°C",  tr.T_core)
        T_skin[i]  = ustrip(u"°C",  tr.T_skin)
        T_insul[i] = ustrip(u"°C",  tr.T_insulation)
        vent[i]    = 60 * 1000 * 1000 * ustrip(u"m^3/s", mf.V_air)        # ml/min     
    end

    if _PLOTS_AVAILABLE
        p1 = Plots.plot(T_air_C, Q_gen;  xlabel = "T_air [°C]", ylabel = "Metabolic rate [W]",
                        lw = 2, label = "Q_gen", title = "Metabolic rate")
        p2 = Plots.plot(T_air_C, m_evap; xlabel = "T_air [°C]", ylabel = "Water loss [g/hr]",
                        lw = 2, label = "total",
                        title = "Evaporative water loss")
        Plots.plot!(p2, T_air_C, m_resp; lw = 2, label = "respiratory")
        Plots.plot!(p2, T_air_C, m_sweat;  lw = 2, label = "cutaneous")
        p3 = Plots.plot(T_air_C, [T_core T_skin T_insul];
                        xlabel = "T_air [°C]", ylabel = "T [°C]",
                        lw = 2, label = ["T_core" "T_skin" "T_insulation"],
                        title = "Body temperatures")
        p4 = Plots.plot(T_air_C, vent; xlabel = "T_air [°C]", ylabel = "Ventilation [ml/min]",
                        lw = 2, label = "ventilation", title = "Ventilation rate")
        display(Plots.plot(p1, p2, p3, p4; layout = (2, 2), size = (1000, 800)))
    else
        println("  T_air  Q_gen  m_evap  m_resp  m_sweat  T_core  T_skin  T_insul vent")
        for i in 1:2:n
            @printf "  %5.1f  %5.3f  %6.3f  %6.3f  %6.3f  %6.2f  %6.2f  %6.2f  %6.2f\n" (
                T_air_C[i], Q_gen[i], m_evap[i], m_resp[i], m_sweat[i], T_core[i], T_skin[i], T_insul[i], vent[i])
        end
    end
end

# ─────────────────────────────────────────────────────────────────────
# 14. Testing mass-made bodies against Conradie data
# ─────────────────────────────────────────────────────────────────────
# Sweeps T_air over the bird's tolerable range and plots:
#   (a) metabolic rate (Q_gen)
#   (b) evaporative water loss (total, respiratory, and cutaneous)
#   (c) body temperatures (T_core / T_skin / T_insulation)
#   (d) panting multiplier and minute ventilation
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 13. 4-panel: bird vs T_air =============")


    # Standard metabolic-chamber conditions, with
    # chamber walls and floor at air temperature (isothermal radiative environment).
    base_micro(T_air_C; rh = 0.05) = microclimate_at_altitude(
        altitude           = 1.0u"m",
        air_temperature    = T_air_C * u"°C",
        ground_temperature = T_air_C * u"°C",   # chamber floor at T_air
        sky_temperature    = T_air_C * u"°C",   # chamber walls at T_air
        zenith_angle       = 20.0u"°",
        global_radiation   = 0.0u"W/m^2",       # no solar in lab
        diffuse_fraction   = 0,
        relative_humidity  = 0.05,       # 5% RH to match Conradie et al. (2023) 
        shade              = 0,
    )

    T_air_C = collect(range(0.0, 50.0; length = 23))
    bb      = build_body_for_mass(0.034)    # change the mass 

    n        = length(T_air_C)
    Q_gen    = Vector{Float64}(undef, n)
    m_evap   = Vector{Float64}(undef, n)
    m_resp = Vector{Float64}(undef, n)
    m_sweat   = Vector{Float64}(undef, n)
    T_core   = Vector{Float64}(undef, n)
    T_skin   = Vector{Float64}(undef, n)
    T_insul  = Vector{Float64}(undef, n)
    vent     = Vector{Float64}(undef, n)

    for (i, Ta) in enumerate(T_air_C)
        # Apply temperature-dependent humidity to match reference lab conditions
        rh  = 0.05
        r  = run_body_thermoregulation(bb, base_micro(Ta; rh = rh); V_air = 0.01u"m/s")
        ef = r.endotherm_out.energy_fluxes
        mf = r.endotherm_out.mass_fluxes
        tr = r.endotherm_out.thermoregulation
        Q_gen[i]   = ustrip(u"W",   ef.Q_gen)
        m_evap[i]  = 3.6e6 * ustrip(u"kg/s", mf.m_evap)             # g/hr
        m_resp[i] = 3.6e6 * ustrip(u"kg/s", mf.m_resp)  # g/hr
        m_sweat[i]   = 3.6e6 * ustrip(u"kg/s", mf.m_sweat)    # g/hr
        T_core[i]  = ustrip(u"°C",  tr.T_core)
        T_skin[i]  = ustrip(u"°C",  tr.T_skin)
        T_insul[i] = ustrip(u"°C",  tr.T_insulation)
        vent[i]    = 60 * 1000 * 1000 * ustrip(u"m^3/s", mf.V_air)        # ml/min     
    end

    if _PLOTS_AVAILABLE
        p1 = Plots.plot(T_air_C, Q_gen;  xlabel = "T_air [°C]", ylabel = "Metabolic rate [W]",
                        lw = 2, label = "Q_gen", title = "Metabolic rate")
        p2 = Plots.plot(T_air_C, m_evap; xlabel = "T_air [°C]", ylabel = "Water loss [g/hr]",
                        lw = 2, label = "total",
                        title = "Evaporative water loss")
        Plots.plot!(p2, T_air_C, m_resp; lw = 2, label = "respiratory")
        Plots.plot!(p2, T_air_C, m_sweat;  lw = 2, label = "cutaneous")
        p3 = Plots.plot(T_air_C, [T_core T_skin T_insul];
                        xlabel = "T_air [°C]", ylabel = "T [°C]",
                        lw = 2, label = ["T_core" "T_skin" "T_insulation"],
                        title = "Body temperatures")
        p4 = Plots.plot(T_air_C, vent; xlabel = "T_air [°C]", ylabel = "Ventilation [ml/min]",
                        lw = 2, label = "ventilation", title = "Ventilation rate")
        display(Plots.plot(p1, p2, p3, p4; layout = (2, 2), size = (1000, 800)))
    else
        println("  T_air  Q_gen  m_evap  m_resp  m_sweat  T_core  T_skin  T_insul vent")
        for i in 1:2:n
            @printf "  %5.1f  %5.3f  %6.3f  %6.3f  %6.3f  %6.2f  %6.2f  %6.2f  %6.2f\n" (
                T_air_C[i], Q_gen[i], m_evap[i], m_resp[i], m_sweat[i], T_core[i], T_skin[i], T_insul[i], vent[i])
        end
    end
end

# ─────────────────────────────────────────────────────────────────────
# 14b. Budgie (34 g) — explicit parameter demo, Conradie chamber conditions
# ─────────────────────────────────────────────────────────────────────
# Same temperature sweep (0–50 °C) as demo 14 but every default is
# written out explicitly so individual parameters are easy to tweak.
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 14b. Budgie explicit-parameter demo (Conradie conditions) =")

    # ── Body mass ─────────────────────────────────────────────────
    m_kg = 0.034    # kg  (budgerigar, Conradie et al. 2023)

    # ── Morphometrics ─────────────────────────────────────────────
    # axis_ratio_allometry(0.034) ≈ 1.1  (near-spherical small passerine)
    axis_ratio      = 1.1
    # feather_depth_allometry(0.034) ≈ 0.00566 m  (d_f = 0.01737 · m^0.33)
    feather_depth_m = 0.01737 * m_kg^0.33
    ρ_flesh         = 1000.0u"kg/m^3"   # body density
    fibre_diameter  = 30.0e-6u"m"       # feather fibre diameter
    fibre_density   = 5.0e7u"1/m^2"     # feather fibre density  (5×10⁷ /m²)
    fat_fraction    = 0.05               # fractional fat layer
    fat_density     = 901.0u"kg/m^3"

    bb = build_body_for_mass(m_kg;
        axis_ratio      = axis_ratio,
        feather_depth_m = feather_depth_m,
        ρ_flesh         = ρ_flesh,
        fibre_diameter  = fibre_diameter,
        fibre_density   = fibre_density,
        fat_fraction    = fat_fraction,
        fat_density     = fat_density,
    )

    # ── Chamber microclimate (Conradie et al. 2023 conditions) ────
    # Isothermal radiative environment (walls = floor = air), no solar,
    # 5 % RH, effectively still air (0.01 m/s).
    base_micro(T_air_C) = microclimate_at_altitude(
        altitude           = 1.0u"m",
        air_temperature    = T_air_C * u"°C",
        ground_temperature = T_air_C * u"°C",   # chamber floor at T_air
        sky_temperature    = T_air_C * u"°C",   # chamber walls at T_air
        zenith_angle       = 20.0u"°",
        global_radiation   = 0.0u"W/m^2",       # no solar
        diffuse_fraction   = 0,
        relative_humidity  = 0.05,               # 5 % RH
        shade              = 0,
    )
    V_air = 0.01u"m/s"   # still-air chamber

    # ── Heat budget parameters ────────────────────────────────────
    T_core          = 311.15u"K"     # 38 °C normothermic setpoint
    Q_basal         = nothing        # nothing → McKechnieWolf() allometry
    # q10 is intentionally NOT passed in organism_kwargs — the auto-Q10
    # logic in run_body_thermoregulation switches to q10_hot=2.0 when
    # T_air ≥ T_CORE_MAX, matching the behaviour of demo 14.
    Δ_breath        = 5.0u"K"        # exhaled air above inhaled
    fO2_extract     = 0.25           # O₂ extraction efficiency
    rq              = 0.80           # respiratory quotient
    rh_exit         = 1.0            # exhaled air saturated
    k_flesh         = 0.9u"W/m/K"   # flesh conductivity (pre-vasodilation)
    refl_dorsal     = 0.248          # dorsal feather reflectance
    refl_ventral    = 0.351          # ventral feather reflectance
    emiss_body      = 0.99           # body emissivity

    # ── Water budget / panting parameters ─────────────────────────
    skin_wetness       = 0.005       # baseline cutaneous evaporation fraction
    skin_wetness_max   = 0.05        # maximum (Conradie ref: 0.05)
    skin_wetness_step  = 0.0025      # increment per thermoregulation step
    pant_current       = 1.0         # starting pant multiplier
    pant_max           = 15.0        # maximum pant multiplier
    pant_step          = 0.01        # increment per thermoregulation step
    pant_multiplier    = 1.0         # cost multiplier on panting metabolic load

    # ── Thermoregulation limits ────────────────────────────────────
    T_core_max   = T_core + 5.0u"K"  # 43 °C hyperthermia ceiling
    T_core_step  = 0.1u"K"
    k_flesh_max  = 2.8u"W/m/K"       # full vasodilation
    k_flesh_step = 0.1u"W/m/K"

    # ── Temperature sweep ─────────────────────────────────────────
    T_air_C = collect(range(0.0, 50.0; length = 23))
    n       = length(T_air_C)

    Q_gen   = Vector{Float64}(undef, n)
    m_evap  = Vector{Float64}(undef, n)
    m_resp  = Vector{Float64}(undef, n)
    m_sweat = Vector{Float64}(undef, n)
    T_core_out  = Vector{Float64}(undef, n)
    T_skin_out  = Vector{Float64}(undef, n)
    T_insul_out = Vector{Float64}(undef, n)
    vent    = Vector{Float64}(undef, n)

    for (i, Ta) in enumerate(T_air_C)
        r = run_body_thermoregulation(bb, base_micro(Ta);
            V_air = V_air,
            organism_kwargs = (
                T_core         = T_core,
                Q_basal        = Q_basal,
                Δ_breath       = Δ_breath,
                fO2_extract    = fO2_extract,
                rq             = rq,
                rh_exit        = rh_exit,
                k_flesh        = k_flesh,
                refl_dorsal    = refl_dorsal,
                refl_ventral   = refl_ventral,
                emiss_body     = emiss_body,
                skin_wetness   = skin_wetness,
                pant_current   = pant_current,
                thermoregulation_kwargs = (
                    T_core_max        = T_core_max,
                    T_core_step       = T_core_step,
                    k_flesh_max       = k_flesh_max,
                    k_flesh_step      = k_flesh_step,
                    pant_max          = pant_max,
                    pant_step         = pant_step,
                    pant_multiplier   = pant_multiplier,
                    skin_wetness_max  = skin_wetness_max,
                    skin_wetness_step = skin_wetness_step,
                ),
            ),
        )
        ef = r.endotherm_out.energy_fluxes
        mf = r.endotherm_out.mass_fluxes
        tr = r.endotherm_out.thermoregulation
        Q_gen[i]      = ustrip(u"W",   ef.Q_gen)
        m_evap[i]     = 3.6e6 * ustrip(u"kg/s", mf.m_evap)
        m_resp[i]     = 3.6e6 * ustrip(u"kg/s", mf.m_resp)
        m_sweat[i]    = 3.6e6 * ustrip(u"kg/s", mf.m_sweat)
        T_core_out[i] = ustrip(u"°C",  tr.T_core)
        T_skin_out[i] = ustrip(u"°C",  tr.T_skin)
        T_insul_out[i]= ustrip(u"°C",  tr.T_insulation)
        vent[i]       = 60 * 1000 * 1000 * ustrip(u"m^3/s", mf.V_air)
    end

    if _PLOTS_AVAILABLE
        p1 = Plots.plot(T_air_C, Q_gen;  xlabel = "T_air [°C]", ylabel = "Metabolic rate [W]",
                        lw = 2, label = "Q_gen", title = "Metabolic rate")
        p2 = Plots.plot(T_air_C, m_evap; xlabel = "T_air [°C]", ylabel = "Water loss [g/hr]",
                        lw = 2, label = "total", title = "Evaporative water loss")
        Plots.plot!(p2, T_air_C, m_resp;  lw = 2, label = "respiratory")
        Plots.plot!(p2, T_air_C, m_sweat; lw = 2, label = "cutaneous")
        p3 = Plots.plot(T_air_C, [T_core_out T_skin_out T_insul_out];
                        xlabel = "T_air [°C]", ylabel = "T [°C]",
                        lw = 2, label = ["T_core" "T_skin" "T_insulation"],
                        title = "Body temperatures")
        p4 = Plots.plot(T_air_C, vent; xlabel = "T_air [°C]", ylabel = "Ventilation [ml/min]",
                        lw = 2, label = "ventilation", title = "Ventilation rate")
        display(Plots.plot(p1, p2, p3, p4; layout = (2, 2), size = (1000, 800)))
    else
        println("  T_air  Q_gen  m_evap  m_resp  m_sweat  T_core  T_skin  T_insul  vent")
        for i in 1:2:n
            @printf "  %5.1f  %5.3f  %6.3f  %6.3f  %6.3f  %6.2f  %6.2f  %6.2f  %6.2f\n" (
                T_air_C[i], Q_gen[i], m_evap[i], m_resp[i], m_sweat[i],
                T_core_out[i], T_skin_out[i], T_insul_out[i], vent[i])
        end
    end
end

# ─────────────────────────────────────────────────────────────────────
# 15. Body thermoregulation — same 100 g bird across three microclimates
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 15. Body thermoregulation — 100 g bird cold/temp/hot =====")

    envs = (
        cold      = microclimate_at_altitude(altitude = 30.0u"m",
                        air_temperature = -10.0u"°C",
                        ground_temperature = -5.0u"°C",
                        sky_temperature = -25.0u"°C",
                        zenith_angle = 60.0u"°",
                        global_radiation = 200.0u"W/m^2",
                        diffuse_fraction = 0.40,
                        relative_humidity = 0.70),
        temperate = microclimate_at_altitude(altitude = 30.0u"m",
                        air_temperature = 15.0u"°C",
                        ground_temperature = 25.0u"°C",
                        sky_temperature = 5.0u"°C",
                        zenith_angle = 10.0u"°",
                        global_radiation = 500.0u"W/m^2",
                        diffuse_fraction = 0.20,
                        relative_humidity = 0.50),
        hot       = microclimate_at_altitude(altitude = 30.0u"m",
                        air_temperature = 38.0u"°C",
                        ground_temperature = 55.0u"°C",
                        sky_temperature = 25.0u"°C",
                        zenith_angle = 5.0u"°",
                        global_radiation = 950.0u"W/m^2",
                        diffuse_fraction = 0.10,
                        relative_humidity = 0.30,
                        shade = 0.5),
    )

    bb = build_body_for_mass(0.100)
    println("  env        Ta[°C]  Q_gen[W]  T_core[°C]  T_skin[°C]  pant   wet     evap[mg/s]")
    for (name, micro) in pairs(envs)
        r  = run_body_thermoregulation(bb, micro; V_air = 0.1u"m/s")
        tr = r.endotherm_out.thermoregulation
        ef = r.endotherm_out.energy_fluxes
        mf = r.endotherm_out.mass_fluxes
        @printf "  %-9s %6.1f  %8.4f  %9.2f  %9.2f  %5.2f  %5.4f  %9.3f\n" String(name) ustrip(u"°C", micro.air_temperature) ustrip(u"W", ef.Q_gen) ustrip(u"°C", tr.T_core) ustrip(u"°C", tr.T_skin) tr.pant tr.skin_wetness (1e6 * ustrip(u"kg/s", mf.m_evap))
    end
end


# ─────────────────────────────────────────────────────────────────────
# 16. Maximum heat dissipation capacity vs body mass
# ─────────────────────────────────────────────────────────────────────
# At each body mass:
#   • bird flies at V_mr (AFPT maximum-range speed) — sets body wind speed
#   • Ward et al. 1999 wind-tunnel microclimate (isothermal walls, no solar)
#   • physiology FORCED to maximum dissipation state:
#       pant         = 15.0  (≈15× resting ventilation)
#       skin_wetness = 0.05  (5 % of skin wetted)
#     via pant_current/skin_wetness + zero step in thermoregulation_kwargs,
#     so the BB controller cannot back them off.
#   • wings held at T_air + 2 K (passive radiator/convector)
#
# Body and wings are solved INDEPENDENTLY then summed:
#   Body: run_body_thermoregulation with V_mr as forced-convection wind
#         speed; physiology forced to maximum dissipation state
#         (pant = 15, skin_wetness = 0.05, full vasodilation).
#   Wings: compute_wingbeat_heatbalance with uniform T_wing = T_air + 2 K,
#          afpt-optimal kinematics at V_mr, MixedPlate convection.
#
# Five outgoing pathways summed (positive = heat dumped to environment).
# Both longwave terms are NET (out − in) so they are directly comparable
# to the (already-net) convection and evaporation terms:
#   1. body convection           (ef.Q_convection)
#   2. body longwave NET         (ef.Q_longwave_out − ef.Q_longwave_in)
#   3. body evaporation          (ef.Q_evaporation; respiratory + cutaneous)
#   4. wing convection           (2 × wf.Q_conv_mean)
#   5. wing longwave NET         (−2 × wf.Q_lw_net_mean)
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 16. Max heat dissipation capacity vs body mass ===========")

    # Wind-tunnel microclimate — Ward et al. (1999) J. Exp. Biol. 202:1589-1602
    # T_air = 22.8 °C, isothermal enclosure, no solar, RH 50 %.
    tunnel_micro = microclimate_at_altitude(
        altitude           = 0.0u"m",
        air_temperature    = 22.8u"°C",
        ground_temperature = 22.8u"°C",
        sky_temperature    = 22.8u"°C",
        zenith_angle       = 60.0u"°",
        global_radiation   = 0.0u"W/m^2",
        diffuse_fraction   = 0,
        relative_humidity  = 0.50,
        shade              = 1,
    )

    T_air_K    = uconvert(u"K", tunnel_micro.air_temperature)
    T_wing_K   = T_air_K + 2.0u"K"           # wings 2 K above air

    # Log-spaced mass sweep, 10 g – 5 kg
    mass_kg = exp10.(range(log10(0.010), log10(5.0); length = 14))
    n       = length(mass_kg)

    Q_body_conv  = Vector{Float64}(undef, n)
    Q_body_lw    = Vector{Float64}(undef, n)
    Q_body_evap  = Vector{Float64}(undef, n)
    Q_wing_conv  = Vector{Float64}(undef, n)
    Q_wing_lw    = Vector{Float64}(undef, n)
    Q_total      = Vector{Float64}(undef, n)
    V_mr_arr     = Vector{Float64}(undef, n)
    pant_arr     = Vector{Float64}(undef, n)
    wet_arr      = Vector{Float64}(undef, n)

    # Maximum-dissipation body physiology: pant and skin_wetness at
    # their ceilings, steps = 0 so the BB controller cannot lower them,
    # full vasodilation, hyperthermic core.
    max_body_kwargs = (
        skin_wetness = 0.05,
        pant_current = 15.0,
        T_core       = 44.0u"°C",
        k_flesh      = 2.8u"W/m/K",
        thermoregulation_kwargs = (
            pant_step         = 0.0,
            skin_wetness_step = 0.0,
        ),
    )

    for (i, m) in enumerate(mass_kg)
        V_mr = afpt_v_mr(m) * u"m/s"
        V_ms = ustrip(u"m/s", V_mr)

        # ── BODY ────────────────────────────────────────────────────
        # run_body_thermoregulation uses V_mr as the wind speed for
        # body convection (overrides micro.wind_speed via build_environment).
        bb   = build_body_for_mass(m)
        br   = run_body_thermoregulation(bb, tunnel_micro;
                                         V_air           = V_mr,
                                         organism_kwargs = max_body_kwargs)
        ef   = br.endotherm_out.energy_fluxes
        tr   = br.endotherm_out.thermoregulation

        Q_body_conv[i] = ustrip(u"W", ef.Q_convection)
        Q_body_lw[i]   = ustrip(u"W", ef.Q_longwave_out - ef.Q_longwave_in)
        Q_body_evap[i] = ustrip(u"W", ef.Q_evaporation)
        pant_arr[i]    = tr.pant
        wet_arr[i]     = tr.skin_wetness

        # ── WINGS ───────────────────────────────────────────────────
        # Afpt-optimal kinematics at V_mr; uniform T_wing = T_air + 2 K;
        # MixedPlate convection (our flat-plate correlation).
        wd = build_wing_for_mass(m; n_elements = 10)
        wt = uniform_temperature(wd, T_wing_K)
        kk = build_kinematics_for_mass(m; V_forward_ms = V_ms)
        wf = compute_wingbeat_heatbalance(kk, wd, wt, tunnel_micro;
                                          n_steps          = 40,
                                          V_forward        = V_mr,
                                          convection_model = MixedPlate())

        # wf.Q_lw_net_mean = Q_lw_in − Q_lw_out (wing struct convention),
        # so negate to get "positive = net loss to environment".
        Q_wing_conv[i] = 2 * ustrip(u"W", wf.Q_conv_mean)
        Q_wing_lw[i]   = -2 * ustrip(u"W", wf.Q_lw_net_mean)

        Q_total[i]  = Q_body_conv[i] + Q_body_lw[i] + Q_body_evap[i] +
                      Q_wing_conv[i] + Q_wing_lw[i]
        V_mr_arr[i] = V_ms
    end

    # ── Confirm pant / sweat held at their maxima ────────────────────
    pant_ok = all(p -> isapprox(p, 15.0; atol = 1e-6), pant_arr)
    wet_ok  = all(w -> isapprox(w, 0.05; atol = 1e-8), wet_arr)
    println("   pant rate    held at max (15.0) for every mass : ",
            pant_ok ? "YES ✓" : "NO ✗")
    println("   skin wetness held at max (0.05) for every mass : ",
            wet_ok  ? "YES ✓" : "NO ✗")
    if !(pant_ok && wet_ok)
        @warn "Forced-max override failed for some mass" pant_arr wet_arr
    end

    if _PLOTS_AVAILABLE
        p = Plots.plot(
            mass_kg .* 1000, Q_total;
            xlabel = "body mass [g]", ylabel = "heat dissipation [W]",
            lw = 3, color = :black, label = "total",
            title  = "Max heat dissipation vs mass\n" *
                     "(Ward et al. 1999 tunnel: T_air = 22.8 °C, RH = 50 %, " *
                     "isothermal walls, no solar; flying at V_mr)",
            legend = :topleft, legendfontsize = 7,
        )
        Plots.plot!(p, mass_kg .* 1000, Q_body_conv; lw = 2,
                    color = :red,    label = "body convection")
        Plots.plot!(p, mass_kg .* 1000, Q_body_lw;   lw = 2,
                    color = :orange, label = "body longwave (net)")
        Plots.plot!(p, mass_kg .* 1000, Q_body_evap; lw = 2,
                    color = :blue,   label = "body evaporation")
        Plots.plot!(p, mass_kg .* 1000, Q_wing_conv; lw = 2,
                    color = :red,    linestyle = :dash, label = "wing convection")
        Plots.plot!(p, mass_kg .* 1000, Q_wing_lw;   lw = 2,
                    color = :orange, linestyle = :dash, label = "wing longwave (net)")
        display(p)
    else
        println("   m[g]    V_mr   Q_b_conv  Q_b_lw  Q_b_evap  " *
                "Q_w_conv  Q_w_lw   Q_total  pant   wet")
        for i in 1:n
            @printf "  %7.1f  %5.2f  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f  %5.2f  %5.4f\n" (
                mass_kg[i]*1000) V_mr_arr[i] Q_body_conv[i] Q_body_lw[i] (
                Q_body_evap[i]) Q_wing_conv[i] Q_wing_lw[i] Q_total[i] (
                pant_arr[i]) wet_arr[i]
        end
    end
end




















