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
# 2. Kinematics demo
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 2. Kinematics + element velocities ======================")

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

    # AFPT-driven kinematics:
    #   • frequency      → Pennycuick2008MinPower delegates to
    #                      AFPT.estimate_frequency, the same `f` used by
    #                      afpt's power model (bird.wingbeatFrequency).
    #   • stroke plane   → optimum from compute_flapping_power.
    #   • amplitude      → AfptOptAmplitude wraps the amplitude returned
    #                      alongside that optimum (degrees).
    kin = build_kinematics_for_mass(m_kg, Pennycuick2008MinPower();
                                    amp              = AfptOptAmplitude(res.amplitude),
                                    stroke_plane_deg = res.strokeplane)

    @printf "V_mr = %.2f m/s,  f = %.2f Hz, amplitude = %.1f°, φ_opt = %.1f°\n" V_mr ustrip(u"Hz", kin.frequency) rad2deg(kin.amplitude) res.strokeplane
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
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 5. Wing heat balance (single microclimate) ==============")

    m_kg = 0.05
    bird  = quick_afpt_bird(m_kg)
    V_mr  = find_maximum_range_speed(bird)
    wd   = build_wing_for_mass(m_kg; n_elements = 10)
    kin  = build_kinematics_for_mass(m_kg, StrouhalFreq(); amp = StrouhalAmplitude(),
                                     stroke_plane_deg = 80.0, V_forward_ms = V_mr)
    wt   = uniform_temperature(wd, 23.0u"°C")
    micro = Microclimate(air_temperature  = 20.0u"°C",
                         wind_speed       = V_mr * u"m/s",
                         global_radiation = 600.0u"W/m^2",
                         zenith_angle     = 30.0u"°")

    wbh = compute_wingbeat_heatbalance(kin, wd, wt, micro;
                                       n_steps = 40, V_forward = V_mr * u"m/s")

    @printf "Q_conv_mean   = %s\n" wbh.Q_conv_mean
    @printf "Q_solar_mean  = %s\n" wbh.Q_solar_mean
    @printf "Q_lw_in_mean  = %s\n" wbh.Q_lw_in_mean
    @printf "Q_lw_out_mean = %s\n" wbh.Q_lw_out_mean
    @printf "Q_lw_net_mean = %s\n" wbh.Q_lw_net_mean
    @printf "Q_net_mean    = %s   (positive ⇒ net heating)\n" wbh.Q_net_mean
    @printf "Both wings   Q_net = %s\n" WingHeatBalance.both_wings(wbh.Q_net_mean)

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
# 5.1 Convection-only comparison: AFPT kinematics vs Strouhal kinematics
#
# Minimal environment — T_air is fixed 3 °C below the wing.  Everything
# else (radiation, microclimate) is omitted so the only heat-loss
# pathway is forced convection.  Both kinematics parameterisations are
# fed the same wing geometry, the same forward airspeed (V_mr from
# AFPT) and the same wing temperature; the only differences are the
# frequency, amplitude and stroke-plane angle.
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 5.1 Convection-only: AFPT vs Strouhal kinematics ========")

    m_kg     = 0.05
    T_wing   = 23.0u"°C"
    ΔT       = 3.0u"K"                      # air is 3 °C cooler than the wing
    T_air    = T_wing - ΔT

    # Shared inputs ----------------------------------------------------
    wd   = build_wing_for_mass(m_kg; n_elements = 10)
    wt   = uniform_temperature(wd, T_wing)
    air  = WingHeatBalance.air_properties(T_air)

    bird = quick_afpt_bird(m_kg)
    V_mr = find_maximum_range_speed(bird)
    res  = compute_flapping_power(bird, V_mr; strokeplane = :opt)

    # AFPT kinematics: Pennycuick freq + afpt-optimised φ + paired amp
    kin_afpt = build_kinematics_for_mass(m_kg, Pennycuick2008MinPower();
                                         amp              = AfptOptAmplitude(res.amplitude),
                                         stroke_plane_deg = res.strokeplane)

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
    @printf "%-12s %8.2f %8.1f %8.1f %10.4f %10.4f %10.4f\n" "AFPT"     f_afpt A_afpt res.strokeplane Q_afpt ustrip(u"W", conv_afpt.Q_max) ustrip(u"W", conv_afpt.Q_min)
    @printf "%-12s %8.2f %8.1f %8.1f %10.4f %10.4f %10.4f\n" "Strouhal" f_str  A_str  80.0            Q_str  ustrip(u"W", conv_str.Q_max)  ustrip(u"W", conv_str.Q_min)
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
# 5c. Bird in several microclimates — radiation + convection pathways # NQR:
# the airspeed etc. plugs in weird
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 5c. Microclimate scenarios ==============================")

    scenarios = [
        ("Desert midday",       Microclimate(air_temperature = 38.0u"°C",
                                              ground_temperature = 55.0u"°C",
                                              sky_temperature = 15.0u"°C",
                                              wind_speed = 3.0u"m/s",
                                              zenith_angle = 10.0u"°",
                                              global_radiation = 1050.0u"W/m^2",
                                              diffuse_fraction = 0.10,
                                              relative_humidity = 0.15)),
        ("Temperate overcast",  Microclimate(air_temperature = 15.0u"°C",
                                              ground_temperature = 14.0u"°C",
                                              sky_temperature = 10.0u"°C",
                                              wind_speed = 4.0u"m/s",
                                              zenith_angle = 45.0u"°",
                                              global_radiation = 200.0u"W/m^2",
                                              diffuse_fraction = 0.85,
                                              shade = 0.5,
                                              relative_humidity = 0.85)),
        ("Alpine cold",         Microclimate(air_temperature = -5.0u"°C",
                                              ground_temperature = -2.0u"°C",
                                              sky_temperature = -30.0u"°C",
                                              wind_speed = 8.0u"m/s",
                                              zenith_angle = 60.0u"°",
                                              global_radiation = 700.0u"W/m^2",
                                              diffuse_fraction = 0.25,
                                              altitude = 3000.0u"m",
                                              atmospheric_pressure = 70110.0u"Pa",
                                              relative_humidity = 0.5)),
        ("Night flight",        Microclimate(air_temperature = 10.0u"°C",
                                              ground_temperature = 8.0u"°C",
                                              sky_temperature = -10.0u"°C",
                                              wind_speed = 2.0u"m/s",
                                              zenith_angle = 90.0u"°",
                                              global_radiation = 0.0u"W/m^2",
                                              diffuse_fraction = 1.0,
                                              relative_humidity = 0.7)),
    ]

    m_kg = 0.05
    @printf "%-20s %-11s %-11s %-11s %-11s %-11s %-11s\n" "Scenario" "Q_conv" "Q_solar" "Q_lw,in" "Q_lw,out" "Q_net(1)" "Q_net(2)"
    println("─"^96)
    for (name, micro) in scenarios
        # NOTE: omitting `amp` and `stroke_plane_deg` so the afpt-
        # optimised values (from compute_flapping_power at V_mr)
        # propagate into the kinematics.
        b = Q_for_mass(m_kg;
                       T_air  = micro.air_temperature,
                       T_wing = 35.0u"°C",
                       altitude = micro.altitude,
                       microclimate = micro,
                       n_elements = 10, n_steps = 40,
                       freq_method = StrouhalFreq())
        Qc = ustrip(u"W", Q_conv_mean(b))
        Qs = ustrip(u"W", Q_solar_mean(b))
        Qli = ustrip(u"W", Q_lw_in_mean(b))
        Qlo = ustrip(u"W", Q_lw_out_mean(b))
        Qn  = ustrip(u"W", Q_net_mean(b))
        @printf "%-20s %+10.4f %+10.4f %+10.4f %+10.4f %+10.4f %+10.4f\n" name Qc Qs Qli Qlo Qn (2*Qn)
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
        Vs = range(0.5, 30.0; length = 200)
        Ps = [compute_flapping_power(bird, V).power_total for V in Vs]
        display(plot(collect(Vs), Ps;
                     xlabel = "V [m/s]", ylabel = "P_mech [W]",
                     title  = "Mechanical flight power (afpt)"))
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
                           T_wing = 23.0u"°C",
                           altitude = 0.0u"m",
                           n_elements = 10,
                           n_steps    = 40,
                           freq_method = StrouhalFreq(),
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
# 8. Interactive Makie panel — sweep microclimate + bird parameters
#    (requires GLMakie; degrades gracefully otherwise)
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 8. Interactive Makie panel ==============================")

    have_makie = try
        @eval using GLMakie
        true
    catch
        @info "GLMakie not available — interactive panel skipped."
        false
    end
    have_makie || return

    GLMakie.activate!()

    fig = Figure(size = (1100, 720))
    Label(fig[0, 1:2], "Wing heat balance — interactive"; fontsize = 18,
          halign = :center)

    # Sliders
    sg = SliderGrid(fig[1, 1],
        (label = "mass [g]",            range = 5:5:1500,   startvalue = 50),
        (label = "T_air [°C]",          range = -10:1:45,   startvalue = 20),
        (label = "T_wing [°C]",         range = 5:1:45,     startvalue = 35),
        (label = "wind [m/s]",          range = 0.5:0.5:15, startvalue = 3),
        (label = "global rad [W/m²]",   range = 0:50:1100,  startvalue = 800),
        (label = "zenith [°]",          range = 0:5:90,     startvalue = 30),
        (label = "RH [-]",              range = 0:0.05:1,   startvalue = 0.5),
        tellheight = false,
    )
    s_m, s_Ta, s_Tw, s_V, s_G, s_Z, s_RH = sg.sliders

    # Live result observable
    result = lift(s_m.value, s_Ta.value, s_Tw.value, s_V.value,
                  s_G.value, s_Z.value, s_RH.value) do mg, Ta, Tw, V, G, Z, RH
        Q_for_mass(mg / 1000;
                   T_air     = Ta * u"°C",
                   T_wing    = Tw * u"°C",
                   wind_speed = V * u"m/s",
                   global_radiation = G * u"W/m^2",
                   zenith_angle    = Z * u"°",
                   relative_humidity = RH,
                   n_elements = 8, n_steps = 24,
                   freq_method = StrouhalFreq())
    end

    labels = ["Q_conv", "Q_solar", "Q_lw,in", "Q_lw,out", "Q_net (1 wing)", "Q_net (both)"]
    vals = lift(result) do b
        [ustrip(u"W", Q_conv_mean(b)),
         ustrip(u"W", Q_solar_mean(b)),
         ustrip(u"W", Q_lw_in_mean(b)),
         ustrip(u"W", Q_lw_out_mean(b)),
         ustrip(u"W", Q_net_mean(b)),
         2 * ustrip(u"W", Q_net_mean(b))]
    end

    ax = Axis(fig[1, 2]; xticks = (1:length(labels), labels),
              ylabel = "Q [W]", title = "Pathway breakdown",
              xticklabelrotation = π/6)
    barplot!(ax, 1:length(labels), vals; color = 1:length(labels),
             colormap = :tab10)
    hlines!(ax, [0]; color = :black, linestyle = :dash)

    info = lift(result) do b
        s = summarize(b)
        string(
            "f = ", round(s.f_Hz, digits = 2), " Hz  |  ",
            "V_mr = ", round(s.V_mr, digits = 2), " m/s  |  ",
            "P_mech = ", round(s.P_mech_W, digits = 3), " W")
    end
    Label(fig[2, 1:2], info; tellwidth = false)

    display(fig)
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
        kin   = build_kinematics_for_mass(m_kg, StrouhalFreq();
                                          amp = StrouhalAmplitude(),
                                          stroke_plane_deg = 80.0,
                                          V_forward_ms = V_mr)
    else
        wing = WingGeometry(wing_length = 0.20u"m",
                            root_chord  = 0.06u"m",
                            tip_chord   = 0.03u"m",
                            thickness   = 0.002u"m")
        wd   = discretize_wing(wing, Discretization(n_elements = n_elements))
        V_mr = 8.0
        kin  = build_kinematics_for_mass(0.05, StrouhalFreq();
                                         amp = StrouhalAmplitude(),
                                         stroke_plane_deg = 80.0,
                                         V_forward_ms = V_mr)
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
        kin  = build_kinematics_for_mass(m_kg, StrouhalFreq();
                                         amp = StrouhalAmplitude(),
                                         stroke_plane_deg = 80.0,
                                         V_forward_ms = V_mr)
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

println("\nAll demos complete.")
