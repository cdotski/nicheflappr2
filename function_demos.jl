# =====================================================================
# function_demos.jl
#
# Consolidated, runnable demonstrations for the streamlined modules:
#
#   • wing_geometry_2.0.jl   →  module WingPlates
#   • wing_kinematics_2.0.jl →  module WingKinematics
#   • wing_convection_2.0.jl →  module WingConvection
#   • wing_power.jl          →  module WingPower
#
# Each demo is wrapped in its own `let … end` block and is independent
# of the others, so individual sections can be evaluated on their own
# in the REPL.  Plotting blocks degrade gracefully if `Plots` is not
# installed.
# =====================================================================

include("wing_power.jl")

using .WingPlates, .WingKinematics, .WingConvection, .WingPower
using Unitful
using Printf

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
    # quick bird build, then either pass V_forward_ms as a kwarg or
    # bake it into the StrouhalFreq struct.
    bird  = build_bird_from_mass(m_kg)
    V_mr  = maximum_range_speed(bird)            # [m/s]

    kin  = build_kinematics_for_mass(m_kg, StrouhalFreq();
                                     amp = 60.0, stroke_plane_deg = 80.0,
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
# 5. Convection snapshot + full wingbeat cycle
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 5. Convective heat loss =================================")

    m_kg = 0.05
    bird  = build_bird_from_mass(m_kg)
    V_mr  = maximum_range_speed(bird)
    wd   = build_wing_for_mass(m_kg; n_elements = 10)
    kin  = build_kinematics_for_mass(m_kg, StrouhalFreq(); amp = StrouhalAmplitude(),
                                     stroke_plane_deg = 80.0, V_forward_ms = V_mr)
    wt   = uniform_temperature(wd, 25.0u"°C")
    air  = air_properties(20.0u"°C"; altitude = 0.0u"m")

    wbc = compute_wingbeat_convection(kin, wd, wt, air;
                                      n_steps = 40, V_forward = V_mr * u"m/s")

    @printf "Q_mean  = %s\n" wbc.Q_mean
    @printf "Q_max   = %s\n" wbc.Q_max
    @printf "Q_min   = %s\n" wbc.Q_min
    @printf "Dorsal/Ventral split: %s / %s\n" wbc.Q_dorsal_mean wbc.Q_ventral_mean

    if _PLOTS_AVAILABLE
        display(plot(ustrip.(u"s", wbc.times), ustrip.(u"W", wbc.Q_timeseries);
                     xlabel = "t [s]", ylabel = "Q [W]",
                     title  = "Whole-wing Q over wingbeat"))
    end
end


# ─────────────────────────────────────────────────────────────────────
# 6. Bird power curve + Vmp / Vmr
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 6. Flight-power curve ===================================")

    bird = build_bird_from_mass(0.05; type = :passerine)
    @printf "Built bird: m = %.3f kg, b = %.3f m, S = %.4f m², BMR = %.3f W\n" bird.mass_total bird.wing_span bird.wing_area bird.basal_metabolic_rate

    V_mp = minimum_power_speed(bird)
    V_mr = maximum_range_speed(bird)
    @printf "V_mp = %.2f m/s,  V_mr = %.2f m/s\n" V_mp V_mr

    if _PLOTS_AVAILABLE
        Vs = range(3.0, 30.0; length = 200)
        Ps = [flapping_power(bird, V).P_mech for V in Vs]
        display(plot(collect(Vs), Ps;
                     xlabel = "V [m/s]", ylabel = "P_mech [W]",
                     title  = "Mechanical flight power"))
    end
end


# ─────────────────────────────────────────────────────────────────────
# 7. Full pipeline: Q_for_mass scaling sweep
# ─────────────────────────────────────────────────────────────────────
let
    println("\n=== 7. Q_for_mass sweep =====================================")

    demo_birds = [
        ("Hummingbird (4 g)",   0.004),
        ("Sparrow (30 g)",      0.030),
        ("Pigeon (300 g)",      0.300),
        ("Crow (500 g)",        0.500),
        ("Owl (1.5 kg)",        1.5),
    ]

    results = [(name = nm, b = Q_for_mass(m;
                                          T_air  = 20.0u"°C",
                                          T_wing = 25.0u"°C",
                                          altitude = 0.0u"m",
                                          n_elements = 10,
                                          n_steps    = 40,
                                          amp        = 60.0,
                                          stroke_plane_deg = 80.0,
                                          freq_method = Greenewalt1975()))
               for (nm, m) in demo_birds]

    println()
    @printf "%-22s %-8s %-8s %-8s %-9s %-9s\n" "Bird" "m[kg]" "f[Hz]" "V_mr" "Q_mean[W]" "q[W/m²]"
    println("─"^70)
    for r in results
        s = summarize(r.b)
        @printf "%-22s %-8.4f %-8.2f %-8.2f %-9.4f %-9.2f\n" r.name s.m_kg s.f_Hz s.V_mr s.Q_mean_W s.q_per_m2
    end

    if _PLOTS_AVAILABLE
        masses = [s.m_kg     for s in summarize.(getfield.(results, :b))]
        Qmean  = [s.Q_mean_W for s in summarize.(getfield.(results, :b))]
        display(bar(string.(getfield.(results, :name)), Qmean;
                    xlabel = "Bird", ylabel = "Q_mean [W]",
                    title  = "Convective heat loss across body sizes",
                    xrotation = 35, legend = false))
    end
end

println("\nAll demos complete.")
