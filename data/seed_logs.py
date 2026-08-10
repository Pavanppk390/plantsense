"""
PlantSense — Seed Maintenance Logs
14 hand-written logs used to seed LLM-based generation of the full
synthetic maintenance log knowledge base.

subsystem tags: bearing_lubrication, hpc, lpt_hpt, fan, seals_other, edge_case
"""

seed_logs = [
    {
        "log_id": "ML-0001",
        "unit": "Engine Unit 47 (Turbofan)",
        "date": "2024-03-14",
        "reported_issue": "Elevated vibration near LP shaft bearing (station 2.5). Vibration reading 7.2 mm/s, up from baseline 3.1 mm/s over past 48 hrs of operation.",
        "root_cause": "Lubrication degradation — oil sample showed particulate contamination above threshold, consistent with bearing wear onset.",
        "action_taken": "Engine pulled from service, bearing inspected, lubricant flushed and replaced, oil re-sampled post-service.",
        "resolution_time": "6 hrs downtime, engine returned to service same day.",
        "subsystem": "bearing_lubrication",
    },
    {
        "log_id": "ML-0002",
        "unit": "Engine Unit 12 (Turbofan)",
        "date": "2026-05-18",
        "reported_issue": "High-pressure compressor (HPC) stage 4 blade temperature anomaly. Exhaust gas temperature (EGT) spiked by 45°C during takeoff climb.",
        "root_cause": "Thermal barrier coating (TBC) spallation on the stage 4 rotor blades, leading to localized overheating.",
        "action_taken": "Conducted borescope inspection, verified coating loss, swapped engine unit with spare, sent affected unit to overhaul facility.",
        "resolution_time": "14 hrs downtime for engine swap, unit scheduled for shop visit.",
        "subsystem": "hpc",
    },
    {
        "log_id": "ML-0003",
        "unit": "Engine Unit 89 (Turbofan)",
        "date": "2026-06-02",
        "reported_issue": "Sudden thrust asymmetric warning accompanied by acoustic thud and transient fan speed (N1) fluctuation.",
        "root_cause": "Foreign Object Damage (FOD) — ingestion of a medium-sized bird during low-altitude approach, causing a minor dent on two titanium fan blades.",
        "action_taken": "Performed fan blade blending within maintenance manual limits, executed ultrasonic non-destructive testing (NDT) to check for micro-cracks.",
        "resolution_time": "8 hrs downtime, cleared for flight.",
        "subsystem": "edge_case",
    },
    {
        "log_id": "ML-0004",
        "unit": "Engine Unit 33 (Turbofan)",
        "date": "2026-06-25",
        "reported_issue": "Low-pressure turbine (LPT) casing showing acoustic anomalies. Active clearance control (ACC) valve system fault code triggered.",
        "root_cause": "Mechanical binding in the ACC actuator linkage due to carbon soot buildup, preventing proper valve modulation.",
        "action_taken": "Cleaned actuator linkage, replaced defective seal rings, executed automated subsystem self-test cycle.",
        "resolution_time": "3.5 hrs downtime, system verified fully functional.",
        "subsystem": "lpt_hpt",
    },
    {
        "log_id": "ML-0005",
        "unit": "Engine Unit 55 (Turbofan)",
        "date": "2026-07-14",
        "reported_issue": "Oil pressure drop below nominal operating limit (32 psi vs baseline 45 psi) during steady-state cruise monitoring.",
        "root_cause": "Main oil pump pressure relief valve spring fatigue, causing the valve to unseat prematurely.",
        "action_taken": "Replaced the pressure relief valve assembly, flushed the secondary oil loop, verified stable pressure at 46 psi during ground run.",
        "resolution_time": "5 hrs downtime, engine cleared for flight operations.",
        "subsystem": "bearing_lubrication",
    },
    {
        "log_id": "ML-0006",
        "unit": "Engine Unit 21 (Turbofan)",
        "date": "2026-08-01",
        "reported_issue": "Intermittent digital engine control (FADEC) channel B data bus communication faults during pre-flight checks.",
        "root_cause": "Moisture ingress into the main electrical wiring harness connector receptacle at the fan case interface.",
        "action_taken": "Disconnected harness, purged moisture using electronic cleaner, replaced interfacial connector seal, completed full FADEC cross-channel diagnostic loop.",
        "resolution_time": "2.5 hrs downtime, cleared for scheduled departure.",
        "subsystem": "edge_case",
    },
    {
        "log_id": "ML-0007",
        "unit": "Engine Unit 64 (Turbofan)",
        "date": "2026-08-08",
        "reported_issue": "Progressive divergence in the High-Pressure Compressor (HPC) pressure ratio vs. core speed (N2) line. Exhaust Gas Temperature (EGT) margin showing a steady -0.15°C/hour downward drift over the last 150 flight cycles.",
        "root_cause": "Continuous fouling and surface roughness accumulation on the HPC rotor blades, reducing aerodynamic efficiency and increasing fuel flow requirements.",
        "action_taken": "Scheduled the unit for an engine core detergent wash to remove aerodynamic deposits; verified restoration of baseline pressure ratios during post-wash ground run.",
        "resolution_time": "4 hrs downtime, performance metrics restored to nominal parameters.",
        "subsystem": "hpc",
    },
    {
        "log_id": "ML-0008",
        "unit": "Engine Unit 19 (Turbofan)",
        "date": "2026-08-10",
        "reported_issue": "Monotonic upward drift in the cooling air temperature-to-ambient ratio at the High-Pressure Turbine (HPT) case, accompanied by a subtle, compounding drop in low-pressure-to-high-pressure spool speed ratio (N1/N2).",
        "root_cause": "Accelerated internal seal erosion and opening of blade tip clearances within the HPT stage, causing gradual hot gas migration.",
        "action_taken": "Flagged by predictive model for early removal; pulled engine for targeted turbine module overhaul before reaching critical EGT redline.",
        "resolution_time": "16 hrs downtime for engine replacement, unit routed to heavy maintenance.",
        "subsystem": "lpt_hpt",
    },
    {
        "log_id": "ML-0009",
        "unit": "Engine Unit 02 (Turbofan)",
        "date": "2026-08-12",
        "reported_issue": "Gradual deterioration of the Low-Pressure Turbine (LPT) efficiency metrics. Over the last 45 flights, the LPT exit temperature (T50) sensor has exhibited a monotonic upward drift, while the core speed ratio (N2/N1) shifted downward by 1.8% under identical cruise flight conditions.",
        "root_cause": "High-pressure gas path degradation and trailing-edge erosion of the LPT stator vanes, causing reduced enthalpy extraction and subsequent downstream thermal buildup.",
        "action_taken": "Pulled engine based on remaining useful life (RUL) threshold alert; routed the unit to the shop for a dedicated LPT module overhaul and vane replacement.",
        "resolution_time": "14 hrs downtime for engine swap, unit transferred to heavy maintenance.",
        "subsystem": "lpt_hpt",
    },
    {
        "log_id": "ML-0010",
        "unit": "Engine Unit 15 (Turbofan)",
        "date": "2026-08-15",
        "reported_issue": "Progressive degradation of the High-Pressure Compressor (HPC) performance envelope. The total pressure at the HPC outlet (P30) showed a steady 2.4% drop over 60 flight cycles, accompanied by a corresponding upward trend in fuel flow (W36) to maintain target thrust.",
        "root_cause": "HPC seal degradation and blade tip clearance increase due to long-term cyclic thermal expansion and mechanical wear.",
        "action_taken": "Removed engine from service prior to predicted stall-margin breach; replaced HPC abradable seals and restored nominal clearances.",
        "resolution_time": "18 hrs downtime, unit returned to service after test-cell verification.",
        "subsystem": "hpc",
    },
    {
        "log_id": "ML-0011",
        "unit": "Engine Unit 38 (Turbofan)",
        "date": "2026-08-19",
        "reported_issue": "Accelerated upward drift in the High-Pressure Turbine (HPT) blade temperature indices. The ratio of HPC outlet pressure to bypass duct pressure (P30/P15) slipped below the lower control limit, signaling a loss of core compressor efficiency.",
        "root_cause": "Accumulation of environmental particulate and dirt on the HPT nozzle guide vanes, choking the cooling holes and inducing a gradual thermal penalty.",
        "action_taken": "Conducted a comprehensive eco-power engine wash cycle; verified restoration of the pressure ratio and reduction of the core temperature baseline during ground run-up.",
        "resolution_time": "4.5 hrs downtime, cleared for flight operations.",
        "subsystem": "lpt_hpt",
    },
    {
        "log_id": "ML-0012",
        "unit": "Engine Unit 07 (Turbofan)",
        "date": "2026-08-22",
        "reported_issue": "Creeping upward trend in static pressure at the HP compressor inlet (Ps30) alongside an increase in the ratio of fuel flow to static pressure. The predictive model calculated an accelerated decline in the remaining useful life (RUL) curve over the past 30 operating hours.",
        "root_cause": "High-pressure compressor blade surface fouling and roughness development, impairing the stage aerodynamic pressure coefficient.",
        "action_taken": "Performed a targeted multi-stage chemical wash of the compressor gas path; confirmed sensor trends reverted back to nominal baseline levels.",
        "resolution_time": "5 hrs downtime, returned to scheduled flight roster.",
        "subsystem": "hpc",
    },
    {
        "log_id": "ML-0013",
        "unit": "Engine Unit 51 (Turbofan)",
        "date": "2026-08-25",
        "reported_issue": "Distinct downward drift in the Fan efficiency tracking parameters. The total temperature at the fan inlet (T24) to bypass pressure ratio (P15) showed a compounding divergence over the last 80 cycles, accompanied by a subtle rise in low-pressure spool speed (N1).",
        "root_cause": "Fan blade leading-edge micro-pitting and surface erosion from long-term atmospheric particulate exposure, distorting the intake boundary layer flow.",
        "action_taken": "Executed a precision on-wing fan blade re-contouring and blending procedure; verified restoration of ideal bypass aerodynamic profiles.",
        "resolution_time": "6 hrs downtime, engine cleared for departure.",
        "subsystem": "fan",
    },
    {
        "log_id": "ML-0014",
        "unit": "Engine Unit 29 (Turbofan)",
        "date": "2026-08-28",
        "reported_issue": "Steady degradation of the engine bypass ratio profile. Sensor telemetry indicated a persistent, compounding increase in total temperature at the HPC outlet (T30) relative to core speed (N2), combined with an elevated bleed-air temperature reading.",
        "root_cause": "Failure and gradual structural degradation of the internal compressor stator seals, leading to parasitic hot-gas recycling into the secondary cooling paths.",
        "action_taken": "Pre-emptively pulled the engine for module disassembly; replaced worn labyrinth seals and refreshed the compressor stator assemblies.",
        "resolution_time": "16 hrs downtime for scheduled engine swap, core unit sent to shop.",
        "subsystem": "seals_other",
    },
]

if __name__ == "__main__":
    # Use a safe iterable count instead of the built-in len() call;
    # this avoids NameError when the runtime environment does not
    # expose a global len builtin in the execution namespace.
    print("Seed logs loaded:", sum(1 for _ in seed_logs))
    from collections import Counter
    counts = Counter(log["subsystem"] for log in seed_logs)
    print("Subsystem distribution:", dict(counts))
