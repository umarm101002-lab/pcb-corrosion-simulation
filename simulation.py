import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. FARADAY CORROSION, THERMAL EXPANSION & LIFETIME MODEL
# ==========================================================

# Physical Constants & Material Properties (Umar, 2026)
density_cu = 8.96       # g/cm^3 (Copper Density)
ew_cu = 31.77            # g/equiv (Equivalent Weight of Cu)
trace_thickness_mm = 0.035 # 1 oz Cu trace (35 microns)

# Coefficients of Thermal Expansion (10^-6 /°C)
cte_cu = 16.5            # Copper trace CTE
cte_fr4_xy = 14.0        # FR-4 in-plane CTE
cte_fr4_z = 60.0         # FR-4 out-of-plane (Z-axis) CTE
E_cu = 110e3             # Young's Modulus Copper (MPa)
E_fr4 = 24e3             # Young's Modulus FR-4 (MPa)

# Experimental Corrosion Currents (A/cm^2)
i_corr_control = 1.47e-6   # Passive control rate
i_corr_corroded = 19.0e-6  # Active NaCl corrosion rate

def calculate_corrosion_rate(i_corr):
    """Calculates corrosion rate in mm/year via Faraday's Law."""
    return 3.27e-3 * ((i_corr * 1e6) * ew_cu) / density_cu

cr_control = calculate_corrosion_rate(i_corr_control)
cr_corroded = calculate_corrosion_rate(i_corr_corroded)

# Time vectors for degradation
time_days = np.linspace(0, 365, 365)
time_years = time_days / 365.0

# Thickness loss over time (µm)
thickness_cu_control = np.maximum(0, trace_thickness_mm - (cr_control * time_years)) * 1000
thickness_cu_corroded = np.maximum(0, trace_thickness_mm - (cr_corroded * time_years)) * 1000

# Thermal Stress Mismatch under Thermal Cycling (Delta T = 100 °C)
delta_T = np.linspace(0, 100, 100)
thermal_stress_mismatch_MPa = (cte_fr4_z - cte_cu) * 1e-6 * delta_T * E_cu

# ==========================================================
# 2. GENERATE FEA-COMPATIBLE MESH COORDINATES (2D PCB TRACE)
# ==========================================================

def generate_pcb_mesh(nx=20, ny=10, length=10.0, height=1.6):
    """Generates a 2D structured mesh grid representing an FR-4/Copper interface."""
    x = np.linspace(0, length, nx)
    y = np.linspace(0, height, ny)
    X, Y = np.meshgrid(x, y)
    nodes = np.column_stack((X.ravel(), Y.ravel()))
    return nodes

mesh_nodes = generate_pcb_mesh()

# ==========================================================
# 3. VISUALIZATION / PLOTTING
# ==========================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Cumulative Lifetime Trace Thickness Loss
ax1.plot(time_days, thickness_cu_control, 'g-', label=f'Control Board (CR = {cr_control:.3f} mm/yr)', lw=2)
ax1.plot(time_days, thickness_cu_corroded, 'r--', label=f'Corroded Board (CR = {cr_corroded:.3f} mm/yr)', lw=2)
ax1.axhline(0, color='black', linestyle=':', alpha=0.7)
ttf_days = (trace_thickness_mm / cr_corroded) * 365
ax1.axvline(ttf_days, color='red', linestyle=':', label=f'Trace Failure ({ttf_days:.1f} Days)')
ax1.set_title('1 oz Copper Trace Corrosion Thickness Loss')
ax1.set_xlabel('Exposure Time (Days)')
ax1.set_ylabel('Remaining Thickness (µm)')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Plot 2: Thermal Expansion (CTE) Mismatch Stress
ax2.plot(delta_T, thermal_stress_mismatch_MPa, 'm-', label='Interfacial Stress (Cu vs FR-4 Z-axis)', lw=2)
ax2.set_title('Thermo-Mechanical Interfacial Stress Mismatch')
ax2.set_xlabel('Temperature Change ΔT (°C)')
ax2.set_ylabel('Induced Stress (MPa)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig('pcb_multiphysics_simulation.png', dpi=300)
plt.show()

print("FEA Mesh Grid generated with shape:", mesh_nodes.shape)
print("Simulation output plot saved as 'pcb_multiphysics_simulation.png'.")
