import numpy as np

# ORDER (fields): velocity (x,y), density, pressure, ORDER (params): mach
mean = np.array([0.560642, -0.000129, 0.903352, 0.637941, 0.700000], dtype=np.float32)
std =  np.array([0.216987, 0.216987, 0.145391, 0.119944, 0.118322], dtype=np.float32)


fields_name = ["velocity " f"$u$",
               "velocity " f"$v$",
               "density " f"$\\rho$",
               "pressure " f"$P$",
               "mach " f"$Ma$"]

# WARNING: order of iteration must match IDs order
IDs = ['y2025_m04_d24_11h_11m_40s', 'y2025_m04_d09_23h_16m_29s', 'y2025_m04_d28_08h_53m_22s',
       'y2025_m04_d24_00h_29m_31s', 'y2025_m04_d24_03h_35m_40s', 'y2025_m04_d24_06h_14m_58s']


# Add as many samples as you can
versions = [1, 2, 3, 4, 5]


fields_name_latex = ['$u$', '$v$', '$\\rho$', '$P$']
