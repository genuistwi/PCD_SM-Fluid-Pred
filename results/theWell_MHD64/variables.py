import numpy as np

# ORDER (fields): velocity (x,y), density, pressure, ORDER (params): mach
mean = np.array([1.0015e+00, 5.3329e-01, 8.7613e-10, 1.9945e-08, 9.5950e-03, 6.7743e-02, 1.9342e-02], dtype=np.float32)
std =  np.array([0.8922, 0.5287, 0.3171, 0.3201, 0.4852, 0.4381, 0.4785], dtype=np.float32)

fields_name = [f"$\\rho$",
               f"$B_x$",
               f"$B_y$",
               f"$B_z$",
               f"$u$",
               f"$v$",
               f"$w$"]

# WARNING: order of iteration must match IDs order

