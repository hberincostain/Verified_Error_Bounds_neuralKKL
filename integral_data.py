import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import torch

def flow(x0, s, f):
    """
    Compute phi_x(x0, s) = flow at time s with initial condition x0.
    Since s <= 0, we integratge backward in time.
    """
    def ode(t, x):
        return f(x)
    
    sol = solve_ivp(ode, [0, s], x0, method='RK45', t_eval=[s]) # s < 0
    return sol.y[:, -1]

def integrand(s, A, B, phi, system):
    """
    Compute the integrand vector at time s.
    """
    hs = system.output(phi)
    eAs = expm(-A * s)    # matrix exponential e^{-As}

    return eAs @ B * hs

def integral_approx(x0, A, B, T, num_points, system):
    """
    Numerically approximate integral from -T to 0 of integrand ds.
    """
    s_vals = np.linspace(0, -T, num_points)
    phi, _ = system.simulate(0, -T, num_points, x0)
    has_nan = np.isnan(phi).any()
    has_inf = np.isinf(phi).any()
    if has_inf or has_nan: #or not is_point_in_box(phi, limits[:, 0], limits[:, 1]):
        return None
    integrand_vals = np.array([-integrand(s_vals[i], x0, A, B, phi[i], system) for i in range(len(s_vals))])  # shape (num_points, n)

    # Integrate each component over s using trapezoidal rule
    integral = np.trapz(integrand_vals, s_vals, axis=0)
    return integral

def ode_approx(x0, A, B, T, num_points, system):
    """
    Numerically approximate integral from -T to 0 of integrand ds.
    """
    A = np.asarray(A)
    B = np.asarray(B).reshape(-1)

    def dynamics_back(z):
        x = z[:2]
        T = z[2:]
        dxdt = system.function(None, x)
        dTdt = A @ T + B * [system.output(x)]
        return np.concatenate([-dxdt, -dTdt])
    
    s_vals = np.linspace(0, T, num_points)
    n = A.shape[0]
    T0 = np.zeros(n) 
    y0 = np.concatenate([x0, T0])
    sol = solve_ivp(dynamics_back, (0, T), y0, t_eval=s_vals, rtol=1e-8, atol=1e-10)
    Ttraj = np.stack(sol.y[2:], axis = 1)
    Xtraj = np.vstack((sol.y[0], sol.y[1])).T 
    has_nan = np.isnan(Xtraj).any() 
    has_inf = np.isinf(Xtraj).any()
    if has_inf or has_nan: #or not is_point_in_box(phi, limits[:, 0], limits[:, 1]):
        return None
    return Ttraj

def is_point_in_box(points, lower_bounds, upper_bounds):
    for point in points:
        point = np.array(point)
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)
        if not np.all((point >= lower_bounds) & (point <= upper_bounds)): return False
    return True

def generate_T_data(x, A, B, sys, N, T=30):
    T_x = []
    good_ic = []
    for x0 in x:
        z = integral_approx(x0, A, B, T, N, sys)
        if z is not None and not np.isnan(z).any() and not np.isinf(z).any():
            T_x.append(z.squeeze())
            good_ic.append(x0)            
    arr = np.array(T_x)
    good_ics = np.array(good_ic)
    return torch.from_numpy(arr), good_ics