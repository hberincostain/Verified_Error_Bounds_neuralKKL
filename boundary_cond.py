import numpy as np
from scipy.integrate import solve_ivp


def solve_T_boundary(A, B, x0, T0, t_span, system, t_eval=None, backwards = False):
    """
    Solve PDE along characteristics with boundary condition T(0,y)=(0,0,0,0,y).
    """

    def dynamics(z):
        x = z[:2]
        T = z[2:]
        dxdt = system.function(None, x)
        dTdt = A @ T + B * system.output(x)
        return np.concatenate([dxdt, dTdt])

    def dynamics_back(z):
        x = z[:2]
        T = z[2:]
        dxdt = system.function(None, x)
        dTdt = A @ T + B * system.output(x)
        return np.concatenate([-dxdt, -dTdt])
    
    A = np.asarray(A)
    B = np.asarray(B).reshape(-1)
    
    z0 = np.concatenate([x0, T0])
    
    if backwards: sol = solve_ivp(dynamics_back, t_span, z0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    else: sol = solve_ivp(dynamics, t_span, z0, t_eval=t_eval, rtol=1e-8, atol=1e-18)
    return sol

def generat_Tx_on_traj(x0, T0, A, B, N, b, system):
    t_span = (0, b)
    t_eval = np.linspace(0, t_span[1], N)
    sol = solve_T_boundary(A, B, x0, T0, t_span, system, t_eval, backwards = False)
    traj_y = sol.y[1]
    Ttraj = np.stack(sol.y[2:], axis = 1)
    Xtraj = np.vstack((sol.y[0], sol.y[1])).T  
    
    return Ttraj[N//2:], Xtraj[N//2:], traj_y[N//2:]
    