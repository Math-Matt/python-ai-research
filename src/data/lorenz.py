"""
Lorenz dynamical system.

This module defines the Lorenz equations and provides utilities
for numerically simulating the system.

The classical chaotic parameter regime is:
    sigma = 10
    rho = 28
    beta = 8 / 3
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class LorenzParameters:
    """Parameters defining the Lorenz system."""

    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0


def lorenz_rhs(
    t: float,
    state: np.ndarray,
    params: LorenzParameters,
) -> np.ndarray:
    """
    Compute the time derivative of the Lorenz system.

    Parameters
    ----------
    t:
        Current time. Included for compatibility with scipy.integrate.solve_ivp.
    state:
        Current state [x, y, z].
    params:
        Parameters of the Lorenz system.

    Returns
    -------
    np.ndarray
        Derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state

    dx_dt = params.sigma * (y - x)
    dy_dt = x * (params.rho - z) - y
    dz_dt = x * y - params.beta * z

    return np.array([dx_dt, dy_dt, dz_dt], dtype=float)


def simulate_lorenz(
    t_span: tuple[float, float],
    t_eval: np.ndarray,
    initial_state: np.ndarray | list[float] | tuple[float, float, float],
    params: LorenzParameters | None = None,
    method: str = "RK45",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the Lorenz system over a specified time interval.

    Parameters
    ----------
    t_span:
        Start and end times as (t_start, t_end).
    t_eval:
        Time points at which the solution should be returned.
    initial_state:
        Initial state [x0, y0, z0].
    params:
        Lorenz system parameters. Uses the classical chaotic
        parameters if None.
    method:
        Numerical integration method passed to scipy.integrate.solve_ivp.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Times and corresponding states.

        times:
            Shape (n_steps,)

        states:
            Shape (n_steps, 3), with columns [x, y, z].

    Raises
    ------
    ValueError
        If the simulation fails.
    """
    if params is None:
        params = LorenzParameters()

    initial_state = np.asarray(initial_state, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    if initial_state.shape != (3,):
        raise ValueError(
            "initial_state must contain exactly three values: [x0, y0, z0]."
        )

    if t_eval.ndim != 1:
        raise ValueError("t_eval must be a one-dimensional array.")

    if len(t_eval) == 0:
        raise ValueError("t_eval cannot be empty.")

    if t_eval[0] < t_span[0] or t_eval[-1] > t_span[1]:
        raise ValueError("All t_eval values must lie inside t_span.")

    solution = solve_ivp(
        fun=lorenz_rhs,
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        args=(params,),
        method=method,
    )

    if not solution.success:
        raise ValueError(f"Lorenz simulation failed: {solution.message}")

    return solution.t, solution.y.T
