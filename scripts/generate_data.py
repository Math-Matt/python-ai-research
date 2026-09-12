from pathlib import Path

import numpy as np

from src.data.lorenz import LorenzParameters, simulate_lorenz


def main():
    # Simulation settings
    t_start = 0.0
    t_end = 100.0
    dt = 0.01

    initial_state = [1.0, 1.0, 1.0]
    params = LorenzParameters()

    # Time points
    t_eval = np.arange(t_start, t_end, dt)

    # Simulate Lorenz system
    t, states = simulate_lorenz(
        t_span=(t_start, t_end),
        t_eval=t_eval,
        initial_state=initial_state,
        params=params,
    )

    # Create output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine time and state variables
    data = np.column_stack((t, states))

    # Save dataset
    output_path = output_dir / "lorenz_trajectory.csv"

    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header="t,x,y,z",
        comments="",
    )

    print(f"Saved dataset to: {output_path}")
    print(f"Number of timesteps: {len(t)}")
    print(f"Time step: {dt}")
    print(f"Final time: {t[-1]}")


if __name__ == "__main__":
    main()
