import matplotlib.pyplot as plt

from src.data.lorenz import LorenzParameters, simulate_lorenz


def plot_lorenz_attractor():
    # Simulation settings
    t_start = 0.0
    t_end = 50.0
    dt = 0.01

    initial_state = [1.0, 1.0, 1.0]
    params = LorenzParameters()

    # Time points
    t_eval = __import__("numpy").arange(t_start, t_end, dt)

    # Simulate
    _, states = simulate_lorenz(
        t_span=(t_start, t_end),
        t_eval=t_eval,
        initial_state=initial_state,
        params=params,
    )

    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]

    # Create 3D figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, linewidth=0.7)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Lorenz Attractor")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_lorenz_attractor()
