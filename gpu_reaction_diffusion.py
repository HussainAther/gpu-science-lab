import time
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
WIDTH = 512
HEIGHT = 512

Du = 0.16   # diffusion rate for U
Dv = 0.08   # diffusion rate for V
F  = 0.060  # feed rate (try 0.035, 0.055, 0.060, 0.022)
k  = 0.062  # kill rate (try 0.065, 0.062, 0.061, 0.051)

dt = 1.0    # time step
dx = 1.0    # space step (assume 1 for simplicity)

# ---------- SIMULATION STATE ----------

U = None
V = None
running = True
last_frame_save_idx = 0

def init_fields():
    """Initialize U and V fields on the GPU."""
    global U, V

    # Start with U = 1 everywhere, V = 0
    U = cp.ones((HEIGHT, WIDTH), dtype=cp.float32)
    V = cp.zeros((HEIGHT, WIDTH), dtype=cp.float32)

    # Add a square of V in the middle
    r = 20  # radius/size of central seed
    cx, cy = WIDTH // 2, HEIGHT // 2
    U[cy - r:cy + r, cx - r:cx + r] = 0.50
    V[cy - r:cy + r, cx - r:cx + r] = 0.25

    # A bit of random noise for more interesting patterns
    U += 0.05 * cp.random.random((HEIGHT, WIDTH))
    V += 0.05 * cp.random.random((HEIGHT, WIDTH))

def laplacian(Z):
    """Discrete 2D Laplacian with periodic boundary conditions."""
    # Roll in four directions and sum; subtract 4 * center
    return (
        -4.0 * Z
        + cp.roll(Z, 1, axis=0)
        + cp.roll(Z, -1, axis=0)
        + cp.roll(Z, 1, axis=1)
        + cp.roll(Z, -1, axis=1)
    ) / (dx * dx)

def step():
    """Perform one simulation timestep (on GPU)."""
    global U, V

    # Compute Laplacians
    Lu = laplacian(U)
    Lv = laplacian(V)

    # Reaction term
    UVV = U * V * V

    # Gray-Scott update
    dU = Du * Lu - UVV + F * (1.0 - U)
    dV = Dv * Lv + UVV - (F + k) * V

    U = U + dU * dt
    V = V + dV * dt

    # Clamp for numerical safety
    U = cp.clip(U, 0.0, 1.0)
    V = cp.clip(V, 0.0, 1.0)

def on_key(event):
    """Keyboard callback for matplotlib."""
    global running, last_frame_save_idx

    if event.key == ' ':
        running = not running
        print("Running:", running)
    elif event.key == 'r':
        print("Resetting fields...")
        init_fields()
    elif event.key == 's':
        # Save current frame
        global im
        # Get CPU copy for saving
        frame = im.get_array()
        fname = f"frame_{last_frame_save_idx:04d}.png"
        plt.imsave(fname, frame, cmap='inferno')
        last_frame_save_idx += 1
        print("Saved:", fname)

def main():
    global im

    init_fields()

    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initial render: visualize V or U-V
    U_cpu = U.get()
    V_cpu = V.get()
    data = V_cpu  # you can experiment with (V_cpu - U_cpu) etc.

    im = ax.imshow(data, cmap='inferno', interpolation='bilinear')
    ax.set_axis_off()
    fig.tight_layout()

    print("Controls:")
    print("  Space: pause/resume")
    print("  r    : reset")
    print("  s    : save current frame")

    # Main loop
    while True:
        if running:
            # Run multiple steps per frame for speed
            for _ in range(10):
                step()

            # Copy from GPU → CPU for display
            V_cpu = V.get()
            im.set_data(V_cpu)

        plt.pause(0.001)  # allow GUI to update

if __name__ == "__main__":
    main()

