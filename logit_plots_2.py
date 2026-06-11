import numpy as np
import matplotlib.pyplot as plt

# Parameters (tweak these)
alpha = 1.0
beta = 1
c = 4.0

# Grid over theta_s and theta_r
theta_s = np.linspace(-5, 5, 100)
theta_r = np.linspace(-5, 5, 100)

Theta_s, Theta_r = np.meshgrid(theta_s, theta_r)

# Compute clipped difference
delta = Theta_r - Theta_s
clipped_delta = np.clip(delta, -c, c)

# Mixed logits
Theta_mixed = alpha * Theta_s + beta * clipped_delta

# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    Theta_s, Theta_r, Theta_mixed,
    cmap='viridis',
    edgecolor='none',
    alpha=0.9
)

# Labels
ax.set_xlabel(r'$\theta_s$ (Safe logits)')
ax.set_ylabel(r'$\theta_r$ (Reference logits)')
ax.set_zlabel(r'$\theta_{mixed}$')

ax.set_title(
    "3D Logit Mixing Surface\n"
    "θ = α·θ_s + β·clip(θ_r − θ_s, ±c)"
)

fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()