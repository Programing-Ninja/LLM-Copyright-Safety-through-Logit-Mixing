import numpy as np
import matplotlib.pyplot as plt
import itertools

theta_r = 1.0
theta_s = np.linspace(-5, 5, 500)

alpha_values = np.linspace(0.5, 2.0, 5)
beta_values = np.linspace(0.0, 1.0, 5)
c_values = np.linspace(0.2, 1.5, 4)

plt.figure(figsize=(12, 7))

# Representative configs to highlight
sampled_configs = [
    (0.5, 0.0, 0.2),
    (1.0, 0.5, 1.0),
    (2.0, 1.0, 1.5)
]

def is_close_config(a, b, c, config):
    return (np.isclose(a, config[0]) and 
            np.isclose(b, config[1]) and 
            np.isclose(c, config[2]))

for alpha, beta, c in itertools.product(alpha_values, beta_values, c_values):
    upper = alpha * theta_s + c
    lower = alpha * theta_s - c
    
    delta = theta_r - theta_s
    clipped_delta = np.clip(delta, -c, c)
    mixed = alpha * theta_s + beta * clipped_delta

    # Default: faint background
    line_alpha = 0.2
    lw = 1

    # Check if this config should be highlighted
    highlight = any(is_close_config(alpha, beta, c, cfg) for cfg in sampled_configs)

    if highlight:
        label = (
            f"α={alpha:.1f}, β={beta:.1f}, c={c:.1f} | "
            "θ = αθ_s + β·clip(θ_r − θ_s)"
        )
        plt.plot(theta_s, mixed, linewidth=3.5, alpha=1.0, label=label)
        plt.plot(theta_s, upper, linestyle='--', linewidth=2.0, alpha=0.8)
        plt.plot(theta_s, lower, linestyle='--', linewidth=2.0, alpha=0.8)
    else:
        plt.plot(theta_s, mixed, linewidth=lw, alpha=line_alpha)
        plt.plot(theta_s, upper, linestyle='--', linewidth=lw, alpha=0.08)
        plt.plot(theta_s, lower, linestyle='--', linewidth=lw, alpha=0.08)

# Reference lines
plt.plot(theta_s, theta_s, color='black', linestyle=':', linewidth=2,
         label="Pure safe model (θ_s)")

plt.axhline(theta_r, color='red', linestyle=':', linewidth=2,
            label="Reference model (θ_r)")

plt.xlabel(r'$\theta_s$ (Safe model logits)')
plt.ylabel(r'$\theta_{mixed}$ (Final mixed logits)')

plt.title(
    "Clipped Logit Mixing: θ = α·θ_s + β·clip(θ_r − θ_s)\n"
    "α: trust safe | β: correction strength | c: max deviation"
)

plt.legend(loc='upper left', fontsize=9)
plt.grid(True)

plt.show()