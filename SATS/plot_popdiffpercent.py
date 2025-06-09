import numpy as np
import matplotlib.pyplot as plt

# (1) Tabulated α vs. diff% from your data:
alpha = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
diff  = np.array([-0.1,  5.9,  4.3, 13.5, 24.6, 29.3, 34.3, 42.0, 50.7, 50.5, 59.0])

# (2) Force the α=0 point to be exactly 0.0 % (instead of –0.1) so that the curve "starts" at zero:
diff[0] = 0.0

# (3) We want the y‐axis to start at zero and then "go down" in the negative direction,
#     so we plot: y = –(diff%).
y = -diff

# (4) Plot
plt.figure(figsize=(6, 4))
plt.plot(alpha, y, '-o', color='tab:blue', linewidth=2, markersize=6, label='–(diff%)')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)            # horizontal line at y=0

plt.xlabel(r'$\alpha$', fontsize=12)
plt.ylabel('Difference %', fontsize=12)
plt.title('Difference % vs. ' + r'$\alpha$', fontsize=14)
plt.ylim(-70, 5)      # adjust so that 0 sits near the top and bottom reaches ~–70
plt.xlim(-0.02, 0.52) # a tiny margin around [0,0.5]
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
