import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_initial_geometry(x_d, y_d, x_b, y_b):
    """Plot domain and boundary collocation points."""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')

    domain_x = x_d.cpu().numpy()
    domain_y = y_d.cpu().numpy()
    boundary_x = x_b.cpu().numpy()
    boundary_y = y_b.cpu().numpy()

    ax.scatter(
        domain_x,
        domain_y,
        c='#1f77b4',
        s=12,
        alpha=0.35,
        linewidths=0,
        label=f'Domain points ({len(x_d)})',
    )
    ax.scatter(
        boundary_x,
        boundary_y,
        c='#d62728',
        s=45,
        alpha=0.85,
        marker='o',
        edgecolors='white',
        linewidths=0.75,
        label=f'Boundary points ({len(x_b)})',
    )

    cylinder = Circle(
        (0.5, 0.0),
        0.1,
        facecolor='none',
        edgecolor='#2c3e50',
        linewidth=1.5,
        linestyle='-',
        alpha=0.85,
        label='Cylinder boundary',
    )
    ax.add_patch(cylinder)

    ax.set_xlabel('x', fontsize=13, fontweight='medium')
    ax.set_ylabel('y', fontsize=13, fontweight='medium')
    ax.set_title('Initial Geometry: Collocation Points', fontsize=18, fontweight='bold', pad=14)

    ax.legend(
        frameon=True,
        framealpha=0.95,
        edgecolor='lightgray',
        fontsize=11,
        loc='upper right',
    )

    ax.grid(True, color='#dcdcdc', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.1, 2.05)
    ax.set_ylim(-0.55, 0.55)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#555555')
    ax.spines['bottom'].set_color('#555555')

    fig.tight_layout(pad=1.0)
    fig.savefig("initial_geometry.png", dpi=300)
    print("Initial geometry saved to initial_geometry.png")
    try:
        plt.show(block=False)
        plt.pause(0.001)
    except Exception:
        pass
