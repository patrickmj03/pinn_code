import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

def generate_mesh(nx=200, ny=100, t_val=0.5, device='cpu'):
    """Erzeugt ein gleichmäßiges Gitter für die Inferenz."""
    x = np.linspace(0.0, 2.0, nx)
    y = np.linspace(-0.5, 0.5, ny)
    X, Y = np.meshgrid(x, y)
    
    x_flat = X.flatten()[:, None]
    y_flat = Y.flatten()[:, None]
    t_flat = np.ones_like(x_flat) * t_val
    
    x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_flat, dtype=torch.float32, device=device)
    t_tensor = torch.tensor(t_flat, dtype=torch.float32, device=device)
    
    return X, Y, x_tensor, y_tensor, t_tensor

def plot_snapshots(model, device, t_val=0.5):
    """Generiert kreative Plots für u, v, p und Stromlinien zu einem Zeitpunkt t."""
    X, Y, x_t, y_t, t_t = generate_mesh(nx=200, ny=100, t_val=t_val, device=device)
    
    print("Berechne Snapshot-Daten für u, v und p...")
    with torch.no_grad():
        inputs = torch.cat([x_t, y_t, t_t], dim=1)
        pred = model(inputs)
        u = pred[:, 0:1].cpu().numpy().reshape(X.shape)
        v = pred[:, 1:2].cpu().numpy().reshape(X.shape)
        p = pred[:, 2:3].cpu().numpy().reshape(X.shape)
    
    inside_cylinder = (X - 0.5)**2 + Y**2 < 0.1**2
    u[inside_cylinder] = np.nan
    v[inside_cylinder] = np.nan
    p[inside_cylinder] = np.nan
    
    print("Zeichne Snapshots...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # FIX: Hier nutzen wir jetzt X[0, :] für x (Breite) und Y[:, 0] für y (Höhe)
    im0 = axes[0].contourf(X, Y, u, levels=50, cmap='jet')
    axes[0].streamplot(X[0, :], Y[:, 0], u, v, color='k', linewidth=0.5, density=1.0)
    axes[0].set_title(f"Horizontale Geschwindigkeit u (t={t_val})")
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].contourf(X, Y, v, levels=50, cmap='seismic')
    axes[1].set_title("Vertikale Geschwindigkeit v (Wirbelindikatoren)")
    fig.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].contourf(X, Y, p, levels=50, cmap='coolwarm')
    axes[2].set_title("Druckfeld p (Unterdruckgebiete hinter dem Zylinder)")
    fig.colorbar(im2, ax=axes[2])
    
    for ax in axes:
        circle = plt.Circle((0.5, 0.0), 0.1, color='black', fill=True)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        
    plt.tight_layout()
    fig.savefig("pinn_flow_results.png", dpi=300)
    print("Snapshots saved to pinn_flow_results.png")
    plt.close(fig)

def create_flow_video(model, device, filename="zylinder_stroemung.mp4", duration=10.0, dt=0.05):
    """Erzeugt ein animiertes Video der Strömung über die Zeit."""
    fig, ax = plt.subplots(figsize=(10, 5))
    X, Y, x_t, y_t, t_t = generate_mesh(nx=150, ny=75, t_val=0.0, device=device)
    time_values = np.arange(0.0, duration + 0.5 * dt, dt)
    
    def animate(frame):
        ax.clear()
        t_current = time_values[frame]
        
        _, _, x_mesh, y_mesh, t_tensor = generate_mesh(nx=150, ny=75, t_val=t_current, device=device)
        
        with torch.no_grad():
            inputs = torch.cat([x_mesh, y_mesh, t_tensor], dim=1)
            pred = model(inputs)
            u = pred[:, 0:1].cpu().numpy().reshape(X.shape)
            
        inside_cylinder = (X - 0.5)**2 + Y**2 < 0.1**2
        u[inside_cylinder] = np.nan
        
        ax.contourf(X, Y, u, levels=40, cmap='jet')
        circle = plt.Circle((0.5, 0.0), 0.1, color='black')
        ax.add_patch(circle)
        ax.set_title(f"Strömungsentwicklung u(x,y,t) | Zeit: {t_current:.2f}s")
        ax.set_aspect('equal')
        return

    ani = animation.FuncAnimation(fig, animate, frames=len(time_values), interval=dt * 1000, blit=False)
    print("Erstelle Strömungsanimation... das kann einige Sekunden dauern.")
    
    try:
        ani.save(filename, writer='ffmpeg', fps=10)
        print(f"\n>>> Video erfolgreich unter '{filename}' gespeichert! <<<")
    except Exception as e:
        print(f"\nffmpeg save failed: {e}")
        html_path = Path(filename).with_suffix('.html')
        print(f"Speichere die Animation stattdessen als HTML-Datei: {html_path}")
        ani.save(str(html_path), writer='html')
        print(f"Animation erfolgreich als HTML gespeichert unter '{html_path}'")
    finally:
        plt.close(fig)


def create_axial_velocity_animation(
    model,
    device,
    filename="axiale_stroemung.html",
    duration=10.0,
    dt=0.05,
    nx=180,
    ny=90,
):
    """Animate the axial velocity component u(x,y,t) over time."""
    time_values = np.arange(0.0, duration + 0.5 * dt, dt)
    X, Y, x_mesh, y_mesh, _ = generate_mesh(nx=nx, ny=ny, t_val=0.0, device=device)
    inside_cylinder = (X - 0.5) ** 2 + Y ** 2 < 0.1 ** 2

    # Use a fixed color scale across the full animation. Without this, matplotlib
    # rescales every frame and can make weak changes look artificially dramatic.
    sample_indices = np.linspace(0, len(time_values) - 1, min(25, len(time_values)), dtype=int)
    u_min = np.inf
    u_max = -np.inf
    with torch.no_grad():
        for idx in sample_indices:
            t_tensor = torch.full_like(x_mesh, float(time_values[idx]))
            pred = model(torch.cat([x_mesh, y_mesh, t_tensor], dim=1))
            u = pred[:, 0:1].cpu().numpy().reshape(X.shape)
            u[inside_cylinder] = np.nan
            u_min = min(u_min, np.nanmin(u))
            u_max = max(u_max, np.nanmax(u))

    if not np.isfinite(u_min) or not np.isfinite(u_max) or np.isclose(u_min, u_max):
        u_min, u_max = -1.0, 1.0

    fig, ax = plt.subplots(figsize=(11, 5))
    levels = np.linspace(u_min, u_max, 60)

    def animate(frame):
        ax.clear()
        t_current = float(time_values[frame])
        t_tensor = torch.full_like(x_mesh, t_current)

        with torch.no_grad():
            pred = model(torch.cat([x_mesh, y_mesh, t_tensor], dim=1))
            u = pred[:, 0:1].cpu().numpy().reshape(X.shape)

        u[inside_cylinder] = np.nan
        ax.contourf(X, Y, u, levels=levels, cmap="turbo", extend="both")
        ax.add_patch(plt.Circle((0.5, 0.0), 0.1, color="black"))
        ax.set_title(f"Axiale Geschwindigkeit u(x,y,t) | t = {t_current:.2f} s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(-0.5, 0.5)
        return []

    initial = ax.contourf(X, Y, np.zeros_like(X), levels=levels, cmap="turbo", extend="both")
    colorbar = fig.colorbar(initial, ax=ax)
    colorbar.set_label("Axiale Geschwindigkeit u")

    ani = animation.FuncAnimation(fig, animate, frames=len(time_values), interval=dt * 1000, blit=False)
    print(f"Erstelle Animation der axialen Geschwindigkeit: {filename}")

    try:
        ani.save(filename, writer="html")
        print(f"Animation der axialen Geschwindigkeit gespeichert unter: {filename}")
    finally:
        plt.close(fig)
