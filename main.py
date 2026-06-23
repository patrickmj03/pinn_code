import torch
import numpy as np
from pathlib import Path

from architecture import PINN
from physics import get_residual
from geometry import get_collocation_points
from plot import plot_initial_geometry
from post import plot_snapshots, create_flow_video

# Setup
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwendetes Gerät: {device}")

model = PINN([3, 64, 64, 64, 64, 3]).to(device)
model_dir = Path("saved_models")
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "pinn_model.pth"

if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Geladenes Modell gefunden und geladen: {model_path}")
else:
    print("Kein gespeichertes Modell gefunden. Training wird gestartet...")

# 1. Daten inklusive Inflow laden
domain_data, boundary_data, inflow_data = get_collocation_points(n_domain=2000, n_boundary=500, n_inflow=400, device=device)
x_d, y_d, t_d = domain_data
x_b, y_b, t_b = boundary_data
x_in, y_in, t_in = inflow_data

# Plot the initial geometry non-blocking and save it.
plot_initial_geometry(x_d, y_d, x_b, y_b)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 1000

if not model_path.exists():
    print("Starte echtes PINN Training...")
    print("-" * 60)

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PHYSIK-LOSS
        f_m, f_x, f_y = get_residual(model, x_d, y_d, t_d)
        loss_pde = torch.mean(f_m**2) + torch.mean(f_x**2) + torch.mean(f_y**2)
        
        # ZYLINDER-LOSS (Wand: u=0, v=0)
        inputs_b = torch.cat([x_b, y_b, t_b], dim=1)
        pred_b = model(inputs_b)
        u_b, v_b = pred_b[:, 0:1], pred_b[:, 1:2]
        loss_bc = torch.mean(u_b**2) + torch.mean(v_b**2)
        
        # NEU: INFLOW-LOSS (Einströmung bei x=0: u=1.0, v=0)
        inputs_in = torch.cat([x_in, y_in, t_in], dim=1)
        pred_in = model(inputs_in)
        u_in, v_in = pred_in[:, 0:1], pred_in[:, 1:2]
        loss_inflow = torch.mean((u_in - 1.0)**2) + torch.mean(v_in**2)
        
        # Gesamter Loss (Gewichtet)
        total_loss = loss_pde + 10.0 * loss_bc + 10.0 * loss_inflow
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoche {epoch:4d} | Total: {total_loss.item():.4f} | PDE: {loss_pde.item():.4f} | Wall: {loss_bc.item():.4f} | Inflow: {loss_inflow.item():.4f}")

    print("-" * 60)
    print("Training abgeschlossen!")
    torch.save(model.state_dict(), model_path)
    print(f"Trainiertes Modell gespeichert unter: {model_path}")
else:
    print("Existierendes Modell gefunden. Training wird übersprungen.")

print("Generiere grafische Auswertungen...")
# Erstellt die schönen statischen Grafiken für u, v und p
plot_snapshots(model, device, t_val=0.5)

print("Erzeuge Strömungsvideo...")
# Erstellt das Video über den zeitlichen Verlauf
create_flow_video(model, device, filename="zylinder_stroemung.mp4")