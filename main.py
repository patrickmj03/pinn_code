import os
from pathlib import Path

import torch

from architecture import PINN
from geometry import get_collocation_points, sample_fluid_points
from physics import get_residual
from plot import plot_initial_geometry
from post import plot_snapshots, create_flow_video, create_axial_velocity_animation


NU = 0.01
INFLOW_U = 1.0
CHECKPOINT_VERSION = "fluid_domain_v2"


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def mse_to_zero(value):
    return torch.mean(value ** 2)


def velocity_loss(pred, target_u, target_v):
    return torch.mean((pred[:, 0:1] - target_u) ** 2) + torch.mean((pred[:, 1:2] - target_v) ** 2)


def grad(outputs, inputs):
    return torch.autograd.grad(outputs.sum(), inputs, create_graph=True)[0]


def compute_losses(model, points):
    x_d, y_d, t_d = points["domain"]
    x_c, y_c, t_c = points["cylinder"]
    x_in, y_in, t_in = points["inflow"]
    x_out, y_out, t_out = points["outflow"]
    x_wall, y_wall, t_wall = points["walls"]

    f_m, f_x, f_y = get_residual(model, x_d, y_d, t_d, nu=NU)
    loss_pde = mse_to_zero(f_m) + mse_to_zero(f_x) + mse_to_zero(f_y)

    pred_cylinder = model(torch.cat([x_c, y_c, t_c], dim=1))
    loss_cylinder = velocity_loss(pred_cylinder, 0.0, 0.0)

    pred_inflow = model(torch.cat([x_in, y_in, t_in], dim=1))
    loss_inflow = velocity_loss(pred_inflow, INFLOW_U, 0.0)

    # Far-field top/bottom boundaries prevent the unconstrained outer box from
    # inventing recirculation that the PDE alone cannot rule out.
    pred_walls = model(torch.cat([x_wall, y_wall, t_wall], dim=1))
    loss_walls = velocity_loss(pred_walls, INFLOW_U, 0.0)

    x_out = x_out.requires_grad_(True)
    y_out = y_out.requires_grad_(True)
    t_out = t_out.requires_grad_(True)
    pred_outflow = model(torch.cat([x_out, y_out, t_out], dim=1))
    u_out, v_out, p_out = pred_outflow[:, 0:1], pred_outflow[:, 1:2], pred_outflow[:, 2:3]
    loss_outflow = mse_to_zero(p_out) + 0.1 * mse_to_zero(grad(u_out, x_out)) + 0.1 * mse_to_zero(grad(v_out, x_out))

    return {
        "pde": loss_pde,
        "cylinder": loss_cylinder,
        "inflow": loss_inflow,
        "walls": loss_walls,
        "outflow": loss_outflow,
    }


def weighted_total(losses):
    return (
        losses["pde"]
        + 30.0 * losses["cylinder"]
        + 20.0 * losses["inflow"]
        + 5.0 * losses["walls"]
        + 2.0 * losses["outflow"]
    )


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwendetes Geraet: {device}")

    model = PINN([3, 64, 64, 64, 64, 3]).to(device)
    model_dir = Path("saved_models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"pinn_model_{CHECKPOINT_VERSION}.pth"
    old_model_path = model_dir / "pinn_model.pth"
    force_retrain = env_flag("PINN_FORCE_RETRAIN")
    train_model = env_flag("PINN_TRAIN")

    if old_model_path.exists() and not model_path.exists():
        print(f"Altes Checkpoint-Format ignoriert: {old_model_path}")
        print("Die Geometrie/Losses wurden geaendert, daher wird dieses alte Modell nicht automatisch genutzt.")

    model_exists = model_path.exists()
    should_train = train_model or force_retrain
    if model_exists and not force_retrain:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Geladenes Modell gefunden und geladen: {model_path}")
    elif not should_train:
        raise FileNotFoundError(
            f"Kein trainiertes Modell gefunden: {model_path}. "
            "Training startet standardmaessig nicht automatisch. "
            "Setze PINN_TRAIN=1, wenn du bewusst neu trainieren willst."
        )

    points = get_collocation_points(
        n_domain=8000,
        n_cylinder=1000,
        n_inflow=800,
        n_outflow=800,
        n_walls=800,
        device=device,
    )

    n_ic = 1000
    x_ic, y_ic = sample_fluid_points(n_ic, device=device, margin=1e-4)
    t_ic = torch.zeros(n_ic, 1, device=device)
    points["initial"] = (x_ic, y_ic, t_ic)

    x_d, y_d, _ = points["domain"]
    x_c, y_c, _ = points["cylinder"]
    plot_initial_geometry(
        x_d,
        y_d,
        x_c,
        y_c,
        extra_points={
            "Inflow/outflow": (
                torch.cat([points["inflow"][0], points["outflow"][0]], dim=0),
                torch.cat([points["inflow"][1], points["outflow"][1]], dim=0),
            ),
            "Top/bottom": (points["walls"][0], points["walls"][1]),
        },
    )

    if should_train:
        print("Kein passendes gespeichertes Modell gefunden. Starte PINN-Training...")
        print("-" * 70)

        optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=2500, gamma=0.5)
        epochs_adam = int(os.environ.get("PINN_EPOCHS_ADAM", "10000"))

        for epoch in range(epochs_adam):
            optimizer_adam.zero_grad()
            losses = compute_losses(model, points)

            pred_ic = model(torch.cat(points["initial"], dim=1))
            losses["initial"] = velocity_loss(pred_ic, 0.0, 0.0)

            total_loss = weighted_total(losses) + 10.0 * losses["initial"]
            total_loss.backward()
            optimizer_adam.step()
            scheduler.step()

            if epoch % 100 == 0:
                current_lr = optimizer_adam.param_groups[0]["lr"]
                print(
                    f"Adam Epoche {epoch:5d}/{epochs_adam} | "
                    f"Total: {total_loss.item():.4f} | PDE: {losses['pde'].item():.4f} | "
                    f"BC: {(losses['cylinder'] + losses['inflow'] + losses['walls']).item():.4f} | "
                    f"LR: {current_lr:.5f}"
                )

        print("\nStarte finale L-BFGS High-Precision Phase...")
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=int(os.environ.get("PINN_LBFGS_MAX_ITER", "1500")),
            line_search_fn="strong_wolfe",
        )

        lbfgs_iter = [0]

        def closure():
            optimizer_lbfgs.zero_grad()
            losses = compute_losses(model, points)
            pred_ic = model(torch.cat(points["initial"], dim=1))
            losses["initial"] = velocity_loss(pred_ic, 0.0, 0.0)
            total_loss = weighted_total(losses) + 10.0 * losses["initial"]
            total_loss.backward()

            if lbfgs_iter[0] % 150 == 0:
                print(
                    f"L-BFGS Iter {lbfgs_iter[0]:4d} | "
                    f"Total Loss: {total_loss.item():.6f} | PDE-Fehler: {losses['pde'].item():.6f}"
                )
            lbfgs_iter[0] += 1
            return total_loss

        optimizer_lbfgs.step(closure)

        print("-" * 70)
        print("Training beendet.")
        torch.save(model.state_dict(), model_path)
        print(f"Modell gesichert unter: {model_path}")
    else:
        print(f"Gespeichertes Modell gefunden: {model_path}. Training wird uebersprungen.")

    print("Generiere grafische Auswertungen...")
    plot_snapshots(model, device, t_val=0.5)

    print("Erzeuge Stroemungsvideo (HTML)...")
    create_flow_video(model, device, filename="zylinder_stroemung.html", duration=10.0, dt=0.05)

    print("Erzeuge Animation der axialen Geschwindigkeit (HTML)...")
    create_axial_velocity_animation(model, device, filename="axiale_stroemung.html", duration=10.0, dt=0.05)


if __name__ == "__main__":
    main()
