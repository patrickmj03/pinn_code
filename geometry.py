import torch


X_MIN, X_MAX = 0.0, 2.0
Y_MIN, Y_MAX = -0.5, 0.5
CYLINDER_CENTER = (0.5, 0.0)
CYLINDER_RADIUS = 0.1


def cylinder_mask(x, y, margin=0.0):
    cx, cy = CYLINDER_CENTER
    radius = CYLINDER_RADIUS + margin
    return (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2


def sample_fluid_points(n_points, device="cpu", margin=0.0):
    """Sample points in the rectangular channel, excluding the solid cylinder."""
    xs = []
    ys = []

    while sum(chunk.shape[0] for chunk in xs) < n_points:
        batch = max(n_points, 1024)
        x = X_MIN + (X_MAX - X_MIN) * torch.rand(batch, 1, device=device)
        y = Y_MIN + (Y_MAX - Y_MIN) * torch.rand(batch, 1, device=device)
        keep = ~cylinder_mask(x, y, margin=margin).squeeze(1)
        xs.append(x[keep])
        ys.append(y[keep])

    x_out = torch.cat(xs, dim=0)[:n_points]
    y_out = torch.cat(ys, dim=0)[:n_points]
    return x_out, y_out


def random_time(n_points, device="cpu"):
    return torch.rand(n_points, 1, device=device)


def get_collocation_points(
    n_domain=2000,
    n_cylinder=500,
    n_inflow=400,
    n_outflow=400,
    n_walls=400,
    device="cpu",
):
    # PDE points must live in the fluid only. Enforcing Navier-Stokes inside the
    # cylinder makes the solid obstacle contradict the no-slip boundary.
    x_domain, y_domain = sample_fluid_points(n_domain, device=device, margin=1e-4)
    t_domain = random_time(n_domain, device=device)

    theta = torch.rand(n_cylinder, 1, device=device) * 2 * torch.pi
    cx, cy = CYLINDER_CENTER
    x_cylinder = cx + CYLINDER_RADIUS * torch.cos(theta)
    y_cylinder = cy + CYLINDER_RADIUS * torch.sin(theta)
    t_cylinder = random_time(n_cylinder, device=device)

    x_inflow = torch.full((n_inflow, 1), X_MIN, device=device)
    y_inflow = Y_MIN + (Y_MAX - Y_MIN) * torch.rand(n_inflow, 1, device=device)
    t_inflow = random_time(n_inflow, device=device)

    x_outflow = torch.full((n_outflow, 1), X_MAX, device=device)
    y_outflow = Y_MIN + (Y_MAX - Y_MIN) * torch.rand(n_outflow, 1, device=device)
    t_outflow = random_time(n_outflow, device=device)

    n_bottom = n_walls // 2
    n_top = n_walls - n_bottom
    x_bottom = X_MIN + (X_MAX - X_MIN) * torch.rand(n_bottom, 1, device=device)
    y_bottom = torch.full((n_bottom, 1), Y_MIN, device=device)
    x_top = X_MIN + (X_MAX - X_MIN) * torch.rand(n_top, 1, device=device)
    y_top = torch.full((n_top, 1), Y_MAX, device=device)
    x_walls = torch.cat([x_bottom, x_top], dim=0)
    y_walls = torch.cat([y_bottom, y_top], dim=0)
    t_walls = random_time(n_walls, device=device)

    return {
        "domain": (x_domain, y_domain, t_domain),
        "cylinder": (x_cylinder, y_cylinder, t_cylinder),
        "inflow": (x_inflow, y_inflow, t_inflow),
        "outflow": (x_outflow, y_outflow, t_outflow),
        "walls": (x_walls, y_walls, t_walls),
    }
