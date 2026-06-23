import torch

def get_collocation_points(n_domain=2000, n_boundary=500, n_inflow=400, device='cpu'):
    # 1. Punkte im Feld (Domain)
    x_domain = torch.rand(n_domain, 1, device=device) * 2.0
    y_domain = (torch.rand(n_domain, 1, device=device) - 0.5)
    t_domain = torch.rand(n_domain, 1, device=device)
    
    # 2. Punkte auf der Zylinderwand (Boundary)
    theta = torch.rand(n_boundary, 1, device=device) * 2 * torch.pi
    x_bound = 0.5 + 0.1 * torch.cos(theta)
    y_bound = 0.0 + 0.1 * torch.sin(theta)
    t_bound = torch.rand(n_boundary, 1, device=device)
    
    # 3. NEU: Punkte am linken Rand (Inflow bei x = 0.0)
    x_inflow = torch.zeros(n_inflow, 1, device=device)
    y_inflow = (torch.rand(n_inflow, 1, device=device) - 0.5)
    t_inflow = torch.rand(n_inflow, 1, device=device)
    
    return (x_domain, y_domain, t_domain), (x_bound, y_bound, t_bound), (x_inflow, y_inflow, t_inflow)