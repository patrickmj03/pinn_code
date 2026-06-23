import torch

def get_residual(model, x, y, t, nu=0.01):
    # 1. Eingaben vorbereiten
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    
    inputs = torch.cat([x, y, t], dim=1)
    pred = model(inputs)
    
    # JETZT RECHTSICHER: Dimension [N, 1] bleibt erhalten
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    
    # 2. Hilfsfunktion für Ableitungen
    def grad(outputs, inputs):
        return torch.autograd.grad(outputs.sum(), inputs, create_graph=True)[0]

    # 3. Erste Ableitungen
    u_t, u_x, u_y = grad(u, t), grad(u, x), grad(u, y)
    v_t, v_x, v_y = grad(v, t), grad(v, x), grad(v, y)
    
    # 4. Zweite Ableitungen
    u_xx, u_yy = grad(u_x, x), grad(u_y, y)
    v_xx, v_yy = grad(v_x, x), grad(v_y, y)
    p_x, p_y = grad(p, x), grad(p, y)
    
    # 5. Navier-Stokes Residuen
    f_mass = u_x + v_y
    f_x = u_t + (u * u_x + v * u_y) + p_x - nu * (u_xx + u_yy)
    f_y = v_t + (u * v_x + v * v_y) + p_y - nu * (v_xx + v_yy)
    
    return f_mass, f_x, f_y