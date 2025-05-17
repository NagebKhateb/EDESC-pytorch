import torch


# Geometric Regularization

# ضمان تعامد الفضاءات الجزئية
class D_constraint1(torch.nn.Module):
    def __init__(self):
        super(D_constraint1, self).__init__()

    def forward(self, d):
        device = d.device
        I = torch.eye(d.shape[1], device=device)
        loss_d1_constraint = torch.norm(torch.mm(d.t(), d) * I - I)
        return 1e-3 * loss_d1_constraint

# ضمان استقلالية الفضاءات الجزئية
class D_constraint2(torch.nn.Module):
    def __init__(self):
        super(D_constraint2, self).__init__()

    def forward(self, d, dim, n_clusters):
        device = d.device
        S = torch.ones(d.shape[1], d.shape[1], device=device)
        zero = torch.zeros(dim, dim, device=device)
        for i in range(n_clusters):
            S[i*dim:(i+1)*dim, i*dim:(i+1)*dim] = zero
        loss_d2_constraint = torch.norm(torch.mm(d.t(), d) * S)
        return 1e-3 * loss_d2_constraint



