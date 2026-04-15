import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os


# Written by Prof. Ameya D. Jagtap, https://sites.google.com/view/paramintelligencelab/home
# April 15, 2026 at 10 am ET
# =====================
# Download dataset
# =====================
if not os.path.exists('burgers_shock.mat'):
    !wget -q https://raw.githubusercontent.com/ParamIntelligence/PINN_Data_Set/main/burgers_shock.mat

# =====================
# Device
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Load dataset
# =====================
data = scipy.io.loadmat('burgers_shock.mat')

x = data['x'].flatten()[:, None]
t = data['t'].flatten()[:, None]
Exact = np.real(data['usol'])

X, T = np.meshgrid(x, t)
u_ref = Exact.T

# =====================
# PINN Model
# =====================
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
        for i in range(len(self.layers)-1):
            X = self.activation(self.layers[i](X))
        return self.layers[-1](X)

# =====================
# PDE Residual
# =====================
def burgers_residual(model, x, t, nu):
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    return u_t + u * u_x - nu * u_xx

# =====================
# Training Data
# =====================
N_f, N_b, N_i = 7000, 200, 100

x_f = torch.rand((N_f,1))*2 - 1
t_f = torch.rand((N_f,1))

x_b1 = -torch.ones((N_b,1))
x_b2 = torch.ones((N_b,1))
t_b = torch.rand((N_b,1))

x_i = torch.rand((N_i,1))*2 - 1
t_i = torch.zeros((N_i,1))

nu = torch.tensor(0.01/np.pi, dtype=torch.float32)

# Initial condition
def initial_condition(x):
    return -torch.sin(np.pi * x)

# =====================
# Model
# =====================
model = PINN([2, 20, 20, 20, 20, 20, 1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Move to device
x_f, t_f = x_f.to(device), t_f.to(device)
x_b1, x_b2, t_b = x_b1.to(device), x_b2.to(device), t_b.to(device)
x_i, t_i = x_i.to(device), t_i.to(device)
nu = nu.to(device)

# =====================
# Training (Adam)
# =====================
adam_loss_history = []
for epoch in range(1500):
    optimizer.zero_grad()

    f = burgers_residual(model, x_f, t_f, nu)
    loss_f = torch.mean(f**2)

    u_b1 = model(x_b1, t_b)
    u_b2 = model(x_b2, t_b)
    loss_b = torch.mean(u_b1**2) + torch.mean(u_b2**2)

    u_i = model(x_i, t_i)
    loss_i = torch.mean((u_i - initial_condition(x_i))**2)

    loss = loss_f + loss_b + loss_i
    loss.backward()
    optimizer.step()
    adam_loss_history.append(loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6e}")

# =====================
# L-BFGS
# =====================
lbfgs_loss_history = []
optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=1500)

def closure():
    optimizer_lbfgs.zero_grad()

    f = burgers_residual(model, x_f, t_f, nu)
    loss_f = torch.mean(f**2)

    u_b1 = model(x_b1, t_b)
    u_b2 = model(x_b2, t_b)
    loss_b = torch.mean(u_b1**2) + torch.mean(u_b2**2)

    u_i = model(x_i, t_i)
    loss_i = torch.mean((u_i - initial_condition(x_i))**2)

    loss = loss_f + loss_b + loss_i
    loss.backward()
    lbfgs_loss_history.append(loss.item())
    return loss

optimizer_lbfgs.step(closure)

# =====================
# Plot Losses
# =====================
plt.figure()
plt.plot(adam_loss_history, label='Adam')
plt.plot(range(len(adam_loss_history), len(adam_loss_history)+len(lbfgs_loss_history)), lbfgs_loss_history, label='L-BFGS')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss: Adam + L-BFGS')
plt.legend()
plt.show()

# =====================
# Prediction
# =====================
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

X_torch = torch.tensor(X_star[:,0:1], dtype=torch.float32).to(device)
T_torch = torch.tensor(X_star[:,1:2], dtype=torch.float32).to(device)

u_pred = model(X_torch, T_torch).detach().cpu().numpy()
u_pred = u_pred.reshape(T.shape)

# =====================
# Error
# =====================
error_l2 = np.linalg.norm(u_ref - u_pred, 2) / np.linalg.norm(u_ref, 2)
print("Relative L2 Error:", error_l2)

error_field = np.abs(u_ref - u_pred)

# =====================
# Plots
# =====================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Predicted")
plt.imshow(u_pred, extent=[-1,1,0,1], aspect='auto', origin='lower')
plt.colorbar()

plt.subplot(1,3,2)
plt.title("Reference")
plt.imshow(u_ref, extent=[-1,1,0,1], aspect='auto', origin='lower')
plt.colorbar()

plt.subplot(1,3,3)
plt.title("Error")
plt.imshow(error_field, extent=[-1,1,0,1], aspect='auto', origin='lower')
plt.colorbar()

plt.tight_layout()
plt.show()

# =====================
# Slice plots 
# =====================
t_values = [0.25, 0.5, 0.75]
plt.figure(figsize=(15,4))

for i, t_val in enumerate(t_values):
    idx = np.argmin(np.abs(t - t_val))
    plt.subplot(1, len(t_values), i+1)
    plt.plot(x, u_ref[idx,:], 'k-', label='Exact')
    plt.plot(x, u_pred[idx,:], 'r--', label='Pred')
    plt.title(f"t = {t_val}")
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()
