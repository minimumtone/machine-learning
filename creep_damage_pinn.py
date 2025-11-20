"""
PINN for Creep/Damage Model Parameter Estimation

This code implements a Physics-Informed Neural Network (PINN) for estimating
parameters in a creep and damage constitutive model. The training is performed
in three phases:
1. Phase 1: Fit NN to data (physics params frozen)
2. Phase 2: Optimize physics params only (NN frozen)
3. Phase 3: Joint fine-tuning with L-BFGS

Refactored to:
- Ensure reproducibility with seed setting
- Eliminate hardcoded values
- Consolidate physics equations
- Externalize loss weights
- Stabilize Phase 2 optimization
- Track and visualize parameter history
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================================
# ============================================================
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

R = 8.314  # Gas constant

# Normalization parameters
T_NORM_OFFSET = 800.0  # Temperature normalization offset (K)
T_NORM_SCALE = 200.0   # Temperature normalization scale (K)
T_MAX_SCALE = 200.0    # Time normalization scale

LOSS_WEIGHT_PHYS_EPS = 1e3  # Weight for physics residual (strain)
LOSS_WEIGHT_PHYS_D = 1e3    # Weight for physics residual (damage)
LOSS_WEIGHT_DATA_PHASE3 = 100.0  # Weight for data loss in Phase 3
LOSS_WEIGHT_DATA_PHASE2 = 0.1    # Weight for data loss in Phase 2 (stabilization)

PHASE1_MAX_EPOCHS = 50000
PHASE1_LOSS_THRESHOLD = 1e-5
PHASE1_LR = 1e-3

PHASE2_MAX_EPOCHS = 20000
PHASE2_LR = 1e-2  # Can be reduced to 1e-3 if unstable
PHASE2_PRINT_INTERVAL = 1000

PHASE3_LR = 0.1
PHASE3_HISTORY_SIZE = 50

params_true = {
    "A1": 0.035, "A2": 0.0010, "A3": 2.5e-4, "C1": 1.0e-5,
    "tau": 22.0, "t0": 110.0,  "Q": 2.0e5
}

# ============================================================
# ============================================================
def compute_rates(t, T, A1, A2, A3, C1, tau, t0, Q, use_softplus=False, beta=0.5):
    """
    Compute strain rate and damage rate based on physics equations.
    
    This function consolidates the physics equations used in both data generation
    and physics loss calculation to avoid duplication.
    
    Args:
        t: Time (Tensor or scalar)
        T: Temperature (Tensor or scalar)
        A1, A2, A3, C1, tau, t0, Q: Physics parameters
        use_softplus: If True, use softplus for smooth transition at t0 (for gradients)
        beta: Softplus beta parameter (only used if use_softplus=True)
    
    Returns:
        deps_dt: Strain rate
        dD_dt: Damage rate
    """
    K = torch.exp(-Q / (R * T)) if torch.is_tensor(T) else np.exp(-Q / (R * T))
    
    if use_softplus:
        tt = F.softplus(t - t0, beta=beta)
    else:
        if torch.is_tensor(t):
            tt = torch.clamp(t - t0, min=0.0)
        else:
            tt = max(0, t - t0)
    
    if torch.is_tensor(t):
        deps_dt = K * ((A1/tau) * torch.exp(-t/tau) + A2 + 4*A3*(tt**3))
    else:
        deps_dt = K * ((A1/tau) * np.exp(-t/tau) + A2 + 4*A3*(tt**3))
    
    dD_dt = K * (C1 * (tt**3))
    
    return deps_dt, dD_dt

def generate_data():
    """
    Generate synthetic training data using the true parameters.
    Uses the consolidated physics kernel to ensure consistency.
    """
    Ts = [873.0, 923.0, 973.0]
    t_line = torch.linspace(0.0, 200.0, 200, device=device)

    t_list, T_list, eps_list, D_list = [], [], [], []

    for T_val in Ts:
        eps = torch.zeros_like(t_line)
        D   = torch.zeros_like(t_line)
        dt  = t_line[1] - t_line[0]

        for i in range(1, len(t_line)):
            ti = t_line[i-1]
            
            deps, dD = compute_rates(
                ti.item(), T_val,
                params_true["A1"], params_true["A2"], params_true["A3"],
                params_true["C1"], params_true["tau"], params_true["t0"],
                params_true["Q"], use_softplus=False
            )

            eps[i] = eps[i-1] + deps * dt
            D[i]   = D[i-1]   + dD   * dt

        # Add noise
        eps_n = eps * (1 + 0.01 * torch.randn_like(eps))
        D_n   = D   * (1 + 0.02 * torch.randn_like(D))
        D_n   = torch.clamp(D_n, 0, 0.99)

        t_list.append(t_line)
        T_list.append(torch.full_like(t_line, T_val))
        eps_list.append(eps_n)
        D_list.append(D_n)

    return (torch.cat(t_list).view(-1,1),
            torch.cat(T_list).view(-1,1),
            torch.cat(eps_list).view(-1,1),
            torch.cat(D_list).view(-1,1))

t_train, T_train, eps_train, D_train = generate_data()
y_eps_train = torch.log(eps_train + 1.0) # Target for NN

# ============================================================
# 1. PINN Model Definition
# ============================================================
class CreepPINN(nn.Module):
    def __init__(self, t_norm_offset=T_NORM_OFFSET, t_norm_scale=T_NORM_SCALE):
        super().__init__()
        
        self.t_norm_offset = t_norm_offset
        self.t_norm_scale = t_norm_scale
        
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 2)
        )

        self.log_A1  = nn.Parameter(torch.tensor(np.log(0.05), dtype=torch.float32))
        self.log_A2  = nn.Parameter(torch.tensor(np.log(0.002), dtype=torch.float32))
        self.log_A3  = nn.Parameter(torch.tensor(np.log(1.0e-4), dtype=torch.float32))
        self.log_C1  = nn.Parameter(torch.tensor(np.log(0.5e-5), dtype=torch.float32))
        self.log_tau = nn.Parameter(torch.tensor(np.log(15.0), dtype=torch.float32))
        self.t0      = nn.Parameter(torch.tensor(130.0, dtype=torch.float32))
        self.log_Q   = nn.Parameter(torch.tensor(np.log(1.8e5), dtype=torch.float32))

    def get_phys_params(self):
        """Return physics parameters (exponential of log parameters)."""
        return (torch.exp(self.log_A1), torch.exp(self.log_A2),
                torch.exp(self.log_A3), torch.exp(self.log_C1),
                torch.exp(self.log_tau), self.t0, torch.exp(self.log_Q))

    def forward(self, t, T):
        """Forward pass through the network."""
        t_n = t / T_MAX_SCALE
        T_n = (T - self.t_norm_offset) / self.t_norm_scale

        out = self.net(torch.cat([t_n, T_n], dim=1))
        y_eps = out[:, 0:1]
        D_raw = out[:, 1:2]

        eps = torch.exp(y_eps) - 1.0
        D   = torch.sigmoid(D_raw)
        return eps, D, y_eps, D_raw

model = CreepPINN().to(device)

# ============================================================
# 2. Loss Functions
# ============================================================
def calc_data_loss(model, t, T, y_eps_true, D_true):
    _, D_pred, y_eps_pred, _ = model(t, T)
    loss_e = torch.mean((y_eps_pred - y_eps_true)**2)
    loss_d = torch.mean((D_pred - D_true)**2)
    return loss_e + loss_d

def calc_phys_loss(model, t, T, weight_eps=LOSS_WEIGHT_PHYS_EPS, 
                   weight_D=LOSS_WEIGHT_PHYS_D):
    """
    Calculate physics-informed loss using consolidated physics kernel.
    
    Args:
        model: PINN model
        t: Time tensor
        T: Temperature tensor
        weight_eps: Weight for strain residual
        weight_D: Weight for damage residual
    """
    t = t.clone().requires_grad_(True)
    eps_pred, D_pred, y_eps_pred, _ = model(t, T)

    dt_eps = torch.autograd.grad(y_eps_pred, t, torch.ones_like(y_eps_pred), 
                                  create_graph=True)[0]
    dt_D   = torch.autograd.grad(D_pred, t, torch.ones_like(D_pred), 
                                  create_graph=True)[0]

    deps_dt = torch.exp(y_eps_pred) * dt_eps
    dD_dt   = dt_D

    A1, A2, A3, C1, tau, t0, Q = model.get_phys_params()
    
    rhs_eps, rhs_D = compute_rates(
        t, T, A1, A2, A3, C1, tau, t0, Q, 
        use_softplus=True, beta=0.5
    )

    res_eps = (deps_dt - rhs_eps) * weight_eps
    res_D   = (dD_dt   - rhs_D)   * weight_D

    return torch.mean(res_eps**2 + res_D**2)

# ============================================================
# ============================================================
def train_phase1(model, t_train, T_train, y_eps_train, D_train):
    """
    Phase 1: Fit NN to data (physics params frozen).
    
    Returns:
        param_history: Dictionary tracking parameter values over epochs
    """
    print("=== Phase 1: Fitting NN to Data (Physics Params Frozen) ===")
    
    for param in [model.log_A1, model.log_A2, model.log_A3, model.log_C1,
                  model.log_tau, model.t0, model.log_Q]:
        param.requires_grad = False
    
    optimizer_nn = optim.Adam(model.net.parameters(), lr=PHASE1_LR)
    
    param_history = {
        'epoch': [],
        'loss': [],
        'A1': [], 'A2': [], 'A3': [], 'C1': [],
        'tau': [], 't0': [], 'Q': []
    }

    for epoch in range(PHASE1_MAX_EPOCHS):
        optimizer_nn.zero_grad()
        loss = calc_data_loss(model, t_train, T_train, y_eps_train, D_train)
        loss.backward()
        optimizer_nn.step()

        current_loss = loss.item()
        
        if epoch % 100 == 0:
            A1, A2, A3, C1, tau, t0, Q = model.get_phys_params()
            param_history['epoch'].append(epoch)
            param_history['loss'].append(current_loss)
            param_history['A1'].append(A1.item())
            param_history['A2'].append(A2.item())
            param_history['A3'].append(A3.item())
            param_history['C1'].append(C1.item())
            param_history['tau'].append(tau.item())
            param_history['t0'].append(t0.item())
            param_history['Q'].append(Q.item())

        if epoch % 1000 == 0:
            print(f"Ep {epoch}: Data Loss = {current_loss:.6f}")

        if current_loss < PHASE1_LOSS_THRESHOLD:
            print(f"\n[Converged] Loss reached {current_loss:.6e} at epoch {epoch}. Moving to Phase 2.")
            break
    
    for param in [model.log_A1, model.log_A2, model.log_A3, model.log_C1,
                  model.log_tau, model.t0, model.log_Q]:
        param.requires_grad = True
    
    return param_history

def train_phase2(model, t_train, T_train, y_eps_train, D_train):
    """
    Phase 2: Optimize physics parameters only (NN frozen).
    Includes data loss for stabilization.
    
    Returns:
        param_history: Dictionary tracking parameter values over epochs
    """
    print("\n=== Phase 2: Optimizing Physics Parameters Only (NN Frozen) ===")
    
    for param in model.net.parameters():
        param.requires_grad = False

    phys_params = [model.log_A1, model.log_A2, model.log_A3, model.log_C1,
                   model.log_tau, model.t0, model.log_Q]
    optimizer_phys = optim.Adam(phys_params, lr=PHASE2_LR)
    
    param_history = {
        'epoch': [],
        'loss': [],
        'loss_phys': [],
        'loss_data': [],
        'A1': [], 'A2': [], 'A3': [], 'C1': [],
        'tau': [], 't0': [], 'Q': []
    }

    for epoch in range(PHASE2_MAX_EPOCHS):
        optimizer_phys.zero_grad()
        
        loss_phys = calc_phys_loss(model, t_train, T_train)
        loss_data = calc_data_loss(model, t_train, T_train, y_eps_train, D_train)
        loss = loss_phys + LOSS_WEIGHT_DATA_PHASE2 * loss_data
        
        loss.backward()
        optimizer_phys.step()
        
        if epoch % 100 == 0:
            A1, A2, A3, C1, tau, t0, Q = model.get_phys_params()
            param_history['epoch'].append(epoch)
            param_history['loss'].append(loss.item())
            param_history['loss_phys'].append(loss_phys.item())
            param_history['loss_data'].append(loss_data.item())
            param_history['A1'].append(A1.item())
            param_history['A2'].append(A2.item())
            param_history['A3'].append(A3.item())
            param_history['C1'].append(C1.item())
            param_history['tau'].append(tau.item())
            param_history['t0'].append(t0.item())
            param_history['Q'].append(Q.item())

        if epoch % PHASE2_PRINT_INTERVAL == 0:
            A1, A2, A3, C1, tau, t0, Q = model.get_phys_params()
            print(f"Ep {epoch}: Total Loss = {loss.item():.5f} | "
                  f"Phys Loss = {loss_phys.item():.5f} | "
                  f"Data Loss = {loss_data.item():.5f}")
            print(f"  A2={A2.item():.5f}, t0={t0.item():.1f}, Q={Q.item():.2e}")
    
    for param in model.net.parameters():
        param.requires_grad = True
    
    return param_history

def train_phase3(model, t_train, T_train, y_eps_train, D_train):
    """
    Phase 3: Joint fine-tuning with L-BFGS.
    
    Returns:
        param_history: Dictionary tracking parameter values
    """
    print("\n=== Phase 3: Joint Fine-tuning (L-BFGS) ===")
    
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=PHASE3_LR,
                                  history_size=PHASE3_HISTORY_SIZE, 
                                  line_search_fn="strong_wolfe")
    
    param_history = {
        'iteration': [],
        'loss': [],
        'loss_data': [],
        'loss_phys': [],
        'A1': [], 'A2': [], 'A3': [], 'C1': [],
        'tau': [], 't0': [], 'Q': []
    }
    
    iteration = [0]
    
    def closure():
        optimizer_lbfgs.zero_grad()
        loss_d = calc_data_loss(model, t_train, T_train, y_eps_train, D_train)
        loss_p = calc_phys_loss(model, t_train, T_train)
        loss = loss_d * LOSS_WEIGHT_DATA_PHASE3 + loss_p
        loss.backward()
        
        A1, A2, A3, C1, tau, t0, Q = model.get_phys_params()
        param_history['iteration'].append(iteration[0])
        param_history['loss'].append(loss.item())
        param_history['loss_data'].append(loss_d.item())
        param_history['loss_phys'].append(loss_p.item())
        param_history['A1'].append(A1.item())
        param_history['A2'].append(A2.item())
        param_history['A3'].append(A3.item())
        param_history['C1'].append(C1.item())
        param_history['tau'].append(tau.item())
        param_history['t0'].append(t0.item())
        param_history['Q'].append(Q.item())
        
        iteration[0] += 1
        
        return loss

    optimizer_lbfgs.step(closure)
    final_loss = closure()
    print(f"Final Loss: {final_loss.item():.5f}")
    
    return param_history

# ============================================================
# ============================================================
history_phase1 = train_phase1(model, t_train, T_train, y_eps_train, D_train)
history_phase2 = train_phase2(model, t_train, T_train, y_eps_train, D_train)
history_phase3 = train_phase3(model, t_train, T_train, y_eps_train, D_train)

# ============================================================
# Result Verification
# ============================================================
est = model.get_phys_params()
labels = ["A1", "A2", "A3", "C1", "tau", "t0", "Q"]
trues  = [params_true[k] for k in labels]
vals   = [p.item() for p in est]

print("\n=== Final Parameter Estimation ===")
print(f"{'Param':<5} | {'Estimated':<12} | {'True':<12} | {'Error %':<8}")
print("-" * 45)
for label, v, t in zip(labels, vals, trues):
    err = abs(v - t) / t * 100
    print(f"{label:<5} | {v:<12.5g} | {t:<12.5g} | {err:<8.2f}")

# ============================================================
# ============================================================
model.eval()
with torch.no_grad():
    eps_pred, D_pred, _, _ = model(t_train, T_train)

    mask = (T_train.flatten() == 923.0)
    fig1 = plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(t_train[mask].cpu(), eps_train[mask].cpu(), 'k.', alpha=0.3, label='Data')
    plt.plot(t_train[mask].cpu(), eps_pred[mask].cpu(), 'r-', label='PINN Fit')
    plt.xlabel('Time')
    plt.ylabel('Strain')
    plt.title("Strain (923K)")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(t_train[mask].cpu(), D_train[mask].cpu(), 'k.', alpha=0.3, label='Data')
    plt.plot(t_train[mask].cpu(), D_pred[mask].cpu(), 'b-', label='PINN Fit')
    plt.xlabel('Time')
    plt.ylabel('Damage')
    plt.title("Damage (923K)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/machine-learning/creep_damage_fit.png', dpi=150)
    print("\nSaved prediction plot to: creep_damage_fit.png")

    fig2 = plt.figure(figsize=(15, 10))
    
    param_names = ['A1', 'A2', 'A3', 'C1', 'tau', 't0', 'Q']
    
    for idx, param_name in enumerate(param_names):
        plt.subplot(3, 3, idx+1)
        
        if len(history_phase1['epoch']) > 0:
            plt.plot(history_phase1['epoch'], history_phase1[param_name], 
                    'b-', alpha=0.5, label='Phase 1')
        
        if len(history_phase2['epoch']) > 0:
            phase2_offset = history_phase1['epoch'][-1] if len(history_phase1['epoch']) > 0 else 0
            phase2_x = [x + phase2_offset for x in history_phase2['epoch']]
            plt.plot(phase2_x, history_phase2[param_name], 
                    'g-', alpha=0.7, label='Phase 2')
        
        if len(history_phase3['iteration']) > 0:
            phase3_offset = phase2_x[-1] if len(history_phase2['epoch']) > 0 else 0
            phase3_x = [x*100 + phase3_offset for x in history_phase3['iteration']]
            plt.plot(phase3_x, history_phase3[param_name], 
                    'r-', alpha=0.7, label='Phase 3')
        
        plt.axhline(y=params_true[param_name], color='k', linestyle='--', 
                   linewidth=2, label='True Value')
        
        plt.xlabel('Training Step')
        plt.ylabel(param_name)
        plt.title(f'{param_name} Convergence')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 8)
    if len(history_phase1['epoch']) > 0:
        plt.semilogy(history_phase1['epoch'], history_phase1['loss'], 
                    'b-', alpha=0.5, label='Phase 1')
    if len(history_phase2['epoch']) > 0:
        phase2_offset = history_phase1['epoch'][-1] if len(history_phase1['epoch']) > 0 else 0
        phase2_x = [x + phase2_offset for x in history_phase2['epoch']]
        plt.semilogy(phase2_x, history_phase2['loss'], 
                    'g-', alpha=0.7, label='Phase 2')
    if len(history_phase3['iteration']) > 0:
        phase3_offset = phase2_x[-1] if len(history_phase2['epoch']) > 0 else 0
        phase3_x = [x*100 + phase3_offset for x in history_phase3['iteration']]
        plt.semilogy(phase3_x, history_phase3['loss'], 
                    'r-', alpha=0.7, label='Phase 3')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Loss Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/machine-learning/creep_damage_param_history.png', dpi=150)
    print("Saved parameter history plot to: creep_damage_param_history.png")

print("\n=== Training Complete ===")
