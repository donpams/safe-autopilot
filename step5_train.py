import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from bc_model import BCPolicy

torch.manual_seed(0)
d = np.load("pid_dataset.npz")
obs, act, is_val = d["obs"], d["act"], d["is_val"]

Xtr, Ytr = torch.tensor(obs[~is_val]), torch.tensor(act[~is_val])
Xva, Yva = torch.tensor(obs[is_val]), torch.tensor(act[is_val])
print(f"train {len(Xtr)} rows, val {len(Xva)} rows")

train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=256, shuffle=True)

model = BCPolicy()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

EPOCHS = 60
train_hist, val_hist = [], []
best_val, best_state, patience, bad = float("inf"), None, 10, 0

for epoch in range(EPOCHS):
    model.train()
    tot, n = 0.0, 0
    for xb, yb in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        tot += loss.item() * len(xb); n += len(xb)
    train_loss = tot / n

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(Xva), Yva).item()
    train_hist.append(train_loss); val_hist.append(val_loss)
    print(f"epoch {epoch+1:2d}/{EPOCHS}  train {train_loss:.6f}  val {val_loss:.6f}")

    if val_loss < best_val - 1e-6:
        best_val, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        bad += 1
        if bad >= patience:
            print(f"early stop at epoch {epoch+1} (best val {best_val:.6f} at epoch {epoch+1-patience})")
            break

model.load_state_dict(best_state)
torch.save(model.state_dict(), "bc_policy.pt")
print(f"saved bc_policy.pt, best val MSE {best_val:.6f}")

with torch.no_grad():
    pred = model(Xva).numpy()
err = pred - Yva.numpy()
print(f"elevator: val MAE {np.abs(err[:,0]).mean():.4f}, PID action std {Yva[:,0].std():.4f}")
print(f"aileron : val MAE {np.abs(err[:,1]).mean():.4f}, PID action std {Yva[:,1].std():.4f}")

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(train_hist, label="train"); ax.plot(val_hist, label="val")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss"); ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("Behavior cloning: training curve")
plt.tight_layout(); plt.savefig("step5_loss_curve.png", dpi=150)
print("saved step5_loss_curve.png")
