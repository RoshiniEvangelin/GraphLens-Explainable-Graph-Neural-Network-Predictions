import torch
import torch.nn.functional as F
from dataset import CORA
from models.GNNs.sage import SAGE

# -------------------------
# Device
# -------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)

# -------------------------
# Load Cora
# -------------------------
data, citeid = CORA.get_cora_casestudy(0)
data = data.to(device)

print("Nodes:", data.num_nodes)
print("Features:", data.x.shape)
print("Classes:", int(data.y.max()) + 1)
print("Edges:", data.edge_index.shape)

# -------------------------
# Create GraphSAGE
# -------------------------
model = SAGE(
    in_channels=data.x.shape[1],
    hidden_channels=128,
    out_channels=7,
    num_layers=2,
    dropout=0.5
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4
)

# -------------------------
# Training
# -------------------------
for epoch in range(1, 101):

    model.train()

    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    loss = F.cross_entropy(
        out[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()
    optimizer.step()

    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    with torch.no_grad():

        pred = out.argmax(dim=1)

        train_acc = (
            pred[data.train_mask] == data.y[data.train_mask]
        ).float().mean()

        val_acc = (
            pred[data.val_mask] == data.y[data.val_mask]
        ).float().mean()

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.4f} | "
            f"Train Acc: {train_acc.item():.4f} | "
            f"Val Acc: {val_acc.item():.4f}"
        )

# -------------------------
# Test
# -------------------------
model.eval()

with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    test_acc = (
        pred[data.test_mask] == data.y[data.test_mask]
    ).float().mean()

print()
# -------------------------
# Save trained model
# -------------------------
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "in_channels": data.x.shape[1],
        "hidden_channels": 128,
        "out_channels": 7,
        "num_layers": 2,
        "dropout": 0.5,
        "test_accuracy": test_acc.item(),
    },
    "sage_cora.pt"
)

print("Saved model to sage_cora.pt")