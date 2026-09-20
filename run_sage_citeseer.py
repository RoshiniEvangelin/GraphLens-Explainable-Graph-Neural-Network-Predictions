import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from models.GNNs.sage import SAGE


# --------------------------------------------------
# Device
# --------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)


# --------------------------------------------------
# Load CiteSeer
# --------------------------------------------------
dataset = Planetoid(
    root="dataset/citeseer",
    name="CiteSeer"
)

data = dataset[0]

print("Dataset: CiteSeer")
print("Nodes:", data.num_nodes)
print("Features:", data.num_features)
print("Classes:", dataset.num_classes)
print("Edges:", data.edge_index.shape)

print("Train nodes:", int(data.train_mask.sum()))
print("Validation nodes:", int(data.val_mask.sum()))
print("Test nodes:", int(data.test_mask.sum()))


# --------------------------------------------------
# Move data to device
# --------------------------------------------------
data = data.to(device)


# --------------------------------------------------
# GraphSAGE
# --------------------------------------------------
model = SAGE(
    in_channels=data.num_features,
    hidden_channels=128,
    out_channels=dataset.num_classes,
    num_layers=2,
    dropout=0.5
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4
)


# --------------------------------------------------
# Training
# --------------------------------------------------
def accuracy(mask):
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        return (
            (pred[mask] == data.y[mask])
            .float()
            .mean()
            .item()
        )


best_val = 0.0
best_state = None

print()
print("Training GraphSAGE...")
print("-" * 60)

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

    train_acc = accuracy(data.train_mask)
    val_acc = accuracy(data.val_mask)

    if val_acc > best_val:
        best_val = val_acc
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )


# --------------------------------------------------
# Restore best model
# --------------------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)

model = model.to(device)

test_acc = accuracy(data.test_mask)

print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Best Validation Accuracy: {best_val:.4f}")
print(f"Test Accuracy:            {test_acc:.4f}")
print("=" * 60)


# --------------------------------------------------
# Save model
# --------------------------------------------------
torch.save(
    model.state_dict(),
    "sage_citeseer.pt"
)

print()
print("Saved model to: sage_citeseer.pt")
