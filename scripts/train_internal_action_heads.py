import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class InternalHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(hidden))


class LayerDataset(Dataset):
    def __init__(self, samples, layer: int, chunk_ids: set[int], max_step: int):
        self.items = [
            (
                sample["hidden_by_layer"][layer].squeeze(0).float(),
                sample["teacher_prediction"].squeeze(0).float(),
                int(sample["step_index"]),
                int(sample["chunk_index"]),
            )
            for sample in samples
            if int(sample["chunk_index"]) in chunk_ids
            and int(sample["step_index"]) <= max_step
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    cosine, rel_l2, mae, mse = [], [], [], []
    by_step = {}
    for hidden, target, steps, _ in loader:
        hidden = hidden.to(device)
        target = target.to(device)
        pred = model(hidden)
        flat_pred = pred.flatten(1)
        flat_target = target.flatten(1)
        diff = flat_pred - flat_target
        batch_cos = F.cosine_similarity(flat_pred, flat_target, dim=1)
        batch_rel = diff.norm(dim=1) / flat_target.norm(dim=1).clamp_min(1e-12)
        batch_mae = diff.abs().mean(dim=1)
        batch_mse = diff.square().mean(dim=1)
        cosine.extend(batch_cos.cpu().tolist())
        rel_l2.extend(batch_rel.cpu().tolist())
        mae.extend(batch_mae.cpu().tolist())
        mse.extend(batch_mse.cpu().tolist())
        for i, step in enumerate(steps.tolist()):
            bucket = by_step.setdefault(step, {"cosine": [], "relative_l2": [], "mae": []})
            bucket["cosine"].append(float(batch_cos[i].cpu()))
            bucket["relative_l2"].append(float(batch_rel[i].cpu()))
            bucket["mae"].append(float(batch_mae[i].cpu()))

    def avg(values):
        return float(sum(values) / len(values)) if values else None

    return {
        "num_samples": len(cosine),
        "cosine": avg(cosine),
        "relative_l2": avg(rel_l2),
        "mae": avg(mae),
        "mse": avg(mse),
        "per_step": {
            str(step): {key: avg(values) for key, values in metrics.items()}
            for step, metrics in sorted(by_step.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 18])
    parser.add_argument("--max-step", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    payload = torch.load(args.dataset, map_location="cpu", weights_only=True)
    samples = payload["samples"]
    all_chunks = sorted({int(sample["chunk_index"]) for sample in samples})
    if len(all_chunks) < 2:
        raise ValueError("Need at least two action chunks for train/validation split.")
    shuffled = list(all_chunks)
    random.Random(args.seed).shuffle(shuffled)
    num_val = max(1, round(len(shuffled) * args.val_fraction))
    val_chunks = set(shuffled[:num_val])
    train_chunks = set(shuffled[num_val:])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    report = {
        "dataset": str(args.dataset),
        "layers": args.layers,
        "max_step": args.max_step,
        "train_chunks": sorted(train_chunks),
        "val_chunks": sorted(val_chunks),
        "results": {},
    }
    checkpoint = {"format_version": 1, "config": vars(args), "heads": {}}

    for layer in args.layers:
        train_set = LayerDataset(samples, layer, train_chunks, args.max_step)
        val_set = LayerDataset(samples, layer, val_chunks, args.max_step)
        if not train_set or not val_set:
            raise ValueError(f"Empty split for layer {layer}.")
        hidden_dim = int(train_set[0][0].shape[-1])
        action_dim = int(train_set[0][1].shape[-1])
        model = InternalHead(hidden_dim, action_dim).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        best_loss = float("inf")
        best_state = None
        for epoch in range(args.epochs):
            model.train()
            for hidden, target, _, _ in train_loader:
                hidden = hidden.to(device)
                target = target.to(device)
                loss = F.mse_loss(model(hidden), target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            metrics = evaluate(model, val_loader, device)
            if metrics["mse"] < best_loss:
                best_loss = metrics["mse"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if epoch in {0, 9, 49, 99, args.epochs - 1}:
                print(
                    f"layer={layer} epoch={epoch + 1}/{args.epochs} "
                    f"val_mse={metrics['mse']:.6f} cos={metrics['cosine']:.6f} "
                    f"rel_l2={metrics['relative_l2']:.6f}"
                )
        model.load_state_dict(best_state)
        train_metrics = evaluate(
            model, DataLoader(train_set, batch_size=args.batch_size), device
        )
        val_metrics = evaluate(model, val_loader, device)
        checkpoint["heads"][int(layer)] = {
            "hidden_dim": hidden_dim,
            "action_dim": action_dim,
            "state_dict": best_state,
        }
        report["results"][str(layer)] = {
            "train": train_metrics,
            "validation": val_metrics,
        }

    torch.save(checkpoint, output)
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"checkpoint: {output}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
