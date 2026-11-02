import copy
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

K, B, LR = 5, 50, 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def make_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(512, 10)
    )


def make_partitions(labels):
    rng = np.random.default_rng(2026)
    by_class = np.stack([
        rng.permutation(np.flatnonzero(labels == c)) for c in range(10)
    ])
    chosen = by_class[:, :1000]
    val_ids = by_class[:, 1000:1200].reshape(-1)
    iid = [chosen[:, 200*k:200*(k+1)].reshape(-1) for k in range(K)]
    shards = chosen.reshape(10, 1000)
    order = np.random.default_rng(2026).permutation(10).reshape(K, 2)
    skew = [shards[pair].reshape(-1) for pair in order]
    return {"IID": iid, "label_skew": skew}, val_ids


transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)
])
datasets.CIFAR10.url = (
    "https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/"
    "resolve/c6e562aecc3a2dd5ee13f9d91630d81219c6a126/"
    "cifar-10-python.tar.gz"
)
train = datasets.CIFAR10("./data", train=True, download=True,
                          transform=transform)
test = datasets.CIFAR10("./data", train=False, download=True,
                         transform=transform)
labels = np.asarray(train.targets)
partitions, val_ids = make_partitions(labels)
val_loader = DataLoader(Subset(train, val_ids.tolist()), batch_size=256)
test_loader = DataLoader(test, batch_size=256)


def forever(loader):
    while True:
        yield from loader


def client_streams(partition, seed):
    return [forever(DataLoader(
        Subset(train, ids.tolist()), batch_size=B, shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(1000 + 100*seed + k)
    )) for k, ids in enumerate(partitions[partition])]


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = 0
    for x, y in loader:
        pred = model(x.to(DEVICE)).argmax(dim=1)
        correct += (pred == y.to(DEVICE)).sum().item()
    return 100.0 * correct / len(loader.dataset)


def local_update(global_model, stream, H):
    local = copy.deepcopy(global_model).train()
    optimizer = torch.optim.SGD(local.parameters(), lr=LR)

    for _ in range(H):
        x, y = next(stream)
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(local(x), y)
        loss.backward()
        optimizer.step()

    return {name: value.detach().clone()
            for name, value in local.state_dict().items()}


def aggregate(states, counts):
    weights = [n / sum(counts) for n in counts]

    averaged = {}
    for name in states[0]:
        averaged[name] = sum(
            weight * state[name]
            for weight, state in zip(weights, states)
        )

    return averaged


def run(partition, H, S=200, seed=0):
    assert S == 200 and S % H == 0
    torch.manual_seed(seed)
    model = make_model().to(DEVICE)
    streams = client_streams(partition, seed)
    counts = [len(ids) for ids in partitions[partition]]
    log = [{"steps": 0, "round": 0, "examples": 0,
            "val_acc": accuracy(model, val_loader)}]
    for r in range(1, S // H + 1):
        states = [local_update(model, stream, H) for stream in streams]
        model.load_state_dict(aggregate(states, counts))
        s = r * H
        if s % 20 == 0 or (r % 5 == 0 and r <= S // 20):
            log.append({"steps": s, "round": r, "examples": K*B*s,
                        "val_acc": accuracy(model, val_loader)})
    return {"partition": partition, "H": H, "S": S, "seed": seed,
            "log": log, "test_acc": accuracy(model, test_loader)}


def one_step_check():
    torch.manual_seed(17)
    base = make_model().to(DEVICE)
    batches = [(torch.randn(B, 3, 32, 32), torch.randint(10, (B,)))
               for _ in range(K)]
    states = [local_update(base, iter([batch]), 1) for batch in batches]
    averaged = aggregate(states, [B] * K)
    reference = copy.deepcopy(base)
    optimizer = torch.optim.SGD(reference.parameters(), lr=LR)
    x = torch.cat([batch[0] for batch in batches]).to(DEVICE)
    y = torch.cat([batch[1] for batch in batches]).to(DEVICE)
    optimizer.zero_grad()
    nn.functional.cross_entropy(reference(x), y).backward()
    optimizer.step()
    return max((averaged[name] - value).abs().max().item()
               for name, value in reference.state_dict().items())


if __name__ == "__main__":
    print("H=1 maximum parameter difference:", one_step_check())
    for partition, clients in partitions.items():
        print(partition, [(len(ids), len(np.unique(labels[ids])))
                          for ids in clients])
    S = 200  # Fixed budget for every configuration.
    results = []
    for partition in partitions:
        for H in (1, 5, 20):
            result = run(partition, H, S=S)
            results.append(result)
            with open("hw2_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print(partition, H, result["log"][-1], result["test_acc"])
