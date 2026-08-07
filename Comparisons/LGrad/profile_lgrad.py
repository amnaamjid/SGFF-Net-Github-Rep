import os
import time

import torch
import torch.nn as nn

from thop import profile, clever_format

from model.lgrad_model import LGradClassifier


# ==========================================================
# CONFIG
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = "LGrad_best_01.pth"
batch_size = 32
input_size = (1, 3, 224, 224)


# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading Model...")

model = LGradClassifier()

checkpoint_data = torch.load(
    checkpoint,
    map_location=device
)

model.load_state_dict(
    checkpoint_data["model_state_dict"]
)

model.to(device)

model.eval()

print("Model Loaded Successfully")


# ==========================================================
# PARAMETER COUNT
# ==========================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

frozen_params = total_params - trainable_params


print("\n====================================")
print("PARAMETER INFORMATION")
print("====================================")

print(f"Total Parameters      : {total_params:,}")

print(f"Trainable Parameters  : {trainable_params:,}")

print(f"Frozen Parameters     : {frozen_params:,}")

print(f"Total Parameters (M)  : {total_params/1e6:.3f} M")

print(f"Trainable Params (M)  : {trainable_params/1e6:.3f} M")


# ==========================================================
# MODEL SIZE
# ==========================================================

model_size = (
    os.path.getsize(checkpoint)
    /
    (1024 * 1024)
)

print("\n====================================")
print("MODEL STORAGE")
print("====================================")

print(f"Checkpoint Size : {model_size:.2f} MB")


# ==========================================================
# FLOPS / MACS
# ==========================================================

dummy = torch.randn(
    input_size
).to(device)

print("\n====================================")
print("COMPUTATIONAL COMPLEXITY")
print("====================================")

macs, params = profile(
    model,
    inputs=(dummy,),
    verbose=False
)

flops = macs * 2

macs_str, params_str = clever_format(
    [macs, params],
    "%.3f"
)

flops_str = clever_format(
    [flops],
    "%.3f"
)[0]

print(f"MACs  : {macs_str}")

print(f"FLOPs : {flops_str}")


# ==========================================================
# LATENCY
# ==========================================================

print("\n====================================")
print("INFERENCE LATENCY")
print("====================================")

warmup = 10

runs = 100

dummy = torch.randn(
    input_size
).to(device)


with torch.no_grad():

    for _ in range(warmup):

        _ = model(dummy)


if torch.cuda.is_available():

    torch.cuda.synchronize()


start = time.time()

with torch.no_grad():

    for _ in range(runs):

        _ = model(dummy)


if torch.cuda.is_available():

    torch.cuda.synchronize()

end = time.time()


latency = (
    (end - start)
    /
    runs
)

fps = 1 / latency


print(f"Average Latency : {latency*1000:.4f} ms/image")

print(f"Throughput      : {fps:.2f} FPS")


# ==========================================================
# GPU MEMORY
# ==========================================================

if torch.cuda.is_available():

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():

        _ = model(dummy)

    torch.cuda.synchronize()

    memory = (
        torch.cuda.max_memory_allocated()
        /
        (1024**2)
    )

    print("\n====================================")
    print("GPU MEMORY")
    print("====================================")

    print(f"Peak GPU Memory : {memory:.2f} MB")


print("\n====================================")
print("PROFILE COMPLETED")
print("====================================")
