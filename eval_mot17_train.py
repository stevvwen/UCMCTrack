import os
import shutil
from eval.eval import eval

# Paths
dataset_path = "/usr/local/data/swen14/Workspace/MOT17/train"  # GT location
train_results_dir = "train"  # Source: raw result .txt files
out_path = "output/mot17"
exp_name = "train"

# Create the output directory for train results
exp_dir = os.path.join(out_path, exp_name)
os.makedirs(exp_dir, exist_ok=True)

# Copy all result files from train/ into the expected tracker directory
for fname in os.listdir(train_results_dir):
    if fname.endswith(".txt"):
        src = os.path.join(train_results_dir, fname)
        dst = os.path.join(exp_dir, fname)
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")

# Get all sequence names from the result files
seq_names = sorted([
    f.replace(".txt", "")
    for f in os.listdir(exp_dir)
    if f.endswith(".txt") and f.startswith("MOT17")
])

print(f"\nSequences to evaluate ({len(seq_names)}):")
for s in seq_names:
    print(f"  {s}")

# Generate seqmap file
seqmap_path = os.path.join(exp_dir, "train_seqmap.txt")
with open(seqmap_path, "w") as f:
    f.write("name\n")
    for seq in seq_names:
        f.write(f"{seq}\n")

print(f"\nSeqmap written to {seqmap_path}")
print(f"Evaluating against GT in {dataset_path}\n")

# Run evaluation (half_eval=False means full gt.txt is used)
HOTA, IDF1, MOTA, AssA = eval(dataset_path, out_path, seqmap_path, exp_name, 1, False)

print(f"\n{'='*50}")
print(f"MOT17 Train Evaluation Results")
print(f"{'='*50}")
print(f"  HOTA: {HOTA}")
print(f"  IDF1: {IDF1}")
print(f"  MOTA: {MOTA}")
print(f"  AssA: {AssA}")
print(f"{'='*50}")
