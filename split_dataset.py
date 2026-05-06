import os
import shutil
import random

SOURCE_DIR = "/data/sj/tic_4_23/105"
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

face_tic_dir = os.path.join(SOURCE_DIR, "face-tic")
none_dir = os.path.join(SOURCE_DIR, "None")

def get_batches(class_dir):
    batches = sorted(os.listdir(class_dir))
    return [os.path.join(class_dir, b) for b in batches]

face_batches = get_batches(face_tic_dir)
none_batches = get_batches(none_dir)

print(f"face-tic batches: {len(face_batches)}")
print(f"None batches: {len(none_batches)}")

def split_batches(batches):
    random.shuffle(batches)
    n_train = round(len(batches) * TRAIN_RATIO)
    return batches[:n_train], batches[n_train:]

face_train, face_test = split_batches(face_batches)
none_train, none_test = split_batches(none_batches)

print(f"face-tic: train={len(face_train)}, test={len(face_test)}")
print(f"None: train={len(none_train)}, test={len(none_test)}")

train_dir = os.path.join(SOURCE_DIR, "train", "105")
test_dir = os.path.join(SOURCE_DIR, "test", "105")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_lines = []
test_lines = []
batch_idx = 1

for src_path in face_train + none_train:
    dst = os.path.join(train_dir, f"batch_{batch_idx}")
    shutil.move(src_path, dst)
    train_lines.append(f"105/batch_{batch_idx}")
    batch_idx += 1

for src_path in face_test + none_test:
    dst = os.path.join(test_dir, f"batch_{batch_idx}")
    shutil.move(src_path, dst)
    test_lines.append(f"105/batch_{batch_idx}")
    batch_idx += 1

with open(os.path.join(SOURCE_DIR, "train_tic.txt"), "w") as f:
    f.write("\n".join(train_lines) + "\n")

with open(os.path.join(SOURCE_DIR, "test_tic.txt"), "w") as f:
    f.write("\n".join(test_lines) + "\n")

shutil.rmtree(face_tic_dir)
shutil.rmtree(none_dir)

print(f"\nDone. Train: {len(train_lines)}, Test: {len(test_lines)}")
print(f"train_tic.txt and test_tic.txt created.")
