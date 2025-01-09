import os
import time
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

start_time = time.time()

# base_dir = "/data4/louxicheng/EEG_data/seizure/v2.0.3/preprocessed/01_tcp_ar_segment_interval_4_sec_normalized/"
base_dir = "/data4/louxicheng/EEG_data/seizure/v2.0.3/preprocessed/01_tcp_ar_segment_interval_4_sec/"
train_dir = os.path.join(base_dir, "Train")
validation_dir = os.path.join(base_dir, "Validation")


def load_samples(directory):
    samples = []
    for file_name_i in os.listdir(directory):
        if file_name_i.endswith(".npz"):
            samples.append(file_name_i)
    return sorted(samples, key=lambda x: int(x.split('_')[1].split('.')[0]))


train_samples = load_samples(train_dir)
validation_samples = load_samples(validation_dir)

if train_samples:
    max_train_index = int(train_samples[-1].split('_')[1].split('.')[0])
else:
    max_train_index = -1

new_validation_samples = []
for idx, file_name in enumerate(validation_samples, start=max_train_index + 1):
    new_name = f"sample_{idx}.npz"
    shutil.move(os.path.join(validation_dir, file_name), os.path.join(train_dir, new_name))
    new_validation_samples.append(new_name)

all_samples = train_samples + new_validation_samples
all_sample_paths = [os.path.join(train_dir, sample) for sample in all_samples]


def extract_label(file_path):
    return int(dict(np.load(file_path))['label'])


all_labels = [extract_label(path) for path in all_sample_paths]

train_files, val_files = train_test_split(
    all_samples, test_size=0.2, stratify=all_labels, random_state=42
)

for file_name in val_files:
    shutil.move(os.path.join(train_dir, file_name), os.path.join(validation_dir, file_name))

end_time = time.time()
print("完成数据集重新划分！")
print(f"运行时间：{end_time - start_time:.2f} 秒")
