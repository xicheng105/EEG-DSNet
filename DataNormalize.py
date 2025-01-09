import os
import time
import shutil
import numpy as np

start_time = time.time()

base_path = "/data4/louxicheng/EEG_data/seizure/v2.0.3/preprocessed/01_tcp_ar_segment_interval_4_sec/"
save_base_path = "/data4/louxicheng/EEG_data/seizure/v2.0.3/preprocessed/01_tcp_ar_segment_interval_4_sec_normalized/"

if os.path.exists(save_base_path):
    shutil.rmtree(save_base_path)
os.makedirs(save_base_path, exist_ok=True)

folders = ["Train", "Validation", "Test"]

max_value = -np.inf
min_value = np.inf

for folder in folders:
    folder_path = os.path.join(base_path, folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(".npz"):
            file_path = os.path.join(folder_path, filename)

            data = np.load(file_path)
            eeg_data = data['segments']

            max_value = max(max_value, eeg_data.max())
            min_value = min(min_value, eeg_data.min())

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    save_folder_path = os.path.join(save_base_path, folder)
    os.makedirs(save_folder_path, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".npz"):
            file_path = os.path.join(folder_path, filename)
            save_file_path = os.path.join(save_folder_path, filename)

            data = np.load(file_path)
            eeg_data = data['segments']
            label = data['label']

            normalized_data = (eeg_data - min_value) / (max_value - min_value)

            np.savez(save_file_path, segments=normalized_data, label=label)

end_time = time.time()

print(f"Normalization complete. Time taken: {end_time - start_time:.2f} seconds.")
print(f"Data saved to {save_base_path}")
