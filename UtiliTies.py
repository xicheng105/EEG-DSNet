import os
import random
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfilt
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, f1_score
from spikingjelly.activation_based import neuron, surrogate, layer, functional


def get_all_session_paths(RootPath):
    session_paths = []
    reference_type_count = {}
    all_patients = os.listdir(RootPath)
    for patient in all_patients:
        patient_sessions = os.listdir(os.path.join(RootPath, patient))
        for patient_session in patient_sessions:
            reference_types = os.listdir(os.path.join(RootPath, patient, patient_session))
            for reference_type in reference_types:
                if reference_type not in reference_type_count:
                    reference_type_count[reference_type] = 1
                else:
                    reference_type_count[reference_type] += 1
                files = os.listdir(os.path.join(RootPath, patient, patient_session, reference_type))
                sessions = []
                for file in files:
                    if file.endswith('.edf'):
                        sessions.append(file.split('.')[0])
                for session in sessions:
                    session_paths.append(
                        os.path.join(RootPath, patient, patient_session, reference_type, session + '.edf')
                    )
    return session_paths, all_patients, reference_type_count


def get_channels_from_raw(raw, ReferenceType='01_tcp_ar'):
    if ReferenceType == '01_tcp_ar':
        montage_1 = [
            'EEG FP1-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG FP2-REF', 'EEG F8-REF', 'EEG T4-REF',
            'EEG T6-REF', 'EEG A1-REF', 'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF',
            'EEG FP1-REF', 'EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF', 'EEG FP2-REF', 'EEG F4-REF', 'EEG C4-REF',
            'EEG P4-REF'
        ]
        montage_2 = [
            'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG O1-REF', 'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF',
            'EEG O2-REF', 'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF', 'EEG A2-REF',
            'EEG F3-REF', 'EEG C3-REF', 'EEG P3-REF', 'EEG O1-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF',
            'EEG O2-REF'
        ]
    elif ReferenceType == '03_tcp_ar_a':
        montage_1 = [
            'EEG FP1-REF', 'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG FP2-REF', 'EEG F8-REF', 'EEG T4-REF',
            'EEG T6-REF', 'EEG T3-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG FP1-REF', 'EEG F3-REF',
            'EEG C3-REF', 'EEG P3-REF', 'EEG FP2-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF'
        ]
        montage_2 = [
            'EEG F7-REF', 'EEG T3-REF', 'EEG T5-REF', 'EEG O1-REF', 'EEG F8-REF', 'EEG T4-REF', 'EEG T6-REF',
            'EEG O2-REF', 'EEG C3-REF', 'EEG CZ-REF', 'EEG C4-REF', 'EEG T4-REF', 'EEG F3-REF', 'EEG C3-REF',
            'EEG P3-REF', 'EEG O1-REF', 'EEG F4-REF', 'EEG C4-REF', 'EEG P4-REF', 'EEG O2-REF'
        ]
    else:
        raise ValueError("Invalid ReferenceType. Expected '01_tcp_ar' or '03_tcp_ar_a'.")
    montage_1_indices = [raw.ch_names.index(ch) for ch in montage_1]
    montage_2_indices = [raw.ch_names.index(ch) for ch in montage_2]
    try:
        signals_1 = raw.get_data(picks=montage_1_indices)
        signals_2 = raw.get_data(picks=montage_2_indices)
    except ValueError:
        print('Something is wrong when reading channels of the raw EEG signal')
        flag_wrong = True
        return flag_wrong, 0
    else:
        flag_wrong = False
    return flag_wrong, signals_1 - signals_2


def butter_bandpass_filter(Data, LowCut, HighCut, SamplingFrequency, order=3):
    nyq = 0.5 * SamplingFrequency
    low = LowCut / nyq
    high = HighCut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    y = sosfilt(sos, Data)
    return y


def slice_signal_to_segments(FilteredSignal, ResamplingFrequency, Annotations, SegmentInterval, OverLappingRatio):
    segments = []
    if Annotations[3] == 'bckg':
        label = 0
    else:
        label = 1

    for i in range(
            int(float(Annotations[1])) * ResamplingFrequency,
            int(float(Annotations[2])) * ResamplingFrequency,
            int(SegmentInterval * (1 - OverLappingRatio[label]) * ResamplingFrequency)
    ):
        if i + SegmentInterval * ResamplingFrequency > int(float(Annotations[2])) * ResamplingFrequency:
            break
        one_window = []
        noise_flag = False
        incomplete_flag = False
        for j in range(FilteredSignal.shape[0]):
            this_channel = FilteredSignal[j, :][i: i + SegmentInterval * ResamplingFrequency]
            if len(this_channel) != SegmentInterval * ResamplingFrequency:
                incomplete_flag = True
                break
            if max(abs(this_channel)) > 500 / 10 ** 6:
                noise_flag = True
                break
            one_window.append(this_channel)

        if incomplete_flag is False and noise_flag is False and one_window and len(
                one_window[0]) == ResamplingFrequency * SegmentInterval:
            segments.append((np.array(one_window), label))
    return segments


class load_data(Dataset):
    def __init__(self, DataDirectory, DownSampling=True):
        self.data_dir = DataDirectory
        self.downSampling = DownSampling
        self.files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        if DownSampling:
            self.balanced_files = self.balance_data()
        else:
            self.balanced_files = self.files

    def __len__(self):
        return len(self.balanced_files)

    def balance_data(self):
        label_0_files = []
        label_1_files = []

        for file in os.listdir(self.data_dir):
            if file.endswith('.npz'):
                file_path = os.path.join(self.data_dir, file)
                data = np.load(file_path, allow_pickle=True)
                label = data['label']

                if label == 0:
                    label_0_files.append(file_path)
                elif label == 1:
                    label_1_files.append(file_path)

        min_samples = min(len(label_0_files), len(label_1_files))

        label_0_files = random.sample(label_0_files, min_samples)
        label_1_files = random.sample(label_1_files, min_samples)

        balanced_files = label_0_files + label_1_files

        return balanced_files

    def __getitem__(self, idx):
        if self.downSampling:
            file_path = os.path.join(self.data_dir, self.balanced_files[idx])
        else:
            file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(file_path, allow_pickle=True)

        eeg_data = data['segments']
        label = data['label']

        return torch.tensor(eeg_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class TransposeLayer(nn.Module):
    def __init__(self, dim0, dim1):
        super(TransposeLayer, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation_rate=(1, 1), bias=False,
                 spiking=False):
        super(SeparableConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.spiking = spiking

        if isinstance(kernel_size, tuple):
            k_h, k_w = kernel_size
        else:
            k_h = k_w = kernel_size
        if len(dilation_rate) != 2 or not all(isinstance(k, int) for k in dilation_rate):
            raise ValueError(f"Invalid kernel_size: {dilation_rate}. Must be int or tuple of two ints.")
        # print(f"dilation_rate: {dilation_rate}, type: {type(dilation_rate)}")
        self.k_h = (k_h - 1) * dilation_rate[0] + 1
        self.k_w = (k_w - 1) * dilation_rate[1] + 1

        self.odd_zero_padding = nn.ZeroPad2d(
            ((self.k_w - 1) // 2, (self.k_w - 1) // 2, (self.k_h - 1) // 2, (self.k_h - 1) // 2)
        )
        self.even_zero_padding = nn.ZeroPad2d(
            (max(self.k_w // 2 - 1, 0), self.k_w // 2, max(self.k_h // 2 - 1, 0), self.k_h // 2)
        )

        self.depthwise = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation_rate, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)

        self.spiking_depthwise = layer.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation_rate, groups=in_channels, bias=bias
        )
        self.spiking_pointwise = layer.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias
        )

    def forward(self, x):
        if self.padding == 0:
            if self.k_h % 2 == 0 or self.k_w % 2 == 0:
                x = self.even_zero_padding(x)
            else:
                x = self.odd_zero_padding(x)
            # print(f"Padded shape: {x.shape}")
        if not self.spiking:
            x = self.depthwise(x)
            # print(f"After depthwise: {x.shape}")
            x = self.pointwise(x)
        else:
            x = self.spiking_depthwise(x)
            x = self.spiking_pointwise(x)
        return x


class IterateSeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation_rate=(1, 1), dilation_layers=4,
                 bias=False):
        super(IterateSeparableConv2D, self).__init__()
        self.iterate_conv = nn.ModuleList([
            SeparableConv2D(
                in_channels=in_channels if i == 0 else 2 * in_channels,
                out_channels=2 * in_channels if i != dilation_layers - 1 else out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation_rate=(dilation_rate[0], dilation_rate[1] + i),
                bias=bias
            ) for i in range(dilation_layers)
        ])

    def forward(self, x):
        for LAYER in self.iterate_conv:
            x = LAYER(x)
        return x


class SqueezeExpand2D(nn.Module):
    def __init__(self, input_length, squeeze_dim, TemporalBranch=True):
        super(SqueezeExpand2D, self).__init__()
        self.temporal_branch = TemporalBranch
        self.squeeze = nn.Linear(in_features=input_length, out_features=squeeze_dim)
        self.relu = nn.ReLU()
        self.expand = nn.Linear(in_features=squeeze_dim, out_features=input_length)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        device = x.device
        self.squeeze = self.squeeze.to(device)
        self.expand = self.expand.to(device)

        if self.temporal_branch:  # (num_channels * 2, 1, ceil(T / 32))
            permute_layer1 = x.permute(0, 3, 2, 1)  # (ceil(T / 32), 1, num_channels * 2)
            pool_layer = nn.AdaptiveAvgPool2d(1)(permute_layer1)  # (ceil(T / 32), 1, 1)
            permute_layer2 = pool_layer.permute(0, 3, 2, 1)  # (1, 1, ceil(T / 32))
            squeeze_layer = self.squeeze(permute_layer2)  # (1, 1, squeeze_dim)
            activate_layer1 = self.relu(squeeze_layer)
            expand_layer = self.expand(activate_layer1)  # (1, 1, ceil(T / 32))
            activate_layer2 = self.sigmoid(expand_layer)
        else:  # (num_channels * 4, 1, ceil(T / 64))
            pool_layer = nn.AdaptiveMaxPool2d(1)(x)  # (num_channels * 4, 1, 1)
            permute_layer1 = pool_layer.permute(0, 3, 2, 1)  # (1, 1, num_channels * 4)
            squeeze_layer = self.squeeze(permute_layer1)  # (1, 1, squeeze_dim)
            activate_layer1 = self.relu(squeeze_layer)  # (1, 1, num_channels * 4)
            expand_layer = self.expand(activate_layer1)
            permute_layer2 = expand_layer.permute(0, 3, 2, 1)  # (num_channels * 4, 1, 1)
            activate_layer2 = self.sigmoid(permute_layer2)
        multiple_layer = activate_layer2 * x
        return multiple_layer


class DilatedConv2d(nn.Module):
    def __init__(self, dilate_layers, dilate_kernel_size, slide_windows, slide_step, num_channels, input_length,
                 squeeze_dim, drop_out=0.25, TemporalBranch=True):
        super(DilatedConv2d, self).__init__()
        self.slide_windows = slide_windows
        self.slide_step = slide_step
        self.input_length = input_length
        self.temporal_branch = TemporalBranch
        self.squeeze_dim = squeeze_dim
        self.dilate_kernel_size = dilate_kernel_size

        self.dilate_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=(1, dilate_kernel_size),
                    padding=0,
                    dilation=(1, dilate_rate + 1)
                ),
                nn.BatchNorm2d(num_channels),
                nn.ELU(),
                nn.Dropout(drop_out)
            )
            for dilate_rate in range(dilate_layers)
        ])

        self.flatten = nn.Flatten()

    def forward(self, x):
        append_layer = []

        for i in range(self.slide_windows):
            start_point = i * self.slide_step
            end_point = self.input_length - (self.slide_windows - i - 1) * self.slide_step

            slide_window = x[:, :, :, start_point:end_point] if self.temporal_branch \
                else x[:, start_point:end_point, :, :]
            # print(f"slide_window shape: {slide_window.shape}")
            squeeze_expansion = SqueezeExpand2D(
                input_length=end_point - start_point,
                squeeze_dim=self.squeeze_dim,
                TemporalBranch=self.temporal_branch
            )(slide_window)

            if not self.temporal_branch:
                squeeze_expansion = squeeze_expansion.permute(0, 3, 2, 1)

            last_layer = squeeze_expansion
            # print(f"last layer shape: {last_layer.shape}")
            for dilation_rate, LAYER in enumerate(self.dilate_conv):
                # print(f"dilation_rate: {dilation_rate}")
                causal_padding = (self.dilate_kernel_size - 1) * (dilation_rate + 1)
                # print(f"causal_padding: {causal_padding}")
                last_layer_padded = F.pad(last_layer, (causal_padding, 0, 0, 0))
                conv_output = LAYER(last_layer_padded)
                # print(conv_output.shape)

                add_layer = conv_output + squeeze_expansion
                last_layer = F.elu(add_layer)

            flattened_layer = self.flatten(last_layer)
            append_layer.append(flattened_layer)

        concat_layer = torch.cat(append_layer, dim=1)
        return concat_layer


def get_gpu_power_usage(device_id):
    try:
        # 获取显卡功耗（单位是瓦特）
        result = subprocess.check_output(
            f"nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i {device_id}",
            shell=True
        )
        return float(result.strip())
    except subprocess.CalledProcessError:
        print(f"Failed to get power usage for device {device_id}")
        return 0.0


def Testing(net, test_loader, criterion, args, out_dir):
    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    test_prediction = []
    test_labels = []

    total_power = 0.0
    power_samples = 0
    device_id = args.device.split(":")[-1]

    with torch.no_grad():
        for sample, label in test_loader:
            sample = sample.to(args.device)
            label = label.to(args.device)

            power_usage = get_gpu_power_usage(device_id)
            total_power += power_usage
            power_samples += 1

            out_fr = net(sample)
            loss = criterion(out_fr, label)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += torch.eq(out_fr.argmax(1), label).float().sum().item()
            test_prediction.extend(out_fr.argmax(1).cpu().numpy())
            test_labels.extend(label.cpu().numpy())

            if args.model_type == 'SNN':
                functional.reset_net(net)

    avg_power_usage = total_power / power_samples if power_samples > 0 else 0.0

    test_loss /= test_samples
    test_acc /= test_samples
    test_F1 = f1_score(test_labels, test_prediction, average='binary')
    test_conf_matrix = confusion_matrix(test_labels, test_prediction)
    test_conf_matrix_percentage = test_conf_matrix.astype('float') / test_conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    results = (
        f"Accuracy: {test_acc * 100:.2f}%\n"
        f"F1 score: {test_F1:.4f}\n"
        f"Confusion matrix:\n{test_conf_matrix}\n"
        # f"Total GPU Power Usage: {total_power:.2f} W\n"
        f"Average GPU Power Usage: {avg_power_usage:.2f} W"
    )

    print(results)

    plt.figure(figsize=(8, 6))
    sns.heatmap(test_conf_matrix_percentage, annot=True, fmt=".2f", cmap="Blues", cbar=True, square=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{args.model} Confusion Matrix Percentage")
    plt.savefig(os.path.join(out_dir, 'confusion_matrix_percentage.png'))
    plt.show()

    with open(os.path.join(out_dir, 'test_results.txt'), 'w', encoding='utf-8') as f:
        f.write(results)

    return test_acc, test_F1, test_conf_matrix



