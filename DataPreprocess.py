import argparse
import os
import shutil
import mne
import numpy as np
import time

from UtiliTies import get_all_session_paths, get_channels_from_raw, butter_bandpass_filter, slice_signal_to_segments
from scipy.signal import resample


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Preprocess the TUSZ v2.0.3 dataset.')
    parser.add_argument('-SegmentInterval', default=4, type=int, help='Segment interval in seconds.')
    parser.add_argument('-LowCut', default=0.5, type=float, help='Low-cut frequency of the signal.')
    parser.add_argument('-HighCut', default=120, type=float, help='High-cut frequency of the signal.')
    parser.add_argument('-ResamplingFrequency', default=250, type=int, help='Resampling frequency.')
    parser.add_argument('-OverLappingRatio', default=[0, 0.75], type=float,
                        help='The ratio of data overlapping.')
    parser.add_argument('-DataDir', default='/data4/louxicheng/EEG_data/seizure/v2.0.3/', type=str,
                        help='The directory of the original data.')
    parser.add_argument('-ReferenceType', default='01_tcp_ar', type=str,
                        choices=['01_tcp_ar', '02_tcp_le', '03_tcp_ar_a'])
    args = parser.parse_args()
    # print(args)

    # Prepare directory.
    output_dir = os.path.join(args.DataDir, 'preprocessed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    preprocessed_data_path = os.path.join(
        output_dir, args.ReferenceType + '_segment_interval_' + str(args.SegmentInterval) + '_sec'
    )

    if os.path.exists(preprocessed_data_path):
        confirm = input(
            f"{preprocessed_data_path} already exists. \nAre you sure you want to delete existing files? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation aborted by user.")
            return
        else:
            for file_name in os.listdir(preprocessed_data_path):
                file_path = os.path.join(preprocessed_data_path, file_name)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.mkdir(preprocessed_data_path)

    # Load data.
    train_root = os.path.join(args.DataDir, 'edf', 'train')
    validation_root = os.path.join(args.DataDir, 'edf', 'eval')
    test_root = os.path.join(args.DataDir, 'edf', 'dev')
    folders = {
        'Train': train_root,
        'Validation': validation_root,
        'Test': test_root
    }
    for folder_name, folder_path in folders.items():
        session_paths, all_patients, reference_type_count = get_all_session_paths(folder_path)
        print(f'\n{folder_name} Set:')
        print('Number of sessions:', len(session_paths))
        print('Number of patients:', len(all_patients))
        print('Reference type count:', reference_type_count)
        count_session = 0
        samples = []
        for data_path in session_paths:
            if folder_name == 'Train':
                reference_type = data_path.split('train/')[1].split('/')[2]
            elif folder_name == 'Validation':
                reference_type = data_path.split('eval/')[1].split('/')[2]
            else:
                reference_type = data_path.split('dev/')[1].split('/')[2]
            if reference_type != args.ReferenceType:
                continue
            count_session += 1
            raw = mne.io.read_raw_edf(data_path, preload=True, verbose='warning')
            flag_wrong, signals = get_channels_from_raw(raw, ReferenceType=args.ReferenceType)
            if flag_wrong:
                continue
            # print(data_path)
            # print(raw.info)

            # Bandpass filter.
            filtered_signals = []
            sampling_frequency = int(raw.info['sfreq'])
            if sampling_frequency == args.ResamplingFrequency:
                resampled_signal = signals
            else:
                resampled_signal = []
                for i in range(signals.shape[0]):
                    resampled_signal_raw = resample(
                        signals[i, :], int(len(signals[i, :]) * args.ResamplingFrequency / sampling_frequency)
                    )
                    resampled_signal.append(resampled_signal_raw)
                resampled_signal = np.array(resampled_signal)
            for i in range(resampled_signal.shape[0]):
                bandpass_filtered_signal = butter_bandpass_filter(
                    resampled_signal[i, :], args.LowCut, args.HighCut, sampling_frequency, order=3
                )
                filtered_signals.append(bandpass_filtered_signal)
            filtered_signals = np.array(filtered_signals)
            # print(filtered_signals.shape)

            # Extract annotation.
            annotation_file_root = data_path[:-4] + '.csv_bi'
            with open(annotation_file_root, 'r') as annotation_file:
                annotation = annotation_file.readlines()
                annotation = annotation[-1]
                annotations = annotation.split(',')
                # print(annotations)

            # Slicing signals.
            segments = slice_signal_to_segments(
                filtered_signals, args.ResamplingFrequency, annotations, args.SegmentInterval, args.OverLappingRatio
            )
            for segment in segments:
                samples.append(segment)

        # save samples
        samples_dir = os.path.join(preprocessed_data_path, folder_name)
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)
        for idx, segment in enumerate(samples):
            data, label = segment
            print(f"Sample {idx} data shape: {data.shape}, Label: {label}")
            segment_file_path = os.path.join(samples_dir, f'sample_{idx}.npz')
            np.savez(segment_file_path, segments=data, label=label)

    end_time = time.time()
    print(f"Preprocessing completed in {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()
