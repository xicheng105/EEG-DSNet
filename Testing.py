import os
import torch
import argparse
import torch.utils.data as data

import models
from UtiliTies import load_data, Testing


def main():
    # Initializing parameters.
    parser = argparse.ArgumentParser(description='Deducting EEG.')
    parser.add_argument('-device', default='cuda:5', help='Which GPU you want to use.')
    parser.add_argument('-batch_size', default=128, type=int, help='Batch size.')
    parser.add_argument('-learning_rate', default=0.0001, type=float, help='Learning rate.')
    parser.add_argument(
        '-out_dir',
        default="/data4/louxicheng/PythonProject/SEI_EEG_P_5/logs/b128_lr0.0001_SCNet_LIFNode_st5/",
        type=str,
        help='dir of checkpoint files.'
    )
    parser.add_argument(
        '-model',
        default='EEG_DSNet',
        choices=[
            'EEGNet', 'EEG_SPNet', 'EEG_DSNet', 'CSNN', 'SCNet', 'LENet', 'EEG_DBNet_V1', 'EEG_DBNet_V2', 'EEG_DBNet_V2'
        ],
        help='Model type to use.'
    )
    parser.add_argument(
        '-model_type',
        default='SNN',
        choices=['ANN', 'SNN'],
        help='Model type to use.'
    )
    parser.add_argument(
        '-T',
        default=5,
        type=int,
        help='simulating time-steps.'
    )
    parser.add_argument(
        '-neuron_type',
        default='AQIFNode',
        choices=['IFNode', 'LIFNode', 'ParametricLIFNode', 'QIFNode', 'EIFNode', 'IzhikevichNode',
                 'LIAFNode', 'KLIFNode', 'AQIFNode'],
        help='Neuron type to use'
    )
    parser.add_argument(
        '-checkpoint_type',
        default='max',
        choices=['max', 'latest'],
        help='Checkpoint type to use.'
    )
    parser.add_argument('-workers', default=48, type=int,
                        help='Number of data loading workers.(Total number of cpu is 96.)')
    parser.add_argument('-data_dir', default='/data4/louxicheng/EEG_data/seizure/v2.0.3/preprocessed',
                        type=str, help='Root dir of TUSZ dataset.')
    parser.add_argument('-AMP', action='store_true', help='Enable automatic mixed precision training.')
    args = parser.parse_args()

    # Chose model.
    # if args.model == 'EEGNet':
    #     net = models.EEGNet()
    # elif args.model == 'EEG_DSNet':
    net = models.EEG_DSNet(T=args.T, neuron_type=args.neuron_type)
    # elif args.model == 'CSNN':
    #     net = models.CSNN(T=args.T)
    # elif args.model == 'SCNet':
    #     net = models.SCNet(T=args.T)
    # elif args.model == 'LENet':
    #     net = models.LENet(T=args.T)
    # else:
    #     net = models.EEG_SPNet(T=args.T, neuron_type=args.neuron_type)
    net.to(args.device)

    # Initializing data loader.
    if args.model_type == 'ANN':
        data_dir = os.path.join(args.data_dir, '01_tcp_ar_segment_interval_4_sec')
    else:
        data_dir = os.path.join(args.data_dir, '01_tcp_ar_segment_interval_4_sec_normalized')
    test_dataset_dir = os.path.join(data_dir, 'Test')
    test_dataset = load_data(test_dataset_dir, DownSampling=False)

    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True
    )

    criterion = torch.nn.CrossEntropyLoss()

    if args.checkpoint_type == 'max':
        checkpoint_path = os.path.join(args.out_dir, 'checkpoint_max.pth')
    else:
        checkpoint_path = os.path.join(args.out_dir, 'checkpoint_latest.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['net'])

    _, _, _ = Testing(net, test_loader, criterion, args, args.out_dir)


if __name__ == '__main__':
    main()
