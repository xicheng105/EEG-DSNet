import os
import time
import argparse
import sys
import datetime
import torch
import torch.utils.data as data

from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import functional

import models
from UtiliTies import load_data


def main():
    start_time_total = time.time()

    # Initializing parameters.
    parser = argparse.ArgumentParser(description='Classify seizure EEG signals.')
    parser.add_argument(
        '-device',
        default='cuda:5',
        help='Which GPU you want to use.'
    )
    parser.add_argument(
        '-epochs',
        default=500,
        type=int,
        help='Number of epochs.'
    )
    parser.add_argument(
        '-batch_size',
        default=128,
        type=int,
        help='Batch size.'
    )
    parser.add_argument(
        '-model_type',
        default='SNN',
        choices=['ANN', 'SNN'],
        help='Model type to use.'
    )
    parser.add_argument(
        '-model',
        default='EEG_DSNet',
        choices=[
            'EEGNet', 'EEG_SPNet', 'EEG_DSNet', 'CSNN', 'SCNet', 'LENet', 'EEG_DBNet_V1', 'EEG_DBNet_V2',
            'EEG_DBNet_V3', 'EEGNeX'
        ],
        help='Model type to use.'
    )
    parser.add_argument(
        '-neuron_type',
        default='AQIFNode',
        choices=['IFNode', 'LIFNode', 'ParametricLIFNode', 'QIFNode', 'EIFNode', 'IzhikevichNode',
                 'LIAFNode', 'KLIFNode', 'AQIFNode'],
        help='Neuron type to use'
    )
    parser.add_argument(
        '-T',
        default=7,
        type=int,
        help='simulating time-steps.'
    )
    parser.add_argument('-learning_rate', default=0.0001, type=float, help='Learning rate.')
    parser.add_argument('-weight_decay', default=0.0001, type=float, help='L2 regularization.')
    parser.add_argument('-workers', default=4, type=int,
                        help='Number of data loading workers.(Total number of cpu is 96.)')
    parser.add_argument('-data_dir', default='/data4/louxicheng/EEG_data/seizure/v2.0.3/preprocessed',
                        type=str, help='Root dir of TUSZ dataset.')
    parser.add_argument('-out_dir', type=str, default='./logs',
                        help='root dir for saving logs and checkpoint')
    parser.add_argument('--resume', action='store_true',
                        help='Whether to resume from the checkpoint.'
                             '(python Training.py -resume /path/to/checkpoint_latest.pth)')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file to resume from.')
    parser.add_argument('-AMP', action='store_true', help='Enable automatic mixed precision training.')
    args = parser.parse_args()

    # Chose model.
    # if args.model == 'EEG_DSNet':
    net = models.EEG_DSNet(T=args.T, neuron_type=args.neuron_type)
    # elif args.model == 'EEGNeX':
    #     net = models.EEGNeX()
    # elif args.model == 'EEG_DSNet':
    #     net = models.EEG_DSNet(T=args.T, neuron_type=args.neuron_type)
    # elif args.model == 'CSNN':
    #     net = models.CSNN(T=args.T)
    # elif args.model == 'SCNet':
    #     net = models.SCNet(T=args.T)
    # elif args.model == 'LENet':
    #     net = models.LENet(T=args.T, neuron_type=args.neuron_type)
    # else:
    #     net = models.EEG_SPNet(T=args.T, neuron_type=args.neuron_type)
    net.to(args.device)

    # Initializing data loader.
    if args.model_type == 'ANN':
        data_dir = os.path.join(args.data_dir, '01_tcp_ar_segment_interval_4_sec')
    else:
        data_dir = os.path.join(args.data_dir, '01_tcp_ar_segment_interval_4_sec_normalized')
    train_dataset_dir = os.path.join(data_dir, 'Train')
    validation_dataset_dir = os.path.join(data_dir, 'Validation')

    train_dataset = load_data(train_dataset_dir, DownSampling=True)
    validation_dataset = load_data(validation_dataset_dir, DownSampling=True)

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True
    )
    validation_loader = data.DataLoader(
        dataset=validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True
    )

    scaler = None
    if args.AMP:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_validation_acc = -1

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)  # , weight_decay=args.weight_decay
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True, min_lr=1e-6)

    if args.resume:
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be specified when resuming training.")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
        max_validation_acc = checkpoint['max_validation_acc']

    out_dir = os.path.join(args.out_dir, f'b{args.batch_size}_lr{args.learning_rate}_{args.model}')

    if args.AMP:
        out_dir += '_amp'
    if args.model_type == 'SNN':
        out_dir += f'_{args.neuron_type}_st{args.T}'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f'dir: {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        # Training
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for sample, label in train_data_loader:
            optimizer.zero_grad()
            sample = sample.to(args.device)
            label = label.to(args.device)
            # label_onehot = F.one_hot(label, 2).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(sample)
                    loss = criterion(out_fr, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(sample)
                loss = criterion(out_fr, label)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()  # Why make loss times number?
            train_acc += torch.eq(out_fr.argmax(1), label).float().sum().item()
            if args.model_type == 'SNN':
                functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        # validation
        net.eval()
        validation_loss = 0
        validation_acc = 0
        validation_samples = 0
        with torch.no_grad():
            for sample, label in validation_loader:
                sample = sample.to(args.device)
                label = label.to(args.device)
                # label_onehot = F.one_hot(label, 2).float()
                out_fr = net(sample)
                loss = criterion(out_fr, label)

                validation_samples += label.numel()
                validation_loss += loss.item() * label.numel()
                validation_acc += torch.eq(out_fr.argmax(1), label).float().sum().item()
                if args.model_type == 'SNN':
                    functional.reset_net(net)

        validation_time = time.time()
        validation_speed = validation_samples / (validation_time - train_time)
        validation_loss /= validation_samples
        validation_acc /= validation_samples
        writer.add_scalar('validation_loss', validation_loss, epoch)
        writer.add_scalar('validation_acc', validation_acc, epoch)

        # scheduler.step(validation_acc)

        save_max = False
        if validation_acc > max_validation_acc:
            max_validation_acc = validation_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'epoch': epoch,
            'max_validation_acc': max_validation_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(
            f'epoch ={epoch}\n'
            f'     train_loss ={train_loss: .4f},      train_acc ={train_acc: .4f}\n'
            f'validation_loss ={validation_loss: .4f}, validation_acc ={validation_acc: .4f}, '
            f'max_validation_acc ={max_validation_acc: .4f}'
        )
        print(
            f'train speed ={int(train_speed)} samples/s, '
            f'validation_speed ={int(validation_speed)} samples/s'
        )
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n'
        )

    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    total_runtime_str = \
        f"Total runtime: {int(total_time // 3600)}h {int((total_time % 3600) // 60)}m {int(total_time % 60)}s"

    print(total_runtime_str)
    with open(os.path.join(out_dir, 'args.txt'), 'a', encoding='utf-8') as args_txt:
        args_txt.write(f"\n{total_runtime_str}")


if __name__ == '__main__':
    main()
