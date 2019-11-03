import os
import argparse
import torch

from cifar10_pytorch.engine import train, test
from cifar10_pytorch.config import cfg
from cifar10_pytorch.data import dataset
from cifar10_pytorch.modeling import build_model



def arg_parser():
    parser = argparse.ArgumentParser(description="CIFAR10 training")
    parser.add_argument(
        "--config",
        default=None,
        help="path to config file",
        type=str
    )
    parser.add_argument(
        '--tfboard', help='tensorboard path for logging', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints',
                        help='directory where checkpoint files are saved')
    parser.add_argument('--resume', type=str,
                        default=None,
                        help='checkpoint file path')
    return parser.parse_args()


def main():
    # args and configs
    args = arg_parser()
    if args.config:
        cfg.merge_from_file(args.config)
    cfg.freeze()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Model definition
    model = build_model(cfg)
    device = cfg.MODEL.DEVICE
    model.to(device)

    # Build dataloader
    dataloader_train, dataloader_test = dataset.prepare_cifar10_dataset(cfg)

    # Optimizer settings
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.SOLVER.BASE_LR,
                                momentum=cfg.SOLVER.MOMENTUM,
                                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        cfg.SOLVER.MILESTONES,
        cfg.SOLVER.GAMMA,
    )

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #scheduler.last_epoch = checkpoint['epoch']
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    # Tensorboard settings
    if args.tfboard:
        print("using tfboard")
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(args.tfboard)
    else:
        tblogger = None

    # Training Loop
    end_epoch = cfg.SOLVER.END_EPOCH
    for epoch in range(start_epoch, end_epoch):
        scheduler.step()
        print("------ start epoch {} / {} with lr = {:.5f} -----".format(
            epoch+1, end_epoch, scheduler.get_lr()[0])
        )
        loss, train_acc = train(model, device, dataloader_train, criterion, optimizer, epoch)
        test_acc = test(model, device, dataloader_test, criterion, epoch)
        if tblogger:
            tblogger.add_scalar('train/loss', loss, epoch+1)
            tblogger.add_scalar('train/acc', train_acc, epoch+1)
            tblogger.add_scalar('test/acc', test_acc, epoch+1)
        torch.save({'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    },
                   os.path.join(args.checkpoint_dir, "snapshot.ckpt"))


if __name__ == "__main__":
    main()