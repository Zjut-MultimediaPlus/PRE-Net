import argparse
import logging
import os

from datetime import datetime


from pathlib import Path
from torch import optim
from torch.nn.utils import parametrize
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import wandb

from train_and_eval import *
from model.unet_model import *
from utils.data_loading import IR_BasicDataset, collate_fn
from utils.utils import layer_parametrization, init_distributed_mode


# 输入文件路径
train_dir_img = Path("/your_dataset_dir/")
val_dir_img = Path("/your_dataset_dir/")

# 设定随机种子
SEED = 2023
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')

def train_model(
        model,
        savedir,
        experiment,
        save_checkpoint: bool = True,
):
    # 1. Create dataset
    train_dataset = IR_BasicDataset(train_dir_img, args.scale, args.classes)
    val_dataset = IR_BasicDataset(val_dir_img, args.scale, args.classes)

    # 2. Split into train / validation partitions
    n_val = len(val_dataset)
    n_train = len(train_dataset)

    # 3. Create data loaders
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, sampler=val_sampler, drop_last=False, batch_size=32, num_workers=8, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, batch_size=32, num_workers=8, pin_memory=True)

    total_steps = len(train_loader) / 4 * args.epochs
    if args.distributed:
        torch.distributed.barrier()

    if args.rank in [-1, 0]:
        logging.info(f'''Starting training:
            Epochs:          {args.epochs}
            Batch size:      {args.batch_size}
            Learning rate:   {args.lr}
            Training size:   {n_train}
            val size:       {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Images scaling:  {args.scale}
            Mixed Precision: {args.amp}
        ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    for name, param in model_without_ddp.fe_block.named_parameters():
        if 'mae' in name and 'lora' not in name:
            param.requires_grad = False

    optimizer = optim.RMSprop(model_without_ddp.parameters(),
                              lr=args.lr * args.world_size, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion_y = nn.CrossEntropyLoss().cuda() if args.classes == 2 else nn.MSELoss().cuda()
    start_epoch = 0

    if args.load != '':
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['lr_scheduler'])
        start_epoch = state_dict['epoch']
        logging.info(f'Model loaded from {args.load}')

    global_step = 0
    # 5. Begin training
    for epoch in range(start_epoch + 1, args.epochs + 1):
        scheduler.step()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
            global_step = train_one_epoch_finetune(model, train_loader, optimizer, grad_scaler, criterion_y,
                                               args, device, pbar, global_step, epoch, experiment)
        if args.distributed:
            torch.distributed.barrier()

        # Evaluation round
        if epoch < 40 and epoch > 1:
            continue
        division_step = 10  # 10epoch 验证一次
        if epoch % division_step == 0 or epoch == args.epochs or epoch == 1:
            if args.rank in [-1, 0]:
                val_str, val_list = evaluate(model, val_loader, device, args)

                with open(recode_txt, 'a') as f:
                    f.write(str(epoch) + '_val_res:' + val_str)

                logging.info('val str score: {}'.format(val_str))

                experiment.log({
                    'rmse': val_list[0],
                    'cc': val_list[1],
                    'pod': val_list[2],
                    'far': val_list[3],
                    'csi': val_list[4],
                    'f1': val_list[7],
                    'step': global_step,
                    'epoch': epoch,
                })

                save_file = {"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
                if args.amp:
                    save_file["scaler"] = grad_scaler.state_dict()
                torch.save(save_file, savedir + 'checkpoint_epoch{}.pth'.format(epoch))
                logging.info(f'Checkpoint {epoch} saved!')

        # scheduler.step()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.999, type=float, metavar='M')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
    parser.add_argument('--gradient-clipping', default=1.0, type=float, metavar='G')
    parser.add_argument('--mask-ratio', default=0.25, type=float)
    parser.add_argument('--mask-strategy', default='self-masktune', type=str)

    parser.add_argument('--load', '-f', type=str, default="", help='Load model from a .pth file')
    parser.add_argument('--fe-path', '-fe', type=str,
                        default="/hy-tmp/PRMD_UNet_KD-v1.0/checkpoints_reg/checkpoint_best.pth", help='Load feature extraction block from a .pth file')

    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    # 是否使用同步BN(在多个GPU之间同步)，默认不开启，开启后训练速度会变慢
    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using SyncBatchNorm')
    # 分布式进程数
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--rank', default=-1)
    parser.add_argument('--gpu', default=0)

    return parser.parse_args()



def pre_net(classes, device):
    mae4 = MAE_ViT(image_size=[128, 24], in_channel=128, mask_ratio=args.mask_ratio)
    mae8 = MAE_ViT(image_size=[64, 12], in_channel=256, mask_ratio=args.mask_ratio)
    mae16 = MAE_ViT(image_size=[32, 6], in_channel=512, mask_ratio=args.mask_ratio)

    mae4 = ViT_Classifier_v3(mae4, num_splits=3)
    mae8 = ViT_Classifier_v3(mae8, num_splits=3)
    mae16 = ViT_Classifier_v4(mae16)

    mae4.num_splits, mae8.num_splits = 1, 1

    fe_block_1 = fe_block_mae_v3(n_channels=8, bilinear=args.bilinear, mae_block=[mae4, mae8, mae16])
    state_dict = torch.load(args.fe_path)
    fe_block_1.load_state_dict(
         {key.split('.', 1)[1]: value for key, value in state_dict['model'].items() if key.startswith('fe_block')})

    model = PRE_Net(n_channels=8, n_classes=classes, bilinear=args.bilinear, fe_block=fe_block_1).to(device)

    items = ['attn', 'mlp']
    for name, module in model.fe_block.named_modules():
        if any(item in name for item in items) and 'parametrizations' not in name:
            try:
                parametrize.register_parametrization(
                    module, "weight",
                    layer_parametrization(module)
                )
            except AttributeError:
                pass
                # logging.warning(f'layer {name} has no weights')
    return model


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    init_distributed_mode(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = pre_net(args.classes, device)

    if args.rank in [-1, 0]:
        logging.info(f'Network:\n'
                     f'\t{model.n_channels} input channels\n'
                     f'\t{model.n_classes} output channels (classes)\n'
                     f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    start_time = get_cur_time()
    savedir = './checkpoints/' + start_time + '/'
    experiment = None
    if args.rank in [-1, 0]:
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        recode_txt = savedir + 'val_and_test_acc.txt'
        with open(recode_txt, mode='a', encoding='utf-8') as f:
            f.write(start_time + '\n' + args.fe_path + '\n')

        # (Initialize logging)
        experiment = wandb.init(project='finetune', resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
                 val_percent=args.val / 100, save_checkpoint=True, img_scale=args.scale, amp=args.amp)
        )

    train_model(
        model=model,
        savedir=savedir,
        experiment=experiment
    )
