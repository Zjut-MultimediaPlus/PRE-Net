import argparse
import logging
import os

from datetime import datetime
from pathlib import Path

import wandb
from torch import optim
from torch.utils.data import DataLoader

from train_and_eval import *
from model.unet_model import *
from model.layers import *
from utils import utils
from utils.F_loss import KL_Loss
from utils.data_loading import BasicDataset

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
        teacher,
        student,
        device,
        experiment
):
    start_time = get_cur_time()
    savedir = f'./checkpoints/{start_time}/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    recode_txt = savedir + 'val_and_test_acc.txt'
    with open(recode_txt, mode='w', encoding='utf-8') as f:
        f.write(start_time + '\n')

    # 1. Create dataset
    train_dataset = BasicDataset(train_dir_img, args.scale, args.classes, geo_norm, geo_channel)
    val_dataset = BasicDataset(val_dir_img, args.scale, args.classes, geo_norm, geo_channel)

    # 2. Split into train / validation partitions
    n_val = len(val_dataset)
    n_train = len(train_dataset)

    # 3. Create data loaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, batch_size=1, num_workers=8, pin_memory=True)

    logging.info(f'''Starting training Feature Extraction:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        val size:       {n_val}
        Device:          {device.type}
        Images scaling:  {args.scale}
        Mixed Precision: {args.amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(student.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    criterion_T = KL_Loss(temperature=args.temperature, reduction='mean').to(device)
    criterion_y = nn.CrossEntropyLoss() if args.classes == 2 else nn.MSELoss()

    start_epoch = 0
    if args.load != '':
        state_dict = torch.load(args.load, map_location=device)
        student.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['lr_scheduler'])
        start_epoch = state_dict['epoch']
        logging.info(f'Model loaded from {args.load}')

    global_step = 0
    # 5. Begin training
    for epoch in range(start_epoch + 1, args.epochs + 1):
        student.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
            global_step = train_one_epoch_distill(teacher, student, train_loader, optimizer, grad_scaler,
                                                  criterion_T, criterion_y,
                                                  args, device, pbar, global_step, experiment)

        # Evaluation round
        if epoch < 40 and epoch > 1:
            continue
        division_step = 20  # 20epoch 验证一次
        if epoch % division_step == 0 or epoch == args.epochs or epoch == 1:
            val_str, val_list = evaluate_distill(student, val_loader, device, args)
            with open(recode_txt, 'a') as f:
                f.write(str(epoch) + '_val_res:' + val_str)

            logging.info('val str score: {}'.format(val_str))

            save_file = {"model": student.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
            if args.amp:
                save_file["scaler"] = grad_scaler.state_dict()
            torch.save(save_file, savedir + f'checkpoint_{epoch}.pth')
            logging.info(f'Checkpoint {epoch} saved!')

        scheduler.step()


def get_args():
    # w KD (ce+32kl) : "/opt/data/private/PRMD_UNet_KD-v1.0/checkpoints_classify/teacher/2024-03-06_13-19CE_two_step_32_with_CE/checkpoint_epoch75.pth"

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-6,
                        help='Learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.999, type=float, metavar='M')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
    parser.add_argument('--gradient-clipping', default=1.0, type=float, metavar='G')
    parser.add_argument('--mask-ratio', default=0.25, type=float, help='length ratio: default(80)')

    parser.add_argument('--load', '-f', type=str, default="", help='Load model from a .pth file')
    parser.add_argument('--load-teacher', '-ft', type=str, default="")
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--temperature', default=3.0, type=float, help='Input the temperature: default(3.0)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 输入高程归一化最大最小值
    with open('geo_norm.txt', 'r') as f:
        geo_str = f.read()

    with open('geo_channel.txt', 'r') as f:
        channel_str = f.read()

    geo_norm = list(map(int, geo_str.split('_')))  # [6000, -300]
    geo_channel = list(map(int, channel_str.split('_')))  # [6000, -300]
    teacher = Teacher(n_channels=197, n_classes=args.classes, geo_channels=geo_channel[1] - geo_channel[0],
                      bilinear=args.bilinear).to(device)
    teacher.load_state_dict(torch.load(args.load_teacher, map_location=device))

    mae4 = MAE_ViT(image_size=[128, 24], in_channel=128, mask_ratio=args.mask_ratio).to(device)
    mae8 = MAE_ViT(image_size=[64, 12], in_channel=256, mask_ratio=args.mask_ratio).to(device)
    mae16 = MAE_ViT(image_size=[32, 6], in_channel=512, mask_ratio=args.mask_ratio).to(device)

    mae4 = ViT_Classifier_v3(mae4, num_splits=3, base_size=(28, 28))
    mae8 = ViT_Classifier_v3(mae8, num_splits=3, base_size=(28, 28))
    mae16 = ViT_Classifier_v4(mae16, base_size=(28, 28))

    fe_block = fe_block_mae_v3(n_channels=8, bilinear=args.bilinear, mae_block=[mae4, mae8, mae16])
    student = PRE_Net(n_channels=8, n_classes=args.classes, bilinear=args.bilinear, fe_block=fe_block).to(device)

    # (Initialize logging)
    experiment = wandb.init(project='distill', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
             val_percent=args.val / 100, save_checkpoint=True, img_scale=args.scale, amp=args.amp)
    )

    train_model(
        teacher,
        student,
        device,
        experiment
    )
