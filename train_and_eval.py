from math import sqrt
import numpy as np
import torch
from tqdm import tqdm
from utils import utils, evaluation_index
from sklearn.metrics import confusion_matrix

import torch.nn.functional as F

def compute_eva_gpu(obj, pred, t=0.1):
    # obj = np.array(obj)
    # pred = np.array(pred)
    mse_loss = torch.nn.MSELoss()
    rmse = sqrt(mse_loss(pred, obj))

    acc = evaluation_index.ACC(obj, pred, t)
    precision = evaluation_index.precision(obj, pred, t)
    f1 = evaluation_index.FSC(obj, pred, t)
    pod = evaluation_index.POD(obj, pred, t)
    far = evaluation_index.FAR(obj, pred, t)
    csi = evaluation_index.CSI(obj, pred, t)
    miou = 9999

    if torch.all(obj == 0) or torch.all(pred == 0):
        cc = 0  # 设置相关系数为0或其他特定值
    else:
        stacked_data = torch.stack([obj, pred], dim=0)
        cc = torch.corrcoef(stacked_data)[0, 1].item()

    # csi_neighborhood = evaluation_index.csi_neighborhood(obj, pred, kappa=2, threshold=t)
    return [rmse, cc, pod, far, csi, 0, acc, precision, f1, miou]


def train_one_epoch_distill(teacher, student, train_loader, optimizer, grad_scaler, criterion_T, criterion_y,
                            args, device, pbar, global_step, experiment):
    mask_ratio = args.mask_ratio
    for batch in train_loader:
        images, geo, label = batch['image'], batch['geo'], batch['label']
        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        geo = geo.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)

        mask = utils.generate_mask((1, 1, 16, 3), mask_ratio).to(device)

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
            logits_t, feat_t = teacher(images, geo)
            logits_s, feat_s, sc_loss = student(images[:, :8], mask)

            kd_feat = []
            kd_feat_mae = []
            for idx, (f_t, f_s) in enumerate(zip(feat_t, feat_s)):
                kd_feat.append(criterion_T(f_s, f_t))
                kd_feat_mae.append(torch.mean((feat_s[idx + 3] - f_t) ** 2))

            nan_index = torch.isnan(label)
            label_no_nan = label.float()[~nan_index]
            if args.classes == 1:
                pred_no_nan = logits_s.squeeze(1)[~nan_index]
            else:
                pred_no_nan = logits_s.permute(1, 0, 2, 3)[:, ~nan_index].T.float().contiguous()

            task_loss = criterion_y(pred_no_nan, label_no_nan)

        kd_feat_loss = 50 * (kd_feat[0] + 2 * kd_feat[1] + 4 * kd_feat[2])
        kd_feat_mae_loss = 0.2 * sum(kd_feat_mae)
        loss = task_loss + kd_feat_mae_loss + kd_feat_loss + sc_loss

        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.gradient_clipping)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        pbar.update(images.shape[0])
        pbar.set_postfix(**{'loss (batch)': loss.item(),
                            'kd_loss (batch)': kd_feat_loss.item(),
                            'task_loss (batch)': task_loss.item(),
                            'sc_loss (batch)': sc_loss.item(),
                            'kd_mae_loss (batch)': kd_feat_mae_loss.item(),
                            })
        global_step += 1
        experiment.log({
            'kd_feat_loss': kd_feat_loss.item(),
            'task_loss': task_loss.item(),
            'sc_loss': sc_loss.item(),
            'loss (all)': loss.item(),
            'kd_mae_loss': kd_feat_mae_loss.item(),
            'step1': global_step,
            'learning rate': optimizer.param_groups[0]['lr'],
        })
    return global_step


@torch.inference_mode()
def evaluate_distill(net, dataloader, device, args):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    total_mask = []
    total_pred = []

    with torch.no_grad():
        for ids_index, batch in tqdm(enumerate(dataloader), total=num_val_batches, desc='eva round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)
            nan_index = torch.isnan(mask_true)

            pic_zero = torch.zeros([1, 197, 256, 49]).to(device=device, dtype=torch.float32)
            mask_pred = torch.zeros([1, args.classes, image.shape[2], image.shape[3]]).to(device=device, dtype=torch.float32)
            mask = torch.ones(1, 1, 16, 3).to(device)

            for i in range(image.shape[2] // 256):
                mask_pred[:, :, i * 256:i * 256 + 256] = net(image[:, :, i * 256:i * 256 + 256], mask)[0]
            if image.shape[2] % 256 != 0:
                pic_zero[:, :, :(image.shape[2] % 256)] = image[:, :, -(image.shape[2] % 256):]
                mask_pred[:, :, -(image.shape[2] % 256):] = net(pic_zero, mask)[0][:, :, :(image.shape[2] % 256)]

            if args.classes == 2:
                mask_pred = mask_pred.argmax(dim=1)

            mask_pred = mask_pred.squeeze(1)

            total_mask.append(mask_true[~nan_index].squeeze())
            total_pred.append(mask_pred[~nan_index].squeeze())

    total_mask = torch.cat(total_mask, dim=0)
    total_pred = torch.cat(total_pred, dim=0)

    total_prmd_acc_list = compute_eva_gpu(total_mask, total_pred, 0.1)
    total_s = " total:\n  rmse    cc   pod  far  csi   acc  precision  f1   miou\n" \
        + 'pred {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}\n'.format(total_prmd_acc_list[0],
                                                                                        total_prmd_acc_list[1],
                                                                                        total_prmd_acc_list[2],
                                                                                        total_prmd_acc_list[3],
                                                                                        total_prmd_acc_list[4],
                                                                                        total_prmd_acc_list[5],
                                                                                        total_prmd_acc_list[6],
                                                                                        total_prmd_acc_list[7],
                                                                                        total_prmd_acc_list[8])
    return total_s, total_prmd_acc_list




def train_one_epoch_finetune(net, train_loader, optimizer, grad_scaler,
                        criterion_y, args, device, pbar, global_step, epoch, experiment):
    mask_ratio = args.mask_ratio

    for idx, batch in enumerate(train_loader):
        indices, images, labels, mask = batch['indices'], batch['images'], batch['labels'], batch['mask_items']
        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        labels = labels.to(device=device, dtype=torch.float32)

        if args.mask_strategy == 'none':
            mask = torch.ones(1, 1, 16, 16).to(device=device)
        elif args.mask_strategy == 'randomly':
            mask = utils.generate_mask((1, 1, 16, 16), mask_ratio)
        elif args.mask_strategy == 'self-masktune':
            if epoch < 40:
                mask = torch.ones(1, 1, 16, 16).to(device=device)
            else:
                mask = mask.view(args.batch_size, 1, 16, 16)
                mask = utils.generate_batch_mask(mask, mask_ratio)

        mask = mask.to(device=device)

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
            pred = net(images, mask)[0]
            nan_index = torch.isnan(labels)
            label_no_nan = labels.float()[~nan_index]
            if args.classes == 1:
                pred_no_nan = pred.squeeze(1)[~nan_index]
            else:
                pred_no_nan = pred.permute(1, 0, 2, 3)[:, ~nan_index].T.float().contiguous()

            loss = criterion_y(pred_no_nan, label_no_nan)

        if args.mask_strategy == 'self-masktune':
            if epoch >= 40:
                new_mask_items = utils.generate_mask_by_loss((8, 16, 16), mask_ratio, label_no_nan, pred_no_nan, args)
                # 更新 Dataset 中的 mask_item
                for i, new_mask_item in zip(indices, new_mask_items):
                    train_loader.dataset.update_mask_item(i, new_mask_item)

        grad_scaler.scale(loss).backward()
        accumulation_steps = args.accumulation_steps
        if ((idx + 1) % accumulation_steps) == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad(set_to_none=True)

        pbar.update(images.shape[0])
        pbar.set_postfix(**{'loss (batch)': loss.item(), 'lr': optimizer.param_groups[0]["lr"]})
        global_step += 1

        if args.rank in [-1, 0]:
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'loss_ALL': loss.item(),
                'step': global_step,
                'epoch': epoch
            })
    return global_step



@torch.inference_mode()
def evaluate(net, dataloader, device, args):
    net.eval()
    num_val_batches = len(dataloader)
    total_mask = []
    total_pred = []

    hits, misses, false_alarms = 0, 0, 0
    # device = net.device
    with torch.no_grad():
        for ids_index, batch in tqdm(enumerate(dataloader), total=num_val_batches, desc='eva round', unit='batch'):
            image, label = batch['image'], batch['label']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            label = label.to(device=device, dtype=torch.float32)

            # predict the mask
            mask = torch.ones(1, 1, 16, 16).to(device=device)
            pred = net(image, mask)[0]

            nan_index = torch.isnan(label)

            if args.classes == 2:
                pred = pred.argmax(dim=1)
                pred = pred.reshape(image.shape[0], 256, -1).squeeze()

            total_mask.append(label[~nan_index].squeeze())
            total_pred.append(pred.squeeze()[~nan_index])

    total_mask = torch.cat(total_mask, dim=0)
    total_pred = torch.cat(total_pred, dim=0)

    total_prmd_acc_list = compute_eva_gpu(total_mask, total_pred, 0.1)
    total_s = (" total:\n  rmse    cc   pod  far  csi   acc  precision  f1   miou\n" \
               + 'pred {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}\n'
               .format(total_prmd_acc_list[0],
                       total_prmd_acc_list[1],
                       total_prmd_acc_list[2],
                       total_prmd_acc_list[3],
                       total_prmd_acc_list[4],
                       total_prmd_acc_list[6],
                       total_prmd_acc_list[7],
                       total_prmd_acc_list[8],
                       total_prmd_acc_list[9]))
    # print(total_s)
    net.train()
    return total_s, total_prmd_acc_list
