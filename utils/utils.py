import os

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import distributed as dist
from scipy.ndimage import zoom

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


def get_current_consistency_weight(current, rampup_length=80):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def generate_mask(shape, zero_ratio):
    batch_size, channels, height, width = shape
    num_elements = height * width
    num_zeros = int(num_elements * zero_ratio)
    num_ones = num_elements - num_zeros

    mask = []
    for _ in range(batch_size * channels):
        # Create an array with the desired ratio of zeros and ones
        channel_mask = np.array([0] * num_zeros + [1] * num_ones)
        # Shuffle the array to randomly distribute zeros and ones
        np.random.shuffle(channel_mask)
        # Reshape the array to the desired height and width
        channel_mask = channel_mask.reshape((height, width))
        mask.append(channel_mask)

    # Convert the list of masks to a torch tensor and reshape to the desired output shape
    mask = np.array(mask)
    mask = mask.reshape((batch_size, channels, height, width))
    mask = torch.tensor(mask, dtype=torch.float32)
    return mask


def generate_mask_by_loss(shape, zero_ratio, y, pred, args):
    batch_size, height, width = shape
    criterion = nn.CrossEntropyLoss(reduction='none') if args.classes == 2 else nn.MSELoss(reduction='none')
    loss = criterion(pred, y).view(batch_size, -1)
    mask_downsampled = []
    random_ratio = 0.1
    for l in loss:
        threshold = torch.quantile(l, 1-zero_ratio)
        mask = l.reshape(256, 256) >= threshold
        mask = mask.float()
        mask = 1 - mask

        # 引入随机性
        random_mask = (torch.rand_like(mask) < random_ratio).float()
        mask = torch.maximum(mask, random_mask)

        mask_downsampled.append(torch.as_tensor(zoom(mask.cpu(), (1/16, 1/16), order=0).reshape(height, width)))

    return mask_downsampled


def generate_batch_mask(masks, mask_ratio):
    summed_mask = masks.sum(dim=0, keepdim=True)  # shape (1, 1, height, width)
    flattened = summed_mask.flatten()

    threshold_bottom = torch.quantile(flattened, mask_ratio)

    new_mask = torch.zeros_like(summed_mask)
    new_mask[summed_mask <= threshold_bottom] = 1

    new_masks = new_mask.repeat(masks.size(0), 1, 1, 1)  # shape (batch, 1, height, width)
    return new_masks


def shuffle_and_split_mask(mask, num_splits=3):
    b, _, h, w = mask.shape
    device = mask.device

    # 获取所有为1的索引
    indices = torch.nonzero(mask, as_tuple=False)

    # 随机打乱索引
    indices = indices[torch.randperm(indices.size(0))]

    # 将打乱的索引均分为 num_splits 块
    split_indices = torch.chunk(indices, num_splits)

    # 创建分割后的mask列表
    split_masks = []
    for split in split_indices:
        # 创建新的空mask
        new_mask = torch.zeros_like(mask)
        # 设置对应的索引为1
        new_mask[split[:, 0], split[:, 1], split[:, 2], split[:, 3]] = 1
        split_masks.append(new_mask)

    return split_masks


def dwt2(x, wavelet='haar'):
    # Haar wavelet filters
    c = x.shape[1]
    if wavelet == 'haar':
        low_filter = torch.tensor([1.0, 1.0]).view(1, 1, 1, 2).contiguous() / math.sqrt(2)
        high_filter = torch.tensor([1.0, -1.0]).view(1, 1, 1, 2).contiguous() / math.sqrt(2)
    else:
        raise ValueError("Currently only supports 'haar' wavelet")

    low_filter = low_filter.to(x.device)
    high_filter = high_filter.to(x.device)

    ll, lh, hl, hh = [], [], [], []
    for i in range(c):
        x_tmp = x[:, i:i + 1, :, :]
        # Apply filters along rows
        low = F.conv2d(x_tmp, low_filter, stride=(1, 2))
        high = F.conv2d(x_tmp, high_filter, stride=(1, 2))

        # Apply filters along columns
        ll.append(F.conv2d(low.transpose(2, 3).contiguous(), low_filter, stride=(1, 2)).transpose(2, 3).contiguous())
        lh.append(F.conv2d(low.transpose(2, 3).contiguous(), high_filter, stride=(1, 2)).transpose(2, 3).contiguous())
        hl.append(F.conv2d(high.transpose(2, 3).contiguous(), low_filter, stride=(1, 2)).transpose(2, 3).contiguous())
        hh.append(F.conv2d(high.transpose(2, 3).contiguous(), high_filter, stride=(1, 2)).transpose(2, 3).contiguous())
    ll = torch.cat(ll, dim=1)
    lh = torch.cat(lh, dim=1)
    hl = torch.cat(hl, dim=1)
    hh = torch.cat(hh, dim=1)

    return ll, (lh, hl, hh)


# Let's create a LoRA class that will add two new matrices A and B to the
# original weights and return them such as B x A yields the same dinmensions as W

class LoRA(nn.Module):
    def __init__(self, features_in, features_out, rank, alpha, device):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        self.scale = alpha / rank
        nn.init.normal_(self.lora_A, mean=0, std=1)

    def forward(self, W):
        return W + torch.matmul(self.lora_B, self.lora_A).view(W.shape).contiguous() * self.scale


# This function takes the layer as the input and sets the features_in.features_out
# equal to the shape of the weight matrix. This will help the LoRA class to
# initialize the A and B Matrices

def layer_parametrization(layer, device, rank=1, lora_alpha=1):
    features_in, features_out = layer.weight.shape
    return LoRA(features_in, features_out, rank=rank, alpha=lora_alpha, device=device)



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.rank = -1
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)


if __name__ == '__main__':
    x, y = [], []
    for epoch in range(1, 201):
        x.append(epoch)
        y.append(get_current_consistency_weight(epoch, rampup_length=30))

    # 绘制折线图
    plt.plot(x, y)

    # 显示图表
    plt.show()
