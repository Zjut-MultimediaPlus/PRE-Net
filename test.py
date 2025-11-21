import argparse
import os.path

from matplotlib import pyplot as plt
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader

# from data_preprocess import Preprocess
from train_and_eval import compute_eva_gpu
from model.unet_model import *
from utils.data_loading import *
from utils.utils import layer_parametrization, init_distributed_mode

# 输入文件路径
test_dir_img = Path("/your_dataset_dir/")

# 设定随机种子
SEED = 2023
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def compute_csi_neighboor(pred, mask, nan_index, kernel_size=4):
    pooled_predictions = F.max_pool2d(pred, kernel_size=kernel_size)
    pooled_labels = F.max_pool2d(mask, kernel_size=kernel_size)

    # 对 nan_index 也进行相同的池化操作
    pooled_nan_index = F.max_pool2d(nan_index.float(), kernel_size=kernel_size).bool()

    # 将 NaN 位置的标签和预测结果忽略
    pooled_labels[pooled_nan_index] = torch.nan
    pooled_predictions[pooled_nan_index] = torch.nan

    # 使用 torch.isfinite 来过滤掉 NaN 值进行计算
    valid_mask = torch.isfinite(pooled_labels)

    pooled_labels = torch.where(pooled_labels >= 0.1, 1, 0)
    pooled_predictions = torch.where(pooled_predictions >= 0.1, 1, 0)

    hit = torch.sum((pooled_predictions == 1) & (pooled_labels == 1) & valid_mask)
    miss = torch.sum((pooled_predictions == 0) & (pooled_labels == 1) & valid_mask)
    false_alarm = torch.sum((pooled_predictions == 1) & (pooled_labels == 0) & valid_mask)
    return [hit, miss, false_alarm]


def test(reg_net, cls_net, dataloader):
    cls_net.eval()
    reg_net.eval()
    num_val_batches = len(dataloader)
    total_mask = []
    total_pred = []

    csi_4 = [0, 0, 0]
    csi_8 = [0, 0, 0]
    # iterate over the validation set
    with tqdm(total=num_val_batches, desc=f'eva round', unit='batch') as pbar:
        with torch.no_grad():
            for ids_index, batch in enumerate(dataloader):
                image, mask_true = batch['image'], batch['label']
                image = image.to(device=device, dtype=torch.float32)
                mask_true = mask_true.to(device=device, dtype=torch.float32)
                mask = torch.ones(1, 1, 16, 16).to(device=device)

                # predict the mask
                reg_mask_pred = reg_net(image[:, :8], mask)[0].squeeze()
                cls_mask_pred = cls_net(image[:, :8], mask)[0]

                cls_res = cls_mask_pred.argmax(dim=1).squeeze(1)  # 分类
                reg_mask_pred = reg_mask_pred.squeeze(1)
                reg_mask_pred[cls_res.squeeze() == 0] = 0

                mask_index = torch.isnan(mask_true)

                csi_4 = [x + y for x, y in zip(csi_4, compute_csi_neighboor(reg_mask_pred, mask_true, mask_index, kernel_size=4))]
                csi_8 = [x + y for x, y in zip(csi_8, compute_csi_neighboor(reg_mask_pred, mask_true, mask_index, kernel_size=8))]

                total_mask.append(mask_true[~mask_index])
                total_pred.append(reg_mask_pred[~mask_index.squeeze()])
                pbar.update(1)

    total_mask = torch.cat(total_mask, dim=0)
    total_pred = torch.cat(total_pred, dim=0)

    total_prmd_acc_list = compute_eva_gpu(total_mask, total_pred, 0.1)

    csi_4 = csi_4[0] / (csi_4[0] + csi_4[1] + csi_4[2] + 1e-10)
    csi_8 = csi_8[0] / (csi_8[0] + csi_8[1] + csi_8[2] + 1e-10)
    total_s = " total:\n  rmse    cc   pod  far  csi  csi_4  csi_8\n" \
              + 'pred {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}\n'.format(
        total_prmd_acc_list[0], total_prmd_acc_list[1], total_prmd_acc_list[2],
        total_prmd_acc_list[3], total_prmd_acc_list[4], csi_4, csi_8)

    logging.info('Validation str score: {}'.format(total_s))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')

    # 权重文件路径
    parser.add_argument('--load-reg', '-reg', type=str,
                        default="checkpoints/reg_checkpoint.pth")
    parser.add_argument('--load-cls', '-cls', type=str,
                        default="checkpoints/cls_checkpoint.pth")
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def pre_net(classes, device):
    mae4 = MAE_ViT(image_size=[128, 24], in_channel=128, mask_ratio=args.mask_ratio)
    mae8 = MAE_ViT(image_size=[64, 12], in_channel=256, mask_ratio=args.mask_ratio)
    mae16 = MAE_ViT(image_size=[32, 6], in_channel=512, mask_ratio=args.mask_ratio)

    mae4 = ViT_Classifier_v3(mae4, num_splits=3)
    mae8 = ViT_Classifier_v3(mae8, num_splits=3)
    mae16 = ViT_Classifier_v4(mae16)

    fe_block_1 = fe_block_mae_v3(n_channels=8, bilinear=args.bilinear, mae_block=[mae4, mae8, mae16])

    model = PRE_Net(n_channels=8, n_classes=classes, bilinear=args.bilinear, fe_block=fe_block_1).to(device)

    items = ['attn', 'mlp']
    for name, module in model.fe_block.named_modules():
        if any(item in name for item in items) and 'parametrizations' not in name:
            try:
                parametrize.register_parametrization(
                    module, "weight",
                    layer_parametrization(module, device)
                )
            except AttributeError:
                pass
                # logging.warning(f'layer {name} has no weights')
    return model



if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 回归模型
    reg_net = pre_net(1, device)
    reg_net_path = args.load_reg
    state_dict = torch.load(reg_net_path, map_location=device)['model']
    new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    reg_net.load_state_dict(new_state_dict)


    # 分类模型
    cls_net = pre_net(2, device)
    cls_net_path = args.load_cls
    state_dict = torch.load(cls_net_path, map_location=device)['model']
    new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    cls_net.load_state_dict(new_state_dict)

    test_dataset = IR_BasicDataset(test_dir_img, 1, 1)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, batch_size=32, num_workers=8,
                             pin_memory=True)

    test(reg_net=reg_net, cls_net=cls_net, dataloader=test_loader)
