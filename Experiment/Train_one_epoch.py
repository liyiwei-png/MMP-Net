import torch.optim
import os
import time
from utils import *
import Config as config
import warnings
from thop import profile, clever_format
warnings.filterwarnings("ignore")


def get_model_complexity_info(model, input_shape):

    input = torch.randn(*input_shape).cuda()
    macs, params = profile(model, inputs=(input,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)

    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)

    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)

def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
    logging_mode = 'Train' if model.training else 'Val'

    if epoch == 0:
        input_shape = (1, config.n_channels, config.img_size, config.img_size)
        macs, params = get_model_complexity_info(model, input_shape)
        logger.info(f'[{logging_mode}] 模型 {model_type} 的计算量(FLOPs): {macs}, 参数量: {params}')
        if writer is not None:
            writer.add_text(f'{logging_mode}_Model_Complexity', f'FLOPs: {macs}, Params: {params}', epoch)

    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0

    dices = []
    for i, (sampled_batch, names) in enumerate(loader, 1):

        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        images, masks = sampled_batch['image'], sampled_batch['label']
        images, masks = images.cuda(), masks.cuda()


        preds = model(images)
        out_loss = criterion(preds, masks.float())  # Loss

        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

        train_iou = iou_on_batch(masks, preds)
        train_dice = criterion._show_dice(preds, masks.float())

        batch_time = time.time() - end
        if epoch % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path + str(epoch) + '/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images, masks, preds, names, vis_path)
        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        iou_sum += len(images) * train_iou
        dice_sum += len(images) * train_dice

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size * (i - 1) + len(images))
            average_time = time_sum / (config.batch_size * (i - 1) + len(images))
            train_iou_average = iou_sum / (config.batch_size * (i - 1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size * (i - 1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0, logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups), logger=logger)

        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if writer is not None:
        writer.add_scalar('epoch_' + logging_mode + '_' + loss_name, average_loss, epoch)

        writer.add_scalar('epoch_' + logging_mode + '_' + 'iou', train_iou_average, epoch)

        writer.add_scalar('epoch_' + logging_mode + '_dice', train_dice_avg, epoch)

    return average_loss, train_dice_avg


if __name__ == '__main__':
    from nets.ACC_UNet import ACC_UNet
    import thop
    from torchsummary import summary
    
    x = torch.randn(1, 3, 224, 224)
    device = torch.device('cpu')
    model = ACC_UNet(n_channels=3, n_classes=1)
    MACs, Params = thop.profile(model, inputs=(x,), verbose=False)
    FLOPs = MACs * 2
    MACs, FLOPs, Params = thop.clever_format([MACs, FLOPs, Params], "%.3f")
    print(summary(model, (3, 224, 224)))
    print(f"MACs:{MACs}")
    print(f"FLOPs:{FLOPs}")
    print(f"Params:{Params}")
    output = model(x)
    print(output.shape)
