import copy
import os
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

working_path = os.path.dirname(os.path.abspath(__file__))

from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity
from utils.utils import accuracy, SCDD_eval_all, AverageMeter

# Data and model choose
###############################################
from datasets import RS_ST as RS
# from models.TED import TED as Net
from models.SCanNet import SCanNet as Net

NET_NAME = 'SCanNet_psd'
DATA_NAME = 'ST'
###############################################
# Training options
###############################################
args = {
    'train_batch_size': 4,
    'val_batch_size': 8,
    'lr': 1e-4,
    'gpu': True,
    'epochs': 50,  # lan 1: 11, lan 2: 20, lan 3: 20
    'lr_decay_power': 1.5,
    'psd_train': True,
    'psd_TTA': True,
    'vis_psd': True,
    'psd_init_Fscd': 0.6,
    'print_freq': 50,
    'predict_step': 5,
    'pseudo_thred': 0.6,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME,
                              'SCanNet_psd_4e_mIoU75.15_Sek26.46_Fscd69.25_OA89.40_Loss0.37.pth')
}
###############################################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
writer = SummaryWriter(args['log_dir'])


class AverageThred(object):
    def __init__(self, num_classes):
        self.threds = np.ones((num_classes), dtype=float) * args['pseudo_thred']
        self.count = np.ones((num_classes), dtype=int)
        self.sum = self.threds * self.count

    def update(self, threds, count):
        self.count += np.array(count, dtype=int)
        self.sum += threds * count
        self.threds = self.sum / self.count

    def value(self):
        return np.clip(self.threds, 0.5, 0.9)


def calc_conf(prob, conf_thred):
    b, c, h, w = prob.size()
    conf, index = torch.max(prob, dim=1)
    index_onehot = F.one_hot(index.long(), num_classes=RS.num_classes).permute((0, 3, 1, 2))
    masked_prob = index_onehot * prob
    threds, len_c = np.zeros(c), np.zeros(c)
    for idx in range(c):
        masked_prob_i = torch.flatten(masked_prob[:, idx])
        masked_prob_i = masked_prob_i[masked_prob_i.nonzero()]
        len = masked_prob_i.size(0)

        if len > 0:
            conf_thred_i = np.percentile(masked_prob_i.cpu().numpy().flatten(), 100 * args['pseudo_thred'])
            threds[idx] = conf_thred_i
            len_c[idx] = len
        else:
            threds[idx] = args['pseudo_thred']
            len_c[idx] = 0

    conf_thred.update(threds, len_c)
    threds = torch.from_numpy(conf_thred.value()).unsqueeze(1).unsqueeze(2).cuda()
    thred_onehot = index_onehot * threds
    thredmap, _ = torch.max(thred_onehot, dim=1)
    conf = torch.ge(conf, thredmap)
    return conf, index


def main():
    net = Net(3, num_classes=RS.num_classes).cuda()
    # net.load_state_dict(torch.load(args['load_path']), strict=False)
    # freeze_model(net.FCN)

    train_set = RS.Data('train', random_flip=True)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], shuffle=True)
    val_set = RS.Data('val')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], shuffle=False)

    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    optimizer = torch.optim.NAdam(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=1e-4)

    # optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'],
    #                                 weight_decay=1e-4, momentum=0.9)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=1e-4,
    #                       momentum=0.9, nesterov=True)
    # optimizer = optim.AdamW(
    #     filter(lambda p: p.requires_grad, net.parameters()),
    #     lr=args['lr'],  # Learning rate
    #     weight_decay=1e-4  # Weight decay
    # )
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
    #                        lr=args['lr'],
    #                        weight_decay=1e-4)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], betas=(0.9, 0.999))
    curr_epoch = 0
    bestFscdV = 0
    train(train_loader, net, criterion, optimizer, val_loader, curr_epoch, bestFscdV)
    writer.close()
    print('Training finished.')


# training tiep từ path weight file da co
# def main():
#     net = Net(3, num_classes=RS.num_classes).cuda()
#
#     # Load model from checkpoint
#     checkpoint = torch.load(args['load_path'])
#     net.load_state_dict(checkpoint['model_state_dict'])
#
#     # Create optimizer
#     # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=5e-4,
#     #                       momentum=0.9, nesterov=True)
#     # optimizer = optim.AdamW(
#     #     filter(lambda p: p.requires_grad, net.parameters()),
#     #     lr=args['lr'],  # Learning rate
#     #     weight_decay=1e-4)  # Weight decay
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
#                            lr=args['lr'],
#                            weight_decay=1e-4)
#
#     # Load optimizer state
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
#     # Load current epoch and best values
#     curr_epoch = checkpoint['epoch'] + 1
#     bestFscdV = checkpoint['bestFscdV']
#
#     # Load datasets and dataloaders
#     train_set = RS.Data('train', random_flip=True)
#     train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], shuffle=True)
#     val_set = RS.Data('val')
#     val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], shuffle=False)
#
#     criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
#
#     # Continue training from the loaded state
#     train(train_loader, net, criterion, optimizer, val_loader, curr_epoch, bestFscdV)
#     writer.close()
#     print('Training finished.')


def train(train_loader, net, criterion, optimizer, val_loader, start_epoch, start_bestFscdV):
    net_psd = None
    conf_thred = AverageThred(RS.num_classes)

    bestaccT = 0
    bestFscdV = 0.0 + start_bestFscdV
    bestloss = 1.0
    bestaccV = 0.0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    criterion_sc = ChangeSimilarity().cuda()
    curr_epoch = 0 + start_epoch

    while True:
        torch.cuda.empty_cache()
        net.train()
        # freeze_model(net.FCN)
        start = time.time()
        acc_meter = AverageMeter()
        train_seg_loss = AverageMeter()
        train_bn_loss = AverageMeter()
        train_sc_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            iter_ratio = running_iter / all_iters
            adjust_lr(optimizer, iter_ratio)
            if len(data) > 4:
                imgs_A, imgs_B, labels_A, labels_B, imgs_id = data
            else:
                imgs_A, imgs_B, labels_A, labels_B = data
            if args['gpu']:
                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels_bn = (labels_A > 0).unsqueeze(1).cuda().float()
                labels_A = labels_A.cuda().long()
                labels_B = labels_B.cuda().long()

            optimizer.zero_grad()
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
            assert outputs_A.size()[1] == RS.num_classes

            if args['psd_train'] and bestFscdV > args['psd_init_Fscd']:
                if net_psd == None:
                    net_psd = copy.deepcopy(net)
                    net_psd.eval()
                with torch.no_grad():
                    out_change_psd, outputsA_psd, outputsB_psd = net_psd(imgs_A, imgs_B)
                    prob_A = F.softmax(outputsA_psd, dim=1)
                    prob_B = F.softmax(outputsB_psd, dim=1)
                    out_change_psd = F.sigmoid(out_change_psd)
                    if args['psd_TTA']:
                        imgs_A_v = torch.flip(imgs_A, [2])
                        imgs_B_v = torch.flip(imgs_B, [2])
                        out_change_v, outputs_A_v, outputs_B_v = net_psd(imgs_A_v, imgs_B_v)
                        outputs_A_v = torch.flip(outputs_A_v, [2])
                        outputs_B_v = torch.flip(outputs_B_v, [2])
                        out_change_v = torch.flip(out_change_v, [2])
                        prob_A += F.softmax(outputs_A_v, dim=1)
                        prob_B += F.softmax(outputs_B_v, dim=1)
                        out_change_psd += F.sigmoid(out_change_v)

                        imgs_A_h = torch.flip(imgs_A, [3])
                        imgs_B_h = torch.flip(imgs_B, [3])
                        out_change_h, outputs_A_h, outputs_B_h = net_psd(imgs_A_h, imgs_B_h)
                        outputs_A_h = torch.flip(outputs_A_h, [3])
                        outputs_B_h = torch.flip(outputs_B_h, [3])
                        out_change_h = torch.flip(out_change_h, [3])
                        prob_A += F.softmax(outputs_A_h, dim=1)
                        prob_B += F.softmax(outputs_B_h, dim=1)
                        out_change_psd += F.sigmoid(out_change_h)

                        imgs_A_hv = torch.flip(imgs_A, [2, 3])
                        imgs_B_hv = torch.flip(imgs_B, [2, 3])
                        out_change_hv, outputs_A_hv, outputs_B_hv = net_psd(imgs_A_hv, imgs_B_hv)
                        outputs_A_hv = torch.flip(outputs_A_hv, [2, 3])
                        outputs_B_hv = torch.flip(outputs_B_hv, [2, 3])
                        out_change_hv = torch.flip(out_change_hv, [2, 3])
                        prob_A += F.softmax(outputs_A_hv, dim=1)
                        prob_B += F.softmax(outputs_B_hv, dim=1)
                        out_change_psd += F.sigmoid(out_change_hv)

                        prob_A = prob_A / 4
                        prob_B = prob_B / 4
                        out_change_psd = out_change_psd / 4
                b, c, h, w = outputsA_psd.shape
                confA, A_index = calc_conf(prob_A, conf_thred)
                confB, B_index = calc_conf(prob_B, conf_thred)
                confAB = torch.logical_and(confA, confB)
                AB_same = torch.eq(A_index, B_index)
                confAB_same = torch.logical_and(confAB, AB_same)
                labels_unchange = torch.logical_not(labels_bn).squeeze()
                confAB_same_unchange = torch.logical_and(confAB_same, labels_unchange)
                pseudo_unchange = A_index * confAB_same_unchange

                labels_A += pseudo_unchange
                labels_B += pseudo_unchange

                if args['vis_psd'] and not running_iter % 1000:
                    psdA_color = RS.Index2Color(labels_A[0].cpu().detach().numpy())
                    psdB_color = RS.Index2Color(labels_B[0].cpu().detach().numpy())
                    io.imsave(os.path.join(args['pred_dir'],
                                           NET_NAME + '_psdA_epoch%diter%d.png' % (
                                               curr_epoch, running_iter)),
                              psdA_color)
                    io.imsave(os.path.join(args['pred_dir'],
                                           NET_NAME + '_psdB_epoch%diter%d.png' % (
                                               curr_epoch, running_iter)),
                              psdB_color)

            loss_seg = criterion(outputs_A, labels_A) + criterion(outputs_B, labels_B)
            loss_bn = weighted_BCE_logits(out_change, labels_bn)
            loss_sc = criterion_sc(outputs_A[:, 1:], outputs_B[:, 1:], labels_bn)

            loss = loss_seg * 0.5 + loss_bn + loss_sc
            loss.backward()
            optimizer.step()

            labels_A = labels_A.cpu().detach().numpy()
            labels_B = labels_B.cpu().detach().numpy()
            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A * change_mask.squeeze().long()).numpy()
            preds_B = (preds_B * change_mask.squeeze().long()).numpy()
            # batch_valid_sum = 0
            acc_curr_meter = AverageMeter()
            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                acc = (acc_A + acc_B) * 0.5
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_seg_loss.update(loss_seg.cpu().detach().numpy())
            train_bn_loss.update(loss_bn.cpu().detach().numpy())
            train_sc_loss.update(loss_sc.cpu().detach().numpy())

            writer.add_scalar('train seg_loss', train_seg_loss.val, running_iter)
            writer.add_scalar('train sc_loss', train_sc_loss.val, running_iter)
            writer.add_scalar('train accuracy', acc_meter.val * 100, running_iter)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

            curr_time = time.time() - start
            if (i + 1) % args['print_freq'] == 0 or (i + 1) == len(train_loader):
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train seg_loss %.4f bn_loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_seg_loss.val, train_bn_loss.val, acc_meter.val * 100))  # sc_loss %.4f, train_sc_loss.val, 

        Fscd_v, mIoU_v, Sek_v, acc_v, loss_v = validate(val_loader, net, criterion, curr_epoch)
        if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
        if Fscd_v > bestFscdV:
            bestFscdV = Fscd_v
            bestaccV = acc_v
            bestloss = loss_v
            net_psd = copy.deepcopy(net)
            conf_thred = AverageThred(RS.num_classes)
            # torch.save(net.state_dict(),
            #            os.path.join(args['chkpt_dir'], NET_NAME + '_%de_mIoU%.2f_Sek%.2f_Fscd%.2f_OA%.2f_Loss%.2f.pth' \
            #                         % (curr_epoch, mIoU_v * 100, Sek_v * 100, Fscd_v * 100, acc_v * 100, loss_v)))
            # save cac thong so de trainning tiep lan sau
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': curr_epoch,
                'bestFscdV': bestFscdV,
                'bestaccV': bestaccV,
                'bestloss': bestloss
            }, os.path.join(args['chkpt_dir'], NET_NAME + '_%de_mIoU%.2f_Sek%.2f_Fscd%.2f_OA%.2f_Loss%.2f.pth' \
                            % (curr_epoch, mIoU_v * 100, Sek_v * 100, Fscd_v * 100, bestaccV * 100, bestloss)))

            print('Save check point')
        print('Total time: %.1fs Best rec: Train acc %.2f, Val Fscd %.2f acc %.2f loss %.4f' % (
            time.time() - begin_time, bestaccT * 100, bestFscdV * 100, bestaccV * 100, bestloss))
        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, criterion, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    for vi, data in enumerate(val_loader):
        if len(data) > 4:
            imgs_A, imgs_B, labels_A, labels_B, imgs_id = data
        else:
            imgs_A, imgs_B, labels_A, labels_B = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()

        with torch.no_grad():
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
            loss_A = criterion(outputs_A, labels_A)
            loss_B = criterion(outputs_B, labels_B)
            loss = loss_A * 0.5 + loss_B * 0.5
        val_loss.update(loss.cpu().detach().numpy())

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A * change_mask.squeeze().long()).numpy()
        preds_B = (preds_B * change_mask.squeeze().long()).numpy()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_meter.update(acc)

        if curr_epoch % args['predict_step'] == 0 and vi == 0:
            pred_A_color = RS.Index2Color(preds_A[0])
            pred_B_color = RS.Index2Color(preds_B[0])
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + str(curr_epoch) + '_A.png'), pred_A_color)
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + str(curr_epoch) + '_B.png'), pred_B_color)
            print('Prediction saved!')

    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, RS.num_classes)

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f' \
          % (curr_time, val_loss.average(), Fscd * 100, IoU_mean * 100, Sek * 100, acc_meter.average() * 100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Fscd', Fscd * 100, curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average() * 100, curr_epoch)

    return Fscd, IoU_mean, Sek, acc_meter.avg, val_loss.avg


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def adjust_lr(optimizer, iter_ratio, init_lr=args['lr']):
    # scale_running_lr = math.sin((iter_ratio)*math.pi/2)
    scale_running_lr = ((1. - iter_ratio) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()
