# -*- coding: utf-8 -*-
"""
Unified training / testing entry point.

Supported tasks (--task):
    cnn                : image + clinical + pathology fusion (Dataset_img_clinical)
    cp                 : clinical + pathology only (Dataset_img_clinical_onlyCP)
    dce                : DCE-MRI only (Dataset_img_clinical_DCE)
    dcemgmlo           : DCE + MG-MLO (Dataset_img)
    dcemgmlogtv        : DCE + MG-MLO + GTV masks (Dataset_img_clinical_org)
    mgmlo              : MG-MLO only (Dataset_img_clinical_MGMLO)
    convnext           : DCE+MG+clinical+pathology, ConvNeXt fusion head
    resnet50           : DCE+MG+clinical+pathology, ResNet50 fusion head
    vgg19              : DCE+MG+clinical+pathology, VGG19 fusion head
    pathology_pretrain : image+clinical+pathology with feature extractor pre-loaded

Run modes (--mode):
    train    : training + validation
    test     : inference on external test set
    prospect : inference on prospective test set
"""
import os
import argparse
import random
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
from dataset import (
    Dataset_img_clinical,
    Dataset_img_clinical_onlyCP,
    Dataset_img_clinical_DCE,
    Dataset_img,
    Dataset_img_clinical_org,
    Dataset_img_clinical_MGMLO,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
from Multimodal_Fusion_Network import (
    MFNet_img_pathology_transcat_l3m_orgTransformer_1,
    MFNet_CNN_img_pathology_transcat_l3m_orgTransformer_1,
    MFNet_pathology_transcat_l3m_orgTransformer_1,
    MFNet_DCE_transcat_l3m_orgTransformer_1,
    MFNet_img_transcat_l3m_orgTransformer_1,
    MFNet_MGMLO_transcat_l3m_orgTransformer_1,
    MFNet_img_pathology_transcat_l3m_convnext,
    MFNet_img_pathology_transcat_l3m_resnet50,
    MFNet_img_pathology_transcat_l3m_vgg19,
    MFNet_img_pathology_transcat_l3m_orgTransformer_1_org,
)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
TASKS = {
    'cnn': dict(
        dataset='Dataset_img_clinical',
        net='MFNet_CNN_img_pathology_transcat_l3m_orgTransformer_1',
        save_subdir='trained_models_CNN',
        forward='full',
        has_clinical=True,
        has_pathology=True,
        has_img_mr=True,
        has_img_mg=True,
    ),
    'cp': dict(
        dataset='Dataset_img_clinical_onlyCP',
        net='MFNet_pathology_transcat_l3m_orgTransformer_1',
        save_subdir='trained_models_CP',
        forward='cp',            # (cln, path) -> 3 outputs
        has_clinical=True,
        has_pathology=True,
        has_img_mr=False,
        has_img_mg=False,
    ),
    'dce': dict(
        dataset='Dataset_img_clinical_DCE',
        net='MFNet_DCE_transcat_l3m_orgTransformer_1',
        save_subdir='trained_models_DCE',
        forward='dce',           # (img_mr) -> 2 outputs
        has_clinical=False,
        has_pathology=False,
        has_img_mr=True,
        has_img_mg=False,
    ),
    'dcemgmlo': dict(
        dataset='Dataset_img',
        net='MFNet_img_transcat_l3m_orgTransformer_1',
        save_subdir='trained_models_img',
        forward='img2',          # (img_mr, img_mg) -> 3 outputs
        has_clinical=False,
        has_pathology=False,
        has_img_mr=True,
        has_img_mg=True,
    ),
    'dcemgmlogtv': dict(
        dataset='Dataset_img_clinical_org',
        net='MFNet_img_pathology_transcat_l3m_orgTransformer_1_org',
        save_subdir='trained_models_org',
        forward='full',          # (img_mr, img_mg, cln, path) -> 5 outputs
        has_clinical=True,
        has_pathology=True,
        has_img_mr=True,
        has_img_mg=True,
    ),
    'mgmlo': dict(
        dataset='Dataset_img_clinical_MGMLO',
        net='MFNet_MGMLO_transcat_l3m_orgTransformer_1',
        save_subdir='trained_models_MGMLO',
        forward='mg',            # (img_mg) -> 2 outputs
        has_clinical=False,
        has_pathology=False,
        has_img_mr=False,
        has_img_mg=True,
    ),
    'convnext': dict(
        dataset='Dataset_img_clinical',
        net='MFNet_img_pathology_transcat_l3m_convnext',
        save_subdir='trained_models_convnext',
        forward='full',
        has_clinical=True,
        has_pathology=True,
        has_img_mr=True,
        has_img_mg=True,
    ),
    'resnet50': dict(
        dataset='Dataset_img_clinical',
        net='MFNet_img_pathology_transcat_l3m_resnet50',
        save_subdir='trained_models_resnet50',
        forward='full',
        has_clinical=True,
        has_pathology=True,
        has_img_mr=True,
        has_img_mg=True,
    ),
    'vgg19': dict(
        dataset='Dataset_img_clinical',
        net='MFNet_img_pathology_transcat_l3m_vgg19',
        save_subdir='trained_models_vgg19',
        forward='full',
        has_clinical=True,
        has_pathology=True,
        has_img_mr=True,
        has_img_mg=True,
    ),
    'pathology_pretrain': dict(
        dataset='Dataset_img_clinical',
        net='MFNet_img_pathology_transcat_l3m_orgTransformer_1',
        save_subdir='trained_models',
        forward='full',
        has_clinical=True,
        has_pathology=True,
        has_img_mr=True,
        has_img_mg=True,
        # Task-specific pre-training logic
        pretrain_from_fold=10,
        pretrain_ckpt_subdir='pretrainparams/',
        pretrain_ckpt_name='best_AUC_val.pth',
        load_prefixes=['extractor_MR_DCE', 'extractor_MG_MLO', 'MLP_cln', 'MLP_pathology'],
    ),
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Unified training script for NSLN classification')
    p.add_argument('--task', type=str, required=True, choices=list(TASKS.keys()),
                   help='which task / architecture to run')
    p.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'prospect'],
                   help='train / external test / prospective test')
    p.add_argument('--gpu', type=str, default='0', help='which gpu is used')
    p.add_argument('--bs', type=int, default=4, help='batch size')
    p.add_argument('--epoch', type=int, default=100, help='number of training epochs')
    p.add_argument('--seed', type=int, default=42, help='random seed')
    p.add_argument('--name', type=str, default=None, help='experiment name (default: task)')
    p.add_argument('--date', type=str, default='',
                   help='date/tag appended to save dir, e.g. all')
    p.add_argument('--dropout', type=float, default=0.0, help='dropout in fusion transformer')
    p.add_argument('--fold', type=int, default=1, help='fold number')
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    p.add_argument('--wd', type=float, default=0.01, help='weight decay')
    p.add_argument('--data_dir', type=str, default=None,
                   help='dataset root; default depends on mode: ../NSLN (train), '
                        '../NSLN_waibu (test), ../NSLN_prospect (prospect)')
    p.add_argument('--split_path', type=str, default=None,
                   help='split pkl path; default: <data_dir>/output_data_fold<fold>.pkl')
    p.add_argument('--num_class', type=int, default=2)
    p.add_argument('--loss_threshold', type=float, default=0.0)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DATASET_MAP = {
    'Dataset_img_clinical': Dataset_img_clinical,
    'Dataset_img_clinical_onlyCP': Dataset_img_clinical_onlyCP,
    'Dataset_img_clinical_DCE': Dataset_img_clinical_DCE,
    'Dataset_img': Dataset_img,
    'Dataset_img_clinical_org': Dataset_img_clinical_org,
    'Dataset_img_clinical_MGMLO': Dataset_img_clinical_MGMLO,
}

NET_MAP = {
    'MFNet_img_pathology_transcat_l3m_orgTransformer_1': MFNet_img_pathology_transcat_l3m_orgTransformer_1,
    'MFNet_CNN_img_pathology_transcat_l3m_orgTransformer_1': MFNet_CNN_img_pathology_transcat_l3m_orgTransformer_1,
    'MFNet_pathology_transcat_l3m_orgTransformer_1': MFNet_pathology_transcat_l3m_orgTransformer_1,
    'MFNet_DCE_transcat_l3m_orgTransformer_1': MFNet_DCE_transcat_l3m_orgTransformer_1,
    'MFNet_img_transcat_l3m_orgTransformer_1': MFNet_img_transcat_l3m_orgTransformer_1,
    'MFNet_MGMLO_transcat_l3m_orgTransformer_1': MFNet_MGMLO_transcat_l3m_orgTransformer_1,
    'MFNet_img_pathology_transcat_l3m_convnext': MFNet_img_pathology_transcat_l3m_convnext,
    'MFNet_img_pathology_transcat_l3m_resnet50': MFNet_img_pathology_transcat_l3m_resnet50,
    'MFNet_img_pathology_transcat_l3m_vgg19': MFNet_img_pathology_transcat_l3m_vgg19,
    'MFNet_img_pathology_transcat_l3m_orgTransformer_1_org': MFNet_img_pathology_transcat_l3m_orgTransformer_1_org,
}

def build_inputs(batch, cfg, device):
    """Split a dataloader batch into (args_for_net, labels)."""
    ID = batch[0]
    labels = batch[-1].to(device).long()
    rest = batch[1:-1]

    # Dataset returns in fixed order: img_MR, img_MG, cln, path (only present ones)
    idx = 0
    net_args = []
    if cfg['has_img_mr']:
        net_args.append(rest[idx].to(device).float()); idx += 1
    if cfg['has_img_mg']:
        net_args.append(rest[idx].to(device).float()); idx += 1
    if cfg['has_clinical']:
        net_args.append(rest[idx].to(device).float()); idx += 1
    if cfg['has_pathology']:
        net_args.append(rest[idx].to(device).float()); idx += 1
    return ID, net_args, labels


def forward_net(net, net_args, cfg):
    """Run the network and return the last output (logits)."""
    return net(*net_args)[-1]


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------
def run_epoch(net, loader, optimizer, loss_fc, cfg, device, num_class,
              train=True, desc='train', epoch=0, total_epoch=0, loss_threshold=0.0):
    net.train() if train else net.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()
    ep_loss, ep_onehot, ep_score, ep_gt, ep_pred, ep_id = [], [], [], [], [], []

    with ctx:
        for i, batch in enumerate(loader):
            ID, net_args, labels = build_inputs(batch, cfg, device)
            labels_onehot = torch.zeros((labels.size(0), num_class), device='cpu') \
                .scatter_(1, labels.cpu().unsqueeze(1), 1).float()

            if train:
                optimizer.zero_grad()
            outputs = forward_net(net, net_args, cfg)
            loss = loss_fc(outputs, labels)
            loss = (loss - loss_threshold).abs() + loss_threshold
            if train:
                loss.backward()
                optimizer.step()

            prob = torch.softmax(outputs, dim=1)
            pred = torch.argmax(prob, dim=1)

            ep_score.append(prob.detach().cpu())
            ep_onehot.append(labels_onehot)
            ep_loss.append(loss.item())
            ep_gt.append(labels.cpu().numpy())
            ep_pred.append(pred.detach().cpu().numpy())
            ep_id.append(ID)

            if train:
                print('[%d/%d, %d/%d] %s_loss: %.3f' %
                      (epoch + 1, total_epoch, i + 1, len(loader), desc, loss.item()))

    onehot = torch.cat(ep_onehot, 0).numpy().astype(np.uint8)
    score = torch.cat(ep_score, 0).numpy()
    gt = np.concatenate(ep_gt)
    pred = np.concatenate(ep_pred)
    ids = np.concatenate(ep_id)
    auc = roc_auc_score(onehot, score)
    acc = accuracy_score(gt, pred)
    mean_loss = float(np.mean(ep_loss))
    return dict(loss=mean_loss, auc=auc, acc=acc,
                label=gt, pred=pred, score=score, ids=ids)


def save_epoch_arrays(save_dir, split, res, epoch):
    out = os.path.join(save_dir, 'log', split)
    np.save(os.path.join(out, 'truelabel_{}.npy'.format(epoch)), res['label'])
    np.save(os.path.join(out, 'predlabel_{}.npy'.format(epoch)), res['pred'])
    np.save(os.path.join(out, 'predscores_{}.npy'.format(epoch)), res['score'])
    np.save(os.path.join(out, 'ID_{}.npy'.format(epoch)), res['ids'])


# ---------------------------------------------------------------------------
# Pre-training loader (only used by `pathology_pretrain`)
# ---------------------------------------------------------------------------
def load_pretrain_if_needed(net, task_cfg, fold):
    if 'pretrain_from_fold' not in task_cfg:
        return

    src_fold = task_cfg['pretrain_from_fold']
    if fold == src_fold:
        print('Current fold is the source fold {}; skip loading pretrained feature extractor.'.format(fold))
        return

    ckpt = os.path.join(task_cfg['pretrain_ckpt_subdir'], task_cfg['pretrain_ckpt_name'])
    if not os.path.isfile(ckpt):
        print('WARNING: pretrained ckpt not found: {}; training from scratch.'.format(ckpt))
        return

    print('=' * 60)
    print('Loading pretrained feature extractor from fold {}:'.format(src_fold))
    print(ckpt)

    org = torch.load(ckpt, map_location='cpu')
    model_dict = net.state_dict()
    prefixes = task_cfg['load_prefixes']

    loaded, skipped = {}, []
    for k, v in org.items():
        if k not in model_dict:
            skipped.append((k, 'not in model')); continue
        if k.split('.', 1)[0] not in prefixes:
            continue  # fusion head etc.; skipped silently
        if v.shape != model_dict[k].shape:
            skipped.append((k, 'shape mismatch {} vs {}'.format(v.shape, model_dict[k].shape)))
            continue
        loaded[k] = v

    model_dict.update(loaded)
    net.load_state_dict(model_dict)

    print('Loaded {} params:'.format(len(loaded)))
    for k in loaded:
        print('   [OK] {}'.format(k))
    if skipped:
        print('Skipped {} params:'.format(len(skipped)))
        for k, r in skipped:
            print('   [SKIP] {}  <- {}'.format(k, r))
    print('=' * 60)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_train(args, task_cfg, device):
    dataset_cls = DATASET_MAP[task_cfg['dataset']]
    data_dir = args.data_dir or '../NSLN'
    split_path = args.split_path or os.path.join(data_dir, 'output_data_fold{}.pkl'.format(args.fold))
    name = args.name or args.task

    save_dir = os.path.join(
        task_cfg['save_subdir'],
        '{}_fold{}'.format(name, args.fold),
        'bs{}_epoch{}_seed{}_{}_drop{}'.format(args.bs, args.epoch, args.seed, args.date, args.dropout),
    )
    os.makedirs(save_dir, exist_ok=True)

    train_writer = SummaryWriter(os.path.join(save_dir, 'log/train'), flush_secs=2)
    train_true_writer = SummaryWriter(os.path.join(save_dir, 'log/train_ture'), flush_secs=2)
    val_writer = SummaryWriter(os.path.join(save_dir, 'log/val'), flush_secs=2)
    print('save_dir:', save_dir)

    train_data = dataset_cls(data_dir, split_path, data_set='train', augment=True)
    train_true_data = dataset_cls(data_dir, split_path, data_set='train', augment=False)
    val_data = dataset_cls(data_dir, split_path, data_set='val', augment=False)

    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    train_true_loader = DataLoader(train_true_data, batch_size=args.bs, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=args.bs, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    print('train_length: %i  val_length: %i' % (train_data.len, val_data.len))

    net_cls = NET_MAP[task_cfg['net']]
    net = net_cls(n_classes=args.num_class, no_cuda=False, drop=args.dropout).cuda()
    load_pretrain_if_needed(net, task_cfg, args.fold)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = MultiStepLR(optimizer,
                               milestones=[int(0.9 * args.epoch)],
                               gamma=0.1, last_epoch=-1)
    loss_fc = nn.CrossEntropyLoss()

    best_auc, best_acc = 0.0, 0.0
    print('training')

    for epoch in range(args.epoch):
        lr = optimizer.param_groups[0]['lr']

        res_train = run_epoch(net, train_loader, optimizer, loss_fc, task_cfg, device,
                              args.num_class, train=True, desc='train',
                              epoch=epoch, total_epoch=args.epoch,
                              loss_threshold=args.loss_threshold)
        lr_scheduler.step()

        res_true = run_epoch(net, train_true_loader, optimizer, loss_fc, task_cfg, device,
                             args.num_class, train=False, desc='train_ture')
        res_val = run_epoch(net, val_loader, optimizer, loss_fc, task_cfg, device,
                            args.num_class, train=False, desc='val')

        print('[%d/%d] train_loss: %.3f train_AUC: %.3f train_ture_AUC: %.3f val_AUC: %.3f '
              'train_ACC: %.3f train_ture_ACC: %.3f val_ACC: %.3f' % (
                  epoch, args.epoch,
                  res_train['loss'], res_train['auc'], res_true['auc'], res_val['auc'],
                  res_train['acc'], res_true['acc'], res_val['acc']))

        if res_val['auc'] > best_auc:
            best_auc = res_val['auc']
            torch.save(net.state_dict(), os.path.join(save_dir, 'best_AUC_val_{}.pth'.format(epoch)))
        if res_val['acc'] > best_acc:
            best_acc = res_val['acc']
            torch.save(net.state_dict(), os.path.join(save_dir, 'best_ACC_val_{}.pth'.format(epoch)))

        train_writer.add_scalar('lr', lr, epoch)
        train_writer.add_scalar('loss', res_train['loss'], epoch)
        train_writer.add_scalar('AUC', res_train['auc'], epoch)
        train_writer.add_scalar('ACC', res_train['acc'], epoch)
        save_epoch_arrays(save_dir, 'train', res_train, epoch)

        train_true_writer.add_scalar('AUC', res_true['auc'], epoch)
        train_true_writer.add_scalar('ACC', res_true['acc'], epoch)
        save_epoch_arrays(save_dir, 'train_ture', res_true, epoch)

        val_writer.add_scalar('loss', res_val['loss'], epoch)
        val_writer.add_scalar('AUC', res_val['auc'], epoch)
        val_writer.add_scalar('ACC', res_val['acc'], epoch)
        val_writer.add_scalar('best_AUC_val', best_auc, epoch)
        save_epoch_arrays(save_dir, 'val', res_val, epoch)

        if epoch + 1 == args.epoch:
            torch.save(net.state_dict(), os.path.join(save_dir, 'epoch{}.pth'.format(epoch + 1)))

    train_writer.close()
    train_true_writer.close()
    val_writer.close()
    print('saved_model_name:', save_dir)


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
def _resolve_ckpt(train_save_dir):
    ckpt = os.path.join(train_save_dir, 'best_AUC_val.pth')
    if not os.path.isfile(ckpt):
        cands = sorted(glob.glob(os.path.join(train_save_dir, 'best_AUC_val_*.pth')))
        if cands:
            ckpt = cands[-1]
    if not os.path.isfile(ckpt):
        raise FileNotFoundError('checkpoint not found in {}'.format(train_save_dir))
    return ckpt


def run_test(args, task_cfg, device, data_dir, save_subdir_prefix):
    dataset_cls = DATASET_MAP[task_cfg['dataset']]
    split_path = args.split_path or os.path.join(data_dir, 'output_data_fold{}.pkl'.format(args.fold))
    name = args.name or args.task

    # Locate the checkpoint saved during training
    train_save_dir = os.path.join(
        task_cfg['save_subdir'],
        '{}_fold{}'.format(name, args.fold),
        'bs{}_epoch{}_seed{}_{}_drop{}'.format(args.bs, args.epoch, args.seed, args.date, args.dropout),
    )
    ckpt = _resolve_ckpt(train_save_dir)

    save_dir = os.path.join(
        save_subdir_prefix,
        '{}_fold{}'.format(name, args.fold),
        'bs{}_epoch{}_seed{}_{}_drop{}'.format(args.bs, args.epoch, args.seed, args.date, args.dropout),
    )
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(save_dir, 'log/test'), flush_secs=2)

    print('save_dir:', save_dir)
    print('loading ckpt:', ckpt)

    test_data = dataset_cls(data_dir, split_path, data_set='test', augment=False)
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)
    print('test_length: %i' % test_data.len)

    net_cls = NET_MAP[task_cfg['net']]
    net = net_cls(n_classes=args.num_class, no_cuda=False, drop=args.dropout).cuda()
    net.load_state_dict(torch.load(ckpt))
    loss_fc = nn.CrossEntropyLoss()

    res = run_epoch(net, test_loader, None, loss_fc, task_cfg, device,
                    args.num_class, train=False, desc='test')
    print('[TEST] loss: %.3f AUC: %.3f ACC: %.3f' % (res['loss'], res['auc'], res['acc']))

    epoch = 0
    writer.add_scalar('loss', res['loss'], epoch)
    writer.add_scalar('AUC', res['auc'], epoch)
    writer.add_scalar('ACC', res['acc'], epoch)
    save_epoch_arrays(save_dir, 'test', res, epoch)
    writer.close()
    print('saved_model_name:', save_dir)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    task_cfg = TASKS[args.task]

    if args.mode == 'train':
        run_train(args, task_cfg, device)
    elif args.mode == 'test':
        data_dir = args.data_dir or '../NSLN_waibu'
        run_test(args, task_cfg, device, data_dir, save_subdir_prefix='trained_models_test')
    elif args.mode == 'prospect':
        data_dir = args.data_dir or '../NSLN_prospect'
        run_test(args, task_cfg, device, data_dir, save_subdir_prefix='trained_models_prospect')
    else:
        raise ValueError(args.mode)


if __name__ == '__main__':
    main()