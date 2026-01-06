import os
import sys
import json
import argparse
import time
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import random

from configs.config_transformer import cfg, merge_cfg_from_file
from datasets.datasets import create_dataset
from models.CORTEX import CORTEX, AddSpatialInfo
from models.transformer_decoder import TransformerDecoder
from utils.logger import Logger
from utils.utils import AverageMeter, accuracy, set_mode, save_checkpoint, \
                        decode_sequence, decode_sequence_transformer, decode_beams, \
                        build_optimizer, coco_gen_format_save, one_hot_encode
from utils.vis_utils import visualize_att
from easydict import EasyDict
from torch.cuda.amp import autocast, GradScaler
from scheduler import build_scheduler


# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--visualize_every', type=int, default=10)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--exp_dir', type=str, default='./exp')
parser.add_argument('--run_name', type=str, default='debugging')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--itda_lambda', type=float, default=0.1)
parser.add_argument('--resume_checkpoint', type=str, default=None, help="Path to checkpoint for resuming training")
args = parser.parse_args()

# Load the config from the YAML file
merge_cfg_from_file(args.cfg)

# Overwrite values from existing config
cfg.exp_dir = args.exp_dir
cfg.exp_name = args.run_name
cfg.train.optim.lr = args.lr
cfg.data.train.batch_size = args.batch_size
cfg.train.optim.weight_decay = args.weight_decay
cfg.train.max_iter = args.max_iter
cfg.train.optim.itda_lambda = args.itda_lambda


# Device configuration
use_cuda = torch.cuda.is_available()
if args.gpu == -1:
    gpu_ids = cfg.gpu_id
else:
    gpu_ids = [args.gpu]
torch.backends.cudnn.enabled  = True
default_gpu_device = gpu_ids[0]
torch.cuda.set_device(default_gpu_device)
device = torch.device("cuda" if use_cuda else "cpu")

# Experiment configuration
exp_dir = cfg.exp_dir
exp_name = cfg.exp_name
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

output_dir = os.path.join(exp_dir, exp_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cfg_file_save = os.path.join(output_dir, 'cfg.json')
json.dump(cfg, open(cfg_file_save, 'w'))

sample_dir = os.path.join(output_dir, 'eval_gen_samples')
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
sample_subdir_format = '%s_samples_%d'

sent_dir = os.path.join(output_dir, 'eval_sents')
if not os.path.exists(sent_dir):
    os.makedirs(sent_dir)
sent_subdir_format = '%s_sents_%d'

snapshot_dir = os.path.join(output_dir, 'snapshots')
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)
snapshot_file_format = '%s_checkpoint_%d.pt'

train_logger = Logger(cfg, output_dir, is_train=True)
val_logger = Logger(cfg, output_dir, is_train=False)

random.seed(1111)
np.random.seed(1111)
torch.manual_seed(1111)

# Create model
change_detector = CORTEX(cfg)
change_detector.to(device)

speaker = TransformerDecoder(cfg)
speaker.to(device)

spatial_info = AddSpatialInfo()

print(change_detector)
print(speaker)

with open(os.path.join(output_dir, 'model_print'), 'w') as f:
    print(change_detector, file=f)
    print(speaker, file=f)
    print(spatial_info, file=f)

# Data loading part
train_dataset, train_loader = create_dataset(cfg, 'train')
val_dataset, val_loader = create_dataset(cfg, 'val')
train_size = len(train_dataset)
val_size = len(val_dataset)

all_params = list(change_detector.parameters()) + list(speaker.parameters())
optimizer = build_optimizer(all_params, cfg)
scaler = GradScaler()

# Scheduler configuration for PolyLR
scheduler_cfg = EasyDict({
    'type_name': 'PolyLR',
    'keywords': {
        'gamma': cfg.train.optim.gamma,        # Learning rate decay ratio
        'n_iteration': args.max_iter      # Total training iterations (use max_iter)
    }
})

lr_scheduler = build_scheduler(scheduler_cfg, optimizer)

# Train loop
t = 0
epoch = 0

# Load checkpoint
if args.resume_checkpoint:
    checkpoint_path = args.resume_checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        change_detector.load_state_dict(checkpoint['change_detector_state'])
        speaker.load_state_dict(checkpoint['speaker_state'])

        # Load Optimizer and Scheduler state
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])

        # Overwrite n_iteration (step_size) for PolyLR in case it was overwritten by state dict
        if hasattr(lr_scheduler, 'step_size') and lr_scheduler.step_size < args.max_iter:
             print(f"Updating scheduler step_size from {lr_scheduler.step_size} to {args.max_iter}")
             lr_scheduler.step_size = args.max_iter
        
        # Force update optimizer LR based on new step_size and current iteration
        # Note: PolyLR uses _step_count which matches iteration count in this context
        with torch.no_grad():
            new_lrs = lr_scheduler.get_lr()
            for param_group, lr in zip(optimizer.param_groups, new_lrs):
                param_group['lr'] = lr

        # # Verify Learning Rate
        # current_lr = optimizer.param_groups[0]['lr']
        # print(f"DEBUG: Current Learning Rate immediately after resume: {current_lr}")

        # Restore AMP Scaler (Important)
        scaler.load_state_dict(checkpoint['scaler_state'])

        # Restore training state
        t = checkpoint['iteration']
        epoch = checkpoint['epoch']
        print(f"Resumed from iteration {t}, epoch {epoch}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Starting fresh.")
else:
    print("No checkpoint provided. Starting fresh.")

set_mode('train', [change_detector, speaker])

while t < cfg.train.max_iter:
    epoch += 1
    print('Starting epoch %d' % epoch)
    speaker_loss_avg = AverageMeter()
    cdcr_loss_avg = AverageMeter()
    sim_loss_avg = AverageMeter()
    img_to_txt_loss_avg = AverageMeter()
    total_loss_avg = AverageMeter()
    
    if epoch > cfg.train.scheduled_sampling_start and cfg.train.scheduled_sampling_start >= 0:
        frac = (epoch - cfg.train.scheduled_sampling_start) // cfg.train.scheduled_sampling_increase_every
        ss_prob_prev = ss_prob
        ss_prob = min(cfg.train.scheduled_sampling_increase_prob * frac,
                      cfg.train.scheduled_sampling_max_prob)
        speaker.ss_prob = ss_prob
        if ss_prob_prev != ss_prob:
            print('Updating scheduled sampling rate: %.4f -> %.4f' % (ss_prob_prev, ss_prob))
    for i, batch in enumerate(train_loader):
        iter_start_time = time.time()

        d_feats, nsc_feats, sc_feats, \
        c_d_feats, c_sc_feats, c_n_feats, \
        labels, labels_with_ignore, no_chg_labels, no_chg_labels_with_ignore, masks, no_chg_masks, aux_labels_pos, aux_labels_neg, \
        d_img_paths, nsc_img_paths, sc_img_paths = batch

        batch_size = d_feats.size(0)
        labels = labels.squeeze(1)
        labels_with_ignore = labels_with_ignore.squeeze(1)
        no_chg_labels = no_chg_labels.squeeze(1)
        no_chg_labels_with_ignore = no_chg_labels_with_ignore.squeeze(1)
        masks = masks.squeeze(1).float()
        no_chg_masks = no_chg_masks.squeeze(1).float()

        d_feats, nsc_feats, sc_feats = d_feats.to(device), nsc_feats.to(device), sc_feats.to(device) # torch.Size([128, 1024, 14, 14])
        c_d_feats = [feat.to(device) for feat in c_d_feats]
        c_sc_feats = [feat.to(device) for feat in c_sc_feats]
        c_n_feats = [feat.to(device) for feat in c_n_feats]
        d_feats, nsc_feats, sc_feats = \
            spatial_info(d_feats), spatial_info(nsc_feats), spatial_info(sc_feats)   # torch.Size([128, 1026, 14, 14])
        labels, labels_with_ignore, masks = labels.to(device), labels_with_ignore.to(device), masks.to(device)
        no_chg_labels, no_chg_labels_with_ignore, no_chg_masks = no_chg_labels.to(device), no_chg_labels_with_ignore.to(device), no_chg_masks.to(device)
        aux_labels_pos, aux_labels_neg = aux_labels_pos.to(device), aux_labels_neg.to(device)

        
        optimizer.zero_grad()

        with autocast():

            encoder_output_pos, dirl_loss_pos, img_to_txt_loss_pos = change_detector(d_feats, sc_feats, c_d_feats, c_sc_feats)
            encoder_output_neg, dirl_loss_neg, img_to_txt_loss_neg = change_detector(d_feats, nsc_feats, c_d_feats, c_n_feats)


            loss_pos, att_pos, ccr_loss_pos = speaker._forward(encoder_output_pos,
                                              labels, masks, labels_with_ignore=labels_with_ignore)
            
            loss_neg, att_neg, ccr_loss_neg = speaker._forward(encoder_output_neg,
                                              no_chg_labels, no_chg_masks, labels_with_ignore=no_chg_labels_with_ignore)
            


            speaker_loss = 0.5 * loss_pos + 0.5 * loss_neg
            speaker_loss_val = speaker_loss.item()

            dirl_loss = 0.5 * dirl_loss_pos + 0.5 * dirl_loss_neg
            dirl_loss_val = dirl_loss.item()

            img_to_txt_loss = 0.5 * img_to_txt_loss_pos + 0.5 * img_to_txt_loss_neg
            img_to_txt_loss_val = img_to_txt_loss.item()

            ccr_loss = 0.5 * ccr_loss_pos + 0.5 * ccr_loss_neg
            ccr_loss_val = ccr_loss.item()

            total_loss = speaker_loss + 0.03 * dirl_loss + 0.05 * ccr_loss + args.itda_lambda * img_to_txt_loss
            total_loss_val = total_loss.item()

        
            speaker_loss_avg.update(speaker_loss_val, 2 * batch_size)
            cdcr_loss_avg.update(dirl_loss_val, 2 * batch_size)
            sim_loss_avg.update(ccr_loss_val, 2 * batch_size)
            img_to_txt_loss_avg.update(img_to_txt_loss_val, 2 * batch_size)
            total_loss_avg.update(total_loss_val, 2 * batch_size)

            stats = {}

            stats['speaker_loss'] = speaker_loss_val
            stats['avg_speaker_loss'] = speaker_loss_avg.avg
            stats['cdcr_loss'] = dirl_loss_val
            stats['avg_cdcr_loss'] = cdcr_loss_avg.avg
            stats['sim_loss'] = ccr_loss_val
            stats['avg_sim_loss'] = sim_loss_avg.avg
            stats['img_to_txt_loss'] = img_to_txt_loss_val
            stats['avg_img_to_txt_loss'] = img_to_txt_loss_avg.avg
            stats['total_loss'] = total_loss_val
            stats['avg_total_loss'] = total_loss_avg.avg

            stats['avg_total_loss'] = total_loss_avg.avg

        scaler.scale(total_loss).backward()
        if cfg.train.grad_clip != -1.0:  # enable, -1 == disable
            scaler.unscale_(optimizer)  # unscale the gradients before clipping
            nn.utils.clip_grad_norm_(change_detector.parameters(), cfg.train.grad_clip)
            nn.utils.clip_grad_norm_(speaker.parameters(), cfg.train.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()


        iter_end_time = time.time() - iter_start_time


        t += 1


        if t % cfg.train.log_interval == 0:
            train_logger.print_current_stats(epoch, i, t, stats, iter_end_time)
            train_logger.plot_current_stats(
                epoch,
                float(i * batch_size) / train_size, stats, 'loss')
            

        if t % cfg.train.snapshot_interval == 0:
            speaker_state = speaker.state_dict()
            chg_det_state = change_detector.state_dict()
            checkpoint = {
                'change_detector_state': chg_det_state,
                'speaker_state': speaker_state,
                'optimizer_state': optimizer.state_dict(),
                'lr_scheduler_state': lr_scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),  # Added: Save AMP state
                'iteration': t,
                'epoch': epoch,
                'model_cfg': cfg
            }
            save_path = os.path.join(snapshot_dir,
                                     snapshot_file_format % (exp_name, t))
            save_checkpoint(checkpoint, save_path)

            
            print('Running eval at iter %d' % t)
            set_mode('eval', [change_detector, speaker])
            with torch.no_grad():
                test_iter_start_time = time.time()

                idx_to_word = train_dataset.get_idx_to_word()

                if args.visualize:
                    sample_subdir_path = sample_subdir_format % (exp_name, t)
                    sample_save_dir = os.path.join(sample_dir, sample_subdir_path)
                    if not os.path.exists(sample_save_dir):
                        os.makedirs(sample_save_dir)
                sent_subdir_path = sent_subdir_format % (exp_name, t)
                sent_save_dir = os.path.join(sent_dir, sent_subdir_path)
                if not os.path.exists(sent_save_dir):
                    os.makedirs(sent_save_dir)


                result_sents_pos = {}
                result_sents_neg = {}

                total_time = 0
                total_images = 0
                for val_i, val_batch in enumerate(val_loader):
                    val_batch_size = val_batch[0].size(0)  # batch size based on d_feats
                                        
                    d_feats, nsc_feats, sc_feats, \
                    c_d_feats, c_sc_feats, c_n_feats, \
                    labels, labels_with_ignore, no_chg_labels, no_chg_labels_with_ignore, masks, no_chg_masks, aux_labels_pos, aux_labels_neg, \
                    d_img_paths, nsc_img_paths, sc_img_paths = val_batch
                    

                    val_batch_size = d_feats.size(0)

                    d_feats, nsc_feats, sc_feats = d_feats.to(device), nsc_feats.to(device), sc_feats.to(device)
                    c_d_feats = [feat.to(device) for feat in c_d_feats]
                    c_sc_feats = [feat.to(device) for feat in c_sc_feats]
                    c_n_feats = [feat.to(device) for feat in c_n_feats]
                    d_feats, nsc_feats, sc_feats = \
                        spatial_info(d_feats), spatial_info(nsc_feats), spatial_info(sc_feats)
                    labels, labels_with_ignore, masks = labels.to(device), labels_with_ignore.to(device), masks.to(device)
                    no_chg_labels, no_chg_labels_with_ignore, no_chg_masks = no_chg_labels.to(device), no_chg_labels_with_ignore.to(device), no_chg_masks.to(device)
                    aux_labels_pos, aux_labels_neg = aux_labels_pos.to(device), aux_labels_neg.to(device)


                    encoder_output_pos, dirl_loss_pos, img_to_txt_loss_pos = change_detector(d_feats, sc_feats, c_d_feats, c_sc_feats)
                    encoder_output_neg, dirl_loss_neg, img_to_txt_loss_neg = change_detector(d_feats, nsc_feats, c_d_feats, c_n_feats)
                    

                    speaker_output_pos, _ = speaker.sample(encoder_output_pos)
                    speaker_output_neg, _ = speaker.sample(encoder_output_neg)

                    gen_sents_pos = decode_sequence_transformer(idx_to_word, speaker_output_pos[:, 1:]) # no start
                    gen_sents_neg = decode_sequence_transformer(idx_to_word, speaker_output_neg[:, 1:])

                    for val_j in range(speaker_output_pos.size(0)):
                        gts = decode_sequence_transformer(idx_to_word, labels[val_j][:, 1:])
                        gts_neg = decode_sequence_transformer(idx_to_word, no_chg_labels[val_j][:, 1:])
                        
                        sent_pos = gen_sents_pos[val_j]
                        sent_neg = gen_sents_neg[val_j]
                        image_id = d_img_paths[val_j].split('_')[-1]
                        result_sents_pos[image_id] = sent_pos
                        result_sents_neg[image_id + '_n'] = sent_neg

                        result_sents_pos[image_id] = sent_pos
                        result_sents_neg[image_id + '_n'] = sent_neg

                result_save_path_pos = os.path.join(sent_save_dir, 'sc_results.json')
                result_save_path_neg = os.path.join(sent_save_dir, 'nsc_results.json')
                coco_gen_format_save(result_sents_pos, result_save_path_pos)
                coco_gen_format_save(result_sents_neg, result_save_path_neg)

            set_mode('train', [change_detector, speaker])
