import os
import sys
import time
import yaml
import shutil
import random
import argparse
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

from model.music_performer import MusicPerformer
from dataloader import REMISkylineToMidiTransformerDataset
from utils import pickle_load

sys.path.append('./model')
train_steps = 0


def log_epoch(log_file, log_data, is_init=False):
    if is_init:
        with open(log_file, 'w') as f:
            f.write('{:4} {:8} {:12} {:12}\n'.format('ep', 'steps', 'recons_loss', 'ep_time'))

    with open(log_file, 'a') as f:
        f.write('{:<4} {:<8} {:<12} {:<12}\n'.format(
            log_data['ep'], log_data['steps'], round(log_data['recons_loss'], 5), round(log_data['time'], 2)
        ))


def train_model(epoch, model, dloader, optim, sched, pad_token):
    model.train()
    recons_loss_rec = 0.
    accum_samples = 0

    print('[epoch {:03d}] training ...'.format(epoch))
    print('[epoch {:03d}] # batches = {}'.format(epoch, len(dloader)))
    st = time.time()

    for batch_idx, batch_samples in enumerate(dloader):
        model.zero_grad()

        batch_dec_inp = batch_samples['dec_input'].cuda(gpuid)
        batch_dec_tgt = batch_samples['dec_target'].cuda(gpuid)
        batch_track_mask = batch_samples['track_mask'].cuda(gpuid)
        batch_inp_lens = batch_samples['length']

        batch_chord_idx = batch_samples['chord_idx'].cuda(gpuid)
        batch_melody_idx = batch_samples['melody_idx'].cuda(gpuid)

        global train_steps
        train_steps += 1

        # get logits from model
        omit_feature_map_draw = random.random() > redraw_prob
        dec_logits = model(
            batch_dec_inp,
            seg_inp=batch_track_mask,
            chord_inp=None,
            attn_kwargs={'omit_feature_map_draw': omit_feature_map_draw}
        )
        losses = model.compute_loss(dec_logits, batch_dec_tgt)

        # clip gradient & update model
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        recons_loss_rec += batch_samples['id'].size(0) * losses['recons_loss'].item()
        accum_samples += batch_samples['id'].size(0)

        total_acc, chord_acc, melody_acc, others_acc = \
            compute_accuracy(dec_logits.cpu(), batch_dec_tgt.cpu(), batch_chord_idx, batch_melody_idx, pad_token)

        # anneal learning rate
        if train_steps < warmup_steps:
            curr_lr = max_lr * train_steps / warmup_steps
            optim.param_groups[0]['lr'] = curr_lr
        else:
            sched.step(train_steps - warmup_steps)

        print(
            ' -- epoch {:03d} | batch {:03d}/{:03d}: len: {}\n   '
            '* loss = {:.4f}, total_acc = {:.4f}, chord_acc = {:.4f}, '
            'melody_acc = {:.4f}, others_acc = {:.4f}, '
            'step = {}, time_elapsed = {:.2f} secs | redraw: {}'.format(
                epoch, batch_idx + 1, len(dloader), batch_inp_lens,
                recons_loss_rec / accum_samples, total_acc,
                chord_acc, melody_acc,  others_acc, train_steps,
                time.time() - st, (not omit_feature_map_draw)
            ))

        if not train_steps % log_interval:
            log_data = {
                'ep': epoch,
                'steps': train_steps,
                'recons_loss': recons_loss_rec / accum_samples,
                'time': time.time() - st
            }
            log_epoch(
                os.path.join(ckpt_dir, 'log.txt'), log_data,
                is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
            )

    print('[epoch {:03d}] training completed\n  -- loss = {:.4f}\n  -- time elapsed = {:.2f} secs.'.format(
        epoch, recons_loss_rec / accum_samples, time.time() - st
    ))
    log_data = {
        'ep': epoch,
        'steps': train_steps,
        'recons_loss': recons_loss_rec / accum_samples,
        'time': time.time() - st
    }
    log_epoch(
        os.path.join(ckpt_dir, 'log.txt'), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
    )

    return recons_loss_rec / accum_samples


def validate(model, dloader, pad_token, rounds=1):
    model.eval()
    loss_rec = []
    total_acc_rec = []
    chord_acc_rec = []
    melody_acc_rec = []
    others_acc_rec = []

    with torch.no_grad():
        for r in range(rounds):
            print(' >> validating ... (round {})'.format(r + 1))
            for batch_idx, batch_samples in enumerate(dloader):
                batch_dec_inp = batch_samples['dec_input'].cuda(gpuid)
                batch_dec_tgt = batch_samples['dec_target'].cuda(gpuid)
                batch_track_mask = batch_samples['track_mask'].cuda(gpuid)
                batch_chord_idx = batch_samples['chord_idx'].cuda(gpuid)
                batch_melody_idx = batch_samples['melody_idx'].cuda(gpuid)

                omit_feature_map_draw = random.random() > redraw_prob
                dec_logits = model(
                    batch_dec_inp,
                    seg_inp=batch_track_mask,
                    chord_inp=None,
                    attn_kwargs={'omit_feature_map_draw': omit_feature_map_draw}
                )

                losses = model.compute_loss(dec_logits, batch_dec_tgt)
                if not batch_idx % 5:
                    print('  -- batch #{:03d}: valloss = {:.4f}'.format(
                        batch_idx + 1, losses['recons_loss'].item()
                    ))
                loss_rec.append(losses['recons_loss'].item())

                total_acc, chord_acc, melody_acc, others_acc = \
                    compute_accuracy(dec_logits.cpu(), batch_dec_tgt.cpu(), batch_chord_idx, batch_melody_idx, pad_token)
                total_acc_rec.append(total_acc)
                chord_acc_rec.append(chord_acc)
                melody_acc_rec.append(melody_acc)
                others_acc_rec.append(others_acc)

    return loss_rec, total_acc_rec, chord_acc_rec, melody_acc_rec, others_acc_rec


def compute_accuracy(dec_logits, dec_target, inp_chord, inp_melody, pad_token):
    dec_pred = torch.argmax(dec_logits, dim=-1)
    dec_target = dec_target
    total_acc = np.mean(np.array((dec_pred[dec_target != pad_token] == dec_target[dec_target != pad_token])))
    chord_acc = np.mean(np.array((dec_pred[inp_chord == 1] == dec_target[inp_chord == 1])))
    melody_acc = np.mean(np.array((dec_pred[inp_melody == 1] == dec_target[inp_melody == 1])))
    others_acc = (total_acc * len(dec_target[dec_target != pad_token]) - chord_acc * len(dec_target[inp_chord == 1]) -
                  melody_acc * len(dec_target[inp_melody == 1])) / \
                 (len(dec_target[dec_target != pad_token]) - len(dec_target[inp_chord == 1]) - len(dec_target[inp_melody == 1]))
    return total_acc, chord_acc, melody_acc, others_acc


if __name__ == "__main__":
    # configuration
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--configuration',
                          choices=['stage2_accompaniment/config/pop1k7_pretrain.yaml',
                                   'stage2_accompaniment/config/emopia_finetune.yaml'],
                          help='configurations of training', required=True)
    required.add_argument('-r', '--representation',
                          choices=['absolute', 'functional'],
                          help='representation for symbolic music', required=True)
    args = parser.parse_args()

    train_conf_path = args.configuration
    train_conf = yaml.load(open(train_conf_path, 'r'), Loader=yaml.FullLoader)
    print(train_conf)
    representation = args.representation

    # training configurations
    train_conf_ = train_conf['training']
    gpuid = train_conf_['gpuid']
    torch.cuda.set_device(gpuid)
    warmup_steps = train_conf_['warmup_steps']
    max_lr = train_conf_['lr']
    min_lr = train_conf_['lr_scheduler']['eta_min']
    lr_decay_steps = train_conf_['lr_scheduler']['T_max']
    redraw_prob = train_conf_['feat_redraw_prob']
    max_epochs = train_conf_['num_epochs']
    ckpt_dir = train_conf_['ckpt_dir'].format(representation)
    pretrained_param_path = train_conf_['trained_params']
    pretrained_optimizer_path = train_conf_['trained_optim']
    ckpt_interval = train_conf_['ckpt_interval']
    val_interval = 1
    log_interval = train_conf_['log_interval']

    # dataloader configurations
    batch_size = train_conf['data_loader']['batch_size']
    train_split = train_conf['data_loader']['train_split']
    val_split = train_conf['data_loader']['val_split']
    data_path = train_conf['data_loader']['data_path'].format(representation)
    vocab_path = train_conf['data_loader']['vocab_path'].format(representation)
    event2idx = pickle_load(vocab_path)[0]
    idx2event = pickle_load(vocab_path)[1]

    # model configurations
    model_conf = train_conf['model']
    predict_key = False

    # load dataset
    dset = REMISkylineToMidiTransformerDataset(
        data_dir=data_path,
        vocab_file=vocab_path,
        model_dec_seqlen=model_conf['max_len'],
        pieces=pickle_load(train_split),
        pad_to_same=True,
        predict_key=predict_key
    )
    val_dset = REMISkylineToMidiTransformerDataset(
        data_dir=data_path,
        vocab_file=vocab_path,
        model_dec_seqlen=model_conf['max_len'],
        pieces=pickle_load(val_split),
        pad_to_same=True,
        predict_key=predict_key
    )
    print('[info] # training pieces:', len(dset.pieces))
    dloader = DataLoader(
        dset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    val_dloader = DataLoader(
        val_dset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    # load model
    model = MusicPerformer(
        dset.vocab_size, model_conf['n_layer'], model_conf['n_head'],
        model_conf['d_model'], model_conf['d_ff'], model_conf['d_embed'],
        use_segment_emb=model_conf['use_segemb'],
        n_segment_types=model_conf['n_segment_types'],
        favor_feature_dims=model_conf['feature_map']['n_dims'],
        use_chord_mhot_emb=False
    ).cuda(gpuid)

    if pretrained_param_path:
        pretrained_dict = torch.load(pretrained_param_path, map_location='cpu')
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if 'feature_map.omega' not in k
        }
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict)

    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# params:', n_params)
    print('segemb:', model.segemb)

    opt_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(opt_params, lr=max_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, lr_decay_steps, eta_min=min_lr
    )
    if pretrained_optimizer_path:
        optimizer.load_state_dict(
            torch.load(pretrained_optimizer_path, map_location='cpu')
        )

    params_dir = os.path.join(ckpt_dir, 'params/')
    optimizer_dir = os.path.join(ckpt_dir, 'optim/')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    if not os.path.exists(optimizer_dir):
        os.makedirs(optimizer_dir)

    shutil.copy(train_conf_path, os.path.join(ckpt_dir, 'config.yaml'))

    for ep in range(max_epochs):
        loss = train_model(ep + 1, model, dloader, optimizer, scheduler, dset.pad_token)
        if not (ep + 1) % ckpt_interval:
            torch.save(model.state_dict(),
                       os.path.join(params_dir, 'ep{:03d}_loss{:.3f}_params.pt'.format(ep + 1, loss))
                       )
            torch.save(optimizer.state_dict(),
                       os.path.join(optimizer_dir, 'ep{:03d}_loss{:.3f}_optim.pt'.format(ep + 1, loss))
                       )
        if not (ep + 1) % val_interval:
            val_losses, total_acc_rec, chord_acc_rec, melody_acc_rec, others_acc_rec = \
                validate(model, val_dloader, val_dset.pad_token)
            with open(os.path.join(ckpt_dir, 'valloss.txt'), 'a') as f:
                f.write("ep{:03d} | loss: {:.3f} | valloss: {:.3f} (±{:.3f}) | total_acc: {:.3f} | "
                        "chord_acc: {:.3f} | melody_acc: {:.3f} | others_acc: {:.3f}\n".format(
                    ep + 1, loss, np.mean(val_losses), np.std(val_losses),
                    np.mean(total_acc_rec), np.mean(chord_acc_rec), np.mean(melody_acc_rec), np.mean(others_acc_rec)
                ))
