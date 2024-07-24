import os
import sys
import shutil
import time
import yaml
import torch
import argparse
import numpy as np
from torch import optim
from torch.utils.data import DataLoader

from model.plain_transformer import PlainTransformer
from dataloader import SkylineFullSongTransformerDataset
from utils import pickle_load

sys.path.append('./model/')


def train(epoch, model, dloader, optim, sched, pad_token):
    model.train()
    recons_loss_rec = 0.
    accum_samples = 0

    print('[epoch {:03d}] training ...'.format(epoch))
    st = time.time()

    for batch_idx, batch_samples in enumerate(dloader):
        # if batch_idx > 4:
        #     break
        mems = tuple()
        # print ('[debug] got batch samples')
        for segment in range(max(batch_samples['n_seg'])):
            # print ('[debug] segment:', segment)

            model.zero_grad()

            dec_input = batch_samples['dec_inp_{}'.format(segment)].permute(1, 0).cuda()
            dec_target = batch_samples['dec_tgt_{}'.format(segment)].permute(1, 0).cuda()
            dec_seg_len = batch_samples['dec_seg_len_{}'.format(segment)].cuda()

            inp_chord = batch_samples['inp_chord_{}'.format(segment)]
            inp_melody = batch_samples['inp_melody_{}'.format(segment)]
            # print ('[debug]', dec_input.size(), dec_target.size(), dec_seg_len.size())
            global train_steps
            train_steps += 1

            # print ('[debug] prior to model forward(), train steps:', train_steps)
            dec_logits, mems = \
                model(
                    dec_input, mems, dec_seg_len=dec_seg_len
                )

            # print ('[debug] got model output')
            # compute loss
            losses = model.compute_loss(
                dec_logits, dec_target
            )

            total_acc, chord_acc, melody_acc, others_acc = \
                compute_accuracy(dec_logits.cpu(), dec_target.cpu(), inp_chord, inp_melody, pad_token)

            # clip gradient & update model
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            recons_loss_rec += batch_samples['id'].size(0) * losses['ce_loss'].item()
            accum_samples += batch_samples['id'].size(0)

            # anneal learning rate
            if train_steps < warmup_steps:
                curr_lr = max_lr * train_steps / warmup_steps
                optim.param_groups[0]['lr'] = curr_lr
            else:
                sched.step(train_steps - warmup_steps)

            if not train_steps % log_interval:
                log_data = {
                    'ep': epoch,
                    'steps': train_steps,
                    'ce_loss': recons_loss_rec / accum_samples,
                    'total': total_acc,
                    'chord': chord_acc,
                    'melody': melody_acc,
                    'others': others_acc,
                    'time': time.time() - st
                }
                log_epoch(
                    os.path.join(ckpt_dir, log_file), log_data,
                    is_init=not os.path.exists(os.path.join(ckpt_dir, log_file))
                )

        print('-- ep {:03d} | batch {:03d}: loss = {:.4f}, total_acc = {:.4f}, '
              'chord_acc = {:.4f}, melody_acc = {:.4f}, others_acc = {:.4f}, '
              'step = {}, time_elapsed = {:.2f} secs'.format(
            epoch,
            batch_idx,
            recons_loss_rec / accum_samples,
            total_acc,
            chord_acc,
            melody_acc,
            others_acc,
            train_steps,
            time.time() - st
        ))

    return recons_loss_rec / accum_samples, time.time() - st


def validate(epoch, model, dloader, pad_token, rounds=1):
    model.eval()
    recons_loss_rec = []
    total_acc_rec = []
    chord_acc_rec = []
    melody_acc_rec = []
    others_acc_rec = []

    print('[epoch {:03d}] validating ...'.format(epoch))
    with torch.no_grad():
        for r in range(rounds):
            for batch_idx, batch_samples in enumerate(dloader):
                # if batch_idx > 4:
                #     break
                mems = tuple()
                for segment in range(max(batch_samples['n_seg'])):
                    dec_input = batch_samples['dec_inp_{}'.format(segment)].permute(1, 0).cuda()
                    dec_target = batch_samples['dec_tgt_{}'.format(segment)].permute(1, 0).cuda()
                    dec_seg_len = batch_samples['dec_seg_len_{}'.format(segment)].cuda()
                    inp_chord = batch_samples['inp_chord_{}'.format(segment)]
                    inp_melody = batch_samples['inp_melody_{}'.format(segment)]

                    dec_logits, mems = \
                        model(
                            dec_input, mems, dec_seg_len=dec_seg_len
                        )
                    # compute loss
                    losses = model.compute_loss(
                        dec_logits, dec_target
                    )

                    total_acc, chord_acc, melody_acc, others_acc = \
                        compute_accuracy(dec_logits.cpu(), dec_target.cpu(), inp_chord, inp_melody, pad_token)

                    if not (batch_idx + 1) % 10:
                        print('  valloss = {:.4f}, total_acc = {:.4f}, chord_acc = {:.4f}, '
                              'melody_acc = {:.4f}, others_acc = {:.4f}, '.format(
                            round(losses['ce_loss'].item(), 3),
                            total_acc,
                            chord_acc,
                            melody_acc,
                            others_acc))
                    recons_loss_rec.append(losses['ce_loss'].item())
                    total_acc_rec.append(total_acc)
                    chord_acc_rec.append(chord_acc)
                    melody_acc_rec.append(melody_acc)
                    others_acc_rec.append(others_acc)

    return recons_loss_rec, total_acc_rec, chord_acc_rec, melody_acc_rec, others_acc_rec


def log_epoch(log_file, log_data, is_init=False):
    if is_init:
        with open(log_file, 'w') as f:
            f.write('{:4} {:8} {:12} {:12} {:12}\n'.format(
                'ep', 'steps', 'ce_loss', 'total_acc', 'chord_acc', 'melody_acc', 'others_acc', 'ep_time', 'total_time'
            ))

    with open(log_file, 'a') as f:
        f.write('{:<4} {:<8} {:<12} {:<12} {:<12}\n'.format(
            log_data['ep'],
            log_data['steps'],
            round(log_data['ce_loss'], 5),
            round(log_data['time'], 2),
            round(time.time() - init_time, 2)
        ))

    return


def compute_accuracy(dec_logits, dec_target, inp_chord, inp_melody, pad_token):
    dec_pred = torch.argmax(dec_logits, dim=-1).permute(1, 0)
    dec_target = dec_target.permute(1, 0)
    total_acc = np.mean(np.array((dec_pred[dec_target != pad_token] == dec_target[dec_target != pad_token])))
    chord_acc = np.mean(np.array((dec_pred[inp_chord == 1] == dec_target[inp_chord == 1])))
    melody_acc = np.mean(np.array((dec_pred[inp_melody == 1] == dec_target[inp_melody == 1])))
    others_acc = (total_acc * len(dec_target[dec_target != pad_token]) - chord_acc * len(dec_target[inp_chord == 1]) -
                  melody_acc * len(dec_target[inp_melody == 1])) / \
                 (len(dec_target[dec_target != pad_token]) - len(dec_target[inp_chord == 1]) - len(dec_target[inp_melody == 1]))
    return total_acc, chord_acc, melody_acc, others_acc


if __name__ == '__main__':
    # configuration
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--configuration',
                          choices=['config/hooktheory_pretrain.yaml', 'config/emopia_finetune.yaml',
                                   'config/pop1k7_pretrain.yaml', 'config/emopia_finetune_full.yaml'],
                          help='configurations of training', required=True)
    required.add_argument('-r', '--representation',
                          choices=['absolute', 'functional'],
                          help='representation for symbolic music', required=True)
    args = parser.parse_args()

    config_path = args.configuration
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    print(config)
    representation = args.representation
    ckpt_dir = config['output']['ckpt_dir'].format(representation)

    torch.cuda.device(config['device'])
    train_steps = 0 if config['training']['trained_steps'] is None else config['training']['trained_steps']
    start_epoch = 0 if config['training']['trained_epochs'] is None else config['training']['trained_epochs']
    warmup_steps = config['training']['warmup_steps']
    log_interval = config['training']['log_interval']
    max_lr = config['training']['max_lr']
    log_file = 'log.txt' if start_epoch == 0 else 'log_from_ep{:03d}.txt'.format(start_epoch)

    optim_ckpt_path = config['pretrained_optim_path']
    param_ckpt_path = config['pretrained_param_path']

    init_time = time.time()

    params_dir = os.path.join(ckpt_dir, 'params/') if start_epoch == 0 \
        else os.path.join(ckpt_dir, 'params_from_ep{:03d}/'.format(start_epoch))
    optimizer_dir = os.path.join(ckpt_dir, 'optim/') if start_epoch == 0 \
        else os.path.join(ckpt_dir, 'optim_from_ep{:03d}/'.format(start_epoch))

    dset = SkylineFullSongTransformerDataset(
        config['data']['data_dir'].format(representation),
        config['data']['vocab_path'].format(representation),
        pieces=pickle_load(config['data']['train_split']),
        # do_augment=True if "lmd" not in config['data']['data_dir'] else False,
        do_augment=False,
        model_dec_seqlen=config['model']['decoder']['tgt_len'],
        max_n_seg=config['data']['max_n_seg'],
        # max_pitch=108, min_pitch=48,
        max_pitch=108, min_pitch=21,
        convert_dict_event=True
    )

    val_dset = SkylineFullSongTransformerDataset(
        config['data']['data_dir'],
        config['data']['vocab_path'],
        pieces=pickle_load(config['data']['val_split']),
        do_augment=False,
        model_dec_seqlen=config['model']['decoder']['tgt_len'],
        max_n_seg=config['data']['max_n_seg'],
        # max_pitch=108, min_pitch=48,
        max_pitch=108, min_pitch=21,
        convert_dict_event=True
    )
    print('[dset lens]', len(dset), len(val_dset))

    dloader = DataLoader(
        dset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=24,
        collate_fn=dset.collate_fn
    )
    val_dloader = DataLoader(
        val_dset,
        batch_size=4,
        num_workers=24,
        collate_fn=val_dset.collate_fn
    )

    mconf = config['model']
    # torch.cuda.set_device(1)
    # print (torch.cuda.current_device())
    model = PlainTransformer(
        mconf['d_word_embed'],
        dset.vocab_size,
        mconf['decoder']['n_layer'],
        mconf['decoder']['n_head'],
        mconf['decoder']['d_model'],
        mconf['decoder']['d_ff'],
        mconf['decoder']['mem_len'],
        mconf['decoder']['tgt_len'],
        dec_dropout=mconf['decoder']['dropout'],
        pre_lnorm=mconf['pre_lnorm']
    ).cuda()
    print('[info] # params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    opt_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(opt_params, lr=config['training']['max_lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['lr_decay_steps'],
        eta_min=config['training']['min_lr']
    )

    if optim_ckpt_path:
        optimizer.load_state_dict(
            torch.load(optim_ckpt_path, map_location=config['device'])
        )

    if param_ckpt_path:
        pretrained_dict = torch.load(param_ckpt_path, map_location=config['device'])
        model.load_state_dict(
            pretrained_dict
        )

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    if not os.path.exists(optimizer_dir):
        os.makedirs(optimizer_dir)

    shutil.copy(config_path, os.path.join(ckpt_dir, 'config.yaml'))

    for ep in range(start_epoch, config['training']['max_epoch']):
        recons_loss, ep_time = train(ep + 1, model, dloader, optimizer, scheduler, dset.pad_token)
        if not (ep + 1) % config['output']['ckpt_interval']:
            torch.save(model.state_dict(),
                       os.path.join(params_dir, 'ep{:03d}_loss{:.3f}_params.pt'.format(ep + 1, recons_loss))
                       )
            torch.save(optimizer.state_dict(),
                       os.path.join(optimizer_dir, 'ep{:03d}_loss{:.3f}_optim.pt'.format(ep + 1, recons_loss))
                       )

        if not (ep + 1) % config['training']['val_interval']:
            val_recons_losses, total_acc_rec, chord_acc_rec, melody_acc_rec, others_acc_rec = \
                validate(ep + 1, model, val_dloader, val_dset.pad_token)
            valloss_file = os.path.join(ckpt_dir, 'valloss.txt') if start_epoch == 0 \
                else os.path.join(ckpt_dir, 'valloss_from_ep{:03d}.txt'.format(start_epoch))

            if os.path.exists(valloss_file):
                with open(valloss_file, 'a') as f:
                    f.write("ep{:03d} | loss: {:.3f} | valloss: {:.3f} (±{:.3f}) | total_acc: {:.3f} | "
                            "chord_acc: {:.3f} | melody_acc: {:.3f} | others_acc: {:.3f}\n".format(
                            ep + 1, recons_loss, np.mean(val_recons_losses), np.std(val_recons_losses),
                            np.mean(total_acc_rec), np.mean(chord_acc_rec), np.mean(melody_acc_rec), np.mean(others_acc_rec)
                    ))
            else:
                with open(valloss_file, 'w') as f:
                    f.write("ep{:03d} | loss: {:.3f} | valloss: {:.3f} (±{:.3f}) | total_acc: {:.3f} | "
                            "chord_acc: {:.3f} | melody_acc: {:.3f} | others_acc: {:.3f}\n".format(
                            ep + 1, recons_loss, np.mean(val_recons_losses), np.std(val_recons_losses),
                            np.mean(total_acc_rec), np.mean(chord_acc_rec), np.mean(melody_acc_rec), np.mean(others_acc_rec)
                    ))

        print('[epoch {:03d}] training completed\n  -- loss = {:.4f}\n  -- time elapsed = {:.2f} secs.'.format(
            ep + 1,
            recons_loss,
            ep_time,
        ))
        log_data = {
            'ep': ep + 1,
            'steps': train_steps,
            'ce_loss': recons_loss,
            'time': ep_time
        }
        log_epoch(
            os.path.join(ckpt_dir, log_file), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, log_file))
        )
