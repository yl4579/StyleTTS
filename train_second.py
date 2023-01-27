import os
import os.path as osp
import re
import sys
import yaml
import shutil
import numpy as np
import torch
import click
import warnings
warnings.simplefilter('ignore')
from torch.utils.tensorboard import SummaryWriter

# load packages
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa

from models import *
from meldataset import build_dataloader
from utils import *
from optimizers import build_optimizer
import time


# for data augmentation
class TimeStrech(nn.Module):
    def __init__(self, scale):
        super(TimeStrech, self).__init__()
        self.scale = scale

    def forward(self, x):
        mel_size = x.size(-1)
        
        x = F.interpolate(x, scale_factor=(1, self.scale), align_corners=False,
                          recompute_scale_factor=True, mode='bilinear').squeeze()
        
        return x.unsqueeze(1)

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):

    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)
    device = config.get('device', 'cpu')
    epochs = config.get('epochs_2nd', 100)
    save_freq = config.get('save_freq', 2)
    train_path = config.get('train_data', None)
    val_path = config.get('val_data', None)
    multigpu = config.get('multigpu', False)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)

    # load data
    train_list, val_list = get_data_path_list(train_path, val_path)

    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        dataset_config={},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=2,
                                      device=device,
                                      dataset_config={})
    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 1e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    model = build_model(Munch(config['model_params']), text_aligner, pitch_extractor)

    _ = [model[key].to(device) for key in model]

    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                      scheduler_params_dict= {key: scheduler_params.copy() for key in model})

    # multi-GPU support
    if multigpu:
        for key in model:
            model[key] = MyDataParallel(model[key])
        
    if config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False):
        model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                    load_only_params=config.get('load_only_params', True))
    else:
        start_epoch = 0
        iters = 0
        
        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            print('Loading the first stage model at %s ...' % first_stage_path)
            model, optimizer, start_epoch, iters = load_checkpoint(model, optimizer, first_stage_path,
                                        load_only_params=True)
        else:
            raise ValueError('You need to specify the path to the first stage model.') 

    best_loss = float('inf')  # best test loss
    loss_train_record = list([])
    loss_test_record = list([])

    loss_params = Munch(config['loss_params'])
    TMA_epoch = loss_params.TMA_epoch
    TMA_CEloss = loss_params.TMA_CEloss

    for epoch in range(epochs):
        running_loss = 0
        start_time = time.time()
        criterion = nn.L1Loss() 
        
        _ = [model[key].eval() for key in model]

        model.predictor.train()
        model.discriminator.train()
        for i, batch in enumerate(train_dataloader):

            batch = [b.to(device) for b in batch]
            texts, input_lengths, mels, mel_input_length = batch

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** model.text_aligner.n_down)).to('cuda')
                mel_mask = length_to_mask(mel_input_length).to('cuda')
                text_mask = length_to_mask(input_lengths).to(texts.device)

                _, _, s2s_attn_feat = model.text_aligner(mels, mask, texts)

                s2s_attn_feat = s2s_attn_feat.transpose(-1, -2)
                s2s_attn_feat = s2s_attn_feat[..., 1:]
                s2s_attn_feat = s2s_attn_feat.transpose(-1, -2)

                with torch.no_grad():
                    text_mask = length_to_mask(input_lengths).to(texts.device)
                    attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                    attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                    attn_mask = (attn_mask < 1)
                
                if TMA_CEloss:
                    s2s_attn = F.softmax(s2s_attn_feat, dim=1) # along the mel dimension
                else:
                    s2s_attn = F.softmax(s2s_attn_feat, dim=-1) # along the text dimension

                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** model.text_aligner.n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                # encode
                m = length_to_mask(input_lengths)

                t_en = model.text_encoder(texts, input_lengths, m)
                asr = (t_en @ s2s_attn_mono)

                d_gt = s2s_attn_mono.sum(axis=-1).detach()

                # compute the style of the entire utterance
                # this operation cannot be done in batch because of the avgpool layer (may need to work on masked avgpool)
                ss = []
                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item())
                    mel = mels[bib, :, :mel_input_length[bib]]
                    s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                    ss.append(s)
                s = torch.stack(ss).squeeze()

            d, _ = model.predictor(t_en, s, 
                                                    input_lengths, 
                                                    s2s_attn_mono, 
                                                    m)
            # augmentation
            with torch.no_grad():
                M = np.random.random()
                ts = TimeStrech(1+ (np.random.random()-0.5)*M*0.5)

                mels = ts(mels.unsqueeze(1)).squeeze(1)
                mels = mels[:, :, :mels.size(-1) // 2 * 2]

                mel_input_length = torch.floor(ts.scale * mel_input_length) // 2 * 2 

                mask = length_to_mask(mel_input_length // (2 ** model.text_aligner.n_down)).to('cuda')
                mel_mask = length_to_mask(mel_input_length).to('cuda')
                text_mask = length_to_mask(input_lengths).to(texts.device)
                
                # might have misalignment due to random scaling
                try:
                    _, _, s2s_attn_feat = model.text_aligner(mels, mask, texts)
                except:
                    continue

                s2s_attn_feat = s2s_attn_feat.transpose(-1, -2)
                s2s_attn_feat = s2s_attn_feat[..., 1:]
                s2s_attn_feat = s2s_attn_feat.transpose(-1, -2)

                with torch.no_grad():
                    text_mask = length_to_mask(input_lengths).to(texts.device)
                    attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                    attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                    attn_mask = (attn_mask < 1)
                
                if TMA_CEloss:
                    s2s_attn = F.softmax(s2s_attn_feat, dim=1) # along the mel dimension
                else:
                    s2s_attn = F.softmax(s2s_attn_feat, dim=-1) # along the text dimension

                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** model.text_aligner.n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                # encode
                asr = (t_en @ s2s_attn_mono)

            _, p = model.predictor(t_en, s, 
                                                    input_lengths, 
                                                    s2s_attn_mono, 
                                                    m)

            # get clips
            mel_len = int(mel_input_length.min().item() / 2 - 1)
            en = []
            gt = []
            p_en = []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start+mel_len])
                p_en.append(p[bib, :, random_start:random_start+mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()

            if gt.size(-1) < 80:
                continue

            with torch.no_grad():
                s = model.style_encoder(gt.unsqueeze(1))

                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()

                asr_real = model.text_aligner.get_feature(gt)

                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)

                mel_rec_gt = model.decoder(en, F0_real, N_real, s)

            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

            mel_rec = model.decoder(en, F0_fake, N_fake, s)

            loss_F0_rec =  (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            # discriminator loss
            optimizer.zero_grad()
            mel_rec_gt.requires_grad_()
            out, _ = model.discriminator(mel_rec_gt.unsqueeze(1))
            loss_real = adv_loss(out, 1)
            loss_reg = r1_reg(out, mel_rec_gt)
            out, _ = model.discriminator(mel_rec.detach().unsqueeze(1))
            loss_fake = adv_loss(out, 0)
            d_loss = loss_real + loss_fake + loss_reg
            d_loss.backward()
            optimizer.step('discriminator')

            # generator loss
            optimizer.zero_grad()
            loss_mel = criterion(mel_rec, mel_rec_gt)
            loss_dur = 0
            for _s2s_pred, _text_input, _text_length in zip(d, d_gt, input_lengths):
                loss_dur += F.l1_loss(_s2s_pred[1:_text_length-1], 
                                            _text_input[1:_text_length-1])
            loss_dur /= texts.size(0)

            with torch.no_grad():
                _, f_real = model.discriminator(mel_rec_gt.unsqueeze(1))
            out_rec, f_fake = model.discriminator(mel_rec.unsqueeze(1))
            loss_adv = adv_loss(out_rec, 1)

            # feature matching loss
            loss_fm = 0
            for m in range(len(f_real)):
                for k in range(len(f_real[m])):
                    loss_fm += torch.mean(torch.abs(f_real[m][k] - f_fake[m][k]))

            g_loss = loss_params.lambda_mel * loss_mel + \
                    loss_params.lambda_F0 * loss_F0_rec + \
                    loss_params.lambda_dur * loss_dur + \
                    loss_params.lambda_norm * loss_norm_rec + \
                    loss_params.lambda_adv * loss_adv + \
                    loss_params.lambda_fm * loss_fm

            running_loss += loss_mel.item()
            g_loss.backward()
            if torch.isnan(g_loss):
                from IPython.core.debugger import set_trace
                set_trace()
            optimizer.step('predictor')

            iters = iters + 1
            if (i+1)%log_interval == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Avd Loss: %.5f,  Disc Loss: %.5f, Dur Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f'
                        %(epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, loss_adv, d_loss, loss_dur, loss_norm_rec, loss_F0_rec))
                
                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/adv_loss', loss_adv.item(), iters)
                writer.add_scalar('train/d_loss', d_loss.item(), iters)
                writer.add_scalar('train/dur_loss', loss_dur, iters)
                writer.add_scalar('train/norm_loss', loss_norm_rec, iters)
                writer.add_scalar('train/F0_loss', loss_F0_rec, iters)

                running_loss = 0
                print('Time elasped:', time.time()-start_time)

        loss_test = 0
        loss_align = 0
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                batch = [b.to(device) for b in batch]
                texts, input_lengths, mels, mel_input_length = batch
                with torch.no_grad():
                    mask = length_to_mask(mel_input_length // (2 ** model.text_aligner.n_down)).to('cuda')
                    text_mask = length_to_mask(input_lengths).to(texts.device)

                    _, _, s2s_attn_feat = model.text_aligner(mels, mask, texts)

                    s2s_attn_feat = s2s_attn_feat.transpose(-1, -2)
                    s2s_attn_feat = s2s_attn_feat[..., 1:]
                    s2s_attn_feat = s2s_attn_feat.transpose(-1, -2)

                    with torch.no_grad():
                        text_mask = length_to_mask(input_lengths).to(texts.device)
                        attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                        attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                        attn_mask = (attn_mask < 1)

                    if TMA_CEloss:
                        s2s_attn = F.softmax(s2s_attn_feat, dim=1) # along the mel dimension
                    else:
                        s2s_attn = F.softmax(s2s_attn_feat, dim=-1) # along the text dimension

                    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** model.text_aligner.n_down))
                    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                    # encode
                    m = length_to_mask(input_lengths)
                    t_en = model.text_encoder(texts, input_lengths, m)
                    asr = (t_en @ s2s_attn_mono)

                    d_gt = s2s_attn_mono.sum(axis=-1).detach()

                # compute the style of the entire utterance
                # this operation cannot be done in batch because of the avgpool layer (may need to work on masked avgpool)
                ss = []
                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item())
                    mel = mels[bib, :, :mel_input_length[bib]]
                    s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                    ss.append(s)
                s = torch.stack(ss).squeeze()
                
                d, p = model.predictor(t_en, s, 
                                                    input_lengths, 
                                                    s2s_attn_mono, 
                                                    m)
                # get clips
                mel_len = int(mel_input_length.min().item() / 2 - 1)
                en = []
                gt = []
                p_en = []


                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item() / 2)

                    random_start = np.random.randint(0, mel_length - mel_len)
                    en.append(asr[bib, :, random_start:random_start+mel_len])
                    p_en.append(p[bib, :, random_start:random_start+mel_len])

                    gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

                en = torch.stack(en)
                p_en = torch.stack(p_en)
                gt = torch.stack(gt).detach()

                s = model.style_encoder(gt.unsqueeze(1))

                F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

                loss_dur = 0
                for _s2s_pred, _text_input, _text_length in zip(d, d_gt, input_lengths):
                    loss_dur += F.l1_loss(_s2s_pred[1:_text_length-1], 
                                                _text_input[1:_text_length-1])
                loss_dur /= texts.size(0)

                mel_rec = model.decoder(en, F0_fake, N_fake, s)
                mel_rec = mel_rec[..., :gt.shape[-1]]

                loss_mel = criterion(mel_rec, gt)

                loss_test += loss_mel
                loss_align += loss_dur
                iters_test += 1

        print('Epochs:', epoch + 1)
        print('Validation loss: %.3f, %.3f' % (loss_test / iters_test, loss_align / iters_test), '\n\n\n')

        writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)
        writer.add_scalar('eval/dur_loss', loss_align / iters_test, epoch + 1)

        if epoch % saving_epoch == 0:
            if (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            print('Saving..')
            state = {
                'net':  {key: model[key].state_dict() for key in model}, 
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            torch.save(state, save_path)
        
            
if __name__=="__main__":
    main()
