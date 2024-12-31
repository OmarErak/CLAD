"""
    Original code: https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
    Some changes were done for added support
"""
import shutil
import sys
import time
import numpy as np
import math
import torch

_, term_width = shutil.get_terminal_size()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def add_awgn_noise(z, snr_db):
    # Calculate the signal power
    signal_power = torch.mean(torch.abs(z**2))
    
    # Calculate noise power based on the SNR
    sigma2 = signal_power * 10**(-snr_db/10)
    
    # Generate real noise using torch.randn
    noise = torch.sqrt(sigma2) * torch.randn(z.size(), dtype=z.dtype, device=z.device)
    
    # Return the signal plus the noise (real-valued)
    return z + noise


def adjust_learning_rate(optimizer, epoch, mode, args):
    if mode == "contrastive":
        lr = args.lr_contrastive
        n_epochs = args.n_epochs_contrastive
    elif mode == "cross_entropy":
        lr = args.lr_cross_entropy
        n_epochs = args.n_epochs_cross_entropy
    else:
        raise ValueError("Unknown mode for learning rate adjustment.")

    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / n_epochs)) / 2
    else:
        steps = np.sum(epoch > np.array(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_model(model, path):
    torch.save({'model_state_dict': model.state_dict()}, path)
    print(f"Model saved at {path}")

def load_phase1_model(encoder_path, encoder_class, args):

    # Initialize models
    encoder = encoder_class(3, args.latent_dim).to(args.device)

    # Load the encoder state dictionary
    checkpoint_encoder = torch.load(encoder_path, map_location=args.device)
    encoder.load_state_dict(checkpoint_encoder["model_state_dict"])

    print(f"Encoder loaded from {encoder_path}")
    
    return encoder