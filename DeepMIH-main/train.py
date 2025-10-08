#!/usr/bin/env python
import sys
import os
from collections import Counter
import torch
import torch.nn
import torch.optim
import torchvision
import torch.nn.functional as F
import math
import numpy as np
import tqdm
from model import *
from imp_subnet import *
import torchvision.transforms as T
import config as c
import importlib

config_module = os.environ.get("DEEPMIH_CONFIG", "config")
c = importlib.import_module(config_module)
from tensorboardX import SummaryWriter
from datasets import trainloader, testloader
import viz
import modules.module_util as mutil
import modules.Unet_common as common
import warnings
from vgg_loss import VGGLoss
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV is optional at runtime
    cv2 = None

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)

    if mse < 1.0e-10:
        return 100

    if mse > 1.0e15:
        return -100

    return 10 * math.log10(255.0 ** 2 / mse)


def gauss_noise(shape):
    noise = torch.zeros(shape, device=device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape, device=device)

    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def imp_loss(output, resi):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, resi)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.L1Loss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)


def distr_loss(noise):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(noise, torch.zeros(noise.shape, device=device))
    return loss.to(device)


# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def _grad_norm(module):
    """Compute the global L2 norm of all gradients in a module."""

    total = 0.0
    for param in module.parameters():
        grad = param.grad
        if grad is not None:
            total += grad.detach().float().pow(2).sum().item()

    if total == 0.0:
        return 0.0

    return math.sqrt(total)


_IDENTITY_INIT_MESSAGE_EMITTED = False
_IDENTITY_LOSS_WARNING_EMITTED = False


def _print_identity_init_message():
    global _IDENTITY_INIT_MESSAGE_EMITTED
    if _IDENTITY_INIT_MESSAGE_EMITTED:
        return
    if getattr(c, "kan_identity_init", True):
        jitter = getattr(c, "kan_identity_jitter", 1e-3)
        print(
            "[InitDebug] KAN coupling blocks start from a near-identity mapping "
            "(kan_identity_init=True). Early guide/reconstruction losses will "
            "be close to zero; set config.kan_identity_init=False or increase "
            f"config.kan_identity_jitter (currently {jitter}) to randomize the start."
        )
        _IDENTITY_INIT_MESSAGE_EMITTED = True


def _explain_identity_loss(i_epoch):
    global _IDENTITY_LOSS_WARNING_EMITTED
    if _IDENTITY_LOSS_WARNING_EMITTED:
        return
    if getattr(c, "kan_identity_init", True):
        jitter = getattr(c, "kan_identity_jitter", 1e-3)
        print(
            f"[LossDebug] Epoch {i_epoch}: near-zero losses are expected because "
            "KAN coupling layers were initialized to the identity. Disable "
            "kan_identity_init or raise kan_identity_jitter to provide a larger "
            f"initial perturbation (currently {jitter})."
        )
        _IDENTITY_LOSS_WARNING_EMITTED = True


def _summarize_skipped_weights(skipped, filtered_count):
    """Print a human-readable explanation for skipped checkpoint weights."""

    if not skipped:
        return

    conv_block_counts = Counter()
    other_mismatch = 0
    for key in skipped:
        parent = key.rsplit('.', 1)[0]
        parts = parent.split('.')
        try:
            dense_idx = parts.index('dense')
        except ValueError:
            other_mismatch += 1
            continue
        block = '.'.join(parts[: dense_idx + 1])
        if 'conv' in parts[dense_idx + 1:]:
            conv_block_counts[block] += 1
        else:
            other_mismatch += 1

    if conv_block_counts:
        affected = ', '.join(sorted(conv_block_counts.keys())[:5])
        if len(conv_block_counts) > 5:
            affected += ', ...'
        print(
            "Notice: {} convolutional dense blocks in the checkpoint do not exist in the "
            "current model (which uses KAN subnetworks).".format(sum(conv_block_counts.values()))
        )
        print(
            "Affected blocks include: {}. Use a checkpoint trained with the same "
            "architecture or disable pretraining when switching to KAN coupling nets.".format(
                affected
            )
        )

    if other_mismatch:
        print(f"Additional {other_mismatch} parameters were skipped due to shape mismatches.")

    if filtered_count:
        print(
            f"{filtered_count} parameters were explicitly skipped because their keys matched "
            "the configured substrings {getattr(c, 'pretrained_skip_substrings', tuple())}."
        )


def _limit_opencv_threads():
    """Prevent OpenCV from spawning excessive background threads."""

    if cv2 is None:
        return

    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def _configure_loader_workers(loader, target_workers):
    """Downscale the worker count of a ``DataLoader`` in-place."""

    if not isinstance(loader, torch.utils.data.DataLoader):
        return loader

    target_workers = max(0, int(target_workers))
    if loader.num_workers == target_workers:
        return loader

    loader.num_workers = target_workers
    if target_workers == 0 and getattr(loader, "persistent_workers", False):
        loader.persistent_workers = False
    return loader


def _release_tensors(*tensors):
    """Explicitly delete tensors and release cached CUDA memory."""

    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            del tensor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load(name, net, optim, skip_substrings=None):
    state_dicts = torch.load(name, map_location=device)
    pretrained_state = state_dicts.get('net', {})
    if skip_substrings is None:
        skip_substrings = getattr(c, 'pretrained_skip_substrings', tuple())
    model_state = net.state_dict()
    filtered_state = {}
    skipped_mismatch = []
    filtered_matches = 0
    for key, value in pretrained_state.items():
        if 'tmp_var' in key:
            continue
        if any(substr in key for substr in skip_substrings):
            print(f"Skipping weight: {key} (matches skip substrings)")
            filtered_matches += 1
            continue
        if key in model_state and model_state[key].shape == value.shape:
            filtered_state[key] = value
        else:
            print(f"Skipping weight: {key} (missing or shape mismatch)")
            skipped_mismatch.append(key)
    load_result = net.load_state_dict(filtered_state, strict=False)
    if load_result.missing_keys:
        print(f"Warning: {len(load_result.missing_keys)} parameters left at initialization (missing in checkpoint)")
    if load_result.unexpected_keys:
        print(f"Warning: {len(load_result.unexpected_keys)} unexpected parameters ignored from checkpoint")
    _summarize_skipped_weights(skipped_mismatch, filtered_matches)
    optimizer_status = None
    if 'opt' in state_dicts:
        try:
            optim.load_state_dict(state_dicts['opt'])
            optimizer_status = True
        except Exception as exc:
            print(f'Cannot load optimizer for some reason or other: {exc}')
            optimizer_status = False
    return {
        'loaded': len(filtered_state),
        'skipped_mismatch': len(skipped_mismatch),
        'filtered_matches': filtered_matches,
        'missing': len(load_result.missing_keys),
        'unexpected': len(load_result.unexpected_keys),
        'optimizer_status': optimizer_status,
    }


def _log_checkpoint_status(label, report):
    if not report:
        return
    skipped = report.get('skipped_mismatch', 0)
    missing = report.get('missing', 0)
    filtered_matches = report.get('filtered_matches', 0)
    if skipped or missing:
        print(
            f"[CheckpointDebug] {label}: {skipped} mismatched parameters were "
            "skipped and {missing} parameters remain randomly initialized. "
            "This happens when loading checkpoints from a different architecture "
            "(e.g., convolutional vs. KAN coupling blocks). Disable pretraining "
            "or provide matching weights to avoid starting from scratch."
        )
    if filtered_matches:
        print(
            f"[CheckpointDebug] {label}: {filtered_matches} parameters were "
            "ignored because they matched pretrained_skip_substrings."
        )
    optimizer_status = report.get('optimizer_status')
    if optimizer_status is False:
        print(
            f"[CheckpointDebug] {label}: optimizer state could not be restored; "
            "training will continue with a freshly initialized optimizer."
        )
    elif optimizer_status is None:
        print(
            f"[CheckpointDebug] {label}: checkpoint did not contain optimizer "
            "state; gradients will build momentum from scratch."
        )


def init_net3(mod):
    for key, param in mod.named_parameters():
        if param.requires_grad:
            param.data = 0.1 * torch.randn(param.data.shape).cuda()


def main():
    _limit_opencv_threads()

    target_workers = getattr(c, "dataloader_num_workers", 0)
    eval_workers = getattr(c, "dataloader_eval_workers", target_workers)
    train_loader = _configure_loader_workers(trainloader, target_workers)
    test_loader = _configure_loader_workers(testloader, eval_workers)

    #####################
    # Model initialize: #
    #####################
    net1 = Model_1().to(device)
    net2 = Model_2().to(device)
    net3 = ImpMapBlock().to(device)
    init_model(net1)
    init_model(net2)
    init_net3(net3)
    net1 = torch.nn.DataParallel(net1, device_ids=c.device_ids)
    net2 = torch.nn.DataParallel(net2, device_ids=c.device_ids)
    net3 = torch.nn.DataParallel(net3, device_ids=c.device_ids)
    para1 = get_parameter_number(net1)
    para2 = get_parameter_number(net2)
    para3 = get_parameter_number(net3)
    print(para1)
    print(para2)
    print(para3)
    params_trainable1 = list(filter(lambda p: p.requires_grad, net1.parameters()))
    params_trainable2 = list(filter(lambda p: p.requires_grad, net2.parameters()))
    params_trainable3 = list(filter(lambda p: p.requires_grad, net3.parameters()))
    optim1 = torch.optim.Adam(
        params_trainable1,
        lr=c.lr,
        betas=c.betas,
        eps=1e-6,
        weight_decay=c.weight_decay,
    )
    optim2 = torch.optim.Adam(
        params_trainable2,
        lr=c.lr,
        betas=c.betas,
        eps=1e-6,
        weight_decay=c.weight_decay,
    )
    optim3 = torch.optim.Adam(
        params_trainable3,
        lr=c.lr3,
        betas=c.betas,
        eps=1e-6,
        weight_decay=c.weight_decay,
    )
    weight_scheduler1 = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
    weight_scheduler2 = torch.optim.lr_scheduler.StepLR(optim2, c.weight_step, gamma=c.gamma)
    weight_scheduler3 = torch.optim.lr_scheduler.StepLR(optim3, c.weight_step, gamma=c.gamma)

    dwt = common.DWT()
    iwt = common.IWT()
    checkpoint_reports = []
    if c.tain_next:
        checkpoint_reports.append(
            ("resume/net1", load(c.MODEL_PATH + c.suffix_load + '_1.pt', net1, optim1))
        )
        checkpoint_reports.append(
            ("resume/net2", load(c.MODEL_PATH + c.suffix_load + '_2.pt', net2, optim2))
        )
        checkpoint_reports.append(
            ("resume/net3", load(c.MODEL_PATH + c.suffix_load + '_3.pt', net3, optim3))
        )

    if c.pretrain:
        checkpoint_reports.append(
            ("pretrain/net1", load(c.PRETRAIN_PATH + c.suffix_pretrain + '_1.pt', net1, optim1))
        )
        checkpoint_reports.append(
            ("pretrain/net2", load(c.PRETRAIN_PATH + c.suffix_pretrain + '_2.pt', net2, optim2))
        )
        if c.PRETRAIN_PATH_3 is not None:
            checkpoint_reports.append(
                (
                    "pretrain/net3",
                    load(c.PRETRAIN_PATH_3 + c.suffix_pretrain_3 + '_3.pt', net3, optim3),
                )
            )

    for label, report in checkpoint_reports:
        _log_checkpoint_status(label, report)

    _print_identity_init_message()

    use_amp = bool(getattr(c, "use_amp", False) and device.type == "cuda")
    grad_accum_steps = max(1, int(getattr(c, "grad_accum_steps", 1)))
    writer = None
    scaler = GradScaler(enabled=use_amp)

    try:
        writer = SummaryWriter(comment='hinet', filename_suffix="steg")

        for epoch_idx in range(c.epochs):
            i_epoch = epoch_idx + c.trained_epoch + 1
            loss_history = []
            loss_history_g1 = []
            loss_history_g2 = []
            loss_history_r1 = []
            loss_history_r2 = []
            loss_history_imp = []
            grad_norm_history_1 = []
            grad_norm_history_2 = []
            grad_norm_history_3 = []
            #################
            #     train:    #
            #################
            vgg_loss = VGGLoss(3, 1, False)
            vgg_loss.to(device)
            net1.train()
            net2.train()
            net3.train()
            optim1.zero_grad(set_to_none=True)
            optim2.zero_grad(set_to_none=True)
            optim3.zero_grad(set_to_none=True)
            grad_clip_norm = getattr(c, 'grad_clip_norm', None)
            for i_batch, data in enumerate(train_loader):
                data = data.to(device, non_blocking=True)
                cover = data[:data.shape[0] // 3]
                secret_1 = data[data.shape[0] // 3: 2 * (data.shape[0] // 3)]
                secret_2 = data[2 * (data.shape[0] // 3): 3 * (data.shape[0] // 3)]
                cover_dwt = dwt(cover)
                cover_dwt_low = cover_dwt.narrow(1, 0, c.channels_in)
                secret_dwt_1 = dwt(secret_1)
                secret_dwt_2 = dwt(secret_2)

                input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1), 1)

                with autocast(enabled=use_amp):
                    output_dwt_1 = net1(input_dwt_1)
                    output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)
                    output_steg_dwt_low_1 = output_steg_dwt_1.narrow(1, 0, c.channels_in)
                    output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)

                    output_steg_1 = iwt(output_steg_dwt_1)

                    if c.use_imp_map:
                        imp_map = net3(cover, secret_1, output_steg_1)
                    else:
                        imp_map = torch.zeros(cover.shape, device=device)

                    impmap_loss = imp_loss(imp_map, cover - output_steg_1)

                    imp_map_dwt = dwt(imp_map)
                    input_dwt_2 = torch.cat((output_steg_dwt_1, imp_map_dwt), 1)
                    input_dwt_2 = torch.cat((input_dwt_2, secret_dwt_2), 1)

                    output_dwt_2 = net2(input_dwt_2)
                    output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)
                    output_steg_dwt_low_2 = output_steg_dwt_2.narrow(1, 0, c.channels_in)
                    output_z_dwt_2 = output_dwt_2.narrow(
                        1, 4 * c.channels_in, output_dwt_2.shape[1] - 4 * c.channels_in
                    )

                    output_steg_2 = iwt(output_steg_dwt_2)

                    output_z_guass_1 = gauss_noise(output_z_dwt_1.shape)
                    output_z_guass_2 = gauss_noise(output_z_dwt_2.shape)

                    output_rev_dwt_2 = torch.cat((output_steg_dwt_2, output_z_guass_2), 1)

                    rev_dwt_2 = net2(output_rev_dwt_2, rev=True)

                    rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)
                    rev_secret_dwt_2 = rev_dwt_2.narrow(
                        1, rev_dwt_2.shape[1] - 4 * c.channels_in, 4 * c.channels_in
                    )

                    rev_steg_1 = iwt(rev_steg_dwt_1)
                    rev_secret_2 = iwt(rev_secret_dwt_2)

                    output_rev_dwt_1 = torch.cat((rev_steg_dwt_1, output_z_guass_1), 1)

                    rev_dwt_1 = net1(output_rev_dwt_1, rev=True)

                    rev_secret_dwt = rev_dwt_1.narrow(
                        1, rev_dwt_1.shape[1] - 4 * c.channels_in, 4 * c.channels_in
                    )
                    rev_secret_1 = iwt(rev_secret_dwt)

                    g_loss_1 = guide_loss(output_steg_1, cover)
                    g_loss_2 = guide_loss(output_steg_2, cover)

                    vgg_on_cov = vgg_loss(cover)
                    vgg_on_steg1 = vgg_loss(output_steg_1)
                    vgg_on_steg2 = vgg_loss(output_steg_2)

                    perc_loss = guide_loss(vgg_on_cov, vgg_on_steg1) + guide_loss(vgg_on_cov, vgg_on_steg2)

                    l_loss_1 = guide_loss(output_steg_dwt_low_1, cover_dwt_low)
                    l_loss_2 = guide_loss(output_steg_dwt_low_2, cover_dwt_low)
                    r_loss_1 = reconstruction_loss(rev_secret_1, secret_1)
                    r_loss_2 = reconstruction_loss(rev_secret_2, secret_2)

                    total_loss = (
                        c.lamda_reconstruction_1 * r_loss_1
                        + c.lamda_reconstruction_2 * r_loss_2
                        + c.lamda_guide_1 * g_loss_1
                        + c.lamda_guide_2 * g_loss_2
                        + c.lamda_low_frequency_1 * l_loss_1
                        + c.lamda_low_frequency_2 * l_loss_2
                    )
                    total_loss = total_loss + 0.01 * perc_loss

                total_loss_value = float(total_loss.detach().item())
                g_loss_1_value = float(g_loss_1.detach().item())
                g_loss_2_value = float(g_loss_2.detach().item())
                r_loss_1_value = float(r_loss_1.detach().item())
                r_loss_2_value = float(r_loss_2.detach().item())
                impmap_loss_value = float(impmap_loss.detach().item())

                scaled_loss = total_loss / grad_accum_steps
                scaler.scale(scaled_loss).backward()

                should_step = (i_batch + 1) % grad_accum_steps == 0
                if should_step:
                    if use_amp:
                        if c.optim_step_1:
                            scaler.unscale_(optim1)
                        if c.optim_step_2:
                            scaler.unscale_(optim2)
                        if c.optim_step_3:
                            scaler.unscale_(optim3)

                    if grad_clip_norm is not None:
                        if c.optim_step_1:
                            clip_grad_norm_(net1.parameters(), grad_clip_norm)
                        if c.optim_step_2:
                            clip_grad_norm_(net2.parameters(), grad_clip_norm)
                        if c.optim_step_3:
                            clip_grad_norm_(net3.parameters(), grad_clip_norm)

                    grad_norm_history_1.append(_grad_norm(net1))
                    grad_norm_history_2.append(_grad_norm(net2))
                    grad_norm_history_3.append(_grad_norm(net3))

                    if c.optim_step_1:
                        scaler.step(optim1)
                    if c.optim_step_2:
                        scaler.step(optim2)
                    if c.optim_step_3:
                        scaler.step(optim3)
                    scaler.update()
                    optim1.zero_grad(set_to_none=True)
                    optim2.zero_grad(set_to_none=True)
                    optim3.zero_grad(set_to_none=True)

                loss_history.append([total_loss_value, 0.0])
                loss_history_g1.append(g_loss_1_value)
                loss_history_g2.append(g_loss_2_value)
                loss_history_r1.append(r_loss_1_value)
                loss_history_r2.append(r_loss_2_value)
                loss_history_imp.append(impmap_loss_value)

                _release_tensors(
                    data,
                    cover,
                    secret_1,
                    secret_2,
                    cover_dwt,
                    cover_dwt_low,
                    secret_dwt_1,
                    secret_dwt_2,
                    input_dwt_1,
                    output_dwt_1,
                    output_steg_dwt_1,
                    output_steg_dwt_low_1,
                    output_z_dwt_1,
                    output_steg_1,
                    imp_map,
                    imp_map_dwt,
                    input_dwt_2,
                    output_dwt_2,
                    output_steg_dwt_2,
                    output_steg_dwt_low_2,
                    output_z_dwt_2,
                    output_steg_2,
                    output_z_guass_1,
                    output_z_guass_2,
                    output_rev_dwt_2,
                    rev_dwt_2,
                    rev_steg_dwt_1,
                    rev_secret_dwt_2,
                    rev_steg_1,
                    rev_secret_2,
                    output_rev_dwt_1,
                    rev_dwt_1,
                    rev_secret_dwt,
                    rev_secret_1,
                    vgg_on_cov,
                    vgg_on_steg1,
                    vgg_on_steg2,
                    perc_loss,
                    total_loss,
                    g_loss_1,
                    g_loss_2,
                    l_loss_1,
                    l_loss_2,
                    r_loss_1,
                    r_loss_2,
                    scaled_loss,
                    impmap_loss,
                )

            remainder = len(loss_history) % grad_accum_steps
            if grad_accum_steps > 1 and remainder != 0:
                if use_amp:
                    if c.optim_step_1:
                        scaler.unscale_(optim1)
                    if c.optim_step_2:
                        scaler.unscale_(optim2)
                    if c.optim_step_3:
                        scaler.unscale_(optim3)

                if grad_clip_norm is not None:
                    if c.optim_step_1:
                        clip_grad_norm_(net1.parameters(), grad_clip_norm)
                    if c.optim_step_2:
                        clip_grad_norm_(net2.parameters(), grad_clip_norm)
                    if c.optim_step_3:
                        clip_grad_norm_(net3.parameters(), grad_clip_norm)

                grad_norm_history_1.append(_grad_norm(net1))
                grad_norm_history_2.append(_grad_norm(net2))
                grad_norm_history_3.append(_grad_norm(net3))

                if c.optim_step_1:
                    scaler.step(optim1)
                if c.optim_step_2:
                    scaler.step(optim2)
                if c.optim_step_3:
                    scaler.step(optim3)
                scaler.update()
                optim1.zero_grad(set_to_none=True)
                optim2.zero_grad(set_to_none=True)
                optim3.zero_grad(set_to_none=True)

            #################
            #     val:    #
            #################
            if i_epoch % c.val_freq == 1:
                with torch.no_grad():
                    psnr_s1 = []
                    psnr_s2 = []
                    psnr_c1 = []
                    psnr_c2 = []
                    net1.eval()
                    net2.eval()
                    net3.eval()
                    for x in test_loader:
                        x = x.to(device, non_blocking=True)
                        cover = x[:x.shape[0] // 3]
                        secret_1 = x[x.shape[0] // 3: 2 * x.shape[0] // 3]
                        secret_2 = x[2 * x.shape[0] // 3: 3 * x.shape[0] // 3]

                        cover_dwt = dwt(cover)
                        secret_dwt_1 = dwt(secret_1)
                        secret_dwt_2 = dwt(secret_2)

                        input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1), 1)

                        output_dwt_1 = net1(input_dwt_1)
                        output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)
                        output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)

                        output_steg_1 = iwt(output_steg_dwt_1)

                        if c.use_imp_map:
                            imp_map = net3(cover, secret_1, output_steg_1)
                        else:
                            imp_map = torch.zeros(cover.shape, device=device)

                        imp_map_dwt = dwt(imp_map)
                        input_dwt_2 = torch.cat((output_steg_dwt_1, imp_map_dwt), 1)
                        input_dwt_2 = torch.cat((input_dwt_2, secret_dwt_2), 1)

                        output_dwt_2 = net2(input_dwt_2)
                        output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)
                        output_steg_dwt_low_2 = output_steg_dwt_2.narrow(1, 0, c.channels_in)
                        output_z_dwt_2 = output_dwt_2.narrow(
                            1, 4 * c.channels_in, output_dwt_2.shape[1] - 4 * c.channels_in
                        )

                        output_steg_2 = iwt(output_steg_dwt_2)

                        output_z_guass_1 = gauss_noise(output_z_dwt_1.shape)
                        output_z_guass_2 = gauss_noise(output_z_dwt_2.shape)

                        output_rev_dwt_2 = torch.cat((output_steg_dwt_2, output_z_guass_2), 1)

                        rev_dwt_2 = net2(output_rev_dwt_2, rev=True)

                        rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)
                        rev_secret_dwt_2 = rev_dwt_2.narrow(
                            1, rev_dwt_2.shape[1] - 4 * c.channels_in, 4 * c.channels_in
                        )

                        rev_steg_1 = iwt(rev_steg_dwt_1)
                        rev_secret_2 = iwt(rev_secret_dwt_2)

                        output_rev_dwt_1 = torch.cat((rev_steg_dwt_1, output_z_guass_1), 1)

                        rev_dwt_1 = net1(output_rev_dwt_1, rev=True)

                        rev_secret_dwt = rev_dwt_1.narrow(
                            1, rev_dwt_1.shape[1] - 4 * c.channels_in, 4 * c.channels_in
                        )
                        rev_secret_1 = iwt(rev_secret_dwt)

                        secret_rev1_255 = rev_secret_1.cpu().numpy().squeeze() * 255
                        secret_rev2_255 = rev_secret_2.cpu().numpy().squeeze() * 255
                        secret_1_255 = secret_1.cpu().numpy().squeeze() * 255
                        secret_2_255 = secret_2.cpu().numpy().squeeze() * 255

                        cover_255 = cover.cpu().numpy().squeeze() * 255
                        steg_1_255 = output_steg_1.cpu().numpy().squeeze() * 255
                        steg_2_255 = output_steg_2.cpu().numpy().squeeze() * 255

                        psnr_s1.append(computePSNR(secret_rev1_255, secret_1_255))
                        psnr_s2.append(computePSNR(secret_rev2_255, secret_2_255))
                        psnr_c1.append(computePSNR(cover_255, steg_1_255))
                        psnr_c2.append(computePSNR(cover_255, steg_2_255))

                        _release_tensors(
                            x,
                            cover,
                            secret_1,
                            secret_2,
                            cover_dwt,
                            secret_dwt_1,
                            secret_dwt_2,
                            input_dwt_1,
                            output_dwt_1,
                            output_steg_dwt_1,
                            output_z_dwt_1,
                            output_steg_1,
                            imp_map,
                            imp_map_dwt,
                            input_dwt_2,
                            output_dwt_2,
                            output_steg_dwt_2,
                            output_steg_dwt_low_2,
                            output_z_dwt_2,
                            output_steg_2,
                            output_z_guass_1,
                            output_z_guass_2,
                            output_rev_dwt_2,
                            rev_dwt_2,
                            rev_steg_dwt_1,
                            rev_secret_dwt_2,
                            rev_steg_1,
                            rev_secret_2,
                            output_rev_dwt_1,
                            rev_dwt_1,
                            rev_secret_dwt,
                            rev_secret_1,
                        )

                    writer.add_scalars("PSNR", {"S1 average psnr": np.mean(psnr_s1)}, i_epoch)
                    writer.add_scalars("PSNR", {"C1 average psnr": np.mean(psnr_c1)}, i_epoch)
                    writer.add_scalars("PSNR", {"S2 average psnr": np.mean(psnr_s2)}, i_epoch)
                    writer.add_scalars("PSNR", {"C2 average psnr": np.mean(psnr_c2)}, i_epoch)

            if not loss_history:
                print(
                    f"[LossDebug] Epoch {i_epoch}: no training batches were processed. "
                    "Check that the training dataset paths are correct and contain images."
                )
                viz.show_loss(np.array([0.0, np.log10(optim1.param_groups[0]['lr'])]))
                continue

            epoch_losses = np.mean(np.array(loss_history), axis=0)
            epoch_losses[1] = np.log10(optim1.param_groups[0]['lr'])

            epoch_losses_g1 = np.mean(np.array(loss_history_g1))
            epoch_losses_g2 = np.mean(np.array(loss_history_g2))
            epoch_losses_r1 = np.mean(np.array(loss_history_r1))
            epoch_losses_r2 = np.mean(np.array(loss_history_r2))
            epoch_losses_imp = np.mean(np.array(loss_history_imp))
            epoch_grad_norm_1 = np.mean(np.array(grad_norm_history_1)) if grad_norm_history_1 else 0.0
            epoch_grad_norm_2 = np.mean(np.array(grad_norm_history_2)) if grad_norm_history_2 else 0.0
            epoch_grad_norm_3 = np.mean(np.array(grad_norm_history_3)) if grad_norm_history_3 else 0.0

            if loss_history:
                total_values = [entry[0] for entry in loss_history]
                mean_total = float(np.mean(total_values))
                if mean_total < 1e-4:
                    min_total = float(np.min(total_values))
                    max_total = float(np.max(total_values))
                    print(
                        f"[LossDebug] Epoch {i_epoch}: total loss nearly zero "
                        f"(mean={mean_total:.6e}, min={min_total:.6e}, max={max_total:.6e})."
                    )
                    print(
                        "[LossDebug] Component means: "
                        f"g1={epoch_losses_g1:.6e}, g2={epoch_losses_g2:.6e}, "
                        f"r1={epoch_losses_r1:.6e}, r2={epoch_losses_r2:.6e}, imp={epoch_losses_imp:.6e}."
                    )
                    print(
                        "[LossDebug] Gradient norms (avg L2): "
                        f"net1={epoch_grad_norm_1:.6e}, "
                        f"net2={epoch_grad_norm_2:.6e}, net3={epoch_grad_norm_3:.6e}."
                    )
                    _explain_identity_loss(i_epoch)
                    if c.pretrain or c.tain_next:
                        print(
                            "[LossDebug] A pre-trained checkpoint is loaded; near-zero losses "
                            "are expected when continuing from a converged model."
                        )
                    else:
                        print(
                            "[LossDebug] Pretraining is disabled. If the gradient norms above "
                            "stay near zero, the model may not be updating. Verify that "
                            "optim_step_* flags are enabled and that the data loader is "
                            "returning diverse images."
                        )

            viz.show_loss(epoch_losses)
            writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)
            writer.add_scalars("Train", {"g1_Loss": epoch_losses_g1}, i_epoch)
            writer.add_scalars("Train", {"g2_Loss": epoch_losses_g2}, i_epoch)
            writer.add_scalars("Train", {"r1_Loss": epoch_losses_r1}, i_epoch)
            writer.add_scalars("Train", {"r2_Loss": epoch_losses_r2}, i_epoch)
            writer.add_scalars("Train", {"imp_Loss": epoch_losses_imp}, i_epoch)

            if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
                torch.save({'opt': optim1.state_dict(),
                            'net': net1.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i_1' % i_epoch + '.pt')
                torch.save({'opt': optim2.state_dict(),
                            'net': net2.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i_2' % i_epoch + '.pt')
                torch.save({'opt': optim3.state_dict(),
                            'net': net3.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i_3' % i_epoch + '.pt')
            weight_scheduler1.step()
            weight_scheduler2.step()
            weight_scheduler3.step()

        torch.save({'opt': optim1.state_dict(),
                    'net': net1.state_dict()}, c.MODEL_PATH + 'model_1' + '.pt')
        torch.save({'opt': optim2.state_dict(),
                    'net': net2.state_dict()}, c.MODEL_PATH + 'model_2' + '.pt')
        torch.save({'opt': optim3.state_dict(),
                    'net': net3.state_dict()}, c.MODEL_PATH + 'model_3' + '.pt')

    except Exception:
        if c.checkpoint_on_error:
            torch.save({'opt': optim1.state_dict(),
                        'net': net1.state_dict()}, c.MODEL_PATH + 'model_ABORT_1' + '.pt')
            torch.save({'opt': optim2.state_dict(),
                        'net': net2.state_dict()}, c.MODEL_PATH + 'model_ABORT_2' + '.pt')
            torch.save({'opt': optim3.state_dict(),
                        'net': net3.state_dict()}, c.MODEL_PATH + 'model_ABORT_3' + '.pt')
        raise

    finally:
        if writer is not None:
            writer.close()
        viz.signal_stop()


if __name__ == "__main__":
    main()