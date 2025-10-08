#!/usr/bin/env python
import sys
import os
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
from tensorboardX import SummaryWriter
from datasets import trainloader, testloader
import viz
import modules.module_util as mutil
import modules.Unet_common as common
import warnings
from vgg_loss import VGGLoss
from kan_adapter import KANLayer, LoKiKANAdapter
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
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

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
    loss = loss_fn(noise, torch.zeros(noise.shape).cuda())
    return loss.to(device)


# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
def _collect_kan_param_groups(model):
    kan_module_types = (KANLayer, LoKiKANAdapter)
    kan_param_ids = set()
    for module in model.modules():
        if isinstance(module, kan_module_types):
            for param in module.parameters():
                kan_param_ids.add(id(param))

    other_params = []
    kan_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in kan_param_ids:
            kan_params.append(param)
        else:
            other_params.append(param)
    return other_params, kan_params


def _set_kan_trainable(model, requires_grad):
    for module in model.modules():
        if isinstance(module, (KANLayer, LoKiKANAdapter)):
            for param in module.parameters():
                param.requires_grad = requires_grad
                if not requires_grad:
                    param.grad = None


def _summarize_keys(keys, limit=10):
    if not keys:
        return "[]"
    keys = sorted(keys)
    if len(keys) <= limit:
        return str(keys)
    displayed = ", ".join(keys[:limit])
    return f"[{displayed}, ... (+{len(keys) - limit} more)]"


def _load_partial_state_dict(module, checkpoint_state, label="model"):
    module_state = module.state_dict()
    compatible_tensors = {}
    missing_due_to_shape = []
    unexpected_keys = []

    for key, tensor in checkpoint_state.items():
        if key not in module_state:
            unexpected_keys.append(key)
            continue
        if module_state[key].shape != tensor.shape:
            missing_due_to_shape.append(key)
            continue
        compatible_tensors[key] = tensor

    load_info = module.load_state_dict(compatible_tensors, strict=False)
    missing_keys = sorted(set(module_state.keys()) - set(compatible_tensors.keys()))

    loaded_count = len(compatible_tensors)
    print(
        f"[{label}] Loaded {loaded_count} tensors; "
        f"missing {len(missing_keys)}; "
        f"shape-mismatch {len(missing_due_to_shape)}; "
        f"unexpected {len(unexpected_keys)}"
    )
    if missing_keys:
        print(f"    Missing keys sample: {_summarize_keys(missing_keys)}")
    if missing_due_to_shape:
        print(f"    Shape mismatch sample: {_summarize_keys(missing_due_to_shape)}")
    if load_info.unexpected_keys:
        print(f"    Unexpected keys from load: {_summarize_keys(load_info.unexpected_keys)}")


def load(name, net, optim, allow_partial=False, label="model"):
    state_dicts = torch.load(name, map_location="cpu")
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    if allow_partial:
        _load_partial_state_dict(net, network_state_dict, label=label)
    else:
        net.load_state_dict(network_state_dict)
    if 'opt' in state_dicts:
        try:
            optim.load_state_dict(state_dicts['opt'])
        except Exception:
            print('Cannot load optimizer for some reason or other')


def init_net3(mod):
    for key, param in mod.named_parameters():
        if param.requires_grad:
            param.data = 0.1 * torch.randn(param.data.shape).cuda()


#####################
# Model initialize: #
#####################
net1 = Model_1()
net2 = Model_2()
net3 = ImpMapBlock()
net1.cuda()
net2.cuda()
net3.cuda()
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
params_trainable1 = (list(filter(lambda p: p.requires_grad, net1.parameters())))

params_trainable3 = (list(filter(lambda p: p.requires_grad, net3.parameters())))
optim1 = torch.optim.Adam(params_trainable1, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
base_params2, kan_params2 = _collect_kan_param_groups(net2)
param_groups2 = []
if base_params2:
    param_groups2.append({"params": base_params2})
if kan_params2:
    param_groups2.append({"params": kan_params2, "lr": c.lr * c.kan_lr_scale})
if not param_groups2:
    raise RuntimeError("No trainable parameters found for stage-2 network.")
optim2 = torch.optim.Adam(param_groups2, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
optim3 = torch.optim.Adam(params_trainable3, lr=c.lr3, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler1 = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
weight_scheduler2 = torch.optim.lr_scheduler.StepLR(optim2, c.weight_step, gamma=c.gamma)
weight_scheduler3 = torch.optim.lr_scheduler.StepLR(optim3, c.weight_step, gamma=c.gamma)

dwt = common.DWT()
iwt = common.IWT()

if c.tain_next:
    load(
        c.MODEL_PATH + c.suffix_load + '_1.pt',
        net1,
        optim1,
        allow_partial=False,
        label='stage1-resume',
    )
    load(
        c.MODEL_PATH + c.suffix_load + '_2.pt',
        net2,
        optim2,
        allow_partial=False,
        label='stage2-resume',
    )
    load(
        c.MODEL_PATH + c.suffix_load + '_3.pt',
        net3,
        optim3,
        allow_partial=False,
        label='impmap-resume',
    )

if c.pretrain:
    load(
        c.PRETRAIN_PATH + c.suffix_pretrain + '_1.pt',
        net1,
        optim1,
        allow_partial=True,
        label='stage1-pretrain',
    )
    load(
        c.PRETRAIN_PATH + c.suffix_pretrain + '_2.pt',
        net2,
        optim2,
        allow_partial=True,
        label='stage2-pretrain',
    )
    if c.PRETRAIN_PATH_3 is not None:
        load(
            c.PRETRAIN_PATH_3 + c.suffix_pretrain_3 + '_3.pt',
            net3,
            optim3,
            allow_partial=True,
            label='impmap-pretrain',
        )

kan_frozen = False
if c.kan_freeze_epochs > 0 and kan_params2:
    _set_kan_trainable(net2, False)
    kan_frozen = True

try:
    writer = SummaryWriter(comment='hinet', filename_suffix="steg")

    for epoch_idx in range(c.epochs):
        if kan_frozen and epoch_idx >= c.kan_freeze_epochs:
            _set_kan_trainable(net2, True)
            kan_frozen = False

        i_epoch = epoch_idx + c.trained_epoch + 1
        loss_history = []
        loss_history_g1 = []
        loss_history_g2 = []
        loss_history_r1 = []
        loss_history_r2 = []
        loss_history_imp = []
        #################
        #     train:    #
        #################
        vgg_loss = VGGLoss(3, 1, False)
        vgg_loss.to(device)
        for i_batch, data in enumerate(trainloader):
            # data preparation
            data = data.to(device)
            cover = data[:data.shape[0] // 3]  # channels = 3
            secret_1 = data[data.shape[0] // 3: 2 * (data.shape[0] // 3)]
            secret_2 = data[2 * (data.shape[0] // 3): 3 * (data.shape[0] // 3)]
            cover_dwt = dwt(cover)  # channels = 12
            cover_dwt_low = cover_dwt.narrow(1, 0, c.channels_in)  # channels = 3
            secret_dwt_1 = dwt(secret_1)
            secret_dwt_2 = dwt(secret_2)

            input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1), 1)  # channels = 24

            #################
            #    forward1:   #
            #################
            output_dwt_1 = net1(input_dwt_1)  # channels = 24
            output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12
            output_steg_dwt_low_1 = output_steg_dwt_1.narrow(1, 0, c.channels_in)  # channels = 3
            output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12

            # get steg1
            output_steg_1 = iwt(output_steg_dwt_1)  # channels = 3

            #################
            #    forward2:   #
            #################
            if c.use_imp_map:
                imp_map = net3(cover, secret_1, output_steg_1)  # channels = 3
            else:
                imp_map = torch.zeros(cover.shape).cuda()

            impmap_loss = imp_loss(imp_map, cover - output_steg_1)

            imp_map_dwt = dwt(imp_map)  # channels = 12
            input_dwt_2 = torch.cat((output_steg_dwt_1, imp_map_dwt), 1)  # 24, without secret2

            input_dwt_2 = torch.cat((input_dwt_2, secret_dwt_2), 1)  # 36

            output_dwt_2 = net2(input_dwt_2)  # channels = 36
            output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
            output_steg_dwt_low_2 = output_steg_dwt_2.narrow(1, 0, c.channels_in)  # channels = 3
            output_z_dwt_2 = output_dwt_2.narrow(1, 4 * c.channels_in, output_dwt_2.shape[1] - 4 * c.channels_in)  # channels = 24

            # get steg2
            output_steg_2 = iwt(output_steg_dwt_2)  # channels = 3

            #################
            #   backward2:   #
            #################

            output_z_guass_1 = gauss_noise(output_z_dwt_1.shape)  # channels = 12
            output_z_guass_2 = gauss_noise(output_z_dwt_2.shape)  # channels = 24

            output_rev_dwt_2 = torch.cat((output_steg_dwt_2, output_z_guass_2), 1)  # channels = 36

            rev_dwt_2 = net2(output_rev_dwt_2, rev=True)  # channels = 36

            rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
            rev_secret_dwt_2 = rev_dwt_2.narrow(1, rev_dwt_2.shape[1] - 4 * c.channels_in, 4 * c.channels_in)  # channels = 12

            rev_steg_1 = iwt(rev_steg_dwt_1)  # channels = 3
            rev_secret_2 = iwt(rev_secret_dwt_2)  # channels = 3

            #################
            #   backward1:   #
            #################
            output_rev_dwt_1 = torch.cat((rev_steg_dwt_1, output_z_guass_1), 1)  # channels = 24

            rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 36

            rev_secret_dwt = rev_dwt_1.narrow(1, rev_dwt_1.shape[1] - 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
            rev_secret_1 = iwt(rev_secret_dwt)

            #################
            #     loss:     #
            #################
            g_loss_1 = guide_loss(output_steg_1.cuda(), cover.cuda())
            g_loss_2 = guide_loss(output_steg_2.cuda(), cover.cuda())

            vgg_on_cov = vgg_loss(cover)
            vgg_on_steg1 = vgg_loss(output_steg_1)
            vgg_on_steg2 = vgg_loss(output_steg_2)

            perc_loss = guide_loss(vgg_on_cov, vgg_on_steg1) + guide_loss(vgg_on_cov, vgg_on_steg2)

            l_loss_1 = guide_loss(output_steg_dwt_low_1.cuda(), cover_dwt_low.cuda())
            l_loss_2 = guide_loss(output_steg_dwt_low_2.cuda(), cover_dwt_low.cuda())
            r_loss_1 = reconstruction_loss(rev_secret_1, secret_1)
            r_loss_2 = reconstruction_loss(rev_secret_2, secret_2)

            total_loss = c.lamda_reconstruction_1 * r_loss_1 + c.lamda_reconstruction_2 * r_loss_2 + c.lamda_guide_1 * g_loss_1\
                         + c.lamda_guide_2 * g_loss_2 + c.lamda_low_frequency_1 * l_loss_1 + c.lamda_low_frequency_2 * l_loss_2
            total_loss = total_loss + 0.01 * perc_loss
            total_loss.backward()

            if c.optim_step_1:
                optim1.step()

            if c.optim_step_2:
                optim2.step()

            if c.optim_step_3:
                optim3.step()

            optim1.zero_grad()
            optim2.zero_grad()
            optim3.zero_grad()

            loss_history.append([total_loss.item(), 0.])
            loss_history_g1.append(g_loss_1.item())
            loss_history_g2.append(g_loss_2.item())
            loss_history_r1.append(r_loss_1.item())
            loss_history_r2.append(r_loss_2.item())
            loss_history_imp.append(impmap_loss.item())

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
                for x in testloader:
                    x = x.to(device)
                    cover = x[:x.shape[0] // 3]  # channels = 3
                    secret_1 = x[x.shape[0] // 3: 2 * x.shape[0] // 3]
                    secret_2 = x[2 * x.shape[0] // 3: 3 * x.shape[0] // 3]

                    cover_dwt = dwt(cover)  # channels = 12
                    secret_dwt_1 = dwt(secret_1)
                    secret_dwt_2 = dwt(secret_2)

                    input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1), 1)  # channels = 24

                    #################
                    #    forward1:   #
                    #################
                    output_dwt_1 = net1(input_dwt_1)  # channels = 24
                    output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                    output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12

                    # get steg1
                    output_steg_1 = iwt(output_steg_dwt_1)  # channels = 3

                    #################
                    #    forward2:   #
                    #################
                    if c.use_imp_map:
                        imp_map = net3(cover, secret_1, output_steg_1)  # channels = 3
                    else:
                        imp_map = torch.zeros(cover.shape).cuda()

                    imp_map_dwt = dwt(imp_map)  # channels = 12
                    input_dwt_2 = torch.cat((output_steg_dwt_1, imp_map_dwt), 1)  # 24, without secret2
                    input_dwt_2 = torch.cat((input_dwt_2, secret_dwt_2), 1)  # 36

                    output_dwt_2 = net2(input_dwt_2)  # channels = 36
                    output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                    output_z_dwt_2 = output_dwt_2.narrow(1, 4 * c.channels_in, output_dwt_2.shape[1] - 4 * c.channels_in)  # channels = 24

                    # get steg2
                    output_steg_2 = iwt(output_steg_dwt_2)  # channels = 3

                    #################
                    #   backward2:   #
                    #################

                    output_z_guass_1 = gauss_noise(output_z_dwt_1.shape)  # channels = 12
                    output_z_guass_2 = gauss_noise(output_z_dwt_2.shape)  # channels = 24

                    output_rev_dwt_2 = torch.cat((output_steg_dwt_2, output_z_guass_2), 1)  # channels = 36

                    rev_dwt_2 = net2(output_rev_dwt_2, rev=True)  # channels = 36

                    rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                    rev_secret_dwt_2 = rev_dwt_2.narrow(1, output_dwt_2.shape[1] - 4 * c.channels_in, 4 * c.channels_in)  # channels = 12

                    rev_steg_1 = iwt(rev_steg_dwt_1)  # channels = 3
                    rev_secret_2 = iwt(rev_secret_dwt_2)  # channels = 3

                    #################
                    #   backward1:   #
                    #################
                    output_rev_dwt_1 = torch.cat((rev_steg_dwt_1, output_z_guass_1), 1)  # channels = 24

                    rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 24

                    rev_secret_dwt = rev_dwt_1.narrow(1, rev_dwt_1.shape[1] - 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
                    rev_secret_1 = iwt(rev_secret_dwt)

                    secret_rev1_255 = rev_secret_1.cpu().numpy().squeeze() * 255
                    secret_rev2_255 = rev_secret_2.cpu().numpy().squeeze() * 255
                    secret_1_255 = secret_1.cpu().numpy().squeeze() * 255
                    secret_2_255 = secret_2.cpu().numpy().squeeze() * 255

                    cover_255 = cover.cpu().numpy().squeeze() * 255
                    steg_1_255 = output_steg_1.cpu().numpy().squeeze() * 255
                    steg_2_255 = output_steg_2.cpu().numpy().squeeze() * 255

                    psnr_temp1 = computePSNR(secret_rev1_255, secret_1_255)
                    psnr_s1.append(psnr_temp1)
                    psnr_temp2 = computePSNR(secret_rev2_255, secret_2_255)
                    psnr_s2.append(psnr_temp2)

                    psnr_temp_c1 = computePSNR(cover_255, steg_1_255)
                    psnr_c1.append(psnr_temp_c1)
                    psnr_temp_c2 = computePSNR(cover_255, steg_2_255)
                    psnr_c2.append(psnr_temp_c2)
                    #
                    # if i_epoch % (c.val_freq * 30) == 0:
                    #     torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + '%.5d.png' % i_epoch)
                    #     torchvision.utils.save_image(secret_1, c.IMAGE_PATH_secret_1 + '%.5d.png' % i_epoch)
                    #     torchvision.utils.save_image(secret_2, c.IMAGE_PATH_secret_2 + '%.5d.png' % i_epoch)
                    #
                    #     torchvision.utils.save_image(output_steg_1, c.IMAGE_PATH_steg_1 + '%.5d.png' % i_epoch)
                    #     torchvision.utils.save_image(rev_secret_1, c.IMAGE_PATH_secret_rev_1 + '%.5d.png' % i_epoch)
                    #
                    #     torchvision.utils.save_image(output_steg_2, c.IMAGE_PATH_steg_2 + '%.5d.png' % i_epoch)
                    #     torchvision.utils.save_image(rev_secret_2, c.IMAGE_PATH_secret_rev_2 + '%.5d.png' % i_epoch)

                writer.add_scalars("PSNR", {"S1 average psnr": np.mean(psnr_s1)}, i_epoch)
                writer.add_scalars("PSNR", {"C1 average psnr": np.mean(psnr_c1)}, i_epoch)
                writer.add_scalars("PSNR", {"S2 average psnr": np.mean(psnr_s2)}, i_epoch)
                writer.add_scalars("PSNR", {"C2 average psnr": np.mean(psnr_c2)}, i_epoch)


        epoch_losses = np.mean(np.array(loss_history), axis=0)
        epoch_losses[1] = np.log10(optim1.param_groups[0]['lr'])

        epoch_losses_g1 = np.mean(np.array(loss_history_g1))
        epoch_losses_g2 = np.mean(np.array(loss_history_g2))
        epoch_losses_r1 = np.mean(np.array(loss_history_r1))
        epoch_losses_r2 = np.mean(np.array(loss_history_r2))
        epoch_losses_imp = np.mean(np.array(loss_history_imp))
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
    writer.close()

except:
    if c.checkpoint_on_error:
        torch.save({'opt': optim1.state_dict(),
                    'net': net1.state_dict()}, c.MODEL_PATH + 'model_ABORT_1' + '.pt')
        torch.save({'opt': optim2.state_dict(),
                    'net': net2.state_dict()}, c.MODEL_PATH + 'model_ABORT_2' + '.pt')
        torch.save({'opt': optim3.state_dict(),
                    'net': net3.state_dict()}, c.MODEL_PATH + 'model_ABORT_3' + '.pt')
    raise

finally:
    viz.signal_stop()
