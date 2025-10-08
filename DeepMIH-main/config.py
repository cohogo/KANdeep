# RRDB
nf = 3
gc = 32

# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -5.0
lr = 10 ** log10_lr
lr3 = 10 ** -5.0
grad_clip_norm = 1.0
pretrained_skip_substrings = ("kan",)
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01

device_ids = [0]

# Super loss
lamda_reconstruction_1 = 2
lamda_reconstruction_2 = 2
lamda_guide_1 = 1
lamda_guide_2 = 1

lamda_low_frequency_1 = 1
lamda_low_frequency_2 = 1

use_imp_map = True
optim_step_1 = True
optim_step_2 = True
optim_step_3 = True

# Train:
batch_size = 2
cropsize = 128
betas = (0.5, 0.999)
weight_step = 200
gamma = 0.98
dataloader_num_workers = 2
dataloader_eval_workers = 1
grad_accum_steps = 1
use_amp = False
# Val:
cropsize_val_coco = 256
cropsize_val_imagenet = 256
cropsize_val_div2k = 1024

batchsize_val = 3
shuffle_val = False
val_freq = 1

# Dataset
Dataset_mode = 'DIV2K'  # COCO / DIV2K /
Dataset_VAL_mode = 'DIV2K'  # COCO / DIV2K / ImageNet

TRAIN_PATH_DIV2K = r'/root/autodl-fs/DeepMIH_main/div2k/DIV2K_train_HR/DIV2K_train_HR'
VAL_PATH_DIV2K   = r'/root/autodl-fs/DeepMIH_main/div2k/DIV2K_valid_HR/DIV2K_valid_HR'

VAL_PATH_COCO = '/media/disk2/jjp/jjp/Dataset/COCO/val2017/'
TEST_PATH_COCO = '/media/disk2/jjp/jjp/Dataset/COCO/test2017/'

VAL_PATH_IMAGENET = '/media/data/jjp/Imagenet/ILSVRC2012_img_val'

# Display and logging:
loss_display_cutoff = 2.0  # cut off the loss so the plot isn't ruined
loss_names = ['L', 'lr']
silent = False
kan_verbose = False
live_visualization = False
progress_bar = False
# Set ``kan_chunk_size`` to balance memory consumption with a tiny amount of
# extra Python looping overhead.  Larger chunks give slightly better
# throughput, while ``None`` disables chunking entirely for maximal speed on
# GPUs with ample memory.
# KAN memory/expressivity trade-offs
kan_hidden_dims = (32,)
# Stage 1 keeps CNN-based subnetworks by default to save VRAM
kan_stage1_use_scale_nets = False
kan_stage1_use_translation_nets = False
# Stage 2 uses KAN for the scale networks but keeps translations as CNNs
kan_stage2_use_scale_nets = True
kan_stage2_use_translation_nets = False
# Set ``kan_chunk_size`` to balance memory consumption with a tiny amount of
# extra Python looping overhead.  Larger chunks give slightly better
# throughput, while ``None`` disables chunking entirely for maximal speed on
# GPUs with ample memory.
kan_chunk_size = 4096
# Disable the near-identity initialization used by the KAN coupling blocks so
# that training does not get stuck with zero losses and gradients.
kan_identity_init = False
# Increase the jitter that perturbs the KAN weights when an identity init is
# requested. A larger value helps break the symmetry if identity init is
# re-enabled for experiments.
kan_identity_jitter = 1e-2
# RRDB
# Saving checkpoints:
MODEL_PATH = '/root/autodl-fs/DeepMIH_main/model'
checkpoint_on_error = True
SAVE_freq = 1


TEST_PATH = '/home/jjp/DeepMIH/image/'

TEST_PATH_cover = TEST_PATH + 'cover/'
TEST_PATH_secret_1 = TEST_PATH + 'secret_1/'
TEST_PATH_secret_2 = TEST_PATH + 'secret_2/'
TEST_PATH_steg_1 = TEST_PATH + 'steg_1/'
TEST_PATH_steg_2 = TEST_PATH + 'steg_2/'
TEST_PATH_secret_rev_1 = TEST_PATH + 'secret-rev_1/'
TEST_PATH_secret_rev_2 = TEST_PATH + 'secret-rev_2/'
TEST_PATH_imp_map = TEST_PATH + 'imp-map/'


# Load:
suffix_load = ''
tain_next = False

trained_epoch = 3000

pretrain = False
# 主模型加载路径
PRETRAIN_PATH = r'/root/autodl-fs/DeepMIH_main/model/'
suffix_pretrain = 'model_checkpoint_03000'

# 第三子网络（ImpMapBlock）预训练路径
# ⚠️ 注意：这里一定要是“model”不是“models”
PRETRAIN_PATH_3 = r'/root/autodl-fs/DeepMIH_main/model/'
suffix_pretrain_3 = 'model_checkpoint_03000'