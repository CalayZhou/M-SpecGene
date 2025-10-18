_base_ = [
    '../_base_/models/upernet_mae_classbalance.py', '../_base_/datasets/ade20k_640x640.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'#160
]
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='./pretrain/0322_imagenet128w_vit-s_epoch_300_seg_transform.pth',
    #pretrained='./pretrain/vit-s_0815_bs960_imagenet_epoch_299_transform_mmseg.pth',
    #pretrained='./pretrain/0831_v8.5_epoch_500_transform.pth',
    #pretrained='./pretrain/crossattention_selfpredict_0620_epoch_500_transform.pth',
    # pretrained='./pretrain/mae_pretrain_vit_base_mmcls.pth',
    backbone=dict(
        type='MAE',
        img_size=(640, 640),
        patch_size=16,
        embed_dims=768,
        num_layers=8,#12,
        num_heads=8,#12,
        mlp_ratio=3,#4,
        init_values=1.0,
        drop_path_rate=0.1,
        out_indices=[1, 3, 5, 7]),#[3, 5, 7, 11]  0 2 4 6
    neck=dict(embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[768, 768, 768, 768], num_classes=13, channels=768),#13 150
    auxiliary_head=dict(in_channels=768, num_classes=13),#13 150
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(341, 341)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65),
    constructor='LayerDecayOptimizerConstructor')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# mixed precision
fp16 = dict(loss_scale='dynamic')

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=3)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
