# Fine-tune RTMPose-m (COCO-17, top-down 256x192) trên HA4M với nhãn giả từ skeleton Azure Kinect.
# Dữ liệu: python -m kaizen.train.pose prepare --out data/pose_ft   (chạy trong env chính kaizen-py311)
# Train  (env RIÊNG có mmpose — xem docs/training.md):
#   mim train mmpose kaizen/train/pose_configs/rtmpose-m_ha4m-kinect_256x192.py --work-dir models/pose_ft/rtmpose-m_ha4m
# Kế thừa config chuẩn của mmpose ≥ 1.1 (cú pháp "mmpose::" cần mmpose đã cài); chỉ ghi đè dữ liệu + lịch học.
_base_ = ['mmpose::body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py']

data_root = 'data/pose_ft/'
max_epochs = 60          # fine-tune: ít epoch hơn nhiều so với 420 của train từ đầu
base_lr = 1e-4           # nhỏ hơn base (4e-3) để không phá trọng số pretrained
val_interval = 5

# Checkpoint RTMPose-m pretrained (AIC + COCO) — kiểm tra lại link trong mmpose/projects/rtmpose/README.md
load_from = ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'
             'rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth')

train_cfg = dict(max_epochs=max_epochs, val_interval=val_interval)
optim_wrapper = dict(optimizer=dict(lr=base_lr))
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05, begin=max_epochs // 2, end=max_epochs,
         T_max=max_epochs // 2, by_epoch=True, convert_to_iter_based=True),
]
# Bỏ hook đổi pipeline giai đoạn 2 của base (switch_epoch=390 > max_epochs); giữ EMA.
custom_hooks = [dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True, priority=49)]

train_dataloader = dict(
    batch_size=64,
    dataset=dict(data_root=data_root, data_mode='topdown',
                 ann_file='annotations/person_keypoints_train.json', data_prefix=dict(img='images/')))
val_dataloader = dict(
    batch_size=32,
    dataset=dict(data_root=data_root, data_mode='topdown', bbox_file=None, test_mode=True,
                 ann_file='annotations/person_keypoints_val.json', data_prefix=dict(img='images/')))
test_dataloader = val_dataloader
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/person_keypoints_val.json')
test_evaluator = val_evaluator
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=2))
