# Huấn luyện model — hướng dẫn chạy tay

Mọi lệnh train trong tài liệu này **do bạn tự chạy**; pipeline không tự train ngầm. Khi chưa có model,
pipeline vẫn dùng đường mặc định (tách bước bằng rule zone + pick, pose MediaPipe pretrained), nên kết quả và artifact
cũ không đổi.

| Model | Học gì | Từ dữ liệu nào | Dùng ở đâu | Môi trường |
|---|---|---|---|---|
| **MS-TCN segmenter** (`kaizen/train/segmenter.py`) | tách 12 bước SOP + lần lặp (2.0/2.1/2.2…) theo từng frame | chuỗi keypoint `.npz` + `Labels.txt` HA4M | bước 04 `--segmenter mstcn` | `kaizen-py311` (torch CPU có sẵn; GPU tuỳ chọn) |
| **RTMPose fine-tune** (`kaizen/train/pose*.py`, `pose_configs/`) | khung xương COCO-17 sát cảnh HA4M hơn | ảnh Color + skeleton Azure Kinect (nhãn giả) | bước 01 `--model rtmpose-custom` | train trong env **riêng** có mmpose |

Kiểm thử phần training (dữ liệu tổng hợp, vài giây, không phải train thật): `python -m pytest tests/test_train.py`.

---

## A. MS-TCN — phân đoạn bước SOP (train · val · benchmark theo epoch · test)

Luồng đầy đủ cho phần thực nghiệm của báo cáo:

```
zip HA4M ──A1──▶ keypoints .npz + Labels.txt ──A3 train──▶ models/segmenter/<name>/
                                                │  ├─ checkpoints/epoch_XXXX.pt  (mỗi --ckpt-every epoch)
                                                │  ├─ tb/  (TensorBoard: loss, CE, smooth, acc, val F1…)
                                                │  └─ model.pt = epoch tốt nhất theo F1@50 trên VAL
                                                ├──A4 benchmark──▶ bảng + biểu đồ chỉ số theo epoch (val / test)
                                                └──A5 test──────▶ dự đoán từng frame, bảng bước, confusion, Gantt
```

### A0. Môi trường

```bash
conda activate kaizen-py311        # python 3.11, torch 2.14.1+cpu, tensorboard 2.21 (đã cài)
cd /storage/huytg/KaizenMotionAnalyzer
python -m pytest tests/test_train.py -q      # 8 test, ~20 s — kiểm vòng train/ckpt/benchmark/test/resume
```

- Torch trong env là bản **CPU**, đủ để train MS-TCN (model ~0,68 triệu tham số). GPU Quadro RTX 8000 của máy
  đang dùng chung (thường 100% util). Muốn dùng GPU thì phải cài torch CUDA, và việc này **thay torch của cả env**:
  `pip install --index-url https://download.pytorch.org/whl/cu121 torch`, rồi thêm `--device cuda` khi train.
- Máy có 4 lõi CPU và đang dùng chung. `--threads 0` (mặc định) tự chọn số lõi đang rảnh, có khi chỉ còn 1.
  Nếu biết máy rảnh, đặt `--threads 4`.

### A1. Chuẩn bị dữ liệu (đĩa chỉ còn ~32 GB trống!)

Mỗi zip một lần lắp (`IDU001V001.zip`) nặng ~12 GB, giải nén đủ ~17 GB. Train chỉ cần `.npz` (~1,3 MB) và `Labels.txt`.
`src/tools/prepare_ha4m.py` xử lý **tuần tự** từng zip: chỉ giải nén `Color/` và `Labels.txt` (~7,6 GB), trích
keypoints MediaPipe, rồi xoá ảnh Color (giữ thư mục rỗng và nhãn). Script kiểm tra chỗ trống trước mỗi zip, và
bỏ qua trial đã xong nên có thể chạy lại an toàn sau khi bị ngắt.

```bash
# zip tải về để ở một thư mục (vd /storage/huytg/ha4m_zips/). Đĩa ít chỗ → tải vài zip, xử lý, --delete-zip, tải tiếp.
python src/tools/prepare_ha4m.py --zips "/storage/huytg/ha4m_zips/IDU00*.zip,/storage/huytg/ha4m_zips/IDU010*.zip" \
    --data-root data/ha4m --delete-zip
python src/tools/prepare_ha4m.py --zips "..." --dry-run        # chỉ liệt kê: trial nào, cần bao nhiêu GB
# Giữ ảnh cho vài trial muốn render overlay / ảnh bằng chứng trong report: thêm --keep-color

python -m kaizen.train.segmenter status    # số trial sẵn sàng (có .npz + nhãn), liệt kê theo subject
```

Tuỳ chọn: `--with-skeletons` giải nén thêm skeleton Kinect (20 MB/trial), cần cho `kaizen.train.pose eval`.

### A2. Thống kê + chia dữ liệu (`kaizen/train/split.py`)

```bash
python -m kaizen.train.split stats          # từng video: READY / NEED_EXTRACT / NO_LABELS, số frame, 17 bước đủ chưa,
                                            # zip đang tải dở (INCOMPLETE) hay chưa giải nén → outputs/dataset/ha4m_stats.md
python -m kaizen.train.split make --name v1 # → data/splits/v1.json (cố định; không ghi đè nếu thiếu --force)
python -m kaizen.train.segmenter train --split-file data/splits/v1.json --name mstcn_v1
bash src/tools/train_ha4m.sh --split v1     # gộp cả 5 bước: stats → 01_extract → split → train → test
```

`--strategy auto` tự chọn theo số video có nhãn: **1 video → `overfit`** (train = val, không có test, chỉ để kiểm tra code
chạy thông; không thể cắt 1 video theo thời gian vì các bước nối tiếp nhau); **≥ 2 video nhưng < 3 người → `trial`**
(chia theo video, có thể rò rỉ theo người, chỉ dùng thử); **≥ 3 người → `subject`**. U001 luôn ở train (`--pin-train`).
Ép tay: `--val U008 --test U009,U010` (subject hoặc video). Có thêm dữ liệu thì tạo split mới (`--name v2`), không sửa v1.

Khuyến nghị cho U001–U010:

Chia **theo người** (không ai xuất hiện ở hai tập), và quyết định split **trước khi** nhìn bất kỳ kết quả test nào:

| Tập | Subject | Vai trò |
|---|---|---|
| train | U001–U007 | học trọng số; thống kê chuẩn hoá z-score chỉ tính trên tập này |
| val | U008 | early stopping, chọn epoch (`model.pt`), chọn siêu tham số / ablation |
| test | U009, U010 | **chỉ chạy một lần ở cuối** để báo cáo; baseline rule cũng đo trên đúng các trial này |

U001 nên nằm trong **train**, vì zone, ROI và ngưỡng của baseline rule (`config/station.yaml`) đã được chỉnh trên U001.
Đưa U001 vào test là rò rỉ dữ liệu. Nếu có thêm dữ liệu, dùng thêm LOSO cross-validation ở A6.

### A3. Train + TensorBoard

```bash
python -m kaizen.train.segmenter train --name mstcn_s0 \
    --subjects U001,U002,U003,U004,U005,U006,U007,U008,U009,U010 \
    --val-subjects U008 --test-subjects U009,U010 \
    --epochs 100 --eval-every 2 --ckpt-every 10 --patience 20 --seed 0
```

Theo dõi trong một terminal khác (VS Code tự forward cổng 6006; qua ssh thì dùng `ssh -L 6006:localhost:6006 <máy>`):

```bash
tensorboard --logdir models/segmenter --port 6006        # so được nhiều model / seed cùng lúc
```

| Tag TensorBoard | Ý nghĩa |
|---|---|
| `train/loss`, `train/loss_ce`, `train/loss_smooth` | loss tổng = Σ_stage (CE + λ·T-MSE), tách hai thành phần |
| `train/train_acc`, `train/lr` | accuracy theo frame trên batch train (đang bật dropout) · learning rate |
| `val/loss`, `val/acc`, `val/edit`, `val/f1@10/25/50`, `val/best_f1@50` | tính mỗi `--eval-every` epoch trên tập val |
| `time/epoch_s` | thời gian một epoch |
| `benchmark_<split>/…` | do lệnh `benchmark` (A4) ghi thêm, trục x là epoch |

Thư mục `models/segmenter/mstcn_s0/` gồm:

| File | Nội dung |
|---|---|
| `model.pt` | trọng số ở **epoch tốt nhất theo val F1@50**. Dùng cho test và cho bước 04 của pipeline |
| `checkpoints/epoch_0010.pt …` | trọng số mỗi `--ckpt-every` epoch, cùng epoch dừng sớm, để benchmark theo epoch |
| `checkpoints/last.pt` | trạng thái đầy đủ (optimizer, RNG, lịch sử), dùng cho `--resume` |
| `config.json` | lớp (18 = 17 khoá bước.lần + nền), thống kê chuẩn hoá, trial/subject của từng tập, siêu tham số, seed, `best_epoch`, `train_seconds`, `n_params`, `val_step_eval` |
| `train_log.csv`, `train_curve.png` | lịch sử từng epoch; hình loss train/val và acc/F1 (đường đứt = best epoch) → **hình đường cong train trong báo cáo** |
| `tb/` | log TensorBoard |

- Train bị ngắt: chạy lại đúng lệnh, thêm `--resume` (đi tiếp từ `last.pt`). Muốn train thêm epoch: `--resume --epochs 150`.
- Ghi đè model cùng tên: `--force`. Nếu không, lệnh sẽ từ chối.
- Chạy thử khi mới có 1 người: `--val-trials U001_V002`. Val khi đó chia theo trial, có rò rỉ theo người, nên
  lệnh in cảnh báo. **Không dùng số này trong báo cáo.**

Siêu tham số (mặc định theo MS-TCN gốc, Farha & Gall 2019):

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--stages` / `--layers` / `--channels` | 4 / 10 / 64 | số stage / số lớp conv giãn mỗi stage / số kênh |
| `--dropout` | 0.5 | |
| `--lambda-smooth` | 0.15 | trọng số loss làm mượt T-MSE (τ = 4 cố định) |
| `--lr` `--weight-decay` `--batch-size` | 5e-4, 1e-4, 4 | Adam, learning rate cố định |
| `--epochs` `--patience` `--eval-every` | 100, 20, 2 | early stopping theo val F1@50 |
| `--ckpt-every` | 10 | chu kỳ lưu checkpoint cho benchmark (0 = tắt) |
| `--sample-rate` | 2 | train ở 15 fps; khi dự đoán lặp lại nhãn về 30 fps |
| `--min-seg-s` | 0.5 | hậu xử lý khi suy luận: mảnh bước ngắn hơn bị nhập vào bước kề |
| `--seed` | 0 | seed cho khởi tạo và thứ tự batch (cả chia val ngẫu nhiên nếu không truyền `--val-subjects`) |

### A4. Benchmark theo epoch (từ các checkpoint)

```bash
python -m kaizen.train.segmenter benchmark --model models/segmenter/mstcn_s0 --split val
python -m kaizen.train.segmenter benchmark --model models/segmenter/mstcn_s0 --split test   # chỉ để vẽ đường cong
```

- Lệnh đánh giá mọi `checkpoints/epoch_*.pt` trên tập lấy từ `config.json`, với cả chỉ số frame/đoạn (acc, Edit,
  F1@{10,25,50}, thang 0–100) và chỉ số mức bước của pipeline (F1@IoU0.5 thang 0–1, MAE thời lượng bước, sai số
  cycle time, tính bằng giây).
- Kết quả ghi vào `benchmark_<split>.csv/.json/.png` (đường chấm là best epoch theo val) và TensorBoard `benchmark_<split>/`.
- **Quy tắc:** chọn epoch và siêu tham số **chỉ theo val**. Benchmark trên test chỉ minh hoạ under/overfitting
  trong báo cáo, không được dùng để chọn model.
- Tập tuỳ ý: `--subjects U009 --tag u009` hoặc `--trials U009_V001,U009_V002 --tag x`.

### A5. Test — suy luận trên tập giữ riêng

```bash
python -m kaizen.train.segmenter test --model models/segmenter/mstcn_s0                   # model.pt (best theo val)
python -m kaizen.train.segmenter test --model models/segmenter/mstcn_s0 --checkpoint epoch_0050   # 1 checkpoint cụ thể
python -m kaizen.train.segmenter predict --model models/segmenter/mstcn_s0 --npz data/keypoints/U009_V001_mediapipe.npz
```

Mặc định lệnh dùng `test_subjects`/`test_trials` đã ghi trong `config.json` và cảnh báo nếu trial test trùng người
với train/val. Kết quả ghi vào `models/segmenter/mstcn_s0/test/`:

| File | Dùng trong báo cáo |
|---|---|
| `summary.json` | mean ± std mọi chỉ số, macro-F1 theo frame, số tham số, **ms suy luận/video** (độ phức tạp tính toán) |
| `per_trial.csv` | bảng kết quả từng video test |
| `per_class.csv` | precision / recall / F1 theo từng lớp `bước.lần` |
| `confusion_frames.csv/.png` | ma trận nhầm lẫn theo frame, phục vụ phân tích nhầm giữa các lần lặp 2.x/3.x và bước 9 |
| `trials/<stem>_gantt.png` | Gantt nhãn HA4M so với MS-TCN, làm **ví dụ dự đoán đúng/sai** |
| `trials/<stem>_frames.csv`, `_steps_pred.csv`, `_steps_oracle.csv` | dự đoán từng frame, bảng bước dự đoán và bảng bước theo nhãn |

Baseline rule trên **cùng trial test** để so sánh công bằng (dùng được cả khi đã xoá ảnh Color, vì 01 bỏ qua trial đã có
`.npz`; chỉ ảnh bằng chứng trong report bị bỏ):

```bash
bash run_all.sh --subjects U009,U010 --trials all --dry-run --tag test_rule
bash run_all.sh --subjects U009,U010 --trials all --dry-run --tag test_mstcn \
     --segmenter mstcn --segmenter-model models/segmenter/mstcn_s0
python src/tools/compare_runs.py --last 2        # F1@IoU0.5, MAE bước, sai số CT: rule vs MS-TCN
```

### A6. Biến thiên theo seed, ablation, cross-validation

```bash
# ≥ 3 seed → báo mean ± std (đề bài yêu cầu "tính biến thiên theo lần chạy")
for s in 0 1 2; do
  python -m kaizen.train.segmenter train --name mstcn_s$s --seed $s \
      --subjects U001,U002,U003,U004,U005,U006,U007,U008,U009,U010 --val-subjects U008 --test-subjects U009,U010
  python -m kaizen.train.segmenter test --model models/segmenter/mstcn_s$s
done

# Ablation (mỗi cấu hình nên chạy cùng 3 seed như trên; chọn/so sánh theo VAL, báo cáo trên TEST):
#   --stages 1             SS-TCN: một stage, không tinh chỉnh → baseline có học để so với MS-TCN
#   --lambda-smooth 0      bỏ loss làm mượt → ảnh hưởng tới Edit/phân mảnh
#   --sample-rate 1 | 4    độ phân giải thời gian 30 / 7.5 fps
#   --layers 8 | --channels 32   receptive field / dung lượng model
python -m kaizen.train.segmenter train --name abl_s1_seed0 --stages 1 --seed 0 \
    --subjects U001,U002,U003,U004,U005,U006,U007,U008,U009,U010 --val-subjects U008 --test-subjects U009,U010

# Cross-validation theo người (mỗi fold train lại từ đầu; TensorBoard ghi ở models/segmenter/<cv>/fold*/tb)
python -m kaizen.train.segmenter cv --folds 0 --subjects U001,U002,U003,U004,U005,U006,U007,U008,U009,U010 --name cv_loso
#   → cv_report.json (mean ± std), cv_per_trial.csv; --save-folds để lưu model + checkpoint từng fold
```

Ánh xạ sang bảng trong `docs/report/sections/05_experiments.tex`:

| Bảng / hình | Lấy từ |
|---|---|
| `tab:mstcn` dòng "Test giữ riêng" | `test/summary.json` của 3 seed → mean ± std các cột acc / edit / F1@k |
| `tab:mstcn` dòng LOSO | `cv_loso/cv_report.json` → `summary` |
| `tab:mstcn` các dòng ablation | `test/summary.json` của từng mô hình ablation |
| `tab:seg` dòng MS-TCN | `test/per_trial.csv` (f1_iou05, mae_step_dur_s, cycle_time_err_s), cùng dòng rule của `compare_runs.py` trên cùng trial |
| `tab:hparams` (phần cứng, thời gian train) | `config.json`: `device`, `train_seconds`, `best_epoch`, `epochs_run`, `n_params` |
| Hình đường cong train | `train_curve.png` (hoặc ảnh chụp TensorBoard), cùng `benchmark_val.png` |
| Ví dụ sai, confusion | `test/confusion_frames.png`, `test/trials/*_gantt.png`, `test/per_class.csv` |

### A7. Thời gian dự kiến (đo trên máy này, ngày 19/11/2026)

Số đo thật:
- Trích keypoints MediaPipe: 7,2–7,6 fps trên CPU, tức khoảng **5,6–5,8 phút/video** (~2.400–2.650 frame).
- Giải nén Color: ~133 MB/s, tức khoảng **1 phút/video**.
- Train MS-TCN: **0,18–0,21 s/epoch** khi có 1 video train và 1 video val, chạy CPU **1 luồng** (máy đang tải 3,5/4 lõi).
- Suy luận: **~0,46 s/video** trên CPU.

Ước tính cho **U001–U010** (HA4M có 217 video / 41 người, trung bình ~5 video/người, tức **~50 video**; số thật xem bằng `status`):

| Việc | Ước tính | Ghi chú |
|---|---|---|
| Tải 50 zip | ~600 GB qua mạng, tuỳ băng thông | đĩa còn 32 GB → tải 1–2 zip, chạy A1 với `--delete-zip`, rồi tải tiếp |
| A1 giải nén + keypoints | **~7 phút/video → ~6 giờ** | phần tốn thời gian nhất, chạy qua đêm được |
| A3 train một model (~35 video train, 100 epoch) | **~10 phút** (CPU 1 luồng); vài phút nếu 4 luồng rảnh; < 1 phút trên GPU | early stopping thường dừng sớm hơn |
| A4 benchmark 10 checkpoint × val ~5 video | ~1 phút | |
| A5 test ~10 video | < 1 phút | |
| 3 seed | ~30 phút | |
| 3 ablation × 3 seed | ~1,5 giờ | |
| LOSO 10 fold | ~1,5–2 giờ | |
| **Tổng** | **~6 giờ chuẩn bị dữ liệu + ~1 giờ cho một lượt train/val/test 3 seed + ~3 giờ nếu chạy đủ ablation và LOSO** | |

Nếu chỉ có **10 video** (không phải 10 người): A1 mất ~70 phút, train ~1–2 phút. Khi đó lại chỉ có ~2 người, nên **không
chia train/val/test theo người được**, và kết quả không đủ cơ sở cho báo cáo.

### A8. Dùng model trong pipeline

```bash
bash run_all.sh --subjects U009 --trials all --dry-run --segmenter mstcn --segmenter-model models/segmenter/mstcn_s0
# Bật mặc định cho run_all và UI: thêm vào .env
#   KAIZEN_SEGMENTER=mstcn
#   KAIZEN_SEGMENTER_MODEL=models/segmenter/mstcn_s0     # bỏ trống = models/segmenter/LATEST
```

Lưu ý:
- Model gắn với **pose model** đã dùng để trích keypoints (mặc định mediapipe). Đổi sang rtmw thì phải trích lại và
  train lại (`--pose-model rtmw`); bước 04 sẽ cảnh báo nếu không khớp.
- Đặc trưng `kp_v1` có toạ độ tuyệt đối trên ảnh. Kết quả chỉ phản ánh khả năng tổng quát sang **người mới trong cùng
  trạm/camera HA4M**.
- `models/` không đưa lên git. Nộp trọng số bằng link riêng, kèm `config.json`.

---

## B. Fine-tune pose RTMPose trên HA4M

Ý tưởng: dùng skeleton Azure Kinect (đã chiếu lên ảnh Color) làm **nhãn giả** cho 17 khớp COCO, fine-tune RTMPose-m
pretrained (COCO + AIC), export ONNX, rồi nạp lại vào pipeline qua backend `rtmpose-custom`.
Giới hạn: nhãn giả mang sai số của Kinect (nhất là cổ tay khi bị che); COCO-17 không có điểm bàn tay, nên chỉ số Muri
cổ tay sẽ báo "không đủ điểm". Chỉ giữ model fine-tune nếu bước B5 cho kết quả tốt hơn baseline.

### B1. Chuẩn bị dữ liệu (env chính, không cần mmpose)

```bash
conda activate kaizen-py311
python -m kaizen.train.pose prepare --data-root data/ha4m --out data/pose_ft \
    --stride 15 --val-subjects U040,U041 --min-conf 0.5
# → data/pose_ft/images/{train,val}/*.jpg (crop quanh người), annotations/person_keypoints_{train,val}.json, stats.json
```

### B2. Đo baseline trước khi fine-tune

```bash
python -m kaizen.train.pose eval --pose-model mediapipe --subjects U040,U041 --out outputs/pose_eval_mediapipe.json
python src/01_extract.py --model rtmw --subjects U040,U041 && \
python -m kaizen.train.pose eval --pose-model rtmw --subjects U040,U041 --out outputs/pose_eval_rtmw.json
```

### B3. Train (env riêng có mmpose, cần GPU)

```bash
conda create -n kaizen-pose python=3.10 -y && conda activate kaizen-pose
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim && mim install mmengine "mmcv==2.1.0" "mmdet>=3.1.0" "mmpose>=1.3.0"
cd /storage/huytg/KaizenMotionAnalyzer
mim train mmpose kaizen/train/pose_configs/rtmpose-m_ha4m-kinect_256x192.py \
    --work-dir models/pose_ft/rtmpose-m_ha4m --gpus 1
```

Config kế thừa `mmpose::body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py` và chỉ ghi đè dữ liệu,
60 epoch, lr 1e-4, checkpoint khởi tạo. Config **chưa được chạy thử** trên máy này (env chính không có mmpose):
kiểm tra link `load_from` trong `mmpose/projects/rtmpose/README.md` trước khi train.

### B4. Export ONNX

```bash
pip install mmdeploy==1.3.1 onnxruntime
git clone -b v1.3.1 https://github.com/open-mmlab/mmdeploy.git ~/mmdeploy
python ~/mmdeploy/tools/deploy.py ~/mmdeploy/configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
    kaizen/train/pose_configs/rtmpose-m_ha4m-kinect_256x192.py \
    models/pose_ft/rtmpose-m_ha4m/best_coco_AP_epoch_*.pth \
    data/pose_ft/images/val/$(ls data/pose_ft/images/val | head -1) \
    --work-dir models/pose_ft/rtmpose-m_ha4m/onnx --device cpu
# → models/pose_ft/rtmpose-m_ha4m/onnx/end2end.onnx
```

### B5. Dùng lại trong pipeline và so với baseline

```bash
conda activate kaizen-py311
export KAIZEN_POSE_ONNX=models/pose_ft/rtmpose-m_ha4m/onnx/end2end.onnx   # KAIZEN_POSE_INPUT=192x256 (W×H, mặc định)
python src/01_extract.py --model rtmpose-custom --subjects U040,U041
python -m kaizen.train.pose eval --pose-model rtmpose-custom --subjects U040,U041 --out outputs/pose_eval_ft.json
# PCK@0.05 / @0.1 và sai số px từng khớp (cổ tay!) so với baseline ở B2. Nếu tốt hơn:
bash run_all.sh --model rtmpose-custom --subjects U040 --trials all --dry-run --tag pose_ft
```
