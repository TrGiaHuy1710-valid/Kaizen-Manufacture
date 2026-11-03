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

## A. MS-TCN — phân đoạn bước SOP

### A0. Môi trường

```bash
conda activate kaizen-py311                      # torch CPU đã có (2.14+cpu) — đủ để train
# (tuỳ chọn) GPU: xem CUDA của driver bằng `nvidia-smi`, rồi cài bản torch tương ứng, vd CUDA 12.1:
#   pip install --index-url https://download.pytorch.org/whl/cu121 torch
# Lưu ý: thao tác này thay torch của cả env (embedder RAG vẫn chạy được trên bản CUDA).
```

### A1. Chuẩn bị dữ liệu

Model cần **nhiều người**: HA4M có 41 subject (~217 video); nên có ≥ 10 subject để kết quả có nghĩa.
Hiện máy mới có `U001_V002` đủ dữ liệu: `IDU001V001` chỉ còn file `/storage/huytg/IDU001V001.zip`, giải nén lại nếu cần.

```bash
# 1) Đặt các thư mục IDU*V* vào data/ha4m/ (hoặc truyền --data-root a,b)
# 2) Trích keypoints cho mọi trial (~6 phút/video trên CPU; bỏ qua trial đã có .npz)
python src/01_extract.py --data-root data/ha4m --subjects all --model mediapipe
#    chạy song song nhiều terminal để nhanh hơn, mỗi terminal một nhóm subject:
#    python src/01_extract.py --subjects U001,U002,U003,U004 …
# 3) Kiểm tra số trial sẵn sàng (có .npz + nhãn), và trial nào còn thiếu
python -m kaizen.train.segmenter status
```

### A2. Train

```bash
# Giữ riêng vài subject làm TEST (không dùng khi train/val) để đánh giá trung thực:
python -m kaizen.train.segmenter train --name mstcn_v1 \
    --test-subjects U037,U038,U039,U040,U041 --val-frac 0.15
# có GPU: thêm --device cuda · máy CPU dùng chung, load cao: --threads 1 (mặc định tự chọn theo lõi rảnh)
```

- Ra: `models/segmenter/mstcn_v1/{model.pt, config.json, train_log.csv}` và `models/segmenter/LATEST` → `mstcn_v1`.
- `config.json` ghi lớp, chuẩn hoá đặc trưng, subject train/val/test, siêu tham số, epoch tốt nhất và
  **đánh giá mức bước trên val** (`val_step_eval`: F1@IoU0.5, MAE thời lượng bước, sai số cycle time — cùng định nghĩa
  với `<stem>_eval.json` của bước 04).
- Early stopping theo F1@50 trên val (`--patience 20` epoch). Ước tính thời gian: ~200 video, 100 epoch → vài chục
  phút trên CPU 4 lõi rảnh, vài phút trên GPU.

Siêu tham số chính (mặc định theo MS-TCN gốc):

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--stages` / `--layers` / `--channels` | 4 / 10 / 64 | số stage tinh chỉnh / lớp conv giãn mỗi stage / số kênh |
| `--sample-rate` | 2 | train ở 15 fps (1/2 frame), dự đoán xong lặp lại về 30 fps |
| `--epochs` `--lr` `--batch-size` | 100, 5e-4, 4 | |
| `--lambda-smooth` | 0.15 | trọng số loss làm mượt (giảm phân mảnh đoạn) |
| `--min-seg-s` | 0.5 | khi suy luận: mảnh bước ngắn hơn bị nhập vào bước kề |

### A3. Đánh giá

```bash
# Trên subject test đã giữ riêng (cảnh báo nếu trial đã dùng khi train):
python -m kaizen.train.segmenter eval --model models/segmenter/mstcn_v1 \
    --subjects U037,U038,U039,U040,U041 --out models/segmenter/mstcn_v1/test_eval.json
# Cross-validation theo subject (tốn thời gian: train K lần):
python -m kaizen.train.segmenter cv --folds 5            # --folds 0 = leave-one-subject-out
# Dự đoán thử 1 file:
python -m kaizen.train.segmenter predict --model latest --npz data/keypoints/U037_V001_mediapipe.npz
```

Chỉ số in ra gồm frame accuracy, Edit, F1@{10,25,50} (chuẩn MS-TCN, thang 0–100) và chỉ số mức bước của pipeline
(F1@IoU0.5, MAE bước, sai số cycle time). Mốc so sánh là baseline rule: F1@IoU0.5 0.75 (V001) và 0.714 (V002).

### A4. Dùng trong pipeline + so với rule

```bash
bash run_all.sh --subjects U037 --trials all --dry-run --tag rule
bash run_all.sh --subjects U037 --trials all --dry-run --tag mstcn --segmenter mstcn --segmenter-model models/segmenter/mstcn_v1
python src/tools/compare_runs.py --last 2          # F1@IoU0.5, MAE, sai số CT: rule vs MS-TCN
# Bật mặc định cho run_all và UI: thêm vào .env
#   KAIZEN_SEGMENTER=mstcn
#   KAIZEN_SEGMENTER_MODEL=models/segmenter/mstcn_v1     # bỏ trống = models/segmenter/LATEST
```

Lưu ý:
- Model gắn với **pose model** đã dùng để trích keypoints (mặc định mediapipe). Đổi sang rtmw/rtmpose-custom thì phải
  trích lại và train lại (`--pose-model rtmw`); 04 sẽ cảnh báo nếu không khớp.
- Đặc trưng có toạ độ tuyệt đối trên ảnh (vị trí khay cố định theo camera HA4M). Trạm/camera mới cần dữ liệu có nhãn của
  trạm đó để train lại hoặc fine-tune.
- Bước 04 vẫn ghi `source=pred`, và manifest của run ghi `segmenter` + đường dẫn model để truy vết.

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
