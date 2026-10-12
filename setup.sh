#!/usr/bin/env bash
# Tạo môi trường conda + tải model MediaPipe/RTMW.
# Chạy:   bash setup.sh            -> Python 3.11 (mặc định)
#         bash setup.sh 3.10       -> Python 3.10
#         ENV_NAME=kaizen bash setup.sh 3.12   -> đặt tên env tùy ý
set -e

PY_VER="${1:-3.11}"
ENV_NAME="${ENV_NAME:-kaizen-py${PY_VER//./}}"   # vd: kaizen-py311

if ! command -v conda >/dev/null 2>&1; then
  echo "Không tìm thấy conda. Hãy cài Miniforge hoặc Miniconda trước." >&2
  exit 1
fi

# Cho phép dùng 'conda activate' bên trong script
eval "$(conda shell.bash hook)"

# Tạo env nếu chưa có (dùng kênh conda-forge, tránh phải chấp nhận ToS của kênh defaults)
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">> Env '$ENV_NAME' đã tồn tại, bỏ qua bước tạo."
else
  conda create -y -n "$ENV_NAME" --override-channels -c conda-forge python="$PY_VER" pip
fi

conda activate "$ENV_NAME"
echo ">> Đang dùng: $(python --version) tại $(which python)"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Thư mục dữ liệu
mkdir -p models data/raw data/keypoints outputs

# Tải file nếu chưa có; tải ra .part rồi mới đổi tên để không để lại file hỏng
download() {
  local out="$1" url="$2"
  if [ -f "$out" ]; then
    echo ">> Đã có $out, bỏ qua."
  else
    wget -O "$out.part" "$url" && mv "$out.part" "$out"
  fi
}

MP=https://storage.googleapis.com/mediapipe-models
download models/pose_landmarker_full.task \
  "$MP/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
download models/pose_landmarker_heavy.task \
  "$MP/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
download models/hand_landmarker.task \
  "$MP/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Kích hoạt tải sẵn model RTMW (rtmlib tự tải khi khởi tạo lần đầu)
python - << 'PY'
import numpy as np
from rtmlib import Wholebody
wb = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
wb(np.zeros((480, 640, 3), dtype=np.uint8))
print("RTMW ready")
PY

echo "Done. Kích hoạt môi trường: conda activate $ENV_NAME"