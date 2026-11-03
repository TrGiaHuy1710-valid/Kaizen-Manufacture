"""Huấn luyện model (chạy THEO LỆNH người dùng, không tự chạy ngầm). Xem docs/training.md.

- segmenter: MS-TCN phân đoạn bước SOP từ chuỗi keypoint + nhãn HA4M (`python -m kaizen.train.segmenter …`)
- pose: chuẩn bị dữ liệu fine-tune RTMPose từ skeleton Kinect + đánh giá pose so với Kinect (`python -m kaizen.train.pose …`)
features/dataset/metrics là numpy thuần; torch chỉ cần cho segmenter/mstcn; mmpose chỉ cần khi fine-tune pose (env riêng).
"""
