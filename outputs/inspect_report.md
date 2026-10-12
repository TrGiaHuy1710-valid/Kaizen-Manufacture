# Inspect report — HA4M

- data_root: `data/ha4m`
- Số video (IDU*V*) tìm thấy: **1** → U001V001
- Video khảo sát: `IDU001V001` (subject=`U001`, trial=`V001`)

## Cây thư mục 2 cấp
```
IDU001V001/
    Color/  (1 mục)
        000124702712/  (2428 mục)
    ColorAligned/  (1 mục)
        000124702712/  (2427 mục)
    Depth/  (1 mục)
        000124702712/  (2427 mục)
    Infrared/  (1 mục)
        000124702712/  (2428 mục)
    PointClouds_DepthGeometry/  (1 mục)
        000124702712/  (2427 mục)
    Skeletons/  (1 mục)
        000124702712/  (2427 mục)
    [1 file] Labels.txt
```

## Labels.txt
- Số dòng: 2427; CRLF: True
```
0 0 0
1 0 0
2 0 0
3 0 0
4 0 0
5 0 0
6 0 0
7 0 0
8 0 0
9 0 0
```
Schema suy ra:
- Phân cách: **space**, 3 cột, toàn số nguyên: True
- Cột 1 tăng liên tục +1 → **frame index** (nhãn theo frame, không phải timestamp): True
- Cột 2 = action_id (0 = don't care, 1..12), cột 3 = repetition (0,1,2 cho action lặp 2/3/11)
- TrialID KHÔNG nằm trong file: mỗi video có Labels.txt riêng trong thư mục `ID<Subject><Trial>`

Các đoạn liên tiếp (action_id, repetition):

| action_id | rep | start_frame | end_frame | n_frames | tên (paper) |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 29 | 30 | Don't care |
| 1 | 0 | 30 | 155 | 126 | Pick up/Place Carrier |
| 2 | 0 | 156 | 260 | 105 | Pick up/Place Gear Bearings (x3) |
| 2 | 1 | 261 | 364 | 104 | Pick up/Place Gear Bearings (x3) |
| 2 | 2 | 365 | 436 | 72 | Pick up/Place Gear Bearings (x3) |
| 3 | 0 | 437 | 514 | 78 | Pick up/Place Planet Gears (x3) |
| 3 | 1 | 515 | 592 | 78 | Pick up/Place Planet Gears (x3) |
| 3 | 2 | 593 | 674 | 82 | Pick up/Place Planet Gears (x3) |
| 4 | 0 | 675 | 788 | 114 | Pick up/Place Carrier Shaft |
| 5 | 0 | 789 | 899 | 111 | Pick up/Place Sun Shaft |
| 6 | 0 | 900 | 997 | 98 | Pick up/Place Sun Gear |
| 7 | 0 | 998 | 1100 | 103 | Pick up/Place Sun Gear Bearing |
| 8 | 0 | 1101 | 1220 | 120 | Pick up/Place Ring Bear |
| 9 | 0 | 1221 | 1473 | 253 | Pick up Block 2 and place it on Block 1 |
| 10 | 0 | 1474 | 1756 | 283 | Pick up/Place Cover |
| 11 | 0 | 1757 | 1916 | 160 | Pick up/Place Screw (x2) |
| 11 | 1 | 1917 | 2047 | 131 | Pick up/Place Screw (x2) |
| 12 | 0 | 2048 | 2366 | 319 | Pick up Allen Key, turn both screws, return Allen Key and the EGT |
| 0 | 0 | 2367 | 2426 | 60 | Don't care |

## Color frames
- Thư mục: `Color/000124702712` (thư mục con = serial camera)
- 5 file đầu:
```
FrameID000000_DeviceTimeStamp450817700us.png
FrameID000001_DeviceTimeStamp450817700us.png
FrameID000002_DeviceTimeStamp450817700us.png
FrameID000003_DeviceTimeStamp450817700us.png
FrameID000004_DeviceTimeStamp450817700us.png
```
- Pattern: `FrameID<6 số>_DeviceTimeStamp<micro giây>us.png`; file lạ bị bỏ qua: ['Thumbs.db']
- Số frame: **2427** (frame 0 → 2426); số dòng Labels: 2427
- Kích thước ảnh: **2048x1536**, 3 kênh, uint8
- fps ước lượng từ DeviceTimeStamp: **30.00** (median Δt = 33.33 ms); paper HA4M ghi 30 fps → dùng `fps: 30` trong config
- Δt = 0 (timestamp trùng): 7 frame (thường ở đầu video); Δt > 1.5×median (rơi frame): 0
- Thời lượng theo frame_idx/30: 80.9 s; theo timestamp: 80.6 s

## Skeletons (Azure Kinect)
- Số file: 2427; ví dụ `FrameID000100_DeviceTimeStamp453917733us.txt` (32 dòng = joint):
```
BodyID	JointID	Confidance	X	Y	Z	Qw	Qx	Qy	Qz	X2DColor	Y2DColor	X2DDepth	Y2DDepth
1	0	0.666667	201.193924	-42.728466	1845.524048	0.661674	-0.342518	0.544618	-0.385045	1113.798462	858.448181	378.147156	325.841736
1	1	0.666667	221.016052	-205.628204	1764.208496	0.654081	-0.353018	0.537872	-0.397807	1127.397705	768.626038	385.942261	279.363251
1	2	0.666667	236.371918	-337.426239	1702.316772	0.714859	-0.276129	0.493969	-0.410760	1138.647827	691.499756	392.045013	239.614914
1	3	0.666667	274.968170	-529.601196	1596.248779	0.745497	-0.235627	0.514522	-0.352110	1167.412109	567.225830	406.383789	177.830109
```
- TSV có header; 32 joint (0–31) theo Azure Kinect Body Tracking; X2DColor/Y2DColor = toạ độ pixel trên ảnh Color → dùng đối chứng cổ tay (joint 7 = WRIST_LEFT, 14 = WRIST_RIGHT).

## Kết luận cho parser (`src/ha4m_io.py`)
- `list_trials(root)` quét thư mục `ID(U\d+)(V\d+)` → (subject, trial).
- `frames_iter` sort theo số `FrameID`, bỏ Thumbs.db.
- `load_labels` gộp frame liên tiếp cùng (action_id, repetition) → đoạn; bỏ action 0.
- Thời gian t = frame_idx / fps (fps = 30, khớp timestamp thiết bị).
