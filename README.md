# CAPONY-HEIGHT-MAP

Dự án dự báo **bản đồ chiều cao tán rừng (Canopy Height Map)** tại tỉnh **Đắk Lắk, Việt Nam**, kết hợp dữ liệu vệ tinh **GEDI**, **Sentinel-1** và **Sentinel-2**. Repository bao gồm hai hướng tiếp cận: mô hình học máy cổ điển trên dữ liệu bảng (ML) và mô hình học sâu dựa trên ảnh vệ tinh (DL).

## Mục tiêu

- Dự báo chiều cao tán rừng (nhãn GEDI RH95) từ các đặc trưng radar và quang học.
- So sánh hiệu quả giữa các thuật toán ML (XGBoost, Random Forest, LightGBM) và kiến trúc CNN (UNet, UNet++).
- Tạo bản đồ chiều cao tán rừng liên tục trên toàn vùng nghiên cứu.

## Cấu trúc thư mục

```
CAPONY-HEIGHT-MAP/
├── ML/
│   ├── ml.ipynb              # Thí nghiệm ML: LightGBM, XGBoost, Random Forest, MLP
│   ├── ml (2).ipynb
│   └── ml (3).ipynb
├── DL/
│   ├── unetplusplus.ipynb    # Pipeline UNet++ đầy đủ (tiền xử lý → huấn luyện → đánh giá)
│   └── unet_canopy_height.ipynb  # Thí nghiệm UNet cơ bản
├── train.py                  # Script huấn luyện UNet++ tối ưu (production-ready)
├── requirement.txt
└── README.md
```

## Dữ liệu

### Nguồn dữ liệu

| Nguồn | Vai trò |
|-------|---------|
| **GEDI** | Nhãn chiều cao tán rừng (RH95), đơn vị mét |
| **Sentinel-1** | VV/VH ascending & descending (radar) |
| **Sentinel-2** | Các band quang học B2–B12 |

### Định dạng dữ liệu

**Deep Learning** — file GeoTIFF đa band (`DakLak_Full_Merged_Final_Final.tif`):

| Band | Tên | Mô tả |
|------|-----|-------|
| 0 | `rh95` | Nhãn GEDI (chiều cao tán rừng) |
| 1–4 | Sentinel-1 | VV/VH ascending, VV/VH descending |
| 5–14 | Sentinel-2 | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 |

**Machine Learning** — file CSV (`DakLak_train.csv`, `DakLak_test.csv`):

- Nhãn: `rh95`
- Đặc trưng thô: band Sentinel-1/Sentinel-2
- Đặc trưng kỹ thuật: `VH_to_VV_asc`, `VH_to_VV_desc`, `VV_diff`, `VH_diff`, `VV_mean`, `VH_mean`, `NDVI`, `EVI`, `SAVI`, `GNDVI`, `MSAVI`

> File `.tif` và `.csv` không được đưa vào git (xem `.gitignore`). Cần tự tải hoặc chuẩn bị dữ liệu trước khi chạy.

### Tải dữ liệu (Google Drive)

Trong notebook `DL/unetplusplus.ipynb`:

```bash
pip install gdown
gdown 19TewnE9ntvK1wQ7seDHeQXOrXt9iAsVk
```

## Pipeline Deep Learning (UNet++)

Notebook `DL/unetplusplus.ipynb` triển khai pipeline hoàn chỉnh:

### 1. Tiền xử lý

1. **Loại outlier GEDI** — cắt giá trị ngoài phân vị 0.1%–99.9% trên band nhãn.
2. **Spatial tiling** — cắt mosaic thành tile 512×512 px (bỏ tile có NaN ở band kiểm tra).
3. **Lọc theo GEDI** — chỉ giữ tile có ít nhất một footprint GEDI hợp lệ.
4. **Chia tập** — train/validation/test theo tỷ lệ **80% / 10% / 10%** ở mức tile.
5. **Trích patch** — cắt patch 256×256 px; validation/test lọc patch không có GEDI.

### 2. Kiến trúc UNet++

Kiến trúc UNet++ lồng nhau với bộ lọc `[32, 64, 128, 256, 512]`.

- **Input:** 14 kênh (Sentinel-1 + Sentinel-2)
- **Output:** 1 kênh (chiều cao tán rừng, mét)
- **Loss:** L1 (MAE) có mask — chỉ tính trên pixel GEDI hợp lệ
- **Optimizer:** Adam + `ReduceLROnPlateau`
- **Augmentation:** lật ngang/dọc, xoay 90°/180°/270°

### 3. Các thí nghiệm trong notebook

| # | Patch | Tile | Augmentation | Ghi chú |
|---|-------|------|--------------|---------|
| 1 | 1024 | 256 | Không | Baseline |
| 2 | 1024 | 256 | Flip H/V | |
| 3 | 1024 | 256 | Flip + xoay | |
| 4 | 512 | 256 | Flip + xoay | Giảm patch size |
| 5 | 512 | 256 | Flip + xoay | Cấu hình tốt nhất |

**Kết quả thí nghiệm 5 (validation):** MAE ≈ **2.67 m**, RMSE ≈ **4.42 m**, R² ≈ **0.80** (16 802 mẫu test pixel GEDI).

### 4. Đánh giá & trực quan hóa

- Biểu đồ scatter Ground Truth vs Prediction
- Histogram phân phối lỗi dự báo
- So sánh trực quan: Sentinel-2 RGB | GEDI Ground Truth | AI Prediction

## Script huấn luyện (`train.py`)

Script Python độc lập, tối ưu hóa so với notebook:

| Tham số | Giá trị |
|---------|---------|
| Patch size | 256 px |
| Batch size | 24 |
| Epochs | 200 (early stopping patience = 15) |
| Input channels | 12 |
| Loss | Masked MAE |
| Optimizer | AdamW + OneCycleLR |
| LR Finder | Có (tự động tìm learning rate) |

```bash
# Cài đặt dependencies
pip install -r requirement.txt
pip install torch torch-lr-finder tqdm

# Đặt file GeoTIFF tại ./DakLak_Full_Merged.tif rồi chạy
python train.py
```

Model tốt nhất được lưu tại `best_unetpp_canopy_height.pth`.

## Pipeline Machine Learning

Notebook `ML/ml.ipynb` triển khai:

1. **Khám phá dữ liệu** — histogram, mật độ, ma trận tương quan RH95
2. **Feature engineering** — chỉ số thực vật và đặc trưng radar tổng hợp
3. **Huấn luyện & tối ưu hyperparameter** (`RandomizedSearchCV`, 5-fold CV)
4. **Kỹ thuật nâng cao:**
   - Weighted training (trọng số ×3 cho cây > 20 m)
   - Quantile regression (alpha = 0.75)
   - Ensemble (Weighted + Quantile)
   - Oversampling nhóm cây cao
   - Lọc nhiễu bằng Autoencoder / Isolation Forest
5. **Mô hình:** LightGBM (GPU), XGBoost, Random Forest, MLP (PyTorch)

```bash
# Đặt DakLak_train.csv và DakLak_test.csv cùng thư mục notebook
jupyter notebook ML/ml.ipynb
```

## Yêu cầu hệ thống

```
Python >= 3.8
CUDA-capable GPU (khuyến nghị cho DL và LightGBM GPU)
```

### Dependencies chính

| Package | Mục đích |
|---------|----------|
| `torch`, `torchvision` | Deep learning |
| `rasterio` | Đọc/ghi GeoTIFF |
| `scikit-learn` | ML metrics, split, CV |
| `lightgbm`, `xgboost` | Gradient boosting |
| `pandas`, `numpy` | Xử lý dữ liệu bảng |
| `matplotlib`, `seaborn` | Trực quan hóa |
| `optuna`, `shap` | Tối ưu & giải thích mô hình |

Cài đặt:

```bash
pip install -r requirement.txt
```

## Metrics đánh giá

Tất cả mô hình được đánh giá bằng:

- **MAE** (Mean Absolute Error) — sai số tuyệt đích trung bình (m)
- **RMSE** (Root Mean Squared Error) — căn bậc hai MSE (m)
- **R²** (Coefficient of Determination) — hệ số xác định

Với DL, metrics chỉ tính trên pixel có footprint GEDI hợp lệ (masked evaluation).

## Tài liệu tham khảo

- Deng et al., 2025 — *Forests* 16(11):1663 (Canopy Height Mapping)
- [UNet++ (Nested U-Net)](https://github.com/4uiiurz1/pytorch-nested-unet)
- GEDI L2A RH95 product — NASA Earthdata
- Sentinel-1 & Sentinel-2 — Copernicus Open Access Hub

## Nhóm thực hiện

**AI UET** — University of Engineering and Technology
