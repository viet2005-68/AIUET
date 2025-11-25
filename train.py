import rasterio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torchvision.transforms.functional as TF
import random
import os
import matplotlib.pyplot as plt 
from torch_lr_finder import LRFinder 

# ================================================================================
# CANOPY HEIGHT MAPPING - UNET++ TRAINING (FINAL OPTIMIZED)
# Paper Ref: Deng et al., 2025 - Forests 16(11):1663
# ================================================================================

# ================= 1. CONFIGURATION =================
class Config:
    # --- File Paths ---
    TIFF_PATH = "./DakLak_Full_Merged.tif"
    MODEL_SAVE_PATH = "best_unetpp_canopy_height.pth"
    
    # --- Training Hyperparameters ---
    PATCH_SIZE = 256
    BATCH_SIZE = 24  # Đã tăng để phù hợp với 2x RTX 3090 (có thể tăng lên 32 nếu cần)
    LR = 1e-4  # LR ban đầu (sẽ được cập nhật nếu USE_LR_FINDER=True)
    EPOCHS = 200
    EARLY_STOP_PATIENCE = 15
    
    # --- Data Params ---
    MAX_CANOPY_HEIGHT = 100.0  
    
    # --- Optimizer & Scheduler ---
    USE_LR_FINDER = True 
    SCHEDULER_TYPE = 'onecycle' 
    
    # --- Band Configuration ---
    # Band 1: GEDI Label (RH98) - Target
    # Band 2-15: Sentinel-1 & Sentinel-2 Inputs (14 kênh)
    LABEL_BAND_IDX = 1
    INPUT_BAND_INDICES = list(range(2, 14))
    INPUT_CHANNELS = 12
    
    # --- System ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4 
    PIN_MEMORY = True
    
    # --- Split Ratio & Seed ---
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    SEED = 42

# ================= 2. DATASET =================
class CanopyHeightDataset(Dataset):
    def __init__(self, tif_path, config, augment=True):
        self.tif_path = tif_path
        self.cfg = config
        self.augment = augment
        self.windows = []
        
        print(f"\n{'='*60}")
        print(f"DATASET INITIALIZATION")
        print(f"{'='*60}")
        
        if not os.path.exists(tif_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file: {tif_path}")

        with rasterio.open(tif_path) as src:
            self.H, self.W = src.height, src.width
            print(f"Image Size: {self.H} x {self.W}")
            print("Loading label band to generate patches...")
            
            try:
                label_data = src.read(config.LABEL_BAND_IDX)
            except Exception as e:
                print(f"❌ Error reading label: {e}")
                return
        
        patch_size = config.PATCH_SIZE
        stride = patch_size
        
        n_rows = (self.H - patch_size) // stride + 1
        n_cols = (self.W - patch_size) // stride + 1
        
        gedi_pixels_total = 0
        
        for i in range(n_rows):
            for j in range(n_cols):
                row_off = i * stride
                col_off = j * stride
                
                window = rasterio.windows.Window(col_off, row_off, patch_size, patch_size)
                patch_label = label_data[row_off:row_off+patch_size, col_off:col_off+patch_size]
                
                # LỌC: Chỉ lấy patch có ít nhất 1 điểm GEDI hợp lệ (>0.5m)
                valid_gedi = ((patch_label > 0.5) & (patch_label < 100)).sum()
                
                if valid_gedi > 0:
                    self.windows.append(window)
                    gedi_pixels_total += valid_gedi
                    
        print(f"✅ Valid Patches found: {len(self.windows)}")
        print(f"✅ Total GEDI Pixels: {gedi_pixels_total}")
        print(f"{'='*60}\n")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        
        with rasterio.open(self.tif_path) as src:
            img = src.read(self.cfg.INPUT_BAND_INDICES, window=window).astype('float32')
            label = src.read(self.cfg.LABEL_BAND_IDX, window=window).astype('float32')
        
        img = np.nan_to_num(img, nan=0.0)
        img = np.clip(img, 0, 1.0)
        
        label = np.nan_to_num(label, nan=0.0)
        mask = ((label > 0.5) & (label < 100.0)).astype('float32')
        
        label = np.clip(label, 0, self.cfg.MAX_CANOPY_HEIGHT) / self.cfg.MAX_CANOPY_HEIGHT
        label = label * mask 
        
        img_t = torch.from_numpy(img)
        label_t = torch.from_numpy(label).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        
        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                img_t = TF.hflip(img_t)
                label_t = TF.hflip(label_t)
                mask_t = TF.hflip(mask_t)
            if random.random() > 0.5:
                img_t = TF.vflip(img_t)
                label_t = TF.vflip(label_t)
                mask_t = TF.vflip(mask_t)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img_t = TF.rotate(img_t, angle)
                label_t = TF.rotate(label_t, angle)
                mask_t = TF.rotate(mask_t, angle)
                
        return img_t, label_t, mask_t

# ================= 3. MODEL (UNet++) =================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=12, out_channels=1):
        super(UNetPlusPlus, self).__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])
        
        self.conv0_1 = ConvBlock(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3]+nb_filter[4], nb_filter[3])
        
        self.conv0_2 = ConvBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2])
        
        self.conv0_3 = ConvBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1])
        
        self.conv0_4 = ConvBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0])
        
        self.final = nn.Sequential(
            nn.Conv2d(nb_filter[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(x0_4)
        return output

# ================= 4. LOSS & METRICS =================
def masked_mae_loss(preds, targets, mask):
    diff = torch.abs(preds - targets) * mask
    loss = diff.sum() / (mask.sum() + 1e-6)
    return loss

# Hàm Loss Wrapper cho LRFinder (KHÔNG DÙNG MASK)
def lr_finder_loss_wrapper(preds, targets):
    return torch.abs(preds - targets).mean()

def calculate_metrics(preds, targets, mask, max_height=100.0):
    preds_m = preds * max_height
    targets_m = targets * max_height
    
    p = preds_m.detach().cpu().numpy().flatten()
    t = targets_m.detach().cpu().numpy().flatten()
    m = mask.detach().cpu().numpy().flatten()
    
    valid_indices = m > 0
    if valid_indices.sum() < 2:
        return {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0}
        
    p_valid = p[valid_indices]
    t_valid = t[valid_indices]
    
    valid_mask = np.isfinite(p_valid) & np.isfinite(t_valid)
    p_valid = p_valid[valid_mask]
    t_valid = t_valid[valid_mask]
    
    if len(p_valid) < 2:
        return {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0}
        
    r2 = r2_score(t_valid, p_valid)
    rmse = np.sqrt(mean_squared_error(t_valid, p_valid))
    mae = mean_absolute_error(t_valid, p_valid)
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae}

# ================= 5. TRAINING ROUTINE =================
def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    epoch_loss = 0
    metrics_accum = {'r2': 0, 'rmse': 0, 'mae': 0}
    
    pbar = tqdm(dataloader, desc="Train")
    for img, label, mask in pbar:
        img, label, mask = img.to(device), label.to(device), mask.to(device)
        
        optimizer.zero_grad()
        preds = model(img)
        
        loss = masked_mae_loss(preds, label, mask)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
            
        epoch_loss += loss.item()
        
        with torch.no_grad():
            batch_metrics = calculate_metrics(preds, label, mask, Config.MAX_CANOPY_HEIGHT)
            for k in metrics_accum:
                metrics_accum[k] += batch_metrics[k]
                
        pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'MAE': f"{batch_metrics['mae']:.2f}m"})
        
    avg_loss = epoch_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_accum.items()}
    return avg_loss, avg_metrics

def validate(model, dataloader, device):
    model.eval()
    val_loss = 0
    metrics_accum = {'r2': 0, 'rmse': 0, 'mae': 0}
    
    with torch.no_grad():
        for img, label, mask in tqdm(dataloader, desc="Val"):
            img, label, mask = img.to(device), label.to(device), mask.to(device)
            
            preds = model(img)
            loss = masked_mae_loss(preds, label, mask)
            
            val_loss += loss.item()
            
            batch_metrics = calculate_metrics(preds, label, mask, Config.MAX_CANOPY_HEIGHT)
            for k in metrics_accum:
                metrics_accum[k] += batch_metrics[k]
                
    avg_loss = val_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_accum.items()}
    return avg_loss, avg_metrics

# ================= 6. LR FINDER LOGIC =================
def find_lr(model_class, train_loader, optimizer_class, device, cfg):
    """Thực hiện tìm kiếm Learning Rate tối ưu thủ công."""
    print("\n🔎 STARTING LR FINDER (Manual Analysis Mode)...")
    
    # Khởi tạo lại model và optimizer cho LR Finder
    lr_finder_model = model_class(in_channels=cfg.INPUT_CHANNELS, out_channels=1).to(device)
    lr_finder_optimizer = optimizer_class(lr_finder_model.parameters(), lr=1e-7, weight_decay=1e-4)
    
    lr_finder = LRFinder(lr_finder_model, lr_finder_optimizer, lr_finder_loss_wrapper, device=device)
    
    lr_finder.range_test(train_loader, end_lr=1, num_iter=100, step_mode="exp")
    
    # Lấy dữ liệu Loss và LR đã thu thập
    lrs = lr_finder.history['lr']
    losses = lr_finder.history['loss']
    
    # ----------------------------------------------------
    # TÌM KIẾM LR TỐI ƯU THỦ CÔNG (Tim điểm dốc nhất)
    # ----------------------------------------------------
    
    gradients = np.gradient(np.array(losses))
    skip_steps = int(len(losses) * 0.15) 
    min_gradient_index = np.argmin(gradients[skip_steps:]) + skip_steps
    
    suggested_lr = lrs[min_gradient_index]
    best_lr = suggested_lr / 10 
    
    print(f"\n✅ LR FINDER COMPLETED.")
    print(f"👉 Suggested Optimal LR (Điểm dốc nhất): {suggested_lr:.2e}")
    print(f"👉 Using Base LR (Suggested/10) for training: {best_lr:.2e}")
    
    lr_finder.reset()
    
    return best_lr

# ================= 7. MAIN =================
def main():
    cfg = Config()
    
    # 1. Reproducibility
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    
    # 2. Dataset Setup
    if not os.path.exists(cfg.TIFF_PATH):
        print(f"❌ Error: Not found {cfg.TIFF_PATH}")
        return

    full_dataset = CanopyHeightDataset(cfg.TIFF_PATH, cfg)
    
    train_size = int(cfg.TRAIN_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.SEED)
    )
    
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    
    print(f"Train size: {len(train_ds)} patches")
    print(f"Val size:   {len(val_ds)} patches")
    
    # 3. Model & Optimizer Setup
    
    # --- LR Finder Logic ---
    if cfg.USE_LR_FINDER:
        # Chạy LR Finder để cập nhật LR tối ưu
        found_lr = find_lr(UNetPlusPlus, train_loader, optim.AdamW, cfg.DEVICE, cfg)
        cfg.LR = found_lr 
        
    # Khởi tạo model và optimizer CHÍNH THỨC
    model = UNetPlusPlus(in_channels=cfg.INPUT_CHANNELS, out_channels=1).to(cfg.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR/10, weight_decay=1e-4) 
    
    # OneCycleLR Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.LR, 
        steps_per_epoch=len(train_loader), epochs=cfg.EPOCHS
    )
    
    # 4. Loop
    best_mae = float('inf')
    patience = 0
    
    print("\n🚀 START TRAINING...")
    for epoch in range(1, cfg.EPOCHS+1):
        train_loss, train_m = train_one_epoch(model, train_loader, optimizer, scheduler, cfg.DEVICE)
        val_loss, val_m = validate(model, val_loader, cfg.DEVICE)
        
        print(f"Epoch {epoch}/{cfg.EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | MAE: {train_m['mae']:.2f}m | RMSE: {train_m['rmse']:.2f}m | R2: {train_m['r2']:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | MAE: {val_m['mae']:.2f}m | RMSE: {val_m['rmse']:.2f}m | R2: {val_m['r2']:.4f}")
        
        # Save Best Model theo MAE
        if val_m['mae'] < best_mae:
            best_mae = val_m['mae']
            patience = 0
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
            print(f"  ✅ Model Saved (New Best MAE: {best_mae:.4f}m)")
        else:
            patience += 1
            print(f"  ⚠️ No improve ({patience}/{cfg.EARLY_STOP_PATIENCE})")
            
        if patience >= cfg.EARLY_STOP_PATIENCE:
            print("🛑 Early Stopping!")
            break

if __name__ == "__main__":
    main()