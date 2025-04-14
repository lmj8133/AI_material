# CUDA 環境設定指南 (Windows)

本指南提供在 Windows 上設定 CUDA、cuDNN 及 PyTorch 環境的步驟。

---

## 1. 確認顯卡版本與驅動程式

- 桌面空白處按右鍵 → 選擇「NVIDIA 控制面板」
- 點擊左下角「系統資訊」：
  - 在「顯示」分頁確認「驅動程式版本」
  - 在「元素」分頁確認「CUDA 版本」（即最高支援的 CUDA 版本）
- 若驅動程式版本過舊或 CUDA 版本不符需求，可前往 [NVIDIA 官方網站更新驅動程式](https://www.nvidia.com/en-us/drivers/)

---

## 2. 安裝 CUDA Toolkit

- 從 [CUDA Toolkit 官網](https://developer.nvidia.com/cuda-toolkit-archive) 下載所需的 CUDA 版本。
- 執行下載好的安裝檔案後，選擇「自訂」安裝模式。
- 建議只勾選安裝「CUDA」（其他選項可依需求自行決定）。
- 依照指示完成安裝。

---

## 3. 確認 CUDA 安裝成功

- 開啟 `cmd` 或 `PowerShell`，輸入以下指令並執行：

  ```bash
  nvcc --version
  ```

- 若能正確顯示 CUDA 的版本資訊，則表示安裝成功。

- 若出現找不到指令的情況，請手動新增環境變數：

  - 在「此電腦」點擊右鍵 → 「內容」→ 「進階系統設定」 → 「環境變數」。
  - 在「系統變數」中的 `Path` 中新增以下路徑：

  ```
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{CUDA版本}\bin
  ```

---

## 4. 安裝 cuDNN（CUDA 深度神經網路函式庫）

- 前往 [cuDNN 官方下載頁面](https://developer.nvidia.com/rdp/cudnn-archive)下載與已安裝的 CUDA 版本相符的 cuDNN 壓縮檔。
- 將下載的壓縮檔解壓縮後，複製檔案至 CUDA Toolkit 安裝目錄，通常為：

  ```
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{CUDA版本}\
  ```

- 將解壓後的資料夾內容（`bin`, `include`, `lib`）分別複製到相對應的位置。

---

## 5. 安裝 CUDA 版 PyTorch

- 確認 Python 版本為 **3.8 以上且低於 3.12 (建議安裝3.10)**
- 建議先建立 Python 虛擬環境（venv）：
  ```bash
  python -m venv cuda_env
  .\cuda_env\Scripts\activate.bat
  ```
- 安裝 PyTorch（以 CUDA 12.1 為例）：
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- 測試 PyTorch 是否成功啟用 CUDA：
  ```python
  import torch

  print(torch.cuda.is_available())  # 若輸出 True 表示設定成功
  ```

---

