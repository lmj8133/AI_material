# 學習計畫

## 1. 基礎能力

1. **深度學習理論**  
   - [李宏毅老師開放式課程](https://www.youtube.com/watch?v=CXgbekl66jc&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49)
   - 了解[評估指標](https://algoltek-my.sharepoint.com/:p:/g/personal/mj_li_algoltek_com_tw/EXcSKOxl_KtEqdlGGa-dWX8BCxJrMz5YLVbz8_18gPkSFQ?e=cFlP5i)（Accuracy、Precision、Recall、mAP 等）

2. **PyTorch 基礎實作**  
   - [Python Virtual Environment (venv)](https://dev.to/codemee/python-xu-ni-huan-jing-venv-nbg)
   - [Google Colab](https://colab.google/)
   - [WSL2 (Windows Subsystem for Linux)](https://bayareanotes.com/wsl-installation/#google_vignette)
   - [Deep Learning with PyTorch](https://algoltek-my.sharepoint.com/:b:/g/personal/allen_chen_algoltek_com_tw/EX4hBF0mgSZOh80cwyKWdz8Bo9dqJrzx8DA6eDqYA_rUIw?e=Ggg6d4)（特別注意 Transfer Learning 跟 Hyperparameter Tuning）

3. **Docker**  
   - 熟悉 Docker
---

## 2. Jetson 平台環境

1. **JetPack SDK**  
   - 瞭解如何在 Jetson Nano 上安裝與設定 JetPack，包含系統與開發套件。

2. **CUDA 及相關庫（CuBLAS、CuDNN）**  
   - 瞭解 GPU 加速原理，熟悉在 Jetson 環境中配置並驗證 CUDA。

3. **TensorRT**  
   - 了解如何使用 TensorRT 進行推論加速、如何將已訓練的模型轉成 TensorRT Engine。

4. **DeepStream**  
   - 熟悉 NVIDIA 提供的流式影像處理框架，瞭解其 Pipeline 與 Plugin。

5. **[Jetson Inference](https://github.com/dusty-nv/jetson-inference)**  
   - 了解 NVIDIA 官方提供的範例與示例程式，快速上手物件偵測、影像分類等應用。

6. **OpenCV**  
   - 了解如何在 Jetson 平台上安裝與使用 OpenCV，進行影像前處理、顯示、攝影機串流等功能。

---

## 3. Edge AI 與模型效能優化

1. **模型壓縮（Model Compression）**
2. **量化（Quantization）**
3. **剪枝（Pruning）**
4. **高效網路（Efficient Network）設計**  
   - 研究 MobileNet、ShuffleNet 等輕量化架構。
5. **YOLOv8**  
   - 常用於物件偵測的高效模型，可嘗試在 Jetson Nano 上進行測試。
6. **MobileNet**  
   - 在影像分類與物件偵測上均有輕量化版本，適合 Edge 部署。
7. **模型格式轉換：PyTorch → ONNX → TensorRT**  
   - 熟悉如何導出模型為 ONNX，再使用 TensorRT 轉檔優化推論效能。

---

## 4. 資料標註

1. **標註工具**  
   - LabelImg、CVAT、Roboflow、LabelMe、VoTT 等。