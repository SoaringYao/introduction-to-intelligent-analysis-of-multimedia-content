# 大作业资料

## 选题

### 帧间预测

- **CVPR 2021: Deep Learning in Latent Space for Video Prediction and Compression**
   Bowen Liu and colleagues propose a deep learning framework for video prediction and compression in the latent space. This approach uses a convolutional long short-term memory (ConvLSTM) network for inter-frame prediction, leveraging latent space representations to efficiently compress and predict video sequences [oai_citation:1,CVPR 2021 Open Access Repository](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Deep_Learning_in_Latent_Space_for_Video_Prediction_and_Compression_CVPR_2021_paper.html).

- **CVPR 2023: Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation**
   The paper propose a novel module to explicitly extract motion and appearance information via a unifying operation. Specifically, they rethink the information process in inter-frame attention and reuse its attention map for both appearance feature enhancement and motion information extraction. Furthermore, for efficient VFI, the proposed module could be seamlessly integrated into a hybrid CNN and Transformer architecture. This hybrid pipeline can alleviate the computational complexity of inter-frame attention as well as preserve detailed low-level structure information.[oai_citation:2,github](https://github.com/MCG-NJU/EMA-VFI)

- **CVPR 2022: Optimizing Video Prediction via Video Frame Interpolation**
   This paper explores the integration of video frame interpolation techniques to enhance video prediction tasks. By leveraging advancements in video frame interpolation, the researchers aim to address challenges in predicting video frames accurately in diverse and dynamic environments. **The idea of optimization is used, and there is no need to train the network** [oai_citation:3,CVPR 2022 Open Access Repository](https://openaccess.thecvf.com/CVPR2022).

#### 视频异常检测

- **ICCV 2021: A Hybrid Video Anomaly Detection Framework via Memory-Augmented Flow Reconstruction and Flow-Guided Frame Prediction**
   This paper by Zhian Liu et al. presents a framework that combines optical flow reconstruction with frame prediction to detect video anomalies. The method uses a Conditional Variational Autoencoder (CVAE) to predict future frames based on reconstructed optical flows, improving anomaly detection by highlighting discrepancies between predicted and actual frames [oai_citation:1,ICCV 2021 Open Access Repository](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_A_Hybrid_Video_Anomaly_Detection_Framework_via_Memory-Augmented_Flow_Reconstruction_ICCV_2021_paper.html).


### 基于扩散模型的视频编辑

- **CCEdit: Creative and Controllable Video Editing via Diffusion Models**
   This paper presents CCEdit, a generative video editing framework based on diffusion models. It uses a novel trident network structure to separate structure and appearance control, ensuring precise and creative editing capabilities. The authors provide a benchmark dataset and extensive comparisons with state-of-the-art methods, showing substantial improvements. The official code is expected to be available, facilitating reproducibility of the results [oai_citation:1,CCEdit: Creative and Controllable Video Editing via Diffusion Models | Papers With Code](https://paperswithcode.com/paper/ccedit-creative-and-controllable-video).

- **EffiVED: Efficient Video Editing via Text-instruction Diffusion Models**
   EffiVED leverages latent diffusion models (LDM) and introduces efficient training data construction strategies to enhance video editing via text instructions. The model incorporates a 3D U-Net for processing video content and ensures temporal consistency while maintaining high computational efficiency. The code for EffiVED is available on [arXiv](https://arxiv.org/abs/2403.11568) [oai_citation:2,[2403.11568] EffiVED:Efficient Video Editing via Text-instruction Diffusion Models](https://ar5iv.org/abs/2403.11568).

- **Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models (vid2vid-zero)**
   This approach employs existing image diffusion models for video editing without any additional training. It introduces several modules, including a null-text inversion module for text-to-video alignment and a cross-frame modeling module for temporal consistency. The official implementation is available on GitHub: [vid2vid-zero](https://github.com/baaivision/vid2vid-zero) [oai_citation:3,Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models | Papers With Code](https://paperswithcode.com/paper/zero-shot-video-editing-using-off-the-shelf).

- **RAVE: Randomized Noise Shuffling for Fast and Consistent Video Editing with Diffusion Models**
   RAVE introduces a method for zero-shot text-guided video editing using randomized noise shuffling and ControlNet for structural guidance. This approach ensures both efficiency and consistency in video editing tasks. The methodology includes a detailed use of latent diffusion models (LDMs) and denoising diffusion implicit models (DDIM). The paper and code can be found on [arXiv](https://arxiv.org/abs/2312.04524) [oai_citation:4,[2312.04524] RAVE: Randomized Noise Shuffling for Fast and Consistent Video Editing with Diffusion Models](https://ar5iv.org/abs/2312.04524).

### 去雨去云去雾去雷

- **AAAI 2021: EfficientDeRain: Learning Pixel-wise Dilation Filtering for High-Efficiency Single-Image Deraining**
   Qing Guo et al. introduce a model-free deraining method that employs pixel-wise dilation filtering. This technique significantly speeds up deraining processes, achieving results over 80 times faster than previous methods while maintaining similar effectiveness. The method is designed to handle both synthetic and real-world rainy images efficiently [oai_citation:1,EfficientDeRain: Learning Pixel-wise Dilation Filtering for High-Efficiency Single-Image Deraining - AAAI](https://aaai.org/papers/01487-efficientderain-learning-pixel-wise-dilation-filtering-for-high-efficiency-single-image-deraining/).

- **CVPR 2021: Frame-Consistent Recurrent Video Deraining With Dual-Level Flow**
   This paper by Wenhan Yang and colleagues addresses video deraining through a two-stage recurrent network that uses dual-level flow regularizations. The framework enhances frame consistency and motion alignment, providing robust deraining performance on both synthetic and real-world videos [oai_citation:2,Frame-Consistent Recurrent Video Deraining With Dual-Level Flow | Papers With Code](https://paperswithcode.com/paper/frame-consistent-recurrent-video-deraining).
