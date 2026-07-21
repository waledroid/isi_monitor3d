# References — running bibliography (grows with each drafted section)

All entries below were verified this session (landing page, DOI, or arXiv abstract
opened/searched and matched on authors + title + venue). Cite in text as [n].

FORMATTING FLAG (FULL.REVIEW-1 m7, carried): at journal-template time, software
citations [22] ONNX Runtime and [23] TensorRT must gain version numbers (1.23.2 /
10.16, as pinned in §2.7) and access dates.

[1] ISO 3691-4:2023, *Industrial trucks — Safety requirements and verification —
Part 4: Driverless industrial trucks and their systems*, International Organization
for Standardization, 2023. URL: https://www.iso.org/standard/83545.html
— cited in §1 (AGV–human coexistence / safety context).

[2] J. Rutinowski, H. Youssef, A. Gouda, C. Reining, M. Roidl, "The Potential of
Deep Learning based Computer Vision in Warehousing Logistics," *Logistics Journal:
Proceedings*, no. 18, 2022. DOI: 10.2195/lj_proc_rutinowski_en_202211_01.
URL: https://proc.logistics-journal.de/article/view/1050
— cited in §1 (vision in warehousing context: re-identification, multi-view pose
estimation for localization, category-agnostic bin-item segmentation).

[3] R. Iguernaissi, D. Merad, K. Aziz, P. Drap, "People tracking in multi-camera
systems: a review," *Multimedia Tools and Applications*, vol. 78, pp. 10773–10793,
2019. DOI: 10.1007/s11042-018-6638-5.
URL: https://link.springer.com/article/10.1007/s11042-018-6638-5
— cited in §1 (multi-camera tracking survey).

[4] F. Fleuret, J. Berclaz, R. Lengagne, P. Fua, "Multicamera People Tracking with
a Probabilistic Occupancy Map," *IEEE Transactions on Pattern Analysis and Machine
Intelligence*, vol. 30, no. 2, pp. 267–282, 2008. DOI: 10.1109/TPAMI.2007.1174.
URL: https://dl.acm.org/doi/10.1109/TPAMI.2007.1174
— cited in §1 (ground-plane occupancy multi-camera tracking).

[5] Y. Hou, L. Zheng, S. Gould, "Multiview Detection with Feature Perspective
Transformation," *ECCV 2020*. arXiv:2007.07247.
URL: https://arxiv.org/abs/2007.07247
— cited in §1 (bird's-eye-view multi-camera detection).

[6] A. Bewley, Z. Ge, L. Ott, F. Ramos, B. Upcroft, "Simple Online and Realtime
Tracking," *IEEE ICIP 2016*, pp. 3464–3468. arXiv:1602.00763.
URL: https://arxiv.org/abs/1602.00763
— cited in §1 (Kalman/Hungarian tracking-by-detection lineage).

[7] Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, X. Wang,
"ByteTrack: Multi-Object Tracking by Associating Every Detection Box," *ECCV 2022*.
DOI: 10.1007/978-3-031-20047-2_1.
URL: https://dl.acm.org/doi/10.1007/978-3-031-20047-2_1
— cited in §1 (two-pass confidence-split association; adapted in §2.5).

[8] R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed.,
Cambridge University Press, 2004. ISBN 978-0-521-54051-3.
URL: https://www.cambridge.org/9780521540513
— cited in §1 (homography, DLT triangulation foundations).

[9] Z. Zhang, "A Flexible New Technique for Camera Calibration," *IEEE Transactions
on Pattern Analysis and Machine Intelligence*, vol. 22, no. 11, pp. 1330–1334, 2000.
DOI: 10.1109/34.888718.
URL: https://dl.acm.org/doi/10.1109/34.888718
— cited in §1 (planar-target camera calibration).

[10] E. Olson, "AprilTag: A robust and flexible visual fiducial system," *IEEE ICRA
2011*, pp. 3400–3407.
URL: https://ieeexplore.ieee.org/document/5979561/
— cited in §1 (fiducial markers; AprilGrid extrinsic target in §2.3).

[11] S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, M. J. Marín-Jiménez,
"Automatic generation and detection of highly reliable fiducial markers under
occlusion," *Pattern Recognition*, vol. 47, no. 6, pp. 2280–2292, 2014.
DOI: 10.1016/j.patcog.2014.01.005.
URL: https://www.sciencedirect.com/science/article/abs/pii/S0031320314000235
— cited in §1 (ArUco markers; ChArUco boards in §2.3).

[12] J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You Only Look Once: Unified,
Real-Time Object Detection," *IEEE CVPR 2016*, pp. 779–788. arXiv:1506.02640.
URL: https://arxiv.org/abs/1506.02640
— cited in §1 (single-stage real-time detection origin).

[13] J. Terven, D.-M. Córdova-Esparza, J.-A. Romero-González, "A Comprehensive
Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and
YOLO-NAS," *Machine Learning and Knowledge Extraction*, vol. 5, no. 4,
pp. 1680–1716, 2023. arXiv:2304.00501.
URL: https://arxiv.org/abs/2304.00501
— cited in §1 (YOLO family evolution).

[14] G. Jocher, J. Qiu, M. Liu, S. Lyu, F. C. Akyon, M. E. Kalfaoglu, "Ultralytics
YOLO26: Unified Real-Time End-to-End Vision Models," arXiv:2606.03748, 2026.
URL: https://arxiv.org/abs/2606.03748 (docs: https://docs.ultralytics.com/models/yolo26)
— cited in §1 (recent NMS-free edge-oriented YOLO generation; used in §2.4).

[15] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, S. Zagoruyko,
"End-to-End Object Detection with Transformers," *ECCV 2020*. arXiv:2005.12872.
URL: https://arxiv.org/abs/2005.12872
— cited in §1 (detection-transformer line origin).

[16] Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, J. Chen, "DETRs Beat
YOLOs on Real-time Object Detection," *IEEE/CVF CVPR 2024*. arXiv:2304.08069.
URL: https://arxiv.org/abs/2304.08069
— cited in §1 (real-time DETR).

[17] I. Robinson, P. Robicheaux, M. Popov, D. Ramanan, N. Peri, "RF-DETR: Neural
Architecture Search for Real-Time Detection Transformers," arXiv:2511.09554, 2025
(accepted ICLR 2026).
URL: https://arxiv.org/abs/2511.09554
— cited in §1 (fine-tuning-oriented real-time DETR; used in §2.4).

[18] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller,
J. Penna, R. Rombach, "SDXL: Improving Latent Diffusion Models for High-Resolution
Image Synthesis," arXiv:2307.01952, 2023.
URL: https://arxiv.org/abs/2307.01952
— cited in §1 (generative backbone of isiGen, §2.6).

[19] L. Zhang, A. Rao, M. Agrawala, "Adding Conditional Control to Text-to-Image
Diffusion Models," *IEEE/CVF ICCV 2023*. arXiv:2302.05543.
URL: https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html
— cited in §1 (geometry-conditioned generation; labels-by-construction premise of §2.6).

[20] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen,
"LoRA: Low-Rank Adaptation of Large Language Models," *ICLR 2022*. arXiv:2106.09685.
URL: https://arxiv.org/abs/2106.09685
— cited in §1 (low-rank adapter fine-tuning; introduced for LLMs, cited here as the
origin of the technique — the text hedges the diffusion transfer explicitly;
per-class adapters in §2.6).

[21] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, P. Abbeel, "Domain
randomization for transferring deep neural networks from simulation to the real
world," *IEEE/RSJ IROS 2017*. DOI: 10.1109/IROS.2017.8202133.
URL: https://dblp.org/rec/conf/iros/TobinFRSZA17.html
— cited in §1 (sim-to-real via appearance randomization).

[22] ONNX Runtime (software), Microsoft. URL: https://onnxruntime.ai/
— cited in §1 and §2.1/§2.4/§2.7 (portable inference runtime).

[23] NVIDIA TensorRT (software), NVIDIA Corporation.
URL: https://developer.nvidia.com/tensorrt
— cited in §1 and §2.1/§2.4 (GPU inference optimizer/runtime; TRT execution provider).

Entries [24]–[28] added 2026-07-21 for the new §2.1 Background subsection; each was
verified this session (arXiv abstract / ASME record / official project page opened and
matched on authors + title + venue).

[24] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer, "High-Resolution
Image Synthesis with Latent Diffusion Models," *IEEE/CVF CVPR 2022*.
arXiv:2112.10752.
URL: https://arxiv.org/abs/2112.10752
— cited in §2.1 (latent diffusion: denoising in autoencoder latent space; the
foundation SDXL [18] scales).

[25] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, et al., "SAM 2: Segment Anything in
Images and Videos," arXiv:2408.00714, 2024.
URL: https://arxiv.org/abs/2408.00714
— cited in §2.1 (promptable segmentation foundation model; isiGen's ground-truth
masker, §2.6 — repo-verified: trainer/isiGen/src/stages/masking/sam2_masker.py,
model facebook/sam2.1-hiera-small, box-prompted).

[26] R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems,"
*Transactions of the ASME — Journal of Basic Engineering*, vol. 82, no. 1,
pp. 35–45, 1960. DOI: 10.1115/1.3662552.
URL: https://asmedigitalcollection.asme.org/fluidsengineering/article/82/1/35/397706
— cited in §2.1 (recursive minimum-variance state estimation; per-track filters in
§2.5).

[27] GStreamer: open source multimedia framework (software), version 1.28.3, GStreamer team.
URL: https://gstreamer.freedesktop.org/ (accessed 2026-07-21)
— cited in §2.1 (pipeline-based media framework; capture pipeline of §2.4).

[28] NVIDIA Video Codec SDK — NVDEC (software), NVIDIA Corporation.
URL: https://developer.nvidia.com/video-codec-sdk (accessed 2026-07-21)
— cited in §2.1 (GPU hardware video-decode engine; hardware decode chain of §2.4).
