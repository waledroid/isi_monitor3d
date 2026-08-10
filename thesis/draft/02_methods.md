# 2. Materials and Methods

## 2.1 Background: underlying architectures and tools

isiMonitor3d integrates components from six complementary research areas: vision inference, multi-object tracking, projective geometry, synthetic-data generation, machine-consumable data delivery, and edge deployment. This subsection summarizes the specific architectures the system builds on — the two detector families (2.1.1–2.1.2), the generative stack of the synthetic-data pipeline (2.1.3–2.1.6), and the messaging protocol of the delivery layer (2.1.7) — and closes with the tracking, geometric, and runtime foundations (2.1.8). Each summary is limited to the architectural properties the system directly exploits, followed by the concrete form the component takes in isiMonitor3d.

### 2.1.1 YOLO26 — single-pass instance segmentation

The YOLO family frames detection as a single forward pass of a fully convolutional network that predicts classes and boxes densely over the image, trading the region-proposal stage of two-stage detectors for throughput [12]. Across its generations the family converged on a three-part structure [13]: a convolutional backbone extracts features at several scales; a neck aggregates these features across pyramid levels, so that small and large objects are each predicted from feature maps of appropriate resolution; and a decoupled head regresses boxes and classifies objects in separate branches. The dense predictions of such one-stage detectors are redundant by construction and have historically required non-maximum suppression (NMS) — a data-dependent, sequential post-processing step — to select final detections.

YOLO26, the generation adopted here, removes this dependency: it is trained for end-to-end prediction, so the network emits a final detection set directly without NMS, and it drops the distribution-focal-loss regression module of earlier generations, simplifying the prediction head [14]. Because this project exports the raw prediction head — no decoding baked into the artifact — the exported graph never contained NMS in any YOLO generation; the benefit the end-to-end head delivers here lies on the application side: detection decoding reduces to confidence filtering, coordinate un-letterboxing, and mask assembly, with no data-dependent suppression stage whose cost varies with scene density (the postprocessing cost breakdown of Section 3.3 reflects this). The release documentation additionally reports up to 43 % faster CPU ONNX inference for the nano variant relative to its YOLO11 predecessor [14]. The instance-segmentation (`-seg`) variants extend the head with the prototype-based mask design introduced by YOLACT [29]: the network predicts one shared bank of prototype masks per image plus a small coefficient vector per detection, and each instance mask is the thresholded sigmoid of a linear combination of the prototypes — adding segmentation at near-constant cost over detection.

In isiMonitor3d, the production detectors are YOLO26-seg models fine-tuned in the isidet trainer on the three-class warehouse corpus (Section 2.6): the headline `yolo26l-seg` (640 px input; 0.977 box mAP@0.5 and 0.948 box mAP@0.5:0.95 on the validation split — Section 3.1, Table T1) and the deployed `yolo26n-seg`, which meets the accuracy target at a 320 px input (0.962 / 0.895). Models are exported to ONNX (opset 17) with the raw prediction head, so box decoding and prototype-mask combination remain under application control (Section 2.4).

### 2.1.2 RF-DETR — detection transformer

DETR recasts detection as direct set prediction [15]. A transformer encoder [30] processes the flattened feature map of a convolutional backbone; in the decoder, a fixed set of learned object queries cross-attends to the encoded image features, each query producing one (class, box) prediction. Training assigns predictions to ground-truth objects one-to-one by bipartite Hungarian matching, and the resulting loss teaches the network itself to suppress duplicates — no anchors, no NMS. The original formulation, however, converged slowly and paid quadratic attention cost over dense feature maps. Deformable DETR addressed both by replacing full attention with multi-scale deformable attention, which samples a small learned set of points around each reference location [31], and subsequent real-time DETRs such as RT-DETR added efficient hybrid encoders and improved query selection, reaching YOLO-class speed [16].

RF-DETR continues this real-time line with a different emphasis: rather than proposing a single fixed architecture, it fine-tunes a pretrained base network and applies weight-sharing neural architecture search over the model's tunable dimensions, discovering accuracy–latency trade-offs per target dataset and explicitly aiming at transferability of DETRs to domains far from the pretraining distribution [17]. This orientation toward fine-tuning on small, domain-specific datasets matches the per-site training regime of Section 1.

In isiMonitor3d, RF-DETR-seg is the alternative detector family behind the same `Detector` interface. The evaluated model is `rfdetr-medium-seg`, fine-tuned on the same corpus; it reads a fixed 432 px square input, and its exported ONNX graph exposes three named outputs — `dets` (normalized boxes), `labels` (per-query class logits), and `masks` (per-query mask logits) — decoded without NMS. At its selected checkpoint it reaches 0.973 box mAP@0.5 and 0.938 box mAP@0.5:0.95 on the validation split (Section 3.1, Table T1). Because both families serve the same detector interface, each deployment adopts whichever fine-tunes better on its data.

### 2.1.3 SDXL — latent diffusion

In the denoising-diffusion formulation [32], image generation inverts a gradual noising process: a fixed forward process corrupts a training image with Gaussian noise over many steps, and a network — typically a UNet — is trained to reverse the corruption step by step, so that sampling reconstructs an image from noise. Latent diffusion made this affordable at high resolution by moving denoising out of pixel space [24]: a variational autoencoder compresses images into a lower-dimensional latent space, the denoising UNet operates entirely on latents — conditioned on text through cross-attention over the prompt's encoder embeddings [30] — and the autoencoder's decoder maps the final latent back to pixels.

SDXL scales this recipe [18]: a UNet roughly three times larger than in earlier Stable Diffusion versions, text conditioning from two encoders whose embeddings are concatenated (OpenCLIP ViT-bigG and CLIP ViT-L), and additional conditioning on the original image size and crop coordinates of each training example, which allows training on the full dataset without discarding small images and suppresses cropping artifacts at sampling time.

In isiMonitor3d, SDXL is the generator of the isiGen synthetic-data pipeline (Section 2.6): the publicly released `stabilityai/stable-diffusion-xl-base-1.0` checkpoint, used with frozen weights and extended only through the two mechanisms described next — spatial conditioning by ControlNet (2.1.4) and class specialization by LoRA (2.1.5).

### 2.1.4 ControlNet — spatial conditioning

ControlNet adds spatial control to a pretrained text-to-image diffusion model without retraining it [19]. The architecture duplicates the encoder and middle blocks of the base UNet into a trainable copy that receives, in addition to the latent, a spatial control map — a depth map, edge map, or pose skeleton aligned with the target image. The copy's outputs are injected into the frozen base network through zero-initialized 1 × 1 convolutions ("zero convolutions"): at the start of training the added branch contributes exactly nothing, and its influence grows only as training demands it. Because the base weights never change, the pretrained generative prior — object appearance, lighting, texture statistics — is preserved intact while the control branch learns to enforce the supplied spatial structure.

This property is what makes isiGen's annotations correct by construction. isiGen uses the depth ControlNet released for SDXL (`diffusers/controlnet-depth-sdxl-1.0`). Its control maps are composite depth maps assembled by the scaffold stage from monocular depth estimates of real backgrounds and object instances, and each scaffold pairs such a control map with the segmentation mask of the objects it places. The generated image's geometry therefore follows the scaffold while the text prompt randomizes appearance — and the scaffold's mask remains a valid ground-truth annotation for the generated image (Section 2.6).

### 2.1.5 LoRA — low-rank adaptation

Low-rank adaptation fine-tunes a large pretrained network by learning only a low-rank update to selected weight matrices [20]. For a pretrained weight $\mathbf{W} \in \mathbb{R}^{d \times k}$, kept frozen, LoRA parameterizes the update as $\Delta\mathbf{W} = \frac{\alpha}{r}\mathbf{B}\mathbf{A}$ with $\mathbf{B} \in \mathbb{R}^{d \times r}$, $\mathbf{A} \in \mathbb{R}^{r \times k}$, rank $r \ll \min(d, k)$, and a constant scaling factor $\alpha$; the adapted layer computes $\mathbf{h} = \mathbf{W}\mathbf{x} + \frac{\alpha}{r}\mathbf{B}\mathbf{A}\mathbf{x}$. The trainable parameter count per layer is $r(d + k)$ — orders of magnitude below full fine-tuning for small $r$ — and after training the update can be merged, $\mathbf{W}' = \mathbf{W} + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$, so inference incurs no additional latency. Applied to a diffusion backbone, this yields a compact per-class adapter file over one shared frozen base model — and, in the few-dozen-image regime of this work, it is this restriction of the trainable capacity that makes per-class specialization practical at all.

In isiMonitor3d, isiGen trains one rank-16 SDXL LoRA adapter per object class from a few dozen real photographs; the full training configuration is reported with the pipeline in Section 2.6, and the adapter's contribution to detector training is evaluated in the ablation of Section 3.4.

### 2.1.6 SAM2 — promptable segmentation

SAM2 is a promptable segmentation foundation model [25]: rather than segmenting a fixed class list, it returns the mask of whatever a prompt designates. Its image encoder is Hiera, a hierarchical vision transformer that produces multi-scale features from a deliberately plain, MAE-pretrained architecture [33]; a lightweight prompt encoder embeds points, boxes, or coarse masks; and a small mask decoder combines the two embedding streams into output masks with predicted quality scores. The expensive image embedding is computed once per image and reused across prompts. SAM2 additionally carries a streaming memory-attention module for propagating masks through video; isiGen uses single-image prediction only, so this component is not exercised.

In isiMonitor3d, SAM2 removes manual mask annotation from the synthetic-data workflow. isiGen loads `facebook/sam2.1-hiera-small` (~185 MB) through the `SAM2ImagePredictor` interface and prompts it with the bounding boxes of the project's trained detector — detector-prompted masking — with a promptless automatic-mask-generation mode available when no prompts exist; the resulting masks become the ground truth attached to the real photographs (Section 2.6).

### 2.1.7 MQTT and the metadata delivery layer

MQTT is a lightweight publish/subscribe messaging protocol standardized by OASIS [34]. Its architecture is broker-mediated: clients publish messages to hierarchical, slash-separated topics on a central broker, and the broker forwards each message to every client whose subscription matches — including through single-level (`+`) and multi-level (`#`) topic wildcards — so producers need no knowledge of consumer identity or count. The protocol defines three delivery guarantees (QoS 0, at most once; QoS 1, at least once; QoS 2, exactly once), per-topic retained messages that the broker replays to newly connecting subscribers, and a keepalive mechanism that bounds failure-detection time. Because filtering happens at the broker on topic names rather than in consumers on payloads, MQTT suits industrial telemetry with many heterogeneous consumers: each subscribes to exactly the object classes or zones it needs.

In isiMonitor3d, MQTT delivery is implemented by `MqttSink`, a `MetadataSink` plugin holding one paho-mqtt client and one background network-loop thread (reconnecting with 1–30 s exponential backoff); the deployed client speaks MQTT 3.1.1, and every protocol feature relied on here is defined identically in the 3.1.1 and 5.0 OASIS standards [34]. It publishes the pydantic-validated schema-version-6 JSON envelopes of Section 2.7 with per-class topic fan-out — `{prefix}/track2d/{cls}`, `{prefix}/track3d/{cls}`, and per-zone state and event topics — keeping topic cardinality proportional to the number of classes, not the number of tracked objects; the `track_id` travels in the payload. Zone occupancy state is published retained at QoS 1, so a late-joining consumer (for example, an AGV controller reconnecting mid-shift) immediately receives the current state of every zone rather than waiting for the next transition. Delivery failures are logged and swallowed: an unreachable broker degrades delivery, never the perception or geometry pipeline. `MqttSink` operates beside the UDP sink behind the same plugin seam, and the deployed configuration runs both (Section 2.7).

### 2.1.8 Tracking, geometry, and runtime substrate

The remaining foundations are summarized briefly here and developed where they are used.

**Tracking.** Detected objects must be associated across frames into persistent identities. isiMonitor3d builds on the standard tracking-by-detection recipe: a constant-velocity Kalman filter [26] per track supplies both the motion prediction that bridges detection gaps and the innovation covariance that scales matching distances, with Hungarian assignment of detections to predictions [6]. Association follows ByteTrack's two-pass confidence-split scheme, which recovers tracks from low-confidence detections — often occluded true objects — instead of discarding them [7]. The property exploited is that nothing in this recipe is tied to pixels: Section 2.5 transposes the state space to metric floor coordinates, so matching thresholds become physical distances and one tracker spans both cameras.

**Geometry.** Converting image detections into metric object locations requires camera calibration and projective geometry [8]. isiMonitor3d relies on three classical results: planar-target calibration recovers each camera's intrinsics from views of a printed board [9]; a plane in the scene induces an invertible homography between that plane and the image, so a single camera with a metrically anchored floor homography suffices for metric localization of anything touching the floor; and direct linear transformation (DLT) triangulation recovers full 3D exactly when a second calibrated view exists. This complementarity — 2D from one camera cheaply, 3D on demand from two — is why one calibration can serve both localization modes side by side (Section 2.5). Fiducial targets make the estimation practical on site: AprilTag [10] and ArUco/ChArUco [11] boards provide uniquely identifiable corner constellations that remain detectable under occlusion and blur, turning correspondence — the hard part of every geometric solve — into a detection problem (Section 2.3).

**Deployment.** Finally, the camera streams must be acquired and the models executed efficiently on industrial edge hardware. Stream handling uses GStreamer, whose element-graph pipeline architecture lets capture policy be expressed as pipeline structure — an explicit codec-matched depayloader, a hardware decoder with software fallback, a newest-frame-only sink — rather than application logic [27]; decoding is delegated to NVDEC, the dedicated decode engine on NVIDIA GPUs, keeping continuous multi-stream H.264/H.265 decoding off the CPU budget [28]. Inference runs on ONNX Runtime, which executes the framework-neutral ONNX artifact through pluggable execution providers [22]; in the as-measured configuration of this article (July 2026), TensorRT acceleration was provided through ONNX Runtime's TensorRT execution provider [23], contributing engine-compiled speed while the exchanged model file remains the hardware-neutral `.onnx` — compiled engines are locally derived accelerator caches, not exchanged artifacts. The properties exploited are portability — the same `.onnx` serves the development workstation and the Jetson production target (Section 2.7) — and swappable acceleration, whose measured speedup is reported in Section 3.3; TensorRT's engine-per-shape compilation in turn imposes the shape discipline described in Section 2.4.

Overall, isiMonitor3d follows a sequential processing pipeline. GStreamer and NVDEC acquire and decode the camera streams; YOLO26-seg or RF-DETR-seg perform object detection; ground-plane homography projects each detection to metric floor coordinates, where Kalman filtering and ByteTrack maintain object identities; stereo triangulation adds full 3D on demand; and the resulting tracks are delivered through the communication layer. Offline, SDXL, ControlNet, LoRA, and SAM2 generate and annotate synthetic training data used to adapt detectors to each deployment.

## 2.2 System architecture

The proposed system, isiMonitor3d, converts video streams from one or two fixed RGB cameras into metric, identity-stable object metadata. At runtime, it continuously produces 2D floor-plane tracks (`Track2D`) and, on demand, 3D tracks (`Track3D`) when stereo observations are available. The resulting metadata are published as versioned JSON messages over UDP and MQTT for consumption by industrial systems. The architecture is designed to satisfy the five industrial acceptance criteria defined in the customer specification and introduced in Section 1. These criteria are treated as design requirements, while their experimental validation is presented in Section 3.

The system consists of five modules with distinct life cycles (Figure F1): (i) **isical**, an operator-guided calibration application that generates a single `calibration.json` file; (ii) **isistream**, the perception producer responsible for image acquisition, decoding, object detection, and pose estimation; (iii) the **backbone metric engine**, which performs all geometric reasoning and identity management; (iv) **isicomms**, the communication layer providing MQTT publication and a REST gateway for polling clients such as AGVs; and (v) the offline model-production pair (**isiGen/isidet**), which performs synthetic data generation and detector training in a separate environment and exports only ONNX models to the runtime system. This partition mirrors the modules' life cycles: calibration is executed once per deployment, model production is performed offline whenever retraining is required, whereas the runtime components execute continuously. Consequently, each module can be deployed, updated, or restarted independently.

[Figure F1: Five-module architecture and the two frozen wire contracts: isistream → engine (per-camera detection sets, UDP loopback); engine → consumers (versioned UDP/JSON + MQTT); isicomms MQTT-in/REST-out for AGVs. isical and isiGen/isidet contribute only file artifacts (`calibration.json`, `.onnx`).]

At runtime, the architecture is intentionally divided into two processes connected through a loopback interface ("Direction 1"). The separation reflects their contrasting computational characteristics: perception is GPU-intensive and throughput-oriented, whereas metric reasoning is CPU-light and latency-sensitive. Combining both within a single Python process introduced contention between the interpreter lock and inference thread pools, as quantified in Section 4.1. Accordingly, isistream performs frame capture, zone-aware detection, and pose estimation before transmitting a `DetectionSetMessage` for each camera to the backbone metric engine through a dedicated UDP loopback port (default 9010) — the configuration referred to as *points mode*. The engine therefore operates exclusively on detection metadata and imports neither CUDA runtimes nor inference libraries.

Each `DetectionSetMessage` carries four pieces of runtime information. First, it includes the frame `capture_ts`, which serves as the single latency reference propagated through all downstream messages. When the motion gate (Section 2.4) suppresses inference, cached detections are re-emitted using the current frame's capture timestamp to preserve temporal continuity. Second, the producer transmits an explicit-empty heartbeat even when no objects are detected, allowing the engine to distinguish an empty scene from producer failure; message silence is therefore interpreted as a camera fault. Third, every message contains a per-camera monotonic sequence counter, enabling packet loss to be detected as measurable gaps despite UDP transport. Finally, a configuration fingerprint identifies inconsistencies in calibration, detection models, or monitoring zones between the producer and the metric engine.

To avoid redundant video decoding, decoded frames are distributed through a decode-once shared-memory frame bus. Frames from each camera are written into a dedicated shared-memory segment (`/dev/shm/isi3d_frame_<cam>`) implemented as a double-buffered seqlock with lock-free readers. This design eliminates duplicate RTSP sessions and decoding operations while ensuring that all consumers — including the operator dashboard and future applications — observe the exact same pixel data processed by the perception pipeline.

Inter-process communication relies on two stable interfaces: the backbone UDP/JSON schema (Section 2.7) and the isicomms MQTT-in/REST-out interface. Modules exchange information exclusively through these message contracts rather than shared code, allowing each component to evolve independently by extending the communication schema. Within the backbone itself, extensibility is restricted to five abstract plugin interfaces — `FrameSource`, `Detector`, `Tracker`, `Triangulator`, and `MetadataSink` — whose implementations self-register through a plugin registry. A dedicated unit test enforces the fixed set of extension points. All remaining components, including projection, fusion, gating, and stabilization, are implemented as concrete classes, limiting substitution to functionality where algorithmic or hardware diversity is genuinely required.

## 2.3 Calibration — isical

Calibration transforms a newly installed camera system into a metrically accurate sensing rig and is designed to be performed by site operators using only printed planar calibration targets, without surveying equipment or computer vision expertise. The procedure is executed once per deployment and produces a single `calibration.json` file containing, for each camera, the intrinsic matrix $\mathbf{K}$, distortion coefficients $\mathbf{D}$ (OpenCV model), camera pose $(\mathbf{R}, \mathbf{t})$ in a common world coordinate system, and two runtime quantities derived from these parameters: the ground-plane homography $\mathbf{H}$ and the projection matrix $\mathbf{P}$. Both runtime geometric methods query this single calibration file, satisfying the "one calibration, two queries" invariant introduced in Section 2.5.

The stereo calibration workflow consists of two sequential stages, with each calibration target used where it provides the highest accuracy. Separating intrinsic and extrinsic estimation prevents errors in one stage from compensating for those in the other.

**Stage 1 — Intrinsic calibration.** Each camera is calibrated independently using multiple handheld views of a ChArUco board (5 × 7 squares, 120 mm square length, 89.1 mm marker length, `DICT_5X5_50`). The dense checkerboard corners provide well-conditioned intrinsic estimation. For the deployed system, 25 accepted images were captured per camera.

**Stage 2 — Extrinsic calibration.** Relative camera poses are estimated through joint bundle adjustment using synchronized image pairs of a multi-board AprilGrid target (6 boards, each containing two tags arranged in a 1 × 2 grid, 180 mm tag length, 0.30 tag-spacing ratio, family `t36h11`) distributed throughout the shared field of view. The intrinsic parameters obtained in Stage 1 remain fixed during optimization (`multical calibrate --fix_intrinsic`), preventing bundle adjustment from absorbing intrinsic errors into the camera poses. The deployed installation required eight synchronized image pairs. Bundle adjustment is performed using the Multical toolbox in an isolated software environment to avoid dependency conflicts with OpenCV.

Image acquisition quality is enforced automatically by the isical Studio application rather than relying on operator judgment. Each captured frame is accepted only if it satisfies four criteria: detection of at least 12 ChArUco corners (or four AprilGrid tags), a Laplacian variance of at least 80 to reject blurred images, a maximum allowable inter-frame motion to ensure stability, and a novelty criterion that rejects near-duplicate board poses.

The generated calibration is validated before being written to disk. Two acceptance thresholds are enforced: a maximum per-camera reprojection RMS error of 0.5 px for single-stage calibration and 2.0 px for the two-stage joint extrinsic optimization, the latter matching the homography-error KPI defined in the customer specification. Any solution exceeding the applicable threshold is rejected, preventing an invalid calibration from entering production.

Although bundle adjustment estimates the relative poses of all cameras, it does not establish their position with respect to the warehouse floor. A separate floor-anchoring procedure therefore defines the global world coordinate system. The ChArUco board is placed flat on the floor at multiple locations (eight placements per camera in the deployed system). Each placement yields a board pose through PnP estimation (`cv2.solvePnP`), after which a RANSAC plane fit (inlier threshold 0.03 m) is computed over all estimated planes. The resulting consensus plane defines the world plane ($Z = 0$), while the world origin and axes are taken from the first placement projected onto this plane. Using multiple placements prevents a single poorly positioned target from biasing the global reference frame.

For single-camera installations, where stereo bundle adjustment is unnecessary, the floor homography is estimated directly from operator-measured pixel-to-floor correspondences (Mode 1, Section 2.5). Although four non-collinear points are sufficient, five or more correspondences are recommended so that the estimation becomes overdetermined. Calibrations with a worst-case residual exceeding 0.10 m are rejected, allowing incorrectly measured points to be detected automatically.

The stored camera pose $(\mathbf{R}, \mathbf{t})$ maps camera coordinates into the world coordinate system. Runtime projection instead uses its inverse, $\mathbf{R}' = \mathbf{R}^{\top}$, $\mathbf{t}' = -\mathbf{R}^{\top}\mathbf{t}$, so that a homogeneous world point $\mathbf{X} = (X, Y, Z, 1)^{\top}$ projects into image coordinates $\mathbf{p} = (u, v, 1)^{\top}$ as

$$
s\,\mathbf{p} = \mathbf{P}\,\mathbf{X}, \qquad
\mathbf{P} = \mathbf{K}\,[\,\mathbf{R}' \mid \mathbf{t}'\,].
$$

Let $\mathbf{R}' = [\mathbf{r}_1\ \mathbf{r}_2\ \mathbf{r}_3]$. For points lying on the warehouse floor ($Z = 0$),

$$
s\,\mathbf{p} = \mathbf{K}\,[\,\mathbf{r}_1\ \ \mathbf{r}_2\ \ \mathbf{t}'\,]
\begin{pmatrix} X \\ Y \\ 1 \end{pmatrix}
\;\triangleq\; \mathbf{H}_{w \to p} \begin{pmatrix} X \\ Y \\ 1 \end{pmatrix},
\qquad
\mathbf{H} = \mathbf{H}_{w \to p}^{-1},
$$

allowing undistorted image points to be mapped directly into metric floor coordinates. Lens distortion is handled separately: image points are first undistorted using $(\mathbf{K}, \mathbf{D})$ before the homography is applied (Section 2.5). Calibration quality is finally inspected in a 3D visualization tool that displays camera and calibration-target poses together with per-view reprojection errors (Figure F3).

[Figure F3: Calibration materials and verification — ChArUco and multi-AprilGrid boards as deployed, and the bundle-adjustment viewer showing camera/board poses with reprojection-error visualization.]

## 2.4 Perception — isistream and detection models

isistream is the perception producer of isiMonitor3d. It owns the complete pixel-processing pipeline — from RTSP video acquisition through object detection and pose estimation — and converts each incoming frame into the compact detection sets consumed by the backbone metric engine (Section 2.2).

**Capture pipeline.** Each camera is ingested over RTSP (TCP transport) through a GStreamer pipeline constructed programmatically using PyGObject (Section 2.1). At startup, the stream codec is detected automatically and the pipeline instantiates the corresponding depayloader (H.264 or H.265/HEVC). Unlike automatic GStreamer plug-in selection, this explicit configuration supports mixed-codec camera installations while eliminating pad-linking race conditions.

Video decoding uses NVIDIA NVDEC hardware acceleration whenever available (`nvh264dec`/`nvh265dec`), including GPU-based color conversion and in-pipeline downscaling. When hardware acceleration is unavailable, decoding falls back automatically to software. The pipeline terminates in an appsink configured to retain only the newest decoded frame (`max-buffers=1, drop=true`). Since warehouse monitoring requires decisions based on the current scene rather than historical frames, discarding stale frames bounds end-to-end latency by design.

Each frame receives a `capture_ts` timestamp immediately upon arrival at the appsink, representing the earliest instant at which the system observes the frame. This timestamp serves as the common temporal reference throughout the remainder of the processing pipeline. Relative to the physical scene, it includes the RTSP jitter-buffer delay (approximately 100 ms) together with the video decoding latency, and therefore forms the basis of all latency measurements reported in Section 3.

**Runtime perception pipeline.** The perception pipeline executes as a paced processing loop. During each iteration, isistream:

1. acquires the most recent frame and corresponding `capture_ts` from every camera;
2. performs zone-scoped object detection across all cameras using a single batched inference;
3. executes person-pose estimation every $n$-th iteration, where $n$ is a configurable stride; and
4. publishes one `DetectionSetMessage` per camera.

Messages are transmitted even when no objects are detected, allowing downstream components to distinguish an empty scene from communication failure. Frames that become stale before processing are discarded, in which case no message is produced and the backbone treats the camera as unavailable.

**Motion gate.** Warehouse environments remain static for long periods, making repeated neural-network inference unnecessary. Since, for compact detection models, per-inference launch overhead is significant relative to arithmetic cost, reducing inference frequency provides a performance lever complementary to accelerating individual inferences.

Accordingly, each camera employs a motion gate (enabled in the deployed configuration). Motion is estimated using a lightweight grayscale signature computed from a 32 × 32 downsampled image. A frame is considered changed when more than 2 % of pixels differ by at least 15 grayscale levels.

Two independent motion gates operate for every camera. Object detection uses signatures computed separately for each monitoring zone so that movement outside operational areas does not trigger inference, whereas person-pose estimation evaluates motion over the full frame to preserve complete personnel monitoring. To prevent gradual drift, inference is forced periodically every two seconds (configurable).

Whenever inference is skipped — or between pose-estimation stride intervals — the previously inferred detections, including bounding boxes and segmentation polygons, are re-emitted using the current `capture_ts`. Consequently, downstream consumers receive a continuous observation stream, while the backbone metric engine repeatedly updates its Kalman filters with the most recent valid measurements under the assumption that the scene remains unchanged. The motion-gate configuration and pose stride used during evaluation are reported in Section 3.

**Zone-scoped detection.** Zone scoping exploits the fixed-camera assumption introduced in Section 1. Because most of each warehouse image contains static floor or shelving, object detection is restricted to operator-defined floor regions, whereas the person-pose model continues to process the entire frame to ensure complete personnel coverage.

During system configuration, the operator declares each monitoring zone by clicking a small number of points (three or more; rectangular zones from two corners) either on the metric floor map or directly on either camera's live view; camera clicks are back-projected to the floor through the same undistortion and homography used by the runtime pipeline. Every zone is therefore stored once, as a floor polygon in metric coordinates — the single source of truth from which all camera-specific geometry is derived. Using the calibration parameters, every polygon is projected into each camera view while accounting for lens distortion and extruded vertically between $z = 0$ m and $z = 2$ m, ensuring that the full height of loaded pallets remains inside the resulting crop; the projected outline is also displayed on every camera view, so the operator sees the same zone on both feeds regardless of where it was drawn. Zones outside a camera's field of view generate no crop.

At runtime, all visible zone crops from every camera are resized by letterboxing to a fixed detector input size (384 px by default; 320 px in the deployed system and throughout the experiments reported in Section 3) before being processed together in a single batched inference. The resulting detections are subsequently transformed back into full-frame image coordinates. This approach implicitly magnifies distant zones, allowing small objects to occupy a larger fraction of the detector input than they would in the original full-resolution image.

Because vertically extruded zones may overlap in image space, the same object can occasionally be detected in multiple crops. A per-camera, per-class cross-crop deduplication stage therefore retains only the highest-confidence detection. When no monitoring zones are configured, the object detector is omitted entirely and the system operates in pose-only mode.

**Cross-camera zone twins.** Because each zone is defined in metric floor coordinates, projecting it into the second camera materializes a *twin*: the same physical floor region observed independently from both viewpoints, each with its own crop and its own detection inference. This redundancy is the zone mechanism's occlusion defense — when a forklift or load blocks one camera's view of a zone, the other camera's twin continues to produce detections, so the zone's tracks and its retained occupancy state (`ZoneState`, Section 2.7) remain valid rather than lapsing; the two observation streams are reconciled downstream by the metric fusion stage (Section 2.5), which merges them when both cameras see an object and falls back to the surviving view when they do not. The operator dashboard derives each twin's outline automatically by round-tripping the drawn polygon through the floor plane — sampling along its edges, since a straight image edge maps to a curve under lens distortion — and regenerates all twins whenever the calibration changes. Segmentation masks are not transferred between cameras: each camera infers its own masks within its own crop, and only the zone geometry itself is projected across views.

**Detection models and inference runtime.** Object detection employs instance-segmentation networks from the two detector families introduced in Section 2.1: YOLO26-seg and RF-DETR-seg. All models are exported to ONNX and executed using ONNX Runtime. Using framework-independent ONNX models rather than vendor-specific engine formats allows identical model files to be deployed on both the development workstation and Jetson production hardware (Section 2.7), while hardware-specific acceleration is provided transparently through interchangeable execution providers.

On NVIDIA systems, in the configuration measured in this article (July 2026), inference used ONNX Runtime's TensorRT execution provider with compiled engines cached on disk. An optional slicing-assisted inference (SAHI) mode subdivides unusually large or elongated zone crops into overlapping square tiles before detection. Because TensorRT generates separate optimized engines for different input dimensions and SAHI introduces variable batch sizes, batches are padded into predefined buckets (1, 2, 4, 8, 16, and 32) to limit the total number of compiled engines.

The same detector interface also supports the CUDA execution provider as a fallback on NVIDIA GPUs, while an OpenVINO-based detector plugin behind the same interface enables deployment on Intel CPUs and integrated GPUs without modifying the remainder of the perception pipeline.

Person pose estimation is performed using a YOLO11-pose ONNX model. Each detected person includes 2D keypoints $(x, y, \text{confidence})$, from which the ankle midpoint is extracted as the representative foot point used by the geometric processing stage. Segmentation masks are polygonized and, together with bounding boxes, foot points, and keypoints, serialized into the per-camera `DetectionSetMessage`.

## 2.5 Geometric core — the backbone metric engine

The backbone metric engine is where image measurements become metric object trajectories. It implements the two complementary geometric methods introduced in Section 2.1 behind a common calibration and a unified identity space (Figure F2). Two architectural invariants govern its design. **One calibration, two queries:** both the homography-based and triangulation-based geometric pipelines read the same `calibration.json` file, ensuring that both methods operate from identical camera parameters. **One identity space:** track identities are created exclusively by the 2D tracker. The 3D pipeline never performs independent data association or re-identification; instead, it augments existing 2D tracks with three-dimensional coordinates.

[Figure F2: The geometric pipeline: always-on homography chain (foot point → undistortion → $\mathbf{H}$ → fusion → disagreement gate → ByteTrack-in-meters → stabilizer → `Track2D`) and subscription-driven triangulation branch (association → two-view DLT → reprojection gate → 3D Kalman → `Track3D`); shared calibration and identity space highlighted.]

**Frame synchronization and operating modes.** Incoming per-camera `DetectionSetMessage`s (or decoded frames in the single-process fallback mode) are synchronized using an approximate-time policy based on `capture_ts`. Whenever the oldest buffered frame from every camera falls within a configurable skew tolerance (33 ms by default, corresponding to one frame at 30 fps), the synchronizer emits a temporally aligned multi-camera observation.

The engine supports two operating modes determined by the configured number of cameras. **Mode 1 (single camera):** only the homography pipeline is instantiated, producing `Track2D`. **Mode 2 (stereo):** the homography pipeline continuously produces `Track2D`, while the triangulation pipeline generates `Track3D` whenever requested.

If one camera fails during stereo operation, the synchronizer waits for a 100 ms grace period before switching automatically to single-camera processing. Buffered stale frames are discarded and only the latest frame from the surviving camera is processed thereafter. Consequently, `Track2D` continues uninterrupted, `Track3D` suspends automatically because its two-camera subscription condition is no longer satisfied, and object identities remain unchanged throughout degradation and recovery. The objective is graceful degradation rather than service interruption.

**Homography pipeline (always active).** The homography pipeline continuously estimates metric floor positions for every detected object. For each detection, only the foot point is projected onto the ground plane: the bottom-center of the bounding box for general objects and the ankle midpoint for persons. Since this point corresponds to the object's physical contact with the floor, it is the only image measurement geometrically consistent with planar projection.

The observed pixel $\mathbf{u}$ is first undistorted using the intrinsic calibration before being mapped into metric floor coordinates,

$$
\begin{pmatrix} X \\ Y \\ 1 \end{pmatrix} \propto \mathbf{H}\,\tilde{\mathbf{p}},
\qquad \tilde{\mathbf{p}} = \Pi^{-1}(\mathbf{u};\, \mathbf{K}, \mathbf{D}),
$$

where $\Pi^{-1}$ denotes OpenCV undistortion.

When multiple cameras observe the same object — including the two independent views of a twinned zone (Section 2.4) — floor observations are fused before tracking. Candidate correspondences are obtained by solving a Euclidean-distance assignment problem in metric space using the Hungarian algorithm. Per-class matching thresholds (0.8 m for persons and pallets, 1.6 m for forklifts by default) determine whether observations represent the same physical object. Accepted pairs are merged by averaging their positions, whereas unmatched detections remain valid single-camera observations to accommodate partial occlusions.

A second, more conservative agreement gate verifies each merged observation. If the estimated positions disagree beyond tighter per-class thresholds (0.4 m for persons and pallets, 0.8 m for forklifts), fusion is rejected. The higher-confidence observation is retained while the conflicting measurement is discarded, implementing the system's "fail honestly" principle within the 2D pipeline.

**Metric-space tracking.** The fused observations are tracked using a modified ByteTrack implementation operating directly in metric coordinates. Unlike conventional image-space tracking, metric-space tracking makes association thresholds independent of camera geometry and enables all cameras to contribute to a single global tracker without requiring cross-camera identity reconciliation.

Each track maintains a constant-velocity Kalman filter with state $[X, Y, v_X, v_Y]$. Tracks are first predicted using the measured inter-frame interval before observations are divided into high- and low-confidence sets (default confidence thresholds: 0.5 and 0.1). Observations below the lower threshold are discarded.

Association proceeds through two Hungarian-assignment stages. Unlike the original ByteTrack formulation, the first stage matches all active tracks — tracked, newly created, and previously lost — to high-confidence detections. Remaining unmatched tracks are subsequently associated with low-confidence detections. Matching uses the Mahalanobis distance,

$$
d^2(\mathbf{z}, \hat{\mathbf{z}}) = (\mathbf{z} - \hat{\mathbf{z}})^{\top} \mathbf{S}^{-1} (\mathbf{z} - \hat{\mathbf{z}}),
$$

where $\mathbf{S}$ is the innovation covariance of the Kalman filter. Unmatched high-confidence observations initialize new tracks. Track identifiers are assigned once and are never reused, preserving identity continuity even during temporary single-camera operation.

To suppress frame-level classification fluctuations, each track's semantic label is stabilized using majority voting over a sliding observation window. Tracks may additionally remain unpublished until they satisfy a configurable confirmation period. The resulting trajectories constitute the continuously published `Track2D` stream.

A parallel rule-based module estimates pallet occupancy by combining two complementary cues — image overlap and metric floor margin — which exhibit different failure modes. The resulting empty/full state is stabilized using the same temporal majority-voting mechanism.

**Triangulation pipeline (on demand).** Three-dimensional reconstruction is performed only for tracks satisfying declarative subscription rules, including object class, minimum number of observing cameras, zone membership, and optional per-track rate limits. This subscription-based design ensures that computational cost scales with consumer demand rather than scene complexity, since most industrial monitoring tasks require only floor-plane trajectories.

For every subscribed `Track2D`, the triangulation pipeline first resolves the corresponding image observations by matching floor projections back to the original per-camera detections from the same synchronized frame. Consequently, the triangulator performs only geometric reconstruction while inheriting object identities directly from the 2D tracker.

Stereo reconstruction uses the standard Direct Linear Transform (DLT) implementation (`cv2.triangulatePoints`). Each camera $i$, with projection matrix $\mathbf{P}_i$ and undistorted observation $\mathbf{p}_i$, contributes two linear equations,

$$
[\mathbf{p}_i]_{\times}\, \mathbf{P}_i\, \mathbf{X} = \mathbf{0},
$$

which are solved jointly by singular value decomposition, with the reconstructed point corresponding to the singular vector associated with the smallest singular value.

Each reconstructed point is subsequently validated through reprojection. The estimated 3D point is projected back into every contributing camera, and reconstruction is rejected whenever the maximum reprojection error,

$$
e_i = \left\lVert \pi(\mathbf{P}_i \hat{\mathbf{X}}) - \mathbf{p}_i \right\rVert_2,
$$

exceeds 5 pixels by default (the customer specification permits thresholds between 5 and 8 pixels).

Accepted observations update a second constant-velocity Kalman filter with state $[X, Y, Z, v_X, v_Y, v_Z]$, indexed by the corresponding `Track2D` identifier. Filters whose associated 2D tracks disappear are removed automatically during each update cycle. The resulting `Track3D` stream therefore carries exactly the same identities as `Track2D`, differing only in the addition of metric height information.

With only two cameras, the DLT system is exactly determined, limiting the types of geometric inconsistencies detectable through reprojection error alone. This limitation, together with the role of the upstream disagreement gate in mitigating it, is discussed in Section 4.

## 2.6 Synthetic training data — isiGen

isiGen addresses the primary deployment bottleneck identified in Section 1: the collection and manual annotation of site-specific training data. Rather than relying solely on large annotated datasets, it expands a small collection of real photographs into a fully labeled synthetic dataset using the diffusion-based generation pipeline introduced in Section 2.1.

The pipeline consists of ten sequential stages implemented as independent plugins: curation, automatic detection, mask generation, control-map generation, captioning, LoRA training, scaffold synthesis, image generation, quality filtering, and dataset export.

The process begins by curating real images while preserving their original warehouse environments. Automatic object detection provides bounding-box prompts that seed mask generation, after which SAM2 produces color-coded segmentation masks. DepthAnythingV2 monocular depth estimation generates the depth control maps used to constrain image synthesis (a Canny edge extractor also runs at this stage, but its maps do not condition generation). Image captions are automatically constructed using a unique trigger token for each object class together with detailed background descriptions, reducing concept bleed during diffusion-based generation.

Class-specific LoRA adapters are then fine-tuned on Stable Diffusion XL, using an fp16 UNet with fp32 LoRA weights. For the black-polybag dataset, training used rank 16, a resolution of 768 px, 2,000 optimization steps, a learning rate of $10^{-4}$, batch size 1 with four-step gradient accumulation, and 53 real training photographs.

Synthetic image generation combines Stable Diffusion XL, a depth-conditioned ControlNet, and the trained LoRA adapter. Generation is guided by procedurally synthesized scaffolds consisting of paired control maps and ground-truth segmentation masks, which may optionally be created by compositing real object instances onto empty warehouse backgrounds. The scaffold constrains scene geometry, while the text prompt randomizes object appearance and environmental variation, producing images whose annotations are known by construction. For the black-polybag dataset, 500 scaffolds and 553 captions produced 500 synthetic images, which were filtered using CLIP similarity scores before being exported in YOLO-seg format (Figure F4).

[Figure F4: isiGen samples — real photograph, derived control map and ground-truth mask scaffold, and synthetic variants generated under geometric control with randomized appearance.]

The exported datasets are consumed by isidet, a dedicated training framework executed in a separate Conda environment. Isolating the training stack from the runtime environment ensures that training dependencies cannot affect the deployed perception system. isidet fine-tunes the YOLO26-seg and RF-DETR-seg models described in Section 2.4 and exports raw-head ONNX models for deployment.

The detector models evaluated in this work were trained exclusively on a real-image dataset comprising three object classes (`palette`, `carton`, `polybag`), containing 5,540 training images and 1,049 validation images. The synthetic images generated by isiGen were packaged into a separate merged dataset and were not used for the detector results reported in this paper. Instead, their contribution is evaluated independently through the training ablation study presented in Section 3.4.

## 2.7 Communication and deployment

The delivery layer converts the metric engine outputs into machine-readable interfaces for three categories of consumers: co-located software modules, remote or multi-node subscribers, and polling-based vehicle controllers. Each communication mechanism is selected according to the requirements of its target consumer.

All engine outputs crossing process boundaries are transmitted as Pydantic-validated JSON envelopes (schema version 6). Each message contains the schema version, message type, and originating `capture_ts` timestamp, ensuring consistent temporal interpretation across all consumers. The message catalogue includes `Track2D`, `Track3D`, zone entry/exit events, image references, persistent zone states (`ZoneState`, used as the WMS/FMS integration signal), per-camera observations, incoming `DetectionSetMessage`s (Section 2.2), oversized-message fragmentation envelopes, proximity alerts, periodic diagnostic heartbeats, and retained configuration advertisements.

For low-latency communication between components deployed on the same site, the system uses UDP/JSON as the primary intra-site transport. Remote and multi-node consumers are served through MQTT, which provides broker-mediated distribution and retained message delivery. MQTT topics follow the hierarchy `<base>/<version>/<node_id>/<suffix>`; the complete per-node topic tree is:

```
{prefix} = isiMonitor3D/v1/<node>
├─ track2d/{cls}          metric floor tracks
├─ track3d/{cls}          on-demand 3D tracks
├─ zone/{zone}            zone occupancy (retained, QoS 1)
│  ├─ passings            entry/exit events
│  └─ images/{track_id}   snapshot references
├─ proximity              person-object pairs (retained, QoS 1)
├─ diagnostics/heartbeat  node health heartbeat
└─ config                 config advertisement (retained)
```

This is the same hierarchy the isicomms probe interface renders live — latest message per topic — during commissioning. The MQTT topic version is intentionally independent from the payload schema version, allowing communication routing and message evolution to progress independently.

The isicomms module provides the interface between the internal messaging system and external industrial clients. It combines an MQTT broker (port 1883) with a REST gateway for polling-based consumers such as AGV fleet controllers. The gateway exposes an optionally token-authenticated HTTP API on port 8080 with endpoints for nodes (`/nodes`), zones (`/zones`), tracks (`/tracks`), and passing events (`/passings`). This abstraction avoids requiring vehicle controllers to implement MQTT clients, which is often incompatible with existing fleet-control software.

**Runtime environment and deployment.** Development and experimental evaluation are performed on a Linux/WSL2 workstation equipped with an NVIDIA RTX 5070 GPU (12 GB, Blackwell architecture, sm_120). The software stack consists of Python 3.10, OpenCV 4.13, GStreamer 1.28, and ONNX Runtime 1.23.2 with GPU acceleration through TensorRT 10.16 and CUDA 12.9.

The production deployment target is an NVIDIA Jetson Orin NX 16 GB module (Seeed reComputer J4012). Python 3.10 is maintained to match the JetPack 6.x software environment. Since the runtime operates on framework-independent ONNX models through ONNX Runtime, which provides a Jetson-compatible backend, migration from development hardware to edge deployment requires only an environment change rather than modifications to application code.

In deployment, the perception producer and metric engine execute as two independent systemd-managed services sharing the same configuration file. This provides deterministic startup, automatic recovery after failure, and fully local operation without dependency on cloud infrastructure.

<!--
Source traceability (per subsection):

2.1 Background (ADDED 2026-07-21, author item 3):
  High-level technology introductions; every claim carried by a verified reference:
  YOLO line [12][13][14]; DETR line [15][16][17]; latent diffusion [24 = Rombach et al.
  CVPR 2022, arXiv 2112.10752, VERIFIED this session]; SDXL [18]; ControlNet [19];
  LoRA [20]; SAM2 [25 = Ravi et al., arXiv 2408.00714, VERIFIED this session — repo
  verification: trainer/isiGen/src/stages/masking/sam2_masker.py uses
  facebook/sam2.1-hiera-small with box prompts from the detection stage, so SAM2 is a
  genuine pipeline component]; GStreamer [27, official site VERIFIED]; NVDEC [28,
  NVIDIA Video Codec SDK page VERIFIED]; ONNX Runtime [22]; TensorRT [23];
  Kalman [26 = Kalman 1960, DOI 10.1115/1.3662552, VERIFIED]; SORT [6]; ByteTrack [7];
  Hartley–Zisserman [8]; Zhang [9]; AprilTag [10]; ArUco/ChArUco [11].
  System-role sentences summarize §§2.3–2.7 content (no new system facts or numbers).
  RT-DETR mechanism ("efficient hybrid encoder, uncertainty-minimal query selection")
  per the RT-DETR abstract [16].

2.2 System architecture (was 2.1):
  /home/aatanda/isi_monitor3d/CLAUDE.md (Direction 1, five seams, seven principles, KPI table, ingest port 9010, /dev/shm bus)
  /home/aatanda/isi_monitor3d/cheatsheet/docs/index.md (four runtime apps, ports, KPI targets)
  /home/aatanda/isi_monitor3d/backbone/comms/schemas.py (DetectionSetMessage contract rules: capture_ts, heartbeat, seq, config_fingerprint)
  /home/aatanda/isi_monitor3d/docs/REUSE.md (two frozen wire contracts, module detachability)

2.3 Calibration (was 2.2):
  /home/aatanda/isi_monitor3d/calibration/calibrate.py (module docstring: two-stage flow, Multical venv isolation, hard limits 0.5 px / 2.0 px, consensus plane fit lines 773-875)
  /home/aatanda/isi_monitor3d/isical/data/c1/calib.yaml (board specs, capture targets/quality gates)
  /home/aatanda/isi_monitor3d/isical/data/c1/{intrinsic,extrinsic,floor}/ (deployed shot counts: 25/cam intrinsic, 8 pairs extrinsic, 8 floor placements/cam)
  /home/aatanda/isi_monitor3d/isical/capture/session.py (floor-anchor capture, quality gating)
  /home/aatanda/isi_monitor3d/backbone/shared/geometry.py (projection_from_K_R_t, floor_homography_from_K_R_t: pose convention and H/P derivation)
  /home/aatanda/isi_monitor3d/calibration/calibrate_single_cam.py (Mode 1: 4-point minimum, 5+ recommended, residual_threshold_m=0.10 refusal)
  /home/aatanda/isi_monitor3d/CLAUDE.md (calibrate-2cam / single-cam commands)

2.4 Perception (was 2.3):
  /home/aatanda/isi_monitor3d/isistream/core.py (tick loop, batched zone detection, pose stride, cached re-emission under new capture_ts, emission policy, TRT batch buckets)
  /home/aatanda/isi_monitor3d/isistream/motion_gate.py (per-zone-crop + full-frame signatures, 32x32 gray, >2% pixels moving >15 levels, refresh_s=2.0 default)
  /home/aatanda/isi_monitor3d/config/backbone.yaml (motion_gate: true — deployed configuration)
  /home/aatanda/isi_monitor3d/isistream/__main__.py (standalone service, shared config)
  /home/aatanda/isi_monitor3d/backbone/ingestion/rtsp.py (TCP RTSP, codec-aware depay, NVDEC chain + software fallback, appsink newest-frame, in-pipeline downscale; docstring: ~100 ms rtspsrc latency + decode lag of capture_ts — the caveat's single full statement now lives HERE per author item 2)
  /home/aatanda/isi_monitor3d/backbone/ingestion/frame_sync.py (capture_ts axis)
  /home/aatanda/isi_monitor3d/backbone/detection/zone_scope.py (z=0..2 m projection, letterboxed crops, cross-crop dedup, pose-only policy)
  /home/aatanda/isi_monitor3d/backbone/detection/yolo_onnx_pose.py (keypoints (x,y,conf), ankle-midpoint foot)
  /home/aatanda/isi_monitor3d/CLAUDE.md (zone_imgsz 384 default, TRT default EP + engine cache, OpenVINO plugin, capture_ts policy)

2.5 Geometric core (was 2.4):
  /home/aatanda/isi_monitor3d/backbone/homography/{__init__,foot_projector,cross_cam_fusion,disagreement_gate,bytetrack,track,temporal_stabilizer,pallet_occupancy}.py (chain order, undistort-then-H, Hungarian fusion, gate semantics, two-pass Mahalanobis matching, id minted once, majority vote, occupancy dual estimator)
  /home/aatanda/isi_monitor3d/backbone/triangulation/{__init__,subscription_manager,keypoint_associator,opencv_dlt,reprojection_gate,tracker_3d}.py (subscription DSL, association, cv2.triangulatePoints DLT, 5 px gate / 5-8 range, 3D Kalman keyed by 2D id, gc)
  /home/aatanda/isi_monitor3d/backbone/ingestion/frame_sync.py (approximate-time pairing, sticky degraded flag, 100 ms grace)
  /home/aatanda/isi_monitor3d/CLAUDE.md (modes table, degradation, one-calibration/one-identity, 2-cam exactly-determined gotcha)

2.6 Synthetic data (was 2.5):
  /home/aatanda/isi_monitor3d/trainer/isiGen/README.md (pipeline phases, SDXL + depth ControlNet + fp16-fix VAE, SAM2/DepthAnythingV2/Canny, anti-bleed captions, copy_paste scaffolds, CLIP filter, YOLO-seg export)
  /home/aatanda/isi_monitor3d/trainer/isiGen/src/stages/ (10 stage packages: captioning, control_maps, curate, detection, exporting, filtering, generation, lora, masking, scaffolds)
  /home/aatanda/isi_monitor3d/trainer/isiGen/runs/lora/black_polybag_r16_18-06-2026_13-24-47/report.md (rank 16, 768 px, 2000 steps, lr 1e-4, batch 1x4 accum, 53 images)
  /home/aatanda/isi_monitor3d/trainer/isiGen/data/black_polybag/ (500 scaffolds [sc000000-sc000499, base+control+inpaint triplets], 553 captions, 500 generated)
  /home/aatanda/isi_monitor3d/trainer/isidet/data/pallet3_yolo_seg/data.yaml + images/{train,val} (3 classes; 5540 train / 1049 val counted on disk)
  /home/aatanda/isi_monitor3d/CLAUDE.md (isi-train env isolation, raw-head ONNX export)

2.7 Comms + deployment (was 2.6):
  /home/aatanda/isi_monitor3d/backbone/comms/schemas.py (SCHEMA_VERSION 6, message classes, MQTT topic convention)
  /home/aatanda/isi_monitor3d/docs/REUSE.md (isicomms MQTT-in/REST-out, Bearer-token polling endpoints, frozen interface)
  /home/aatanda/isi_monitor3d/cheatsheet/docs/index.md (ports 1883/8080)
  /home/aatanda/isi_monitor3d/CLAUDE.md (RTX 5070 12 GB WSL2 sm_120, Jetson Orin NX 16 GB Seeed J4012, Python 3.10/JetPack rationale, OpenCV 4.13, GStreamer 1.28, ONNX Runtime 1.23.2, TensorRT 10.16, CUDA 12.9, ONNX portability, systemd/no-cloud)

Deviations from the task brief (repo evidence overrode the brief):
  - "8 placements, consensus plane fit" for the floor anchor: confirmed 8 shots/cam in isical/data/c1/floor/, but the code default target is 4 (isical/capture/session.py FLOOR_TARGET=4); text states 8 as the deployed-rig count.
  - Brief said "2001 scaffolds, 553 captions, 500 generated" (PLAN T2): on disk the scaffolds directory holds 2001 FILES = 500 scaffolds x (base+control+inpaint) + index.jsonl; text states 500 scaffolds.
  - KPI targets stated as design requirements only (cahier des charges); all measured values reserved for Section 3 per the plan.

Revision log (cv-reviewer pass 1, 2026-07-20 — all points re-verified in code before applying):
  - M1 ACCEPTED: motion gate added as a first-class mechanism (verified: config/backbone.yaml
    motion_gate: true; isistream/core.py tick lines ~285-330 re-emit cached _wire_obj/_wire_person
    under the new frame's capture_ts; _wire_person cache re-sent every tick). Architecture (a)
    capture_ts claim qualified; "persons coast on the engine's Kalman prediction" REMOVED — the
    draft had inherited the stale isistream/core.py module docstring; actual behavior (repeated
    identical measurements re-anchoring the Kalman) now stated.
  - M2 ACCEPTED: first Hungarian pass pool corrected to TRACKED+NEW+LOST (bytetrack.py line 101),
    departure from Zhang et al.'s original association explicitly declared.
  - M3 ACCEPTED: "Sections 4 and 9" -> "Section 4 (Discussion, Limitations)".
  - M4 ACCEPTED: Mode-1 minimum corrected to 4 correspondences, >=5 recommended, 0.10 m worst-residual
    refusal gate stated (calibration/calibrate_single_cam.py).
  - M5 ACCEPTED: ten stages now match the ten src/stages/ packages ("pipeline initialization" removed —
    it is a README workflow phase (P5), not a stage package; "automatic detection" added). README's
    8-phase operator view not cited in text (word budget); noted here.
  - Must-add values ACCEPTED, all re-verified: max_skew_ms 33.0 (frame_sync.py:50; "one frame at
    30 fps" is our gloss), fusion 0.8/1.6 m (cross_cam_fusion.py:27), agreement 0.4/0.8 m
    (disagreement_gate.py:25), conf split 0.5/0.1 (bytetrack.py:57-58), RANSAC plane inlier 0.03 m
    (calibrate.py:838).
  - Minor 1 ACCEPTED ("two independent equations"); Minor 2 ACCEPTED (SAHI introduced in one
    sentence — tiling.py/zone_scope.py); Minor 3 ACCEPTED ("transport's safe payload size");
    Minor 5 ACCEPTED ("2 tags per board (1 x 2 grid)"); Minor 6 ACCEPTED (provenance bridge added;
    exact real/synthetic composition deferred to §3 T2 — not fully derivable from data.yaml alone).
  - Minor 4 (number homes vs Results tables) and Minor 7 (F2 must depict or exclude the occupancy
    module) are DEFERRED to §3 assembly and figure production respectively — no text change required;
    the F2 caption as written does not claim the occupancy module.
  - No review points rejected.

Post-campaign corrections (G0/G1 provenance findings, 2026-07-20):
  - §2.6 merged-corpus sentence REPLACED: G0_data_provenance.md proves every reported
    detector trained on the ALL-REAL pallet3 corpus (5,540/1,049; zero synthetic images;
    RF-DETR headline run predates the first synthetic image by 11 days). The genuinely
    merged corpus (dataset_v2, 2,243 real + 500 synthetic) was never trained on. The text now
    presents the pipeline as a capability quantified by the ablation (Section 3.4).
  - §2.3 hard-limit sentence corrected: the 0.5 px limit gates single-stage assembly;
    the two-stage joint extrinsic solve is gated at 2.0 px (calibrate.py
    assemble_calibration rms_limit_px) — the deployed rig's intrinsic-stage RMS (0.603 px
    cam_a) would otherwise appear to contradict a written calibration.
  - Verified: this section contains no "held-out test split" claim (G1: pallet3_coco/test
    is a byte-identical duplicate of valid).

Author corrections 2026-07-21 (this session):
  - Item 1: "ISI Monitor 3D" -> "isiMonitor3d" (§2.2 opening). MQTT topic string
    `isiMonitor3D/v1/...` in §2.7 kept verbatim (literal wire string, per author exception).
  - Item 2 (de-repetition): §2.2 no longer restates the full KPI list (now "the five
    industrial acceptance criteria ... listed in Section 1"); the capture-clock ~100 ms
    caveat's single full statement moved to §2.4 Capture (was split across §3.3/§4.1);
    §2.3 "cannot drift independently" sentence now a pointer to §2.5's invariant.
    The five-module enumeration remains ONLY in §1 and §2.2 (the M&M head).
  - Item 3: §2.1 Background added; all subsections renumbered +1; internal cross-refs
    updated (2.3->2.4 motion gate; 2.6->2.7 schema; 2.4->2.5 geometry; 2.1->2.2
    DetectionSet; 2.3->2.4 models; 2.5->2.6 isiGen).
  - Item 4 (design rationale added, 1-2 sentences each, no new numbers): module
    partition by life cycle (§2.2); process split = opposed resource profiles +
    contention, pointer to §4.1 (§2.2); frame bus = decode cost + operator/model pixel
    parity (§2.2); schema-only evolution = independent deployability (§2.2); operator
    tooling + no surveying (§2.3); two-stage separation = no intrinsic-error absorption
    (§2.3); refusal gates = no silent bad calibration (§2.3); consensus plane = one bad
    placement cannot tilt the frame (§2.3); newest-frame sink = act on current scene
    (§2.4); zone scoping = scene prior (§2.4); ONNX over vendor-locked = portability
    (§2.4); degradation = reduced service never silence (§2.5); tracking in meters =
    physical thresholds + one identity space (§2.5); subscription = cost proportional
    to declared needs (§2.5); isiGen = attacks annotation cost (§2.6); trainer isolation
    rationale (§2.6); per-transport consumer classes (§2.7).
  - Item 5 (flow): each subsection now opens with a what-it-is/why-it-matters sentence
    (§2.4, §2.5, §2.6, §2.7 openings added/reworked); facts and numbers preserved exactly.

§2.1 REWRITE (author guideline, 2026-07-21 second pass — "no tutorial" verdict):
  - Pattern reversed per paragraph: need → technology → property exploited; all
    "how it works" teaching removed (history capped at one sentence per family).
  - New block order matching data flow, with explicit transitions: Vision inference
    (YOLO26 primary: throughput + NMS-free TRT deployability, raw-head export;
    RF-DETR: small-dataset fine-tuning, same interface) → Tracking (Kalman/ByteTrack,
    pixels-agnostic property → meters in §2.5) → Geometry (calibration/homography/DLT
    complementarity = one-calibration-two-modes; fiducials = correspondence-as-detection)
    → Synthetic training (SDXL latent-space affordability; ControlNet
    labels-by-construction; LoRA few-photo specialization; SAM2 prompt-based GT masks)
    → Deployment (GStreamer structure-as-policy; NVDEC CPU offload; ORT portability;
    TRT swappable EP + shape discipline).
  - Opening and closing paragraphs based on the author's samples; closing summary
    corrected for factual order (homography projection precedes ByteTrack-in-meters;
    triangulation on demand) — the author's sample had tracking before geometry.
  - Length: 1,555 → 839 words (-46%).
  - ORPHAN CHECK: all 22 §2.1 citations retained ([6]-[20], [22]-[28] minus [21]
    which lives in §1) — verified programmatically, none orphaned.
  - Cross-references INTO §2.1 from §2.4 (PyGObject/GStreamer; two families),
    §2.5 (two geometric methods), §2.6 (diffusion stack) remain valid.

§2.2 REPLACED VERBATIM with the author's own rewrite (2026-07-21 third pass).
  Departures from his text, all minimal and coordinator-sanctioned:
  1. PRECISION FIX (coordinator-flagged): "and, when stereo observations are
     available, 3D tracks (Track3D)" -> "and, on demand, 3D tracks (Track3D) when
     stereo observations are available" — Track3D is subscription-driven (§2.5),
     not automatic under stereo.
  2. FACT FIX: "projection, fusion, motion gating, and stabilization" ->
     "projection, fusion, gating, and stabilization" — the motion gate lives in
     isistream (isistream/motion_gate.py), not among the backbone's concrete
     classes; "gating" = the disagreement/reprojection gates, as before.
  3. FACT FIX: "Each camera writes frames into..." -> "Frames from each camera
     are written into..." — the producer writes the segments, not the cameras.
  4. CONVENTION MAPPING only (no wording change): "(Fig. 1)" -> "(Figure F1)" in
     markdown / \\figref{fig:arch} in the .tex; code identifiers set in
     backticks/\\texttt per document convention (Track2D/3D, calibration.json,
     DetectionSetMessage, capture_ts, /dev/shm path, the five seam names);
     module names bolded as elsewhere.
  Facts re-verified in repo: port 9010 (points_in.py:123, backbone.yaml:44);
  /dev/shm/isi3d_frame_<cam> (frame_shm.py); five seam ABCs (interfaces.py);
  explicit-empty heartbeat / seq gap counting / config_fingerprint warn-on-drift
  (points_in.py docstring, schemas.py DetectionSetMessage).
  DROPPED-CONTENT DIFF vs the previous §2.2 (flagged, NOT re-added):
  - The term "points mode" is no longer introduced anywhere in §2 — §3.3 still
    uses it ("points mode with the isistream producer and metric engine as
    separate processes...") with an inline gloss, so the sentence remains
    readable, but the term's formal introduction is gone. FLAG for author.
  - "silence triggers runtime degradation as if the camera had failed" softened
    to "interpreted as a camera fault" — the degradation mechanism remains fully
    specified in §2.5; dependency resolves.
  - Frame-bus cost rationale ("most expensive per-consumer operation") compressed
    into "To avoid redundant video decoding" — rationale substance preserved.
  - "one RTSP session and one decode per camera system-wide" rephrased as
    "eliminates duplicate RTSP sessions and decoding operations" — equivalent.
  No numbers were involved in this section.

§2.4 REPLACED VERBATIM with the author's own rewrite (2026-07-21 fourth pass).
  Verification verdicts (coordinator's 5 checks, all repo-verified this session):
  1. OpenVINO — his "CUDA and OpenVINO execution providers" was IMPRECISE:
     yolo_openvino is a SEPARATE detector plugin running the OpenVINO runtime
     on an OpenVINO IR (backbone/detection/yolo_openvino.py, registered via
     @detector_registry), not an ONNX Runtime EP. CORRECTED minimally to
     "the CUDA execution provider as a fallback on NVIDIA GPUs, while an
     OpenVINO-based detector plugin behind the same interface enables
     deployment on Intel CPUs and integrated GPUs..." (deployment-breadth
     point preserved).
  2. Stale frames — EXACT as written (isistream/core.py:277 'stale frame →
     SILENCE (degradation signal)': no message that tick, engine degrades the
     camera). Kept verbatim.
  3. SAHI — CONFIRMED producer-side (isistream/core.py sahi_cfg →
     ZoneScopedDetector(sahi=..., batch_buckets=...); tiling in
     backbone/detection/{tiling,zone_scope}.py). Buckets (1,2,4,8,16,32) =
     _BATCH_BUCKETS, isistream/core.py:51 (applied when TRT+SAHI both on).
     Sentence kept.
  4. Launch-overhead — TENSION with Table T5 (TRT gave 3.0x/2.8x ISOLATED
     inference gains, so "primarily limited by launch overhead rather than
     arithmetic throughput" + "larger benefit than accelerating" overstated).
     SOFTENED per coordinator: "per-inference launch overhead is significant
     relative to arithmetic cost, reducing inference frequency provides a
     performance lever complementary to accelerating individual inferences."
  5. Remaining facts — ALL CONFIRMED: pose stride = every n-th tick
     (tick_count % pose_every_n, core.py), deployed n=1 (backbone.yaml:58,
     declared in §3.3); forced refresh 2 s default + configurable
     (motion_gate.py refresh_s=2.0, isistream.motion_refresh_s); 32×32 / >2% /
     15 levels (motion_gate.py); z=0–2 m (zone_scope.py crop_height_m=2.0);
     384 default (core.py) / 320 deployed (backbone.yaml:64); per-camera
     per-class highest-confidence dedup (_dedup_across_crops); pose-only when
     no zones (core.py warning path).
  Back-reference targets INTO §2.4 verified surviving: capture-clock full
  statement (~100 ms jitter buffer + decode, "basis of all latency measurements
  reported in Section 3") → §3.3 and §4.1 back-refs resolve; motion-gate full
  statement (§2.2 and §3.3 pointers) resolves; zone-magnification sentence
  (§4.1 "more model pixels" back-ref) resolves in his wording.
  DROPPED-CONTENT DIFF vs previous §2.4 (flagged, NOT re-added):
  - "and is used consistently across cameras" (capture_ts cross-camera
    consistency clause) — dropped; the pairing-on-capture-ts mechanism is
    still fully stated in §2.5's synchronizer paragraph, so the dependency
    resolves. Minor.
  - "with an identical decode path" for the CUDA/OpenVINO alternatives —
    dropped; no other section depends on it. Minor.
  - "raw prediction head" export note now lives ONLY in §2.1 (Vision
    inference) — previous §2.4 did not restate it either post-trim; no break.
  Convention mapping only: subsection titles set as bold run-in headings
  (document convention); code identifiers in backticks/\\texttt; math-mode n,
  z, (x, y, confidence). No numbers changed.

§2.5 REPLACED VERBATIM with the author's own rewrite (2026-07-21 fifth pass).
  Equations: his paste mangled the math; per coordinator, the CURRENT verified
  four equations were slotted at his [EQUATION] callouts unchanged — projection
  with undistortion, Mahalanobis, DLT cross-product, reprojection error. In the
  built PDF they number (3)-(6) automatically ((1)-(2) live in §2.3), matching
  his callout numbers. "(Fig. 3)" mapped to Figure F2 (md placeholder name) /
  \\figref{fig:pipeline} (tex) — which renders as Figure 3 in the PDF (arch=1,
  calibration=2, pipeline=3), consistent with his numbering.
  Symbol-definition retentions (minimal, flagged): "where Π⁻¹ denotes OpenCV
  undistortion" kept after eq (3); his DLT lead-in extended to define P_i and
  p_i ("Each camera i, with projection matrix P_i and undistorted observation
  p_i, contributes two linear equations") — the equation is unreadable without
  the symbol bindings.
  Verification verdicts (coordinator's 9 checks, all repo-verified):
  1. Thresholds EXACT: fusion person 0.8 / pallet 0.8 / forklift 1.6
     (cross_cam_fusion.py:28-30, default 0.8); agreement person 0.4 /
     pallet 0.4 / forklift 0.8 (disagreement_gate.py:26-28, default 0.4).
     Class attribution correct as written.
  2. Merge is a PLAIN AVERAGE ("averaged" xy_m, cross_cam_fusion.py docstring)
     — his "merged by averaging their positions" exact.
  3. Gate outcome EXACT: highest-confidence camera kept as single-cam
     observation, other dropped (disagreement_gate.py:85-95).
  4. ByteTrack EXACT: F/Q rebuilt per predict with measured dt (track.py:50,62);
     split 0.5/0.1, below-low discarded (bytetrack.py:8,57-58); first pass
     TRACKED+NEW+LOST (line 101, departure declared); IDs from count(1), never
     reused (bytetrack.py:74).
  5. Stabilizer EXACT: majority vote over sliding window + min_frames_confirmed
     (default 1, configurable) — "configurable confirmation period" holds
     (temporal_stabilizer.py:35-46; period counted in frames).
  6. Occupancy EXACT: two independent estimators, image overlap + metric floor
     margin, majority-vote stabilizer (pallet_occupancy.py:1-18).
  7. Rate limits EXIST: rate_hz per rule, enforced per (rule_name, track_id)
     (subscription_manager.py:8,64,137,168-171; config/subscriptions.yaml
     rate_hz: 10.0) — his "optional per-track rate limits" KEPT.
  8. Synchronizer EXACT: 33 ms default (frame_sync.py:50), 100 ms grace,
     latest-frame-only after degradation (sticky flag); "(or decoded frames in
     the single-process fallback mode)" consistent with §2.2/frames-mode.
  9. Triangulation chain EXACT: nearest floor-projection match on same-frame
     same-class detections (keypoint_associator.py:11,50-53);
     cv2.triangulatePoints (opencv_dlt.py:30); smallest-singular-vector; 5 px
     default / 5-8 spec range (reprojection_gate.py:5,27); 3D KF keyed by 2D
     track_id; gc(active_track_ids) removes dead filters each cycle
     (tracker_3d.py:12,146).
  NO factual corrections were needed to his prose — all nine verdicts pass.
  DROPPED-CONTENT DIFF vs previous §2.5 (flagged, NOT re-added):
  - The principle name "subscription, not polling" and the fall-detection
    example (persons seen by two cameras inside danger zones) — dropped; no
    other section references either. Minor.
  - "with a metric gating distance" (Mahalanobis gating cutoff) — dropped;
    mechanism detail only, nothing depends on it. Minor.
  - "an industrial deployment must degrade to reduced service, never to
    silence" → his "graceful degradation rather than service interruption" —
    equivalent.
  Inbound back-references verified resolving: §2.1 tracking block ("Section 2.5
  transposes the state space to metric floor coordinates") → Metric-space
  tracking ✓; §2.3's invariant + Mode-1 + undistort-then-H pointers ✓;
  §2.4 "geometric processing stage" ✓; §4.1/§4.2 exactly-determined pointer →
  his final paragraph ✓. No numbers changed.

§2.6 REPLACED VERBATIM with the author's own rewrite (2026-07-21 sixth pass).
  Verification verdicts (coordinator's 8 checks, all repo-verified):
  1. Ten-stage list 1:1 with src/stages/ packages: curation→curate, automatic
     detection→detection, mask generation→masking, control-map
     generation→control_maps, captioning→captioning, LoRA training→lora,
     scaffold synthesis→scaffolds, image generation→generation, quality
     filtering→filtering, dataset export→exporting. Nothing missing/invented.
  2. DepthAnythingV2 CONFIRMED (control_maps/depth_anything_v2.py, registered
     "depth_anything_v2"; base.py names DepthAnythingV2 + Canny).
  3. Captioner CONFIRMED: template "{trigger} {class_phrase} {background}"
     (captioning/template.py) — unique per-class trigger + background bank;
     anti-concept-bleed purpose per README. Kept verbatim.
  4. LoRA precision EXACT: fp16 UNet, LoRA params cast to fp32
     (lora/diffusers_sdxl.py:10-11,153). Kept verbatim.
  5. Compositing mode CONFIRMED: scaffolds/copy_paste.py pastes masked real
     objects onto empty-scene backgrounds (preferred; falls back to object
     images). Kept verbatim.
  6. CORRECTION (flagged): class names "(pallet, carton, and polybag)" →
     literal dataset names "(`palette`, `carton`, `polybag`)" matching
     data.yaml, §3.1, and Tables T1/T2 (which use `palette` verbatim).
  7. CLIP-filter count 267: NOT re-added — §3.4 carries it reader-facing
     ("267 CLIP-filtered ... generations of the isiGen export"), so his
     count-free filtering sentence stands as written.
  8. "Separate merged dataset ... not used" CONFIRMED consistent with the
     provenance audit (dataset_v2 never trained on) and §3.1's provenance
     paragraph.
  Convention mapping: "(Fig. 4)" → Figure F4 placeholder (md) /
  \\figref{fig:isigen} (tex) — renders as Figure 4 in the PDF (arch=1,
  calibration=2, pipeline=3, isigen=4), matching his numbering; $10^{-4}$
  paste artifact set as proper math; code identifiers in backticks/\\texttt.
  DROPPED-CONTENT DIFF vs previous §2.6: none of substance — every fact,
  number, and rationale of the previous version is present in his text
  (53/rank 16/768 px/2,000 steps/1e-4/batch 1×4; 500/553/500; 5,540/1,049;
  trainer isolation rationale; labels-by-construction; ablation pointer).
  No numbers changed.

§2.7 REPLACED VERBATIM with the author's own rewrite (2026-07-21 seventh pass).
  Verification verdicts (coordinator's 7 checks):
  1. Message catalogue EXACT — 1:1 with the 11 message classes in
     backbone/comms/schemas.py: Track2DMessage, Track3DMessage,
     PassingEventMessage (zone entry/exit), ImageRefMessage, ZoneStateMessage
     (WMS/FMS signal), ObservationsMessage (per-camera), DetectionSetMessage,
     FragmentMessage (oversized-message envelopes), ProximityMessage,
     DiagnosticsMessage (heartbeats), ConfigMessage (retained advertisement).
     Nothing invented, none missing.
  2. CORRECTION (flagged): "token-authenticated HTTP API" → "optionally
     token-authenticated HTTP API" — Bearer auth is optional
     (isicomms/isicomms/config.py:42 "Optional bearer-token auth (None = open)").
  3. Endpoints EXACT — /nodes, /zones, /tracks, /passings all exist
     (isicomms/isicomms/api/routes_{nodes,zones,tracks,passings}.py). /ui,
     /docs, /healthz exist but are unmentioned (fine per coordinator).
  4. Topic EXACT — track2d_topic "{prefix}/track2d/{cls}", default prefix
     "isiMonitor3D/v1/node" (mqtt_sink.py:77,87): his example is well-formed
     (node_id "zone_a" cosmetic). Version-orthogonality claim matches
     schemas.py:36 ("MQTT topic convention (orthogonal to schema_version)")
     and :82-83 (topic-layout vs payload-shape bumps).
  5. UDP/JSON primary intra-site transport — consistent with §2.2 ✓.
  6. Hardware/stack paragraph — all values identical to the previously
     verified set (RTX 5070 12 GB sm_120; Python 3.10, OpenCV 4.13, GStreamer
     1.28, ORT 1.23.2, TRT 10.16, CUDA 12.9; Jetson Orin NX 16 GB Seeed
     J4012, JetPack 6.x). Nothing drifted.
  7. Two systemd services, one shared config file — consistent with §2.2 and
     CLAUDE.md deployment defaults ✓.
  Inbound back-refs verified resolving: §2.1 portability pointer → his
  Jetson/ONNX paragraph; §2.2 "backbone UDP/JSON schema (Section 2.7)" → his
  envelope paragraph; §2.4 Jetson "(Section 2.7)" → same.
  Convention mapping: topic hierarchy + example set inline as code (his display
  lines folded into the sentence per coordinator); "Runtime environment and
  deployment" as bold run-in; code identifiers in backticks/\\texttt.
  DROPPED-CONTENT DIFF vs previous §2.7: none of substance — consumer-class
  framing, full message set, transport rationale, ports 1883/8080, endpoint
  list, hardware stack, Jetson portability argument, and systemd defaults all
  carried; "deterministic restart" → "deterministic startup, automatic
  recovery after failure" (equivalent). His ADDITION "often incompatible with
  existing fleet-control software" is design rationale (unmeasured, hedged
  with "often") — kept verbatim, noted here. No numbers changed.

§2.3 REPLACED VERBATIM with the author's own rewrite (2026-07-21 eighth pass —
  completes his Methods sweep: all of §2.2–2.7 now author-voiced).
  Equations: his paste mangled the math; the CURRENT verified pose-inversion
  line + equations (1)–(2) restored at his callouts, byte-identical to the
  prior section (verified \\triangleq form). Global numbering intact: (1)–(2)
  here precede §2.5's (3)–(6) — confirmed in the built PDF.
  "(Fig. 2)" → Figure F3 placeholder (md) / \\figref{fig:calibration} (tex),
  which typesets as Figure 2 (arch=1, calibration=2, pipeline=3, isigen=4) —
  matching his numbering.
  Verification verdicts (coordinator's 6 checks):
  1. Capture QA gates EXACT — deployed config isical/data/c1/calib.yaml:
     min_charuco_corners: 12, min_april_tags: 4, blur_min_var: 80.0,
     steady_max_motion: 2.5, novelty_min_dist: 0.06; SnapGate enforces exactly
     these four criteria in order (isical/capture/detect.py:162-196). His
     ">=12 corners (or four tags) / Laplacian >=80 / max inter-frame motion /
     novelty" is correct. (session.py's _TEXTURE_MIN_BLUR_VAR=60 belongs to
     the separate texture-score helper, not the snap gate — not a conflict.)
  2. Tag spacing EXACT — calib.yaml tag_spacing: 0.30000000000000004 → "0.30
     tag-spacing ratio".
  3. Floor-anchor origin EXACT — calibrate.py consensus-plane block: "Origin:
     the first board's origin projected onto the consensus plane"; gauge
     (origin + in-plane X) from the first placement's board pose.
  4. Rationale sentences CONSISTENT with code docs (two-stage separation,
     --fix_intrinsic semantics, per-target strengths). Sanity pass.
  5. Restated values — ALL match the previously verified set, nothing drifted
     (board specs, 25 shots, 6×(1×2)/180 mm/t36h11, 8 pairs, Multical
     isolation, 0.5/2.0 px gates with CORRECT attribution, solvePnP, 0.03 m,
     8 placements, Mode-1 4-min/>=5-recommended/0.10 m rejection).
  6. 3D viewer CONFIRMED — Multical's built-in viewer (calibrate.py --vis /
     `vis` command; camera + board poses, per-view reprojection); Figure F3's
     caption content matches the claim.
  Symbol-definition retention (minimal, flagged): X = (X,Y,Z,1)^T and
  p = (u,v,1)^T bindings folded into his pose-inversion sentence — eq (1) is
  unreadable without them.
  Spelling normalization (flagged): his "judgement" → "judgment" (document
  uses American English throughout).
  DROPPED-CONTENT DIFF vs previous §2.3 (flag only): none of substance —
  every gate number, board spec, and rationale survives; "the multi-board
  target covers the shared volume in few synchronized pairs" rationale
  compressed to "distributed throughout the shared field of view" +
  "required eight synchronized image pairs" (substance preserved). Inbound
  back-refs verified: §2.1 fiducial pointer "(Section 2.3)"; §2.2 module (i);
  §2.5 reciprocal invariant (his first paragraph cites Section 2.5
  explicitly); §3.2 "two-stage workflow of Section 2.3". No numbers changed.

ZONE-TWIN MECHANISM ADDED (2026-07-21 ninth pass — author flagged it missing).
  Phase-1 repo findings backing every sentence:
  - DECLARATION: draw_mode.js target ∈ {'map','cam_a','cam_b'} — zones drawn
    on the metric map OR EITHER camera (author claim (a) CONFIRMED; >=3 points,
    rectFromCorners gives 4-point rects). Camera clicks back-projected via
    POST /api/project/pixel-to-floor (cv2.undistortPoints then H — the SAME
    backbone.shared.geometry helpers as production, per routes_projection.py
    docstring). Stored ONCE in metric floor coords (zones.yaml;
    floor_zone_sync.py derives entries from non-twin patches, "one floor zone,
    not two"). Outline re-projected onto every camera distortion-aware
    (floor_to_pixel_distorted; raw-frame accurate under k1≈-0.45 barrel).
  - TWIN MATERIALIZATION (claim (b)/(c) CONFIRMED, two layers): backbone
    zone_scope.py — one metric polygon → every camera (z=0..2 m,
    distortion-aware) → one crop per (camera, zone), each in the camera's own
    batched 320 px inference; dashboard routes_zone_patches.py — literal
    auto-derived "__twin" patches ("occlusion persistence ... an occluded
    camera never blinds the zone", regenerated on every save +
    ensure_twins_current on calibration change; edge-densified 8 samples/edge
    because "a straight edge in one camera is CURVED after the floor
    round-trip"; undistortion-divergence + lens-fold guards). Zone workers
    assign each camera's detections to base-or-twin by METRIC foot membership
    (zone_worker.py:331-335).
  - OCCLUSION PATH (claim (d) consequence CONFIRMED): per-camera streams →
    §2.5 metric fusion merges when both see; unmatched remain first-class
    single-camera observations → tracks + ZoneState survive one blocked view.
  - MASK VERDICT — AUTHOR CLAIM (d) "twins also get masks projected" is
    IMPRECISE: masks are NOT geometrically projected cross-camera. Each
    camera's detector produces its own crop-relative masks
    (decode_masks/mask_offset_xy; polygonized per camera into its own
    ObservationsMessage.mask_poly). What IS projected across cameras is the
    ZONE GEOMETRY (outline/twin polygon). The added text states exactly this
    ("Segmentation masks are not transferred between cameras..."). FLAGGED
    for the author.
  - 320 px crop path: already documented, unchanged.
  TEXT CHANGES (all in the author's voice/style):
  1. §2.4 zone-scope config paragraph EXTENDED with the declaration workflow
     (either-camera/map authoring, back-projection, metric single source of
     truth, outline shown on both feeds).
  2. §2.4 NEW run-in paragraph "Cross-camera zone twins." (mechanism → 
     occlusion consequence → dashboard twin derivation → exact mask
     statement).
  3. §2.5 fusion sentence: minimal clause "— including the two independent
     views of a twinned zone (Section 2.4) —" inserted into the author's
     verbatim text to close the loop (FLAGGED as an insertion into his §2.5).

§2.1 EXPANDED into per-architecture subsections 2.1.1–2.1.8 (2026-07-31,
  author request: "more technical grounding on every AI architecture used";
  15-page cap explicitly waived for now — shrink pass later).
  - Structure: condensed lead-in → 2.1.1 YOLO26 → 2.1.2 RF-DETR → 2.1.3 SDXL
    → 2.1.4 ControlNet → 2.1.5 LoRA → 2.1.6 SAM2 → 2.1.7 MQTT/MqttSink →
    2.1.8 Tracking/Geometry/Deployment (the three prior paragraphs preserved
    verbatim except "must then be" → "must be" in Tracking) → closing
    pipeline-overview paragraph unchanged. The prior "Vision inference" and
    "Synthetic training data" paragraphs are superseded by 2.1.1–2.1.2 and
    2.1.3–2.1.6; every claim and citation they carried survives in the
    subsections (orphan check: [12]–[20], [22]–[28] all still cited in §2.1).
  - New citations [29] YOLACT, [30] Vaswani, [31] Deformable DETR, [32] DDPM,
    [33] Hiera, [34] MQTT v5.0 OASIS — all verified 2026-07-31 (arXiv
    abstracts / OASIS page fetched; details in references.md). Existing
    numbering untouched; global renumber deferred to final assembly.
  - Repo verification (this session) of every project-specific number:
    * yolo26l-seg 640 px 0.977/0.948 and yolo26n-seg 320 px 0.962/0.895 —
      MANUSCRIPT Table T1 (unchanged home of the numbers);
      trainer/isidet/configs/train_pallet3_seg_yolo26.yaml exists
      (export_nms: false, export_opset: 17; on-disk instance currently set to
      the nano weights at imgsz 320).
    * Prototype/coefficient seg decode: backbone/detection/postprocess.py
      (protos (nm,mh,mw) + per-detection mask coeffs).
    * RF-DETR: 432 px fixed square, outputs dets/labels/masks, NMS-free —
      backbone/detection/rfdetr_onnx_seg.py docstring + output-name mapping;
      0.973/0.938 best-EMA epoch 23/41 — Table T1.
    * SDXL id stabilityai/stable-diffusion-xl-base-1.0 + ControlNet id
      diffusers/controlnet-depth-sdxl-1.0 — generation/sdxl_{controlnet,
      inpaint}.py, lora/diffusers_sdxl.py, configs/project_template.yaml.
    * Composite depth control maps — scaffolds/copy_paste.py ("control:
      composite depth (bg depth + pasted object's depth)").
    * LoRA rank 16 / 768 px / 2000 steps / lr 1e-4 / batch 1×4 / 53 photos —
      §2.6 (already verified); final loss 0.127 NEW in §2.1.5, from
      runs/lora/black_polybag_r16_18-06-2026_13-24-47/report.md ("final loss
      (mean of last 100): 0.1270").
    * SAM2: facebook/sam2.1-hiera-small ~185 MB, SAM2ImagePredictor, box
      prompts + SAM2AutomaticMaskGenerator promptless path —
      trainer/isiGen/src/stages/masking/sam2_masker.py.
    * MqttSink: one paho client + background loop thread, 1–30 s backoff,
      per-class topics {prefix}/track2d/{cls} (O(classes) fan-out, track_id
      in payload), zone state retained QoS 1 (zone_state_qos=1), failures
      logged and swallowed, registered beside udp in metadata_sink_registry —
      backbone/comms/mqtt_sink.py; deployed config runs BOTH sinks
      (config/backbone.yaml metadata.sinks: udp 9001 + mqtt 1883).
    * YOLO26 claims (NMS-free end-to-end, DFL removal, up-to-43%-faster CPU
      ONNX for nano vs YOLO11, -seg variants) — Ultralytics YOLO26 docs
      fetched 2026-07-31; attributed to [14].
    * RF-DETR claims limited to its abstract (pretrained base network,
      weight-sharing NAS, transferability to diverse target domains) —
      DINOv2-backbone claim deliberately NOT asserted (not in the abstract).
  - New forward references §2.1 → Section 3.1/Table T1 (detector accuracies)
    and §2.1.5 → Section 3.4 (ablation): grounding paragraphs cite measured
    values where they live; no numbers duplicated inconsistently.

§2.1 REVIEW APPLIED (02.1_arch.REVIEW.md, 2026-07-31 — verdict "minor
  revision, two mandatory harmonizations"; all M1–M4 + m1–m12 applied):
  - M1 RESOLVED BY REPO EVIDENCE — the benchmarked/deployed export ran the
    END-TO-END (NMS-free) decode path. Proof: the G4 model
    (trainer/isidet/runs/segment/models/yolo/yolo26n-seg_e100_320px_
    03-07-2026_15-09-28/weights/best.fp16.onnx) was run on CPU EP this
    session; output0 = (1, 300, 38) = (num_det, 6+nm) — the end-to-end head.
    decode_yolo11_seg dispatches on head.shape[1]==6+nm → _decode_seg_end2end,
    whose docstring/code contain NO NMS (confidence/class filter + letterbox
    inversion + mask assembly only; the config's iou 0.45 is ignored on this
    path). Therefore §2.1.1 keeps NMS-free as an exploited property; T5's
    method note (§3.3) and §4.1 were WRONG to bill "NMS" and now say
    "detection decoding / mask assembly" (fixed in 03_results.md,
    04_discussion.md, MANUSCRIPT). The archived G4_trt_vs_cuda.md record still
    says "NMS" in its labels — record left untouched as a frozen measurement
    artifact; this note is the correction of record. §2.1.1's "pure tensor
    program" sentence REWRITTEN per the reviewer: the raw-head export never
    contained NMS in any generation; the honest benefit is removing the
    application-side suppression stage.
  - M2 APPLIED (option a — pin as-measured, no re-measure): §2.1.8 Deployment
    and §2.4 now date the TRT-EP description ("as-measured configuration of
    this article (July 2026)", past tense); portability sentence reworded
    around the artifact (exchanged file = .onnx; engines = locally derived
    accelerator caches). The 2026-07-23 native-.engine migration is NOT
    described in the paper; T4/T5 untouched.
  - M3 APPLIED: §2.1.5 grounding reduced to rank-16 + few-dozen photographs +
    pointer to §2.6 (full hyperparameters live ONLY in §2.6); "final loss
    0.127" DROPPED (not comparable across runs; the adapter's evaluation is
    the §3.4 ablation).
  - M4 APPLIED: §2.6 Canny sentence corrected — only DepthAnythingV2 depth
    maps condition generation (sdxl_controlnet.py loads only
    diffusers/controlnet-depth-sdxl-1.0; no canny consumer in
    scaffolds/generation code); Canny stated as extracted-but-not-conditioning.
  - Minors: m1 six research areas (added machine-consumable data delivery);
    m2 MQTT 3.1.1 stated (paho Client default MQTTv311; no protocol= override
    in MqttSink), [34] kept with both-versions clause, last-will DROPPED from
    the feature list (no will_set in MqttSink); m3 "topic cardinality";
    m4 "thresholded sigmoid of a linear combination"; m5 "encoder and middle
    blocks"; m6 α/r scaling added to all three LoRA equations + small-data
    sentence recast as system rationale ("in the few-dozen-image regime of
    this work"), not a [20] claim; m7 "In the denoising-diffusion formulation
    [32]"; m8 "a single camera with a metrically anchored floor homography";
    m9 43 % figure kept vendor-attributed ("the release documentation
    reports"); m10 §2.1.3 opening compressed to one sentence (sampling-
    mechanics sentence dropped), §2.1.7 trimmed via the last-will cut;
    m11 private-repo file paths removed from §2.1 reader-facing text
    (train_pallet3_seg_yolo26.yaml, mqtt_sink.py, sam2_masker.py,
    control_maps/, scaffolds/ — all retained in THIS comment for
    traceability); m12 [29] bibliography annotation aligned with the M1
    resolution (both references.md and the manuscript bibliography).
-->
