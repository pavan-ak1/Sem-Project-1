# NephroAI: Deep Learning Renal Lesion Segmentation & Clinical Decision Support System

An end-to-end medical AI platform for automated segmentation of kidney parenchyma, renal cell carcinoma (RCC) tumors, and cysts from abdominal CT scans. The system couples a **4-stage 2D Residual U-Net** trained on the **KiTS23 benchmark** with an automated **clinical analytics engine** (AJCC TNM staging, 3D PCA caliper morphometry) and an interactive **React-based radiological web cockpit**.

---

## Key Highlights

- **Multi-Class CT Segmentation**: Simultaneously segments **Healthy Kidney Parenchyma**, **Renal Tumors/Neoplasms**, and **Benign Renal Cysts** across patient CT slice series.
- **Clinically Validated Architecture**: 4-stage Medical Residual U-Net with `InstanceNorm2d` and `PReLU` activations (~8.11M parameters), achieving **75.67% Foreground Macro Dice** on held-out test data.
- **Anatomical Post-Processing Engine**: 
  - 3D Connected Component Analysis (CCA) morphological filtering.
  - Kidney-absence rule eliminating false-positive lesion noise on non-renal slices.
  - Exophytic tumor preservation envelope safeguarding large outward retroperitoneal tumor margins.
- **Automated Clinical Staging & Calipers**:
  - 3D Principal Component Analysis (PCA) orthogonal caliper measurements (Length $\times$ Width $\times$ Thickness in cm).
  - AJCC/UICC 8th Edition clinical **TNM tumor staging** (cT0 to cT3a) and nephron-sparing surgery (partial vs. radical nephrectomy) recommendations.
  - Quantitative slice-by-slice surface area measurements ($\text{cm}^2$) for kidney, tumor, and cyst structures.
- **High-Performance Full-Stack Delivery**:
  - **FastAPI backend** with tiered caching (in-memory + disk cache) and PyTorch mixed precision (CUDA AMP and CPU bfloat16).
  - **React 18 + Vite dashboard** featuring zero-latency slice scrubbing, opacity controls, layer toggling, split views, and slice summary CSV exports.
  - **Docker Compose deployment** supporting both GPU acceleration and multi-core CPU inference.

---

## Model Benchmark Performance

Evaluated on the held-out **KiTS23 (Kidney Tumor Segmentation)** benchmark test cohort:

| Anatomical Structure | Class Index | Dice Similarity Coefficient (DSC) | Precision (PPV) | Recall (Sensitivity) | Color Indicator |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Background** | `0` | **99.93%** | 99.89% | 99.96% | Transparent |
| **Kidney Parenchyma** | `1` | **94.55%** | 95.23% | 93.88% | 🟢 Emerald Green (`#10B981`) |
| **Renal Tumor** | `2` | **72.33%** | 83.73% | 63.67% | 🔴 Crimson Red (`#EF4444`) |
| **Renal Cyst** | `3` | **60.13%** | 60.38% | 59.89% | 🔵 Cerulean Blue (`#3B82F6`) |
| **Foreground Macro Average** | `1-3` | **75.67%** | **79.78%** | **72.48%** | — |

* **Final Model Weights**: `notebookV3/checkpoints/best_unet_kits23.pth`
* **Test Loss**: `0.0737` (Class-Weighted Focal + Asymmetric Tversky Loss)

---

## Clinical Workflow Pipeline

```
Abdominal CT Slices (DICOM / PNG)
                   │
                   ▼
       [Preprocessing & Normalization]
  Center-Crop / Pad to (512, 512) ➔ Intensity Normalization
                   │
                   ▼
     [Residual U-Net Deep Inference]
   4-Stage ResNet-like Encoder-Decoder (~8.11M params)
   Optional Test-Time Augmentation (4-Fold TTA)
                   │
                   ▼
    [Morphological & Anatomical Cleaning]
   Kidney-Absence Filtering ➔ 3D Connected Component Analysis
   Exophytic Boundary Dilation ➔ Noise Suppression
                   │
                   ▼
       [Automated Clinical Analytics]
   ┌───────────────────────┴────────────────────────┐
   ▼                                                ▼
3D PCA Calipers (L × W × T)                  AJCC 8th Ed.
Slice Surface Areas (cm²)                    TNM Tumor Staging
Tumor Burden Ratio (%)                       & Surgical Recommendations
                   │
                   ▼
     [Radiological Web Interface]
  Instant Interactive Slice Scrubbing & CSV Export
```

---

## Repository Structure

```text
Sem-Project-1/
├── backend/                             # FastAPI Deep Learning Engine
│   ├── app.py                          # REST API routes & static mount
│   ├── clinical_analytics.py           # 3D PCA calipers & TNM clinical staging
│   ├── inference_engine.py             # Inference pipeline, TTA, CCA, caching
│   ├── model_loader.py                 # Residual U-Net PyTorch architecture
│   ├── schemas.py                      # Pydantic data models & request schemas
│   ├── slice_summary.py                # Per-slice connected component analytics
│   ├── benchmark_inference.py          # Batch size & hardware speed benchmark CLI
│   ├── Dockerfile                      # Production backend container image
│   └── tests/                          # Backend unit & integration tests
│
├── frontend/                            # React 18 + Vite Web Application
│   ├── src/
│   │   ├── App.jsx                     # Primary UI, canvas viewer, study loader
│   │   ├── components/
│   │   │   ├── SliceAnalysisPanel.jsx  # Per-slice metrics & layer controls
│   │   │   └── ScanWideOverview.jsx    # Clinical staging & slice distribution chart
│   │   └── index.css                   # Medical dashboard theme styling
│   ├── package.json                    # Frontend dependencies
│   ├── vite.config.js                  # Vite server & backend proxy configuration
│   └── Dockerfile                      # Frontend preview container image
│
├── notebookV3/                          # Jupyter Research & Exploration Pipeline
│   ├── 01_dataset_analysis.ipynb       # KiTS23 cohort exploration & distribution
│   ├── 02_preprocessing_and_augmentation.ipynb # CT normalization & augmentation
│   ├── 03_model_architecture_and_training.ipynb # Residual U-Net training pipeline
│   ├── 04_volumetric_analysis_and_clinical_metrics.ipynb # Exploratory staging & metrics notebook
│   └── checkpoints/
│       └── best_unet_kits23.pth        # Final trained model checkpoint (8.11M params)
│
├── reports/                             # Runtime Caches & Analytical Outputs
│   ├── case_bundles/                   # Precomputed JSON bundles for instant UI loading
│   └── slice_summaries/                # Exported per-case slice summary CSVs
│
├── uploaded_cases/                      # Storage for user/clinician uploaded CT studies
├── docker-compose.yml                   # Multi-container stack (NVIDIA GPU mode)
├── docker-compose.cpu.yml               # Multi-container stack (CPU fallback mode)
├── requirements.txt                     # Host Python dependencies
├── requirements-docker.txt              # Containerized Python dependencies
├── run_command.txt                      # Command cheat-sheet for running all services
└── README.md                            # Project documentation
```

---

## Getting Started

### Prerequisites

- **Python**: 3.11 or higher
- **Node.js**: 20.x or higher (for local frontend development)
- **Docker & Docker Compose** (optional, recommended for isolated deployment)
- **NVIDIA GPU** with CUDA support *(optional; CPU fallback is fully supported with bfloat16)*

---

### Option 1: Docker Deployment (Recommended)

#### A. GPU Mode (NVIDIA GPU with Container Toolkit)
```bash
# Build and start services in the background
docker compose up --build -d

# Check service health
docker compose ps
docker compose logs -f backend
```

#### B. CPU Mode (Standard Multi-Core CPU)
```bash
# Start CPU-optimized stack with bfloat16 inference
docker compose -f docker-compose.cpu.yml up --build -d
```

Once running, access:
- **Clinical Frontend Cockpit**: [http://localhost:3000](http://localhost:3000)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

To stop services:
```bash
docker compose down
# or for CPU mode:
docker compose -f docker-compose.cpu.yml down
```

---

### Option 2: Local Development Setup

#### 1. Setup Python Virtual Environment
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r requirements.txt
```

#### 2. Launch FastAPI Backend
```powershell
.\venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

#### 3. Launch React Frontend
In a new terminal:
```powershell
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

### Option 3: Unified Single-Port Production Serving

To serve the pre-built React application directly from FastAPI on a single port:
```powershell
# 1. Build production static bundle
cd frontend
npm run build
cd ..

# 2. Run backend (automatically mounts and serves frontend/dist on root '/')
.\venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) to access the complete application.

---

## REST API Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | System health check, device verification (CUDA vs. CPU), and model status |
| `GET` | `/api/cases` | Lists available test cohort cases and pre-computed patient studies |
| `GET` | `/api/case/{case_id}/full` | Returns complete multi-slice bundle (images, overlays, masks, 3D metrics) |
| `GET` | `/api/case/{case_id}/slices` | Returns slice index list and metadata for a patient case |
| `GET` | `/api/case/{case_id}/slice/{slice_idx}` | Returns single-slice image, AI overlay, mask, and area metrics |
| `GET` | `/api/case/{case_id}/slice-summary.csv` | Downloads tabular per-slice morphological analytics CSV |
| `POST` | `/api/predict/slice` | Runs live inference on a single uploaded CT slice image |
| `POST` | `/api/predict/case-upload` | Runs full 3D pipeline on an uploaded patient case folder / slice collection |
| `GET` | `/docs` | Interactive Swagger UI API documentation and testing playground |

---

## Running Automated Tests

To run the backend test suite (testing kidney-absence logic, slice summaries, and morphological filtering):

```powershell
.\venv\Scripts\python.exe -m pytest backend/tests
```

---

## Research & Exploration Notebooks

The research and experimentation notebooks located in `notebookV3/` document the full training and evaluation methodology:

1. **`01_dataset_analysis.ipynb`**: Analyzes the KiTS23 dataset distributions, slice thicknesses, and voxel spacing.
2. **`02_preprocessing_and_augmentation.ipynb`**: Demonstrates intensity windowing, resampling, and spatial augmentations.
3. **`03_model_architecture_and_training.ipynb`**: Defines the Residual U-Net architecture, loss functions, and optimization routines.
4. **`04_volumetric_analysis_and_clinical_metrics.ipynb`**: Exploratory research notebook evaluating post-processing routines, connected component analysis, and clinical staging concepts.

---

## License & Acknowledgments

- **Dataset**: Developed using data from the [KiTS23 Kidney Tumor Segmentation Challenge](https://kits-challenge.org/kits23/).
- **Clinical Staging Guidelines**: Based on the *AJCC Cancer Staging Manual (8th Edition)* for Renal Cell Carcinoma.
