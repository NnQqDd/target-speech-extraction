# Audio-Visual Target Speech Extraction

This repository implements **Target Speech Extraction (TSE)** using the [SepReformer architecture](https://github.com/dmlguq456/SepReformer).


* **Python version:** 3.10+ recommended. Ubuntu preferred.
* **GPU:** Ensure CUDA-compatible GPU and drivers installed with at least 12GB VRAM per GPU.
* **Logs & Weights:** Stored in `wandb` and `weights` directories; consider mounting them for persistence in Docker.


---

## 1. Virtual Environment

**Create and activate the environment:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Install dependencies:**

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Deactivate when done:**

```bash
deactivate
```

---

## 2. Docker Setup

**Build the Docker image:**

```bash
docker build -t avtse:latest .
```

**Run container with GPU access:**

```bash
docker run -it --gpus all --name avtse avtse:latest
```

**Subsequent runs:**

```bash
docker start avtse
docker exec -it avtse /bin/bash
```

**Copy files from container to host:**

```bash
docker cp avtse:/app/weights/<run_id>/best.pth ./best.pth
```

**Optional - mount host directories for persistence:**

```bash
docker run -it --gpus all \
  -v ./weights:/app/weights \
  -v ./wandb:/app/wandb \
  --name avtse avtse:latest /bin/bash
```

---

## 3. Prepare Dataset

Download **VoxCeleb2** dataset from [Kaggle](https://www.kaggle.com/datasets/e1lephant/voxceleb2).

Generate CSV for training:

```bash
python prepare_VoxCeleb2.py --input /path/to/dataset --output VoxCeleb2.csv
```

---

## 4. Training

**List latest checkpoints:**

```bash
python list.py
```

**Start training with defaults:**

```bash
python train.py
```

**With custom config/checkpoint:**

```bash
python train.py --config config.yaml --ckpt /path/to/checkpoint/epoch_16.pth
```

**Disable WandB, specify CUDA devices, and log output:**

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python train.py | tee -a logs.txt
```

