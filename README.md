# DeepResearch-9K

This repository contains the dataset and codebase for the DeepResearch-9K project. All environment configuration files are stored in the `env/` directory.

---

## 📊 Environment Overview

| Environment | Python | Key Features | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **react_infer_env** | 3.10 | OpenAI SDK, Data processing | Inference & Data Modification |
| **search** | 3.9 | vLLM, Verl, Flash-Attn 2 | Model Training & RL Tasks |
| **retrieval** | 3.10 | Faiss-GPU, Pyserini, FastAPI | Vector Search & Knowledge Retrieval |

---

## 🛠 1. Inference Environment (`react_infer_env`)
**Purpose:** Optimized for running basic inference and large-scale data processing/modification scripts.

```
# Create and activate environment
conda create -n react_infer_env python=3.10.0 -y
conda activate react_infer_env

# Install dependencies from the env folder
conda install --file env/react_infer_requirements.txt

```
## 🚀 2. Search & Training Environment (searchr1)
**Purpose:** Designed for model training (Verl), Reinforcement Learning (RL) tasks, and high-performance inference via vLLM.

```

# Create and activate environment
conda create -n searchr1 python=3.9 -y
conda activate searchr1

# Install PyTorch and vLLM
pip install torch==2.4.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install vllm==0.6.3

# Install Verl Framework & Flash Attention 2
pip install -e .
pip install flash-attn --no-build-isolation
pip install wandb

```
## 🔍 3. Retriever Environment (retriever)
**Purpose:** Specialized for knowledge retrieval, vector database management, and hosting API services.
```

# Create and activate environment
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch and CUDA (Conda recommended for Faiss-GPU compatibility)
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Faiss-GPU to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# Install additional retrieval components
pip install transformers datasets pyserini uvicorn fastapi

```
## 📊 Dataset & Rollouts

You can access the complete dataset and model rollouts on Hugging Face. We provide two versions based on evaluation results:

* **Full Dataset**: [artillerywu/DeepResearch-9K](https://huggingface.co/datasets/artillerywu/DeepResearch-9K)
    * Contains **9,000 high-quality samples** covering three difficulty levels.
* **Hard Subset**: [artillerywu/DeepResearch-Hard](https://huggingface.co/datasets/artillerywu/DeepResearch-Hard)
    * A curated subset of **3,974 challenging samples** (filtered by `INCORRECT` verdicts).
      
Note: After downloading the dataset files, please place them in the data/ directory of the project root

### Data Format
Each sample follows a standardized structure for seamless integration with SFT scripts:
* `question`: The initial user query.
* `difficulty`: Difficulty level (1-3).
* `search trajectory`: Full reasoning and tool-use rollouts.
* `final answer`: The definitive response enclosed within `<answer></answer>` tags.

---

## 🚀 Supervised Fine-Tuning (SFT)

We provide optimized scripts for supervised fine-tuning on 3B-parameter base models.

### 1. Base Models
The scripts are compatible with the following models:
* **Qwen2.5-3B**: [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)
* **Llama-3.2-3B**: [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)

### 2. Training Commands
Launch the training process using the following scripts:
* **Llama 3.2**: `python sft_llama3b.py`
* **Qwen 2.5**: `python sft_qwen3b.py`
