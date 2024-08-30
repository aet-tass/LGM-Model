# Large Multi-View Gaussian Model (LGM)

This is the official implementation of *LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation*.

### [Project Page](https://me.kiui.moe/lgm/) | [Arxiv](https://arxiv.org/abs/2402.05054) | [Weights](https://huggingface.co/ashawkey/LGM) | <a href="https://huggingface.co/spaces/ashawkey/LGM"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a>

### Overview

The Large Multi-View Gaussian Model (LGM) is designed for high-resolution 3D content creation by leveraging multi-view Gaussian representations. The model generates detailed 3D content from multiple views by using a deep neural network architecture that integrates Gaussian splatting techniques with novel rendering methods. This implementation allows for efficient training and inference, providing tools for generating, visualizing, and converting 3D content.

### Installation

To set up the environment for LGM, follow these steps:

```bash
# xformers is required! please refer to https://github.com/facebookresearch/xformers for details.
# for example, we use torch 2.1.0 + cuda 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# for mesh extraction
pip install git+https://github.com/NVlabs/nvdiffrast

# other dependencies
pip install -r requirements.txt


Pretrained Weights
Pretrained weights are available for download on Hugging Face.

To download the fp16 model for inference:

mkdir pretrained && cd pretrained
wget https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors
cd ..


For MVDream and ImageDream, weights are automatically downloaded through a diffusers implementation.

Inference
Inference requires approximately 10GB of GPU memory to load all required models.

### Gradio app for text/image to 3D
python app.py big --resume pretrained/model_fp16.safetensors

### Test
# --workspace: folder to save output (*.ply and *.mp4)
# --test_path: path to a folder containing images or a single image
python infer.py big --resume pretrained/model_fp16.safetensors --workspace workspace_test --test_path data_test 

### Local GUI to visualize saved PLY files
python gui.py big --output_size 800 --test_path workspace_test/saved.ply

### Mesh conversion
python convert.py big --test_path workspace_test/saved.ply


For more options, please check options.

Training
NOTE: Since the dataset used in our training is based on AWS, it cannot be directly used for training in a new environment. We provide the necessary training code framework, please check and modify the dataset implementation!

We also provide the ~80K subset of Objaverse used to train LGM in objaverse_filter.

# Debug training
accelerate launch --config_file acc_configs/gpu1.yaml main.py big --workspace workspace_debug

# Training (use slurm for multi-nodes training)
accelerate launch --config_file acc_configs/gpu8.yaml main.py big --workspace workspace

Team
This project is an internship at 3D Smart Factory. The team members are:

AYOUB ET-TASS GitHub
Elhaimer Salma GitHub
Choukhantri Ikram GitHub

