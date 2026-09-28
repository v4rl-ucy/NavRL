#!/bin/bash

# Exit immediately if a command fails
set -e

# Define environment name
ENV_NAME="NavRL"

# Load Conda environment handling
eval "$(conda shell.bash hook)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Step 1: Create conda env with python3.10
echo "Setting up conda env..."
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "Removing existing ${ENV_NAME} environment..."
    conda env remove -n "$ENV_NAME" -y
fi
conda create -n $ENV_NAME python=3.10 -c conda-forge -y
conda activate "$ENV_NAME"
python -m pip install numpy==1.26.4

TORCH_INSTALL="https://developer.download.nvidia.cn/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
python -m pip install --no-cache-dir "$TORCH_INSTALL"

#pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
python -m pip install "pydantic!=1.7,!=1.7.1,!=1.7.2,!=1.7.3,!=1.8,!=1.8.1,<2.0.0,>=1.6.2"
python -m pip install imageio-ffmpeg==0.4.9
python -m pip install moviepy==1.0.3
python -m pip install hydra-core --upgrade
python -m pip install einops
python -m pip install pyyaml
python -m pip install rospkg
python -m pip install matplotlib

# Step 2: Install TensorDict and dependencies
echo "Installing TensorDict dependencies..."
python -m pip uninstall -y tensordict
python -m pip uninstall -y tensordict
python -m pip install tomli  # If missing 'tomli'
#cd ./third_party/tensordict
python -m pip install --no-deps --no-build-isolation -e "$SCRIPT_DIR/third_party/tensordict"
# Step 3: Install TorchRL
echo "Installing TorchRL..."
#cd ../rl
python -m pip install --no-deps --no-build-isolation -e "$SCRIPT_DIR/third_party/rl"

# Check which torch is being used
#python -c "import torch; print(torch.__path__)"
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

echo "Setup completed successfully!"
