#!/bin/bash
set -e

ENV_NAME="NavRL"

eval "$(conda shell.bash hook)"

echo "Setting up conda env..."

if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "Environment ${ENV_NAME} already exists, removing it first..."
    conda env remove -n "$ENV_NAME" -y
fi

conda create -n $ENV_NAME python=3.10 -c conda-forge -y
conda activate $ENV_NAME

python -m pip install --no-cache-dir "setuptools==69.5.1"
python -m pip install numpy==1.26.4
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
python -m pip install "pydantic!=1.7,!=1.7.1,!=1.7.2,!=1.7.3,!=1.8,!=1.8.1,<2.0.0,>=1.6.2"
python -m pip install imageio-ffmpeg==0.4.9
python -m pip install moviepy==1.0.3
python -m pip install hydra-core --upgrade
python -m pip install einops
python -m pip install pyyaml
python -m pip install rospkg
python -m pip install matplotlib

echo "Installing TensorDict dependencies..."
python -m pip uninstall -y tensordict || true
python -m pip uninstall -y tensordict || true
python -m pip install tomli

cd ./third_party/tensordict
python -m pip install --no-deps --no-build-isolation -e .

echo "Installing TorchRL..."
cd ../rl
python -m pip install --no-deps --no-build-isolation -e .

python -c "import torch; print(torch.__path__)"
echo "Setup completed successfully!"
