curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
git submodule update --init --recursive
apt update
apt install zip