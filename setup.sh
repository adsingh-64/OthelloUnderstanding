python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
git submodule update --init --recursive
cd /tmp
wget https://nodejs.org/dist/v20.19.0/node-v20.19.0-linux-x64.tar.xz
tar -xf node-v20.19.0-linux-x64.tar.xz
sudo cp -r node-v20.19.0-linux-x64/* /usr/local/
rm -rf node-v20.19.0-linux-x64*
npm install -g @anthropic-ai/claude-code