# prepare python
# python3 -m venv $SCRATCH/envs/
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
# pip install torch
# pin exact versions
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade transformers
pip install --upgrade pandas
pip install --upgrade huggingface_hub
pip install python-dotenv
pip install accelerate
pip install protobuf
pip install pillow
pip install kagglehub
pip install datasets
pip install matplotlib
pip install seaborn
pip install tensorflow
pip install scikit-learn
pip install --index-url https://pypi.clarin-pl.eu/ lambo
