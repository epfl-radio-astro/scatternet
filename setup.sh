module load daint-gpu
module load TensorFlow
export PYTHONPATH="$PWD"
export PYTHONPATH="/usr/local/opt/sparse2d/python:$PYTHONPATH"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt