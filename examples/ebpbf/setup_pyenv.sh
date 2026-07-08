python -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install grpcio-tools
pip install --force-reinstall "setuptools==69.5.1"
pip install --no-build-isolation obplib
pip install numpy
deactivate