# Converting CSV to OBP
The different optimization examples in this directoy generate CSV files of spot scan locations and dwell times. In order to run these sequences on the Freemelt ONE, they must be converted.

First, install a virtual environment with all of the necessary dependencies. This can be done by running
```
bash examples/ebpbf/setup_pyenv.sh
```

Next, activate the environment and run the python script for conversion
```
source venv/bin/activate
python examples/ebpbf/ebamareaprint.py [CSV to process]
```