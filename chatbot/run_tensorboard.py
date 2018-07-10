import os
import sys
sys.path.insert(0, os.getcwd())
from chatbot.config import PATHS
import subprocess

run_tensorboard_command = 'tensorboard --port 22222 --logdir {}'.format(PATHS['log_dir'])
activate_conda_command = 'source activate tensorflow'
p = subprocess.Popen(f"{activate_conda_command}; {run_tensorboard_command}",
                     stdout=subprocess.PIPE, shell=True, bufsize=1, executable="/bin/bash")
for line in iter(p.stdout.readline, b''):
    print(line),
p.stdout.close()
p.wait()
