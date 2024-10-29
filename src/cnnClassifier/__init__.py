# When you run setup.py to install your package (especially in editable mode), setup.py identifies which directories should be treated as packages. If a directory contains an __init__.py file, it will be included in the installation as a package.
# This means that setup.py relies on __init__.py to define the structure of your package and determine what will be available for import when users install your package.

import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout) # to print the messages in the terminal too.
    ]
)

logger = logging.getLogger("cnnClassifierLogger") # name of the logger.

# In Python, if a module (like cnnClassifier) has variables, functions, or classes defined in it, they become part of that module's namespace. So, if you define logger in __init__.py, it's available for import as part of the cnnClassifier module's namespace.