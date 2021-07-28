import logging
import random
import os
import numpy as np
import torch

# def set_logger(log_path):
#     """
#     Set the logger to log info in terminal and file `log_path`.
#     Args:
#         log_path: (string) where to log
#     """
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)

#     if not logger.handlers:
#         # Logging to a file
#         file_handler = logging.FileHandler(log_path)
#         file_handler.setFormatter(
#             logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
#         logger.addHandler(file_handler)
        
def setup_logger(logger_name, log_file, level=logging.INFO):
    """
    ...
    """
    l = logging.getLogger(logger_name)
    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))

    #streamHandler = logging.StreamHandler()

    l.setLevel(level)
    l.addHandler(fileHandler)
    #l.addHandler(streamHandler)    
    
def seed_everything(seed=1234):
    """
    Ensure reproducible results
    
    Args
        seed: random seed (an integer)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
