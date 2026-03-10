import logging
import os

def get_logger():

    log_dir = "experiment/experiment_logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "experiment_log.txt")

    logger = logging.getLogger("experiment_logger")

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(message)s")

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger