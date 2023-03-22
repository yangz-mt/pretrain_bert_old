import os
import logging
import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Logger:
    def __init__(self, log_path, local_rank, cuda=False, debug=False):
        self.logger = logging.getLogger(__name__)
        self.cuda = cuda
        self.local_rank = local_rank
        self.log_path = log_path
        self.debug = debug

    def info(self, message, log_=True, print_=True, *args, **kwargs):
        if (self.cuda and self.local_rank in [0, -1]) or not self.cuda:
            if print_:
                self.logger.info(message, *args, **kwargs)

            if log_:
                with open(self.log_path, "a+") as f_log:
                    f_log.write(message + "\n")

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)
