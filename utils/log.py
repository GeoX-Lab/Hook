import logging
import os


class MY_Logger(object):
    
    def __init__(self, workdir):
        self.workdir = workdir
        
        # logger
        formatter = logging.Formatter("%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(workdir)
        self.logger.setLevel(logging.DEBUG)
        
        fileHandler = logging.FileHandler(os.path.join(workdir, 'log.log'))
        fileHandler.setFormatter(formatter)
        self.logger.addHandler(fileHandler)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        self.logger.addHandler(streamHandler)