import logging
from logging import getLogger, Formatter, StreamHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL
from typing import Text

class Logger:
    def __init__(self, level: Text='debug'):
        fmt = '%(asctime)s [%(levelname)s]: %(message)s'
        formatter = Formatter(fmt) # this is better than basicConfig
        #logging.basicConfig(format=fmt)
        if level.upper() == 'DEBUG':
            lvl = DEBUG
        elif level.upper() == 'INFO':
            lvl = INFO
        elif level.upper() == 'WARNING':
            lvl = WARNING
        elif level.upper() == 'ERROR':
            lvl = ERROR
        elif level.upper() == 'CRITICAL':
            lvl = CRITICAL
        self.logger = getLogger(__name__)
        self.handler = StreamHandler()
        self.handler.setFormatter(formatter)
        self.handler.setLevel(lvl)
        self.logger.setLevel(lvl)
        self.logger.addHandler(self.handler)

    def debug(self, msg: Text):
        self.logger.debug(msg)

    def info(self, msg: Text):
        self.logger.info(msg)

    def warning(self, msg: Text):
        self.logger.warning(msg)

    def error(self, msg: Text):
        self.logger.error(msg)

    def critical(self, msg: Text):
        self.logger.critical(msg)

    def exception(self, msg: Text):
        self.logger.exception(msg)

    def log(self, msg: Text):
        self.logger.log(msg)

