# -*- coding: utf-8 -*-

import logging
import sys

from loguru import logger


class InterceptHandler(logging.Handler):
    """ Intercept logging messages and reroute them to the loguru. """

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        (logger.opt(depth=depth, exception=record.exc_info)
               .log(level, record.getMessage()))


logging.basicConfig(handlers=[InterceptHandler()], level=0)


# Put together a formatting string for the logger.
# Split into pieces to improve legibility.
tim_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
lvl_fmt = "<level>{level: <8}</level>"
src_fmt = "<cyan>{module}</cyan>:<cyan>{function}</cyan>"
msg_fmt = "<level>{message}</level>"

fmt = " | ".join([tim_fmt, lvl_fmt, src_fmt, msg_fmt])

config = {
    "handlers": [
        {"sink": sys.stderr,
         "level": "INFO",
         "format": fmt},
        # {"sink": "{time:YYYYMMDD_HHmmss}_crystalball.log",
        #  "level": "DEBUG",
        #  "format": fmt,
        #  }
    ],
}

logger.configure(**config)
