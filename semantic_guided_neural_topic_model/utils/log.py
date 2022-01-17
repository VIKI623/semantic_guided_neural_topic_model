import logging
from logging import config
from datetime import datetime
from os.path import join, abspath, dirname


def get_today():
    return datetime.now().strftime('%Y-%m-%d')


LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'verbose': {
            'format': "[%(asctime)s] %(levelname)s %(process)d [%(filename)s::%(funcName)s:%(lineno)s] %(message)s",
            'datefmt': "%Y-%m-%d %H:%M:%S"
        },
        'simple': {
            'format': '%(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose'
        },
        'file': {
            'level': 'INFO',
            'class': 'concurrent_log_handler.ConcurrentRotatingFileHandler',
            'maxBytes': 1024 * 1024 * 10,
            'backupCount': 50,
            'delay': True,
            'filename': f'{join(dirname(abspath(__file__)), "..", f"logs/{get_today()}.log")}',
            'formatter': 'verbose'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
        },
    }
}

config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("hello")
