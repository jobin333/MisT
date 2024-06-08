import logging

## Configurations
logger_configs_ = {
    'file_logging_level': logging.DEBUG,
    'console_logging_level': logging.DEBUG,
    'logging_file': 'mvit_debugging.log'
}

def get_logger_():

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(logger_configs_['logging_file'], 'w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logger_configs_['file_logging_level'])

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logger_configs_['console_logging_level'])

    logger = logging.getLogger('MVIT')
    logger.addHandler(file_handler)
    logger.setLevel(logger_configs_['file_logging_level'])
    logger.addHandler(console_handler)
    return logger


logger = get_logger_()