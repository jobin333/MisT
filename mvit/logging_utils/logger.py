import logging

## Configurations
logger_configs_ = {
    'logging_level': logging.DEBUG,
    'logging_file': 'mvit_debugging.log'
}

def get_logger_():

    file_handler = logging.FileHandler(logger_configs_['logging_file'], 'w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logger_configs_['logging_level'])
    logger = logging.getLogger('MVIT')
    logger.addHandler(file_handler)
    logger.setLevel(logger_configs_['logging_level'])
    return logger


logger = get_logger_()