import logging

def get_logger(name=None):
    default = "__app__"
    formatter = logging.Formatter('%(levelname)s: %(asctime)s %(funcName)s(%(lineno)d) -- %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
    log_map = {"__app__": "app.log"}
    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(default)
    if not logger.handlers:
        fh = logging.FileHandler(log_map[default])
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)
    return logger
