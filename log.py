import logging


def logger(loglevel="INFO"):
    """
    Store your logging files in a beautiful format.

    Parameters
    ----------
    loglevel : :obj:`str`, optional
        Input log level

    Returns
    -------
    object
        An object for storing system logs
    """

    logger = logging.getLogger("MAIN")
    logginglevel = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    logfile = {"DEBUG": "debug.log", "INFO": "info.log", "WARNING": "warning.log", "ERROR": "error.log"}
    logger.setLevel(logginglevel[loglevel])
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logginglevel[loglevel])
    f_handler = logging.FileHandler(logfile[loglevel])
    f_handler.setLevel(logginglevel[loglevel])
    f_format = logging.Formatter('face_identification_api - %(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                 datefmt='%d-%b-%y %H:%M:%S')
    c_handler.setFormatter(f_format)
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    logger.addHandler(c_handler)

    return logger
