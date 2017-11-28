import logging
from datetime import datetime
import os
import inspect

_name = os.path.basename(inspect.stack()[-1][1]).replace(".py", "")
_start_time = None
_setup = False

def get_start_time():
    return _start_time if _start_time else datetime.now().strftime("%Y.%m.%d.%H.%M.%S.%f")

_start_time = get_start_time()

# Import and use this
def get_logger():
    if not(_setup):
        setup_logger()
    return logging.getLogger(_name)

# Never expose this
# Create a custom logger, because ai gym environment seems to hijack default logger
def setup_logger(level = logging.DEBUG):
    _setup = True
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    cli_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    log = logging.getLogger(_name)

    cli = logging.StreamHandler()
    cli.setFormatter(cli_formatter)

    fl = logging.FileHandler("./logs/{}-{}.txt".format(get_start_time(), log.name))
    fl.setFormatter(file_formatter)

    log.handlers.clear()
    log.addHandler(cli)
    log.addHandler(fl)

    log.setLevel(level)
    log.propagate = False
    log.debug("Log setup")
    log.info("Running on {}".format(":".join(os.uname())))
    return log
