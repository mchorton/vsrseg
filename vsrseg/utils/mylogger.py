import sys
from logging import *
_formatstr = '%(asctime)s %(name)s %(levelname)s]: %(message)s'
_level = INFO

def configure():
    basicConfig(stream=sys.stderr, format=_formatstr, level=_level)

def reconfigure():
    log = getLogger()
    for handler in log.handlers[:]:
        log.removeHandler(handler)
    configure()

configure()
