import logging

class IgnoreStereoFilter(logging.Filter):
    def filter(self, record):
        return "(not error because allow_undefined_stereo=True)" not in record.getMessage()

offlogger = logging.getLogger("openff")
offlogger.setLevel(logging.ERROR)
offlogger.propagate = False
# offlogger.addFilter(IgnoreStereoFilter())


from .polymetrizer import Polymetrizer
from .monomer import Monomer
from .oligomer import Oligomer, HYDROGEN_CAP
