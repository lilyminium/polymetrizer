"""
polymetrizer
Generate force fields for polymer-like molecules
"""

# Add imports here
from .polymetrizer import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
