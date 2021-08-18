"""
Tests for command line utilities.
"""

import pytest
import hjson
from rich.console import Console

from c3.c3objs import hjson_decode
from c3.utils.log_reader import show_table

SAMPLE_LOG = "test/sample_optim_log.log"


@pytest.mark.unit
def test_log_viewer():
    """
    Check that the log is read and processed without error. This does not check if
    the output is correct.
    """
    console = Console()
    with open(SAMPLE_LOG) as logfile:
        show_table(hjson.load(logfile, object_pairs_hook=hjson_decode), console)
