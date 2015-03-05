#!/usr/bin/env python
import sys
from pyrlcade.runner.main_runner import main_runner
from pyrlcade.misc.autoconvert import autoconvert

if __name__ == '__main__':
    m = main_runner()
    m.run_from_cmd(sys.argv)

