#!/usr/bin/env python
import sys
from pyrlcade.runner.main_rerunner import main_rerunner
from pyrlcade.misc.autoconvert import autoconvert

if __name__ == '__main__':
    m = main_rerunner()
    m.run_from_cmd(sys.argv)

