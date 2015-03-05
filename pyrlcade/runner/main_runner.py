import sys
from pyrlcade.env.pyrlcade_environment import pyrlcade_environment
from pyrlcade.runner.rl_runner import rl_runner
from pyrlcade.misc.autoconvert import autoconvert


class main_runner(object):
    def run_from_cmd(self,argv=[]):
        #Get the parameters file from the command line
        #use mnist_train__forget_params.py by default (no argument given)
        if(len(argv) > 1):
            params_file = argv[1]
        else:
            print("No Parameter File Specified")
            print("Usage: ./run_pyrlcade.py <Parameter_File> <parameter_1=value> ... <parameter_n=value>")
            print("Running with default parameter file: pyrlcade_default_params.py")
            params_file = 'pyrlcade_default_params.py'
        p = {}
        self.run(params_file,p,argv)

    def run(self,params_file,params=None,argv=None):
        if(params is None):
            params = {}
        p = {}
        execfile(params_file,p)

        #merge elements from the parameters file, and overriding with parameters in params.
        p = dict(p.items() + params.items())
        if(argv is None):
            argv = []

        #grab extra parameters from command line
        for i in range(2,len(argv)):
            (k,v) = argv[i].split('=')
            v = autoconvert(v)
            p[k] = v
            print(str(k) + ":" + str(v))

        self.results = None
        #the runtype tells exactly what type of simulation we want to run
        #if(p['runtype'].lower() == "game"):
            #only import if we need it, since it links to pygame and we don't want to require it to be installed
        #    from pyrlcade.runner.runner_game import runner_game
        #    run = runner_game()
        #    run.run_sim(p)
            #The "game" returns no results

        if(p['runtype'].lower() == "sarsa"):
            run = rl_runner()
            run.run_sim(p)
            self.results = run.results
        else:
            print("Unknown Run Type: " + str(p['runtype']) + " only 'sarsa' and 'game' are supported right now");
        return self.results


