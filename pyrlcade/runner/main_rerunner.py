import sys
from pyrlcade.env.pyrlcade_environment import pyrlcade_environment
from pyrlcade.runner.rl_runner import rl_runner
from pyrlcade.misc.autoconvert import autoconvert
from pyrlcade.misc.save_h5py import save_results,load_results


class main_rerunner(object):
    def run_from_cmd(self,argv=[]):
        #Get the parameters file from the command line
        #use mnist_train__forget_params.py by default (no argument given)
        if(len(argv) > 2):
            h5py_file = argv[1]
            random_seed = int(argv[2])
        else:
            print("No h5py File Specified")
            print("Usage: ./run_pyrlcade_rerun.py <h5py_results_file> <random_seed> <param1=value1> <param2=value2> ...")
            return
        p = {}
        self.run(h5py_file,random_seed,argv)

    def run(self,h5py_file,random_seed,argv=None):
        f = load_results(h5py_file)
        p = f['parameters']


        #grab extra parameters from command line
        for i in range(3,len(argv)):
            (k,v) = argv[i].split('=')
            v = autoconvert(v)
            p[k] = v
            print(str(k) + ":" + str(v))

        #Give this sim a new random seed, and append it to the version
        p['version'] = p['version'] + '_' + str(random_seed)
        p['random_seed'] = random_seed
        if(not p.has_key('use_float32')):
            p['use_float32'] = True

        self.results = None
        #TODO: Change this to only accept 'rl-sim' and change all parameter files also
        if(p['runtype'].lower() == "sarsa"):
            run = rl_runner()
            run.run_sim(p)
            self.results = run.results
        else:
            print("Unknown Run Type: " + str(p['runtype']) + " only 'sarsa' is supported right now");
        return self.results


