import importlib
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
#def fcn(conf):
#    mdl_root = 'fws'
#    name = 'fcn'
#    mdl_path = '{}.{}.{}'.format(mdl_root, conf.backend, name)
#    mdl = importlib.import_module(mdl_path)
#    fcn_ = getattr(mdl, name)
#    return fcn_(conf.fcn_layers)

#class Fcn:
#    def __init__(self, inputs, outputs, constraints, weights_conf, backend):
#        self.inputs = inputs
#        self.outputs = outputs
#        self.constraints = constraints
#        self.weights_conf = weights_conf
#        mdl_root = 'fws'
#        name = 'fcn'
#        mdl_path = '{}.{}.{}'.format(mdl_root, backend, name)
#        mdl = importlib.import_module(mdl_path)
#        self.fcn_ = getattr(mdl, name)
#
#    def load(self):
#        return self.fcn_(self.inputs, self.outputs, self.constraints, self.weights_conf)
