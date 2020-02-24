import utils.constants as const
import utils.imports as imp


class Optimizer:
    def __init__(self, conf, logger):
        self.logger = logger

        # variables used in this class
        self.backend = const.backend_mdls[conf.backend]
        self.optimizer_name = conf.optimizer
        self.parameters = dict()
        self.parameters['learning_rate'] = conf.learning_rate

    def load(self):
        optimizer_mdl = imp.import_fws_module(self.backend, 'optimizer')
        opt_func = getattr(optimizer_mdl, 'load')
        return opt_func(
                self.optimizer_name,
                self.parameters
                )
