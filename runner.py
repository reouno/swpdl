import utils.constants as const
import utils.imports as imp
from optimizer import Optimizer

class Runner:
    '''run training/test/inference
    '''

    def __init__(self, model, conf, logger):
        self.logger = logger

        # variables used in this class
        self.model = model
        self.backend = const.backend_mdls[conf.backend]
        self.learning_rate = conf.learning_rate
        self.decay_rate = conf.learning_rate_decay
        self.gpu_conf = conf.gpu_conf
        self.save_dir = conf.save_dir
        self.train_steps_per_epoch = conf.train_steps_per_epoch
        self.test_steps_per_epoch = conf.test_steps_per_epoch
        self.validation_steps_per_epoch = conf.validation_steps_per_epoch
        self.num_epochs = conf.num_epochs
        self.optimizer = Optimizer(conf, logger).load()

    def train(self, data):
        self.logger.info('start training')
        runner_mdl = imp.import_fws_module(self.backend, 'runner')
        runner_class = getattr(runner_mdl, 'Runner')
        return runner_class(
                self.model,
                self.optimizer,
                self.learning_rate,
                self.decay_rate,
                self.gpu_conf,
                self.save_dir,
                self.train_steps_per_epoch,
                self.test_steps_per_epoch,
                self.validation_steps_per_epoch,
                self.num_epochs
                ).train(data)

    def test(self, data):
        self.logger.info('start test')
        pass

    #def __import_module(self, name):
    #    mdl_path = 'fws.{}.{}'.format(const.backend_mdls[self.backend], name)
    #    return import_module(mdl_path)

