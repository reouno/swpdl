from config import Config
from dataset import Dataset
from logger import Logger
from model import Model
from runner import Runner

def main(conf):
    '''train deep learning model
    '''
    log = Logger()
    with conf.gpu_conf.strategy.scope(): # TODO: remove this line
        model = Model(conf, log).load()
    log.info('model created')
    model.summary()

    data = Dataset(conf, log).load()
    log.info('data loaded')

    # TODO: How to hide this dirty code?
    conf.set_num_samples_and_steps_per_epoch(
            data.num_train_samples,
            data.num_test_samples,
            data.num_validation_samples
            )

    runner = Runner(model, conf, log)
    log.info('runner created')
    runner.train(data)
    log.info('training done')
    runner.test(data)
    log.info('test done')


if __name__ == '__main__':
    conf = Config()
    main(conf)
