import os
import tensorflow as tf
import time

from typing import Text

class Runner:
    def __init__(
            self,
            model,
            optimizer,
            learning_rate,
            decay_rate,
            gpu_conf,
            save_dir: Text,
            train_steps_per_epoch: int,
            test_steps_per_epoch: int,
            validation_steps_per_epoch: int,
            epochs: int
            ):
        '''
        :param model: model object to be trained
        :param optimizer: optimizer object
        :param learning_rate: initial learning rate
        :param decay_rate: learning rate decay rate, 1.0 means no decay
        :param gpu_conf: gpu configuration and context
        :param save_dir: directory to save logs and checkpoint files
        :param train_steps_per_epoch: train steps per epoch
        :param test_steps_per_epoch: test steps per epoch
        :param epochs: number of training epochs
        '''
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.strategy = gpu_conf.strategy

        # save dirs
        self.checkpoint_dir = os.path.join(save_dir, 'training_checkpoints')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt_{epoch}')
        self.log_dir = os.path.join(save_dir, 'logs')
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.log_dir)

        self.train_steps_per_epoch = train_steps_per_epoch
        #self.test_steps_per_epoch = test_steps_per_epoch
        self.validation_steps_per_epoch = validation_steps_per_epoch
        self.epochs = epochs

        #self.loss_object = None
        #self.train_loss = None
        #self.test_loss = None
        #self.validation_loss = None
        #self.train_accuracy = None
        #self.test_accuracy = None
        #self.validation_accuracy = None
        #self.optimizer = None
        #self.checkpoint = None
        #self.__do_all_tasks_before_training()

    def train(self, data):

        # have to be done in distributed strategy scope
        with self.strategy.scope():
            self.model.compile(loss='sparse_categorical_crossentropy',
                        optimizer=self.optimizer,
                        metrics=['accuracy'])

        callbacks = self.set_callbacks()
        self.model.fit(
                data.train_dataset,
                epochs=self.epochs,
                steps_per_epoch=self.train_steps_per_epoch,
                callbacks=callbacks,
                validation_data=data.validation_dataset,
                validation_steps=self.validation_steps_per_epoch
                )

        # TODO: do test in the test method instead of doing in the train method
        # test
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        test_loss, test_acc = model.evaluate(eval_dataset)
        print('Test loss: {}, Test Accuracy: {}'.format(test_loss, test_acc))

    def set_callbacks(self):
        decay_func = lambda epoch: decay(epoch, self.learning_rate, self.decay_rate)
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir),
            tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_prefix,
                                               save_weights_only=True),
            tf.keras.callbacks.LearningRateScheduler(decay_func),
            PrintLR()
        ]
        return callbacks

    def _depricated_train(self, data):
        '''
        '''
        # TODO: need redesigning and refactoring
        with self.strategy.scope():
            # Train step
            def train_step(inputs):
                images, labels = inputs

                with tf.GradientTape() as tape:
                    predictions = self.model(images, training=True)
                    loss = self.loss_object(labels, predictions)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                self.train_loss(loss)
                self.train_accuracy(labels, predictions)

            # Validation step
            def validation_step(inputs):
                images, labels = inputs

                predictions = self.model(images, training=False)
                t_loss = self.loss_object(labels, predictions)

                self.validation_loss(t_loss)
                self.validation_accuracy(labels, predictions)

        with self.strategy.scope():
            # `experimental_run` replicates the provided computation and runs it
            # with the distributed input.
            @tf.function
            def distributed_train():
                return self.strategy.experimental_run(train_step, data.train_iterator)

            @tf.function
            def distributed_validation():
                return self.strategy.experimental_run(validation_step, data.validation_iterator)

            start = time.time()
            print(self.epochs)
            for epoch in range(self.epochs):
                print('learning_rate:', self.optimizer.lr.numpy())
                print('epoch:{}, train steps:{}, validation steps:{}'.format(epoch, self.train_steps_per_epoch, self.validation_steps_per_epoch))
                # Note: This code is expected to change in the near future.
          
                # TRAIN LOOP
                # Initialize the iterator
                data.train_iterator.initialize()
                print('Steps [{}]:'.format(self.train_steps_per_epoch), end='', flush=True)
                prev_progress = 0
                for i in range(self.train_steps_per_epoch):
                    distributed_train()
                    progress = int(i / self.train_steps_per_epoch * 100)
                    print('#' * (progress - prev_progress), end='', flush=True)
                    prev_progress = progress
                print() # return line
                #self.__update_learning_rate(epoch)

                # VALIDATION LOOP
                data.validation_iterator.initialize()
                prev_progress = 0
                print('Validation steps [{}]:'.format(self.validation_steps_per_epoch), end='', flush=True)
                for i in range(self.validation_steps_per_epoch):
                    distributed_validation()
                    progress = int(i / self.validation_steps_per_epoch * 100)
                    print('#' * (progress - prev_progress), end='', flush=True)
                    prev_progress = progress
                print() # return line
          
                if epoch % 2 == 0:
                  self.checkpoint.save(self.checkpoint_prefix)
          
                template = ("Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, "
                            "Validation Accuracy: {}")
                print (template.format(epoch+1, self.train_loss.result(),
                                       self.train_accuracy.result()*100, self.validation_loss.result(),
                                       self.validation_accuracy.result()*100))
          
                self.train_loss.reset_states()
                self.validation_loss.reset_states()
                self.train_accuracy.reset_states()
                self.validation_accuracy.reset_states()
            elapsed = time.time() - start
            print('training time: {}sec'.format(elapsed))
          
    def __depricated__do_all_tasks_before_training(self):
        '''
        '''
        # TODO: delete this method and redesign&reimplement
        with self.strategy.scope():
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.test_loss = tf.keras.metrics.Mean(name='test_loss')
            self.validation_loss = tf.keras.metrics.Mean(name='validation_loss')

            self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                    name='train_accuracy')
            self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                    name='test_accuracy')
            self.validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                    name='validation_accuracy')

        with self.strategy.scope():
            learning_rate = tf.Variable(1e-4, name='learning_rate')
            #self.optimizer = tf.keras.optimizers.Adam()
            self.optimizer = tf.optimizers.RMSprop(
                    learning_rate = 1e-4,
                    rho = 0.9,
                    momentum = 0.9,
                    epsilon = 1e-10,
                    centered = True)
                    #decay = 0.98
                    #)
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

def decay(epoch, initial_lr, decay_rate):
    return initial_lr * (decay_rate ** epoch)

class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print('LR:', self.model.optimizer.lr.numpy())

#    def __update_learning_rate(self, epoch):
#        '''update learning rate each epoch
#        '''
#        self.optimizer.
