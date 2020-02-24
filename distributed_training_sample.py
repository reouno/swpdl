import tensorflow as tf
import tensorflow_datasets as tfds

import os


def main():

    datasets, info = load_dataset('mnist')
    train_data, test_data = datasets['train'], datasets['test']

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # You can also do info.splits.total_num_examples to get the total
    # number of examples in the dataset.

    num_train_examples = info.splits['train'].num_examples
    #num_test_examples = info.splits['test'].num_examples

    BUFFER_SIZE = 10000

    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    train_steps_per_epoch = num_train_examples // BATCH_SIZE
    #test_steps_per_epoch = num_test_examples // BATCH_SIZE

    train_dataset = train_data.repeat().map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = test_data.map(scale).batch(BATCH_SIZE)

    with strategy.scope():
        model = tf.keras.Sequential([
          tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy'])

    # Define the checkpoint directory to store the checkpoints
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                           save_weights_only=True),
        tf.keras.callbacks.LearningRateScheduler(decay),
        #PrintLR()
    ]

    # train
    model.fit(
            train_dataset,
            epochs=12,
            steps_per_epoch=train_steps_per_epoch,
            callbacks=callbacks
            )
    
    # test
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    eval_loss, eval_acc = model.evaluate(eval_dataset)
    print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

def load_dataset(name):
    if name == 'mnist':
        datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
        #train_data, test_data = datasets['train'], datasets['test']
    else:
        raise RuntimeError('no such dataset')

    return datasets, info

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label

# Function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
              model.optimizer.lr.numpy()))

if __name__ == '__main__':
    main()
