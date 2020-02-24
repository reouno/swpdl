import numpy as np
import os
import pathlib
import random
import tensorflow as tf

from typing import List, Text, Tuple, Union

AUTOTUNE = tf.data.experimental.AUTOTUNE

class Dataset:
    #train_iterator = None
    #test_iterator = None
    #validation_iterator = None

    # number of samples
    num_train_samples = None
    num_test_samples = None
    num_validation_samples = None

    #input_shape = None # (H, W, D)
    #output_units = None # int

    def __init__(
            self,
            data: Text,
            batch_size: int,
            input_size: Tuple[int, int],
            gpu_conf,
            test_data_dir: Union[None, Text]=None,
            validation_data_dir: Union[None, Text]=None
            ):
        '''
        :param data: dataset name or directory path.
        :param batch_size: global batch size
        :param input_size: (height, width)
        :param gpu_conf: configurations of gpus
        :param test_data_dir: test data directory if any
        :param validation_data_dir: validation data directory if any
        '''
        self.data = data
        self.batch_size = batch_size
        self.input_size = input_size
        self.test_data_dir = test_data_dir
        self.validation_data_dir = validation_data_dir
        self.gpu_conf = gpu_conf
        if self.data.upper() == 'FASHION_MNIST':
            self.__fashion_mnist()
        elif os.path.isdir(self.data):
            self.__flow_from_directory()
        else:
            # to be implemented
            raise RuntimeError('invalid dataset name or no such directory, "{}"'.format(self.data))

    def __fashion_mnist(self):
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

        # Adding a dimension to the array -> new shape == (28, 28, 1)
        # We are doing this because the first layer in our model is a convolutional
        # layer and it requires a 4D input (batch_size, height, width, channels).
        # batch_size dimension will be added later on.
        train_images = train_images[..., None]
        test_images = test_images[..., None]

        # Getting the images in [0, 1] range.
        train_images = train_images / np.float32(255)
        test_images = test_images / np.float32(255)

        self.num_train_samples = len(train_images)
        self.num_test_samples = len(test_images)

        #with self.gpu_conf.strategy.scope():
        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(self.num_train_samples).batch(self.batch_size) 
        #self.train_iterator = self.gpu_conf.strategy.make_dataset_iterator(train_dataset)
        
        self.test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(self.batch_size) 
        #self.test_iterator = self.gpu_conf.strategy.make_dataset_iterator(test_dataset)

    def __flow_from_directory(self):
        '''load image dataset from directory
        refer: https://www.tensorflow.org/alpha/tutorials/load_data/images

        Assuming the following directory structure
        dataset_root (dir)
         |--label1 (dir)
         |   |--image1.jpg
         |   |--image2.jpg
         |--label2 (dir)
         ...
        '''
        #with self.gpu_conf.strategy.scope():
        self.train_dataset, self.num_train_samples \
                = self.__load_data_from_directory(self.data, mode='training')
        #self.train_iterator = self.gpu_conf.strategy.make_dataset_iterator(train_dataset)
        if self.test_data_dir is not None:
            self.test_dataset, self.num_test_samples \
                    = self.__load_data_from_directory(self.test_data_dir, mode='test')
            #self.test_iterator = self.gpu_conf.strategy.make_dataset_iterator(test_dataset)
        if self.validation_data_dir is not None:
            self.validation_dataset, self.num_validation_samples \
                    = self.__load_data_from_directory(self.validation_data_dir, mode='validation')
            #self.validation_iterator = self.gpu_conf.strategy.make_dataset_iterator(validation_dataset)

    def __load_data_from_directory(self, dataset_root, mode='training'):
        '''load dataset from directory
        :param dataset_root: root directory of the datset
        :param mode: 'training', 'test', 'validation'
        '''
        data_root_dir = pathlib.Path(dataset_root)
        all_image_paths = list_files(data_root_dir, randomize=True)
        image_count = len(all_image_paths)
        #print(all_image_paths[:10])
        #print(image_count)
        all_image_labels = list_each_data_labels(data_root_dir, all_image_paths)
        #print('First 10 labels indices: ', all_image_labels[:10])

        # create datset
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        # refer for num_parallel_calls:
        # https://www.tensorflow.org/alpha/guide/data_performance#parallelize_data_transformation
        image_ds = path_ds.map(lambda x:load_and_preprocess_image(x, self.input_size), num_parallel_calls=AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
        #print(label_ds)
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        #print(image_label_ds)

        ds = image_label_ds
        if mode == 'training':
            # The order is important: shuffle first --> repeat and batch
            # refer: https://www.tensorflow.org/alpha/tutorials/load_data/images#basic_methods_for_training
            ds = ds.shuffle(buffer_size=image_count)
            ds = ds.repeat()
            ds = ds.map(
                    lambda x,y:(flip(x), y)
                    ).map(
                    lambda x,y:(color(x), y)
                    ).map(
                    lambda x,y:(rotate(x), y)
                    ).map(
                    lambda x,y:(random_crop(x), y)
                    )
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        #print(ds)
        return ds, image_count

##### utils

def list_files(root_dir: pathlib.Path, randomize=True):
    file_paths = list(root_dir.glob('*/*'))
    file_paths = [str(path) for path in file_paths]
    if randomize:
        random.shuffle(file_paths)
    return file_paths

def list_each_data_labels(root_dir: pathlib.Path, data_paths: List[Text]):
    '''list labels for each samples.

    Assuming the following directory structure
    dataset_root (dir)
     |--label1 (dir)
     |   |--image1.jpg
     |   |--image2.jpg
     |--label2 (dir)
     ...
    '''
    label_names = sorted(item.name for item in root_dir.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index,name in enumerate(label_names))
    #print('label_to_index:',label_to_index)
    return [label_to_index[pathlib.Path(path).parent.name] for path in data_paths]

def preprocess_image(image, sizehw=[224,224]):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, sizehw)
    image /= 255.0
    return image

def load_and_preprocess_image(path, sizehw):
    image = tf.io.read_file(path)
    return preprocess_image(image, sizehw=sizehw)


##### augmentation utils
def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x

def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

def create_box(crop_hws):
    y1 = np.random.uniform(0, 1-crop_hws[0], 1)
    x1 = np.random.uniform(0, 1-crop_hws[1], 1)
    y2 = y1+crop_hws[0]
    x2 = x1+crop_hws[1]
    return np.concatenate([y1,x1,y2,x2])

def random_box():
    crop_hw = np.random.uniform(0.75, 1.0, 2)
    return create_box(crop_hw)

def random_crop(x: tf.Tensor) -> tf.Tensor:
    '''random crop
    input tensor have to be rank-3 with (batch, h, w, c)
    '''
    size = x.shape[:2]
    crop_hw = np.random.uniform(0.75, 1.0, 2)
    boxes = [list(create_box(crop_hw))]
    cropped_imgs = tf.image.crop_and_resize([x], boxes=boxes, box_indices=[0], crop_size=size)
    cropped_img = tf.reshape(cropped_imgs, x.shape)
    #print(cropped_img.shape)
    return cropped_img

