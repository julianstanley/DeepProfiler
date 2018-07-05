import gc
import os
import random

import keras
import numpy as np
import pandas as pd
import pytest
import skimage.io
import tensorflow as tf

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import deepprofiler.imaging.cropping
import deepprofiler.learning.models
import deepprofiler.learning.training
import deepprofiler.learning.validation


def __rand_array():
    return np.array(random.sample(range(100), 12))


@pytest.fixture(scope='function')
def out_dir(tmpdir):
    return os.path.abspath(tmpdir.mkdir("test_validation"))


@pytest.fixture(scope='function')
def config(out_dir):
    return {
        "model": {
            "type": "convnet"
        },
        "sampling": {
            "images": 12,
            "box_size": 16,
            "locations": 10,
            "locations_field": 'R'
        },
        "image_set": {
            "channels": ['R', 'G', 'B'],
            "mask_objects": False,
            "width": 128,
            "height": 128,
            "path": out_dir
        },
        "training": {
            "learning_rate": 0.001,
            "output": out_dir,
            "epochs": 0,
            "steps": 12,
            "minibatch": 2
        },
        "validation": {
            "minibatch": 2,
            "save_features": True,
            "sample_first_crops": False,
            "frame": "val",
            "top_k": 2
        },
        "queueing": {
            "loading_workers": 2,
            "queue_size": 2,
            "min_size": 0
        },
        "profiling": {
            "feature_layer": "pool5"  # TODO: make this work with any model
        }
    }


@pytest.fixture(scope='function')
def metadata(out_dir):
    filename = os.path.join(out_dir, 'metadata.csv')
    df = pd.DataFrame({
        'Metadata_Plate': __rand_array(),
        'Metadata_Well': __rand_array(),
        'Metadata_Site': __rand_array(),
        'R': [str(x) + '.png' for x in __rand_array()],
        'G': [str(x) + '.png' for x in __rand_array()],
        'B': [str(x) + '.png' for x in __rand_array()],
        'Class': ['0', '1', '2', '3', '0', '1', '2', '3', '0', '1', '2', '3'],
        'Sampling': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'Split': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    }, dtype=int)
    df.to_csv(filename, index=False)
    meta = deepprofiler.dataset.metadata.Metadata(filename)
    train_rule = lambda data: data['Split'].astype(int) == 0
    val_rule = lambda data: data['Split'].astype(int) == 1
    meta.splitMetadata(train_rule, val_rule)
    return meta


@pytest.fixture(scope='function')
def target():
    return deepprofiler.dataset.target.MetadataColumnTarget("Class", ["0", "1", "2", "3"])


@pytest.fixture(scope='function')
def dataset(metadata, target, out_dir):
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(metadata, 'Sampling', ['R', 'G', 'B'], out_dir, keygen)
    dset.add_target(target)
    return dset


@pytest.fixture(scope='function')
def locations(out_dir, metadata, config):
    for i in range(len(metadata.data.index)):
        meta = metadata.data.iloc[i]
        path = os.path.join(out_dir, meta['Metadata_Plate'], 'locations')
        os.makedirs(path, exist_ok=True)
        path = os.path.abspath(os.path.join(path, '{}-{}-{}.csv'.format(meta['Metadata_Well'],
                                                  meta['Metadata_Site'],
                                                  config['sampling']['locations_field'])))
        locs = pd.DataFrame({
            'R_Location_Center_X': np.random.randint(0, 128, (config['sampling']['locations'])),
            'R_Location_Center_Y': np.random.randint(0, 128, (config['sampling']['locations']))
        })
        locs.to_csv(path, index=False)


@pytest.fixture(scope='function')
def data(metadata, out_dir):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, metadata.data['R'][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, metadata.data['G'][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, metadata.data['B'][i // 3]), images[:, :, i + 2])


@pytest.fixture(scope='function')
def validation(config, dataset):
    return deepprofiler.learning.validation.Validation(config, dataset)


def test_save_images(out_dir):
    images_shape = (4, 128, 128, 5)
    images = np.random.randint(0, 256, images_shape, dtype=np.uint8)
    deepprofiler.learning.validation.save_images(out_dir, images)
    for i in range(images_shape[0]):
        filename = os.path.join(out_dir, '{}.jpg'.format(i))
        assert os.path.exists(filename)
        image = skimage.io.imread(filename)
        assert image.shape == (images_shape[1] // 2, images_shape[2] // 2, 3)


def test_init(config, dataset, out_dir):
    validation = deepprofiler.learning.validation.Validation(config, dataset)
    config["queueing"]["min_size"] = 0
    assert validation.config == config
    assert validation.dset == dataset
    assert validation.save_features == config["validation"]["save_features"]
    assert validation.val_dir == os.path.join(out_dir, 'validation/')
    assert validation.metrics == []


def test_output_base(validation, metadata, out_dir):
    filebase = validation.output_base(metadata.data)
    val_dir = os.path.join(out_dir, 'validation/')
    expected = os.path.join(
            val_dir,
            str(metadata.data["Metadata_Plate"]) + "_" +
            str(metadata.data["Metadata_Well"]) + "_" +
            str(metadata.data["Metadata_Site"])
        )
    assert filebase == expected


def test_configure(validation, config, dataset, data, locations, out_dir):
    sess = tf.Session()
    epoch = 0
    deepprofiler.learning.training.learn_model(config, dataset, epoch)
    checkpoint = os.path.join(out_dir, "checkpoint_0000.hdf5")
    assert os.path.exists(checkpoint)
    validation.configure(sess, checkpoint)
    assert validation.config["training"]["minibatch"] == config["validation"]["minibatch"]
    assert isinstance(validation.model, keras.models.Model)
    assert isinstance(validation.crop_generator, deepprofiler.imaging.cropping.SingleImageCropGenerator)
    assert validation.num_features == validation.model.get_layer(config['profiling']['feature_layer']).output.shape[-1]
    assert isinstance(validation.feat_extractor, keras.backend.tensorflow_backend.Function)
    assert len(validation.metrics) == len(dataset.targets)
    assert validation.val_dir == os.path.join(out_dir, 'validation/')
    assert os.path.exists(validation.val_dir)
    assert validation.session == sess


def test_load_batches(validation, metadata):  # TODO: implement these
    pass


def test_process_batches():
    pass


def test_predict():
    pass


def test_report_results():
    pass


def test_validate(config, dataset, data, locations, out_dir):
    pass  # TODO: validation is hardcoded to use 5 channels
    # epoch = 0
    # deepprofiler.learning.training.learn_model(config, dataset, epoch)
    # checkpoint = os.path.join(out_dir, "checkpoint_0000.hdf5")
    # assert os.path.exists(checkpoint)
    # gc.collect()
    # deepprofiler.learning.validation.validate(config, dataset, checkpoint)
