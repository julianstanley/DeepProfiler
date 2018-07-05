import gc
import os
import numpy as np
import tensorflow as tf
import pickle

import deepprofiler.dataset.utils
import deepprofiler.imaging.boxes
import deepprofiler.imaging.cropping
import deepprofiler.learning.metrics
import deepprofiler.learning.models
import deepprofiler.learning.training

import keras

import skimage.io
import skimage.transform
import warnings

def save_images(fname, images):
    colors = np.asarray(
         [[0,   192, 64],
         [192, 0,   64],
         [0,   0,   255],
         [255, 0,   0], 
         [0,   255, 0]], dtype=np.uint8)
    colors = colors[np.newaxis, np.newaxis, np.newaxis, :]
    result = images[...,np.newaxis] * colors
    result = np.sum(result, axis=3)
    maxs = np.max(np.max(np.max(result, axis=1, keepdims=True), axis=2, keepdims=True),axis=3, keepdims=True)
    result = result / maxs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.makedirs(fname, exist_ok=True)
        for i in range(result.shape[0]):
            img = skimage.transform.rescale(result[i,...], 0.5)
            skimage.io.imsave(fname + "/" + str(i) + ".jpg", img)

class Validation(object):

    def __init__(self, config, dset):
        self.config = config
        self.dset = dset
        self.config["queueing"]["min_size"] = 0
        self.save_features = config["validation"]["save_features"] #and config["validation"]["sample_first_crops"]
        self.val_dir = self.config["training"]["output"] + "/validation/"
        self.metrics = []


    def output_base(self, meta):
        filebase = os.path.join(
            self.val_dir,
            str(meta["Metadata_Plate"]) + "_" +
            str(meta["Metadata_Well"]) + "_" +
            str(meta["Metadata_Site"])
        )
        return filebase


    def configure(self, session, checkpoint_file):
        gc.collect()
        # Create model and load weights
        batch_size = self.config["validation"]["minibatch"]
        self.config["training"]["minibatch"] = batch_size
        feature_layer = self.config["profiling"]["feature_layer"]

        if self.config["model"]["type"] in ["convnet", "mixup", "same_label_mixup"]:
            input_shape = (
                self.config["sampling"]["box_size"],      # height
                self.config["sampling"]["box_size"],      # width
                len(self.config["image_set"]["channels"]) # channels
            )
            self.model = deepprofiler.learning.models.create_keras_resnet(input_shape, self.dset.targets, is_training=False)
            self.crop_generator = deepprofiler.imaging.cropping.SingleImageCropGenerator(self.config, self.dset)
        elif self.config["model"]["type"] == "recurrent":
            print("RECURRENT MODEL")
            input_shape = (
                self.config["model"]["sequence_length"],  # time
                self.config["sampling"]["box_size"],      # height
                self.config["sampling"]["box_size"],      # width
                len(self.config["image_set"]["channels"]) # channels
            )
            self.model = deepprofiler.learning.models.create_recurrent_keras_resnet(input_shape, self.dset.targets, is_training=False)
            self.crop_generator = deepprofiler.imaging.cropping.SingleImageCropSetGenerator(self.config, self.dset)
       
        print("Checkpoint:", checkpoint_file)
        self.model.load_weights(checkpoint_file)

        # Create feature extraction function
        feature_embedding = self.model.get_layer(feature_layer).output
        self.num_features = feature_embedding.shape[-1]
        self.feat_extractor = keras.backend.function([self.model.input], [feature_embedding])

        # Configure metrics for each target
        for i in range(len(self.dset.targets)):
            tgt = self.dset.targets[i]
            mtr = deepprofiler.learning.metrics.Metrics(name=tgt.field_name, k=self.config["validation"]["top_k"])
            mtr.configure_ops(tgt.index)
            self.metrics.append(mtr)

        # Prepare output directory
        self.val_dir = self.config["training"]["output"] + "/validation/"
        if not os.path.isdir(self.val_dir):
            os.mkdir(self.val_dir)

        # Initiate generator
        self.crop_generator.start(session)
        self.session = session


    def load_batches(self, meta):
        filebase = self.output_base(meta)
        if os.path.isfile(filebase + ".npz"):
            print(filebase, "is done")
            return False
        if os.path.isfile(filebase + ".pkl"):
            with open(filebase + ".pkl", "rb") as input_file:
                batches = pickle.load(input_file)
                # Hack to replicate validation crops N times
                # Only works if crops were cached before with a non-recurrent model
                #if self.config["model"]["type"] == "recurrent":
                #    for i in range(len(batches["batches"])):
                #        batches["batches"][i][0] = np.tile(batches["batches"][i][0], (self.config["model"]["sequence_length"], 1, 1, 1, 1))
                #        batches["batches"][i][0] = np.swapaxes(batches["batches"][i][0], 0, 1)
                self.predict(batches, meta)
            return False
        else:
            return True


    def process_batches(self, key, image_array, meta):
        # Prepare image for cropping
        s = deepprofiler.dataset.utils.tic()
        batch_size = self.config["validation"]["minibatch"] 
        total_crops = self.crop_generator.prepare_image(
                                   self.session, 
                                   image_array, 
                                   meta, 
                                   self.config["validation"]["sample_first_crops"]
                            )
        if total_crops > 0:
            # We expect all crops in a single batch
            filebase = self.output_base(meta)
            batches = [b for b in self.crop_generator.generate(self.session)]
            self.predict(batches[0], meta)
        deepprofiler.dataset.utils.toc(str(total_crops)+" crops", s)

    def predict(self, batch, meta):
        gc.collect()
        # batch[0] contains images, batch[i+1] contains the targets
        features = np.zeros(shape=(batch[0].shape[0], self.num_features))

        # Forward propagate crops into the network and get the outputs
        output = self.model.predict(batch[0])
        if type(output) is not list:
            output = [output]

        # Compute performance metrics for each target
        for i in range(len(self.metrics)):
            metric_values = self.session.run(
                    self.metrics[i].get_ops(), 
                    feed_dict=self.metrics[i].set_inputs(batch[i+1], output[i])
                )
            self.metrics[i].update(metric_values, batch[0].shape[0])

        # Extract features (again) 
        # TODO: compute predictions and features at the same time
        if self.save_features:
            f = self.feat_extractor((batch[0], 0))
            while len(f[0].shape) > 2: # 2D mean spatial pooling
                f[0] = np.mean(f[0], axis=1)
            batch_size = batch[0].shape[0]
            features[:, :] = f[0]

        # Save features and report performance
        filebase = self.output_base(meta)
        if self.save_features:
            #features = features[:-batch_data["pads"], :]
            np.savez_compressed(filebase + ".npz", f=features)
            save_images(filebase, batch[0])
            print(filebase, features.shape)


    def report_results(self):
        status = ""
        for metric in self.metrics:
            status += " " + metric.result_string()
        print(status)


def validate(config, dset, checkpoint_file):
    configuration = tf.ConfigProto()
    #configuration.gpu_options.allow_growth = True
    configuration.gpu_options.visible_device_list = "0"
    session = tf.Session(config=configuration)
    keras.backend.set_session(session)
    keras.backend.set_learning_phase(0)

    validation = Validation(config, dset)
    validation.configure(session, checkpoint_file)
    dset.scan(validation.process_batches, frame=config["validation"]["frame"], check=validation.load_batches)
    validation.report_results()

    print("Validation: done")

