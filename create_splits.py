import argparse
import glob
import os

import tensorflow.compat.v1 as tf

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records where each file is opened and the images are split into
    test-train-val in 80%-10%010%

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # create three directories
    os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "val"), exist_ok=True)

    for ds_path in glob.glob(os.path.join(data_dir, "segment*.tfrecord")):
        raw_dataset = tf.data.TFRecordDataset(ds_path)
        output_file = os.path.basename(ds_path)
        train_writer = tf.python_io.TFRecordWriter(f"{data_dir}/train/{output_file}")
        test_writer = tf.python_io.TFRecordWriter(f"{data_dir}/test/{output_file}")
        val_writer = tf.python_io.TFRecordWriter(f"{data_dir}/val/{output_file}")
        for index, data in enumerate(raw_dataset):
            writer = train_writer
            if (index + 1) % 9 == 0:
                writer = test_writer
            elif (index + 1) % 10 == 0:
                writer = val_writer
            parsed = tf.train.Example.FromString(data.numpy())
            writer.write(parsed.SerializeToString())
        train_writer.close()
        test_writer.close()
        val_writer.close()

        # delete each tf record in the current folder
        os.remove(ds_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into training / validation / testing")
    parser.add_argument("--data_dir", required=True, help="data directory")
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info("Creating splits...")
    split(args.data_dir)
