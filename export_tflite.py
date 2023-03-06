import os
import argparse
import tensorflow as tf


def main(input_model_dir_path, output_model_file_path):
    os.makedirs(os.path.dirname(output_model_file_path), exist_ok=True)
    tf.compat.v1.enable_eager_execution()

    converter = tf.lite.TFLiteConverter.from_saved_model(input_model_dir_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_quant_model = converter.convert()
    open(output_model_file_path, "wb").write(tflite_quant_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export')
    parser.add_argument('--input_model_dir_path', type=str,
                        default='~/output_model/2023-03-06-11-05-09/step-5000_batch-8_epoch-8_loss_0.1199_one_hot_mean_io_u_0.7299_val_loss_0.0929_val_one_hot_mean_io_u_0.7313',
                        help="input tensor model dir path")
    parser.add_argument('--output_model_file_path', type=str,
                        default='~/.vaik-segmentation-tflite-exporter/model_fp.tflite',
                        help="output tflite model file path")
    args = parser.parse_args()

    args.input_model_dir_path = os.path.expanduser(args.input_model_dir_path)
    args.output_model_file_path = os.path.expanduser(args.output_model_file_path)

    main(args.input_model_dir_path, args.output_model_file_path)
