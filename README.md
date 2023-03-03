# vaik-segmentation-tflite-exporter

Export from segmentation pb model to tflite model

## Usage

```shell
pip install -r requirements.txt
python export_tflite.py --input_model_dir_path ~/output_model/2023-03-03-14-32-08/step-5000_batch-8_epoch-9_loss_0.0027_one_hot_mean_io_u_0.8001_val_loss_0.0174_val_one_hot_mean_io_u_0.4809 \
                --output_model_file_path ~/.vaik-segmentation-pb-trainer/model.tflite \
                --sample_max_num 10000
```

## Output

- ```~/.vaik-segmentation-pb-trainer/model.tflite```