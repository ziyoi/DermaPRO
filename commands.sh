#!/usr/bin/env bash

# Tutorial:
# https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193

# Configure
gcloud auth login

export PROJECT="st-model"
export GCS_BUCKET="molenet"
gcloud config set project $PROJECT

export TPU_ACCOUNT=service-623465556844@cloud-tpu.iam.gserviceaccount.com

# Copy data
gsutil -m cp -r /tensorflow/data/training_data.tfrecords-?????-of-00010 gs://${GCS_BUCKET}/data
gsutil -m cp -r /tensorflow/data/validation_data.tfrecords-?????-of-00010 gs://${GCS_BUCKET}/data
gsutil cp /tensorflow/data/label_map_dict.pbtxt gs://${GCS_BUCKET}/data/label_map_dict.pbtxt

# Copy pre-trained model.
cd /tmp
curl -O http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz
tar xzf ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz
gsutil -m cp -r /tmp/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03 gs://${GCS_BUCKET}/data/

# Copy config
gsutil cp /tensorflow/work/pipeline.config gs://${GCS_BUCKET}/data/pipeline.config

# Prepare pycocotools
cd /tensorflow/models/research
bash object_detection/dataset_tools/create_pycocotools_package.sh /tmp/pycocotools
python setup.py sdist
(cd slim && python setup.py sdist)

# Run training
export NAME=`whoami`_object_detection_3class_`date +%s`
export EVAL_NAME=`whoami`_object_detection_eval_validation_3class_`date +%s`
export JOB_DIR=$NAME

gcloud ml-engine jobs submit training $NAME \
--job-dir=gs://${GCS_BUCKET}/$JOB_DIR \
--packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
--module-name object_detection.model_tpu_main \
--runtime-version 1.12 \
--scale-tier BASIC_TPU \
--region us-central1 \
-- \
--model_dir=gs://${GCS_BUCKET}/$JOB_DIR \
--tpu_zone us-central1 \
--pipeline_config_path=gs://${GCS_BUCKET}/data/pipeline.config

# Run evalution (start right after starting training run)
gcloud ml-engine jobs submit training $EVAL_NAME \
--job-dir=gs://${GCS_BUCKET}/$JOB_DIR \
--packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
--module-name object_detection.model_main \
--runtime-version 1.12 \
--scale-tier BASIC_GPU \
--region us-central1 \
-- \
--model_dir=gs://${GCS_BUCKET}/$JOD_DIR \
--pipeline_config_path=gs://${GCS_BUCKET}/data/pipeline.config \
--checkpoint_dir=gs://${GCS_BUCKET}/$JOB_DIR

# Run tensorboard to see results.
gcloud auth application-default login
tensorboard --logdir=gs://${GCS_BUCKET}

# Things to try adjusting:
# Number of steps (listed in two places)
# data_augmentation_options (add some)
# Read documentation: https://github.com/tensorflow/models/tree/master/research/object_detection/g3doc
# eval_config { num_examples: 300

# export model for inference on a computer:
export RUN=root_object_detection_3class_1548123831
export model_checkpoint=gs://${GCS_BUCKET}/${RUN}/model.ckpt-14300.meta
export model_save_dir=/tensorflow/data/saved/${RUN}
gsutil -m cp -r gs://${GCS_BUCKET}/${RUN} /tensorflow/data/checkpoints
export model_checkpoint=/tensorflow/data/checkpoints/${RUN}/model.ckpt-14300
cd /tensorflow/models/research/object_detection
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /tensorflow/work/pipeline.config \
    --trained_checkpoint_prefix $model_checkpoint \
    --output_directory $model_save_dir

# Run inference for evaluation images.
python inference/infer_detections.py \
  --input_tfrecord_paths=/tensorflow/data/validation_data.tfrecords-00000-of-00010,/tensorflow/data/validation_data.tfrecords-00001-of-00010,/tensorflow/data/validation_data.tfrecords-00002-of-00010,/tensorflow/data/validation_data.tfrecords-00003-of-00010,/tensorflow/data/validation_data.tfrecords-00004-of-00010,/tensorflow/data/validation_data.tfrecords-00005-of-00010,/tensorflow/data/validation_data.tfrecords-00006-of-00010,/tensorflow/data/validation_data.tfrecords-00007-of-00010,/tensorflow/data/validation_data.tfrecords-00008-of-00010,/tensorflow/data/validation_data.tfrecords-00009-of-00010 \
  --output_tfrecord_path=/tensorflow/data/inference/inference_${RUN}.tfrecords \
  --inference_graph=/tensorflow/data/saved/root_object_detection_fulldata_10ksteps_1548004752_bak/frozen_inference_graph.pb \
  --discard_image_pixels=True


# Calculate confusion matrix
# Comment out lines 91 and 92 in ../models/research/object_detection/metrics/tf_example_parser.py
cd /tensorflow/work
python confusion_matrix.py \
  --detections_record=/tensorflow/data/inference/inference_${RUN}.tfrecords \
  --label_map=/tensorflow/data/label_map_dict.pbtxt

#Export to tflite
export OUTPUT_DIR=/tensorflow/data/saved/${RUN}
export CONFIG_FILE=/tensorflow/work/pipeline.config
export CHECKPOINT_PATH=/tensorflow/data/checkpoints/${RUN}/model.ckpt-14300
cd /tensorflow/models/research
python object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path=$CONFIG_FILE \
  --trained_checkpoint_prefix=$CHECKPOINT_PATH \
  --output_directory=$OUTPUT_DIR \
  --add_postprocessing_op=true

# Optimize for tensorflow lite
cd /tensorflow
bazel run -c opt tensorflow/lite/toco:toco -- \
--input_file=$OUTPUT_DIR/tflite_graph.pb \
--output_file=$OUTPUT_DIR/detect.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops
# This results in a suspicious warning:
# W tensorflow/lite/toco/graph_transformations/quantize.cc:126] Constant array anchors lacks MinMax information.
# To make up for that, we will now compute the MinMax from actual array elements. That will result in quantization parameters that
# probably do not match whichever arithmetic was used during training, and thus will probably be a cause of poor inference accuracy.

cp $OUTPUT_DIR/detect.tflite /tensorflow/work/android/app/src/main/assets

bazel build -c opt --config=android_arm{,64} --cxxopt='--std=c++11' \
//work/android:tflite_demo

adb install bazel-bin/work/android/tflite_demo.apk
