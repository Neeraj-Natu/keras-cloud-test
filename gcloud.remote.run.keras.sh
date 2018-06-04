export BUCKET_NAME=neeraj-text-bucket
export JOB_NAME="keras_train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-east1

gcloud ml-engine jobs submit training keras-train_04-06-2018 \
  --job-dir gs://neeraj-text-bucket/keras-train_04-06-2018 \
  --runtime-version 1.0 \
  --module-name trainer.keras-model \
  --package-path ./trainer \
  --region us-east1 \
  --config=trainer/cloudml-gpu.yaml \
  -- \
  --train-file gs://neeraj-text-bucket/data.txt