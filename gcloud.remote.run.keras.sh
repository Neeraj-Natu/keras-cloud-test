gcloud ml-engine jobs submit training JOB1 
--module-name=trainer.cnn_with_keras 
--package-path=./trainer 
--job-dir=gs://keras-on-cloud
--region=us-central1 
--config=trainer/cloudml-gpu.yaml
