project=`gcloud config get-value project 2> /dev/null`
folder=.
name=forager-index-${PWD##*/}
gcr_region=us-west1
root_path=../../..

# Copy shared resources in
cp -r $root_path/mihir_run/src/knn $folder

# Submit build from within subdirectory
gcloud config set builds/use_kaniko True
(cd $folder; gcloud builds submit --tag gcr.io/$project/$name)

# Remove shared resources
rm -rf $folder/knn

# Deploy to Cloud Run
gcloud run deploy $name --image gcr.io/$project/$name \
                        --platform managed \
                        --concurrency 1 \
                        --cpu 1 \
                        --max-instances 1000 \
                        --memory 1Gi \
                        --timeout 900 \
                        --region $gcr_region \
                        --allow-unauthenticated
