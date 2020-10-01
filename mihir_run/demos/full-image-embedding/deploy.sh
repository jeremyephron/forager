project=`gcloud config get-value project 2> /dev/null`
folder=mapper
name=mihir-demo-${PWD##*/}
region=us-central1
root_path=../..

# Copy shared resources in
cp -r $root_path/src/knn $folder

# Submit build from within subdirectory
gcloud config set builds/use_kaniko True
(cd $folder; gcloud builds submit --tag gcr.io/$project/$name)

# Remove shared resources
rm -rf $folder/knn
