SRC_DIR=/home/mihir/waymo/val
NEW_SIZE=300x200
DST_DIR=gs://foragerml/thumbnails/2d2b13f9-3b30-4e51-8ab9-4e8a03ba1f03/

cp -r $SRC_DIR thumbs
(cd thumbs && rename.ul .jpeg .jpg *.jpeg && mogrify -resize $NEW_SIZE -monitor *.jpg)
gsutil -m cp -r thumbs/* $DST_DIR
