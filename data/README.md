For new Image-caption datasets the high level process should be as explained below. For existing datasets please follow the respective README files in the dataset directories.

1. Download dataset
2. Run Docker image: 
    - Need to have mapped to the data directory inside Docker container
    - Need to have faster_rcnn model pretrained and in a folder that is mapped
    - Need to have generate_tsv_v2.py and zip_helper.py files inside a mapped directory so that it can be run inside the container
    - In generate_tsv_v2.py need to add the path to /opt/butd which is where the docker repo stores all butd scripts
    - Need to modify generate_tsv_v2.py to work with the dataset
3. Run generate_tsv_v2.py
    - Example: python generate_tsv_v2.py --gpu 3 --cfg /opt/butd/experiments/cfgs/faster_rcnn_end2end_resnet.yml --def /opt/butd/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --net /workspace/data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split flickr30k_train --data_root /workspace/data  --out /workspace/data/flickr30k/train_frcnn/
4. Zip with no loss compression the frcnn json files generated:
    - Example:      cd ../train_frcnn
                    zip -0 ../train_frcnn.zip ./*
