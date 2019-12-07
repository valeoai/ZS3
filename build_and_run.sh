sudo bash -c "DOCKER_BUILDKIT=1 docker build -t zs3_test ."
sudo docker run --rm --gpus "device=4" \
                -v /media/data/datasets/VOC2012:/ZS3/data/VOC2012 \
                zs3_test \
                bash -c "python ./ZS3/train_pascal.py"
