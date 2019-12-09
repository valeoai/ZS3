sudo bash -c "DOCKER_BUILDKIT=1 docker build -t zs3_test ."
sudo docker run --rm --gpus "device=4" \
                -v /media/data/datasets/zsd_pascalVOC/VOC2012:/ZS3/zs3/data/VOC2012:ro \
                -v /mnt/data/workspace/mbucher/zs3_temp/ZS3/checkpoint:/ZS3/zs3/checkpoint:ro \
                -e DRY_RUN=1 \
                --ipc=host \
                zs3_test \
                bash -c "python ./train_pascal.py"
