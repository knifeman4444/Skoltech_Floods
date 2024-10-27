mkdir -p dataset/worldfloodsv2

if [ -z "$(ls -A dataset/train)" ]; then
    cd dataset
    wget https://lodmedia.hb.bizmrg.com/case_files/1166565/train_dataset_skoltech_train.zip
    unzip train_dataset_skoltech_train.zip
    rm train_dataset_skoltech_train.zip
    cd ..
fi

if [ -z "$(ls -A dataset/test)" ]; then
    cd dataset
    wget https://lodmedia.hb.bizmrg.com/case_files/1166565/test_dataset_test_scoltech.zip
    unzip test_dataset_test_scoltech.zip
    rm test_dataset_skoltech_test.zip
    cd ..
fi

# Please install the huggingface-cli if it is not installed
if [ "$(ls -A dataset/worldfloodsv2)" ]; then
    echo "dataset/worldfloodsv2 is not empty. Skipping download."
    exit 0
fi
huggingface-cli download --cache-dir /tmp --local-dir dataset/worldfloodsv2 --repo-type dataset isp-uv-es/WorldFloodsv2