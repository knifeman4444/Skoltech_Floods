mkdir -p dataset/worldfloodsv2

# Install the huggingface-cli if it is not installed
huggingface-cli download --cache-dir /tmp --local-dir dataset/worldfloodsv2 --repo-type dataset isp-uv-es/WorldFloodsv2