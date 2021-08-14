# python3 k_means.py --data-format points \
#                    --options hard \
#                    --num_clusters 2 \
#                    --num_iterations 5

python3 k_means.py --data-format images \
                   --options soft \
                   --input-image "Parrot.jpg" \
                   --num_clusters 5 \
                   --num_iterations 5
