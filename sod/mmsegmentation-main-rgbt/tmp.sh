bash tools/dist_train.sh configs/mae/mae-small_upernet_8xb2-amp-160k_ade20k-768x768_classbalance.py 2

sleep 30


bash tools/dist_train.sh configs/mae/mae-small_upernet_8xb2-amp-160k_ade20k-768x768_classbalance_copy.py 2
