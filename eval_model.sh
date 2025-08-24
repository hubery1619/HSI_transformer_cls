# python eval.py --model proposed --dataset_name sa --device 0 --trans_type 0 --weights ./checkpoints/proposed/sa/0 
# python eval.py --model cnn3d --dataset_name sa --device 0 --weights ./checkpoints/cnn3d/sa/0
# python eval.py --model dffn --dataset_name sa --device 0 --weights ./checkpoints/dffn/sa/0
# python eval.py --model m3ddcnn --dataset_name sa --device 0 --weights ./checkpoints/m3ddcnn/sa/0
# python eval.py --model rssan --dataset_name sa --device 0 --weights ./checkpoints/rssan/sa/0
# python eval.py --model speformer --dataset_name sa --device 0 --weights ./checkpoints/speformer/sa/0



# python eval.py --model cnn3d --dataset_name bot --device 0 --trans_type 8 --patch_size 7 --weights ./checkpoints/cnn3d/bot/300/8/0.2/0
# python eval.py --model dffn --dataset_name bot --device 0 --trans_type 8 --patch_size 7 --weights ./checkpoints/dffn/bot/300/8/0.2/0
# python eval.py --model m3ddcnn --dataset_name bot --device 0 --trans_type 8 --patch_size 7 --weights ./checkpoints/m3ddcnn/bot/300/8/0.2/0
# python eval.py --model rssan --dataset_name bot --device 0 --trans_type 8 --patch_size 7 --weights ./checkpoints/rssan/bot/300/8/0.2/0
# python eval.py --model speformer --dataset_name bot --device 0 --trans_type 8 --patch_size 7 --weights ./checkpoints/speformer/bot/300/8/0.2/0
# python eval.py --model ssftt --dataset_name bot --device 0 --trans_type 8 --patch_size 7 --weights ./checkpoints/ssftt/bot/300/8/0.2/0

python eval.py --model group_transformer --dataset_name bot --device 0 --trans_type 0 --patch_size 7 --weights ./checkpoints/group_transformer/bot/300/8/0.2/0


# python eval.py --model proposed --dataset_name bot --device 0 --trans_type 0 --patch_size 5 --weights ./checkpoints/proposed/bot/300/0/0.2/0
# python eval.py --model proposed --dataset_name bot --device 0 --trans_type 1 --patch_size 5 --weights ./checkpoints/proposed/bot/300/1/0.2/0
# python eval.py --model proposed --dataset_name bot --device 0 --trans_type 2 --patch_size 5 --weights ./checkpoints/proposed/bot/300/2/0.2/0
# python eval.py --model proposed --dataset_name bot --device 0 --trans_type 7 --patch_size 5 --weights ./checkpoints/proposed/bot/300/7/0.2/0
# python eval.py --model proposed --dataset_name bot --device 0 --trans_type 8 --patch_size 5 --weights ./checkpoints/proposed/bot/300/8/0.2/0


