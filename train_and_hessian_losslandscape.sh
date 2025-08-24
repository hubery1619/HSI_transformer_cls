# transtype = 1
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 3
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 5
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 7
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 9
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 5
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 13
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 15

# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 3
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 5
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 7
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 9
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 5
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 13
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 15

# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 3
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 5
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 7
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 9
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 13
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 15

# python hessian_matrics_version_sota.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 1 --weights ./checkpoints/proposed/bot
# python loss_landscape_analysis.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 1 --weights ./checkpoints/proposed/bot

# # transtype = 2

# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 2
# python hessian_matrics_version_sota.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 2 --weights ./checkpoints/proposed/bot
# python loss_landscape_analysis.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 2 --weights ./checkpoints/proposed/bot

# # transtype = 7

# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 7
# python hessian_matrics_version_sota.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 7 --weights ./checkpoints/proposed/bot
# python loss_landscape_analysis.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 7 --weights ./checkpoints/proposed/bot

# # transtype = 8
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8
# python hessian_matrics_version_sota.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/proposed/bot
# python loss_landscape_analysis.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/proposed/bot

# python main.py --model cnn3d --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8 --patch_size 7
# python hessian_matrics_version_sota.py --model cnn3d --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/cnn3d/bot --patch_size 7
# python loss_landscape_analysis.py --model cnn3d --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/cnn3d/bot --patch_size 7


# python main.py --model dffn --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8 --patch_size 7
# python hessian_matrics_version_sota.py --model dffn --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/dffn/bot --patch_size 7
# python loss_landscape_analysis.py --model dffn --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/dffn/bot --patch_size 7

# python main.py --model m3ddcnn --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8 --patch_size 7
# python hessian_matrics_version_sota.py --model m3ddcnn --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/m3ddcnn/bot --patch_size 7
# python loss_landscape_analysis.py --model m3ddcnn --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/m3ddcnn/bot --patch_size 7

# python main.py --model rssan --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8 --patch_size 7
# python hessian_matrics_version_sota.py --model rssan --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/rssan/bot --patch_size 7
# python loss_landscape_analysis.py --model rssan --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/rssan/bot --patch_size 7

# python main.py --model speformer --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8 --patch_size 7
# python hessian_matrics_version_sota.py --model speformer --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/speformer/bot --patch_size 7
# python loss_landscape_analysis.py --model speformer --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/speformer/bot --patch_size 7

# python main.py --model group_transformer --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8 --patch_size 7
# python hessian_matrics_version_sota.py --model group_transformer --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/group_transformer/bot --patch_size 7
# python loss_landscape_analysis.py --model group_transformer --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/group_transformer/bot --patch_size 7

# python main.py --model ssftt --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8 --patch_size 7
python hessian_matrics_version_sota.py --model ssftt --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/ssftt/bot --patch_size 7
# python loss_landscape_analysis.py --model ssftt --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/ssftt/bot --patch_size 7

# # transtype = 0
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 0 --patch_size 11
# python hessian_matrics_version_sota.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 0 --weights ./checkpoints/proposed/bot --patch_size 11
# python loss_landscape_analysis.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 0 --weights ./checkpoints/proposed/bot --patch_size 11

# # # transtype = 1
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 11
# python hessian_matrics_version_sota.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 1 --weights ./checkpoints/proposed/bot --patch_size 11
# python loss_landscape_analysis.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 1 --weights ./checkpoints/proposed/bot --patch_size 11

# # # transtype = 2

# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 2 --patch_size 11
# python hessian_matrics_version_sota.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 2 --weights ./checkpoints/proposed/bot --patch_size 11
# python loss_landscape_analysis.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 2 --weights ./checkpoints/proposed/bot --patch_size 11

# # # transtype = 7

# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 7 --patch_size 11
# python hessian_matrics_version_sota.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 7 --weights ./checkpoints/proposed/bot --patch_size 11
# python loss_landscape_analysis.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 7 --weights ./checkpoints/proposed/bot --patch_size 11

# # # transtype = 8
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8 --patch_size 11
# python hessian_matrics_version_sota.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/proposed/bot --patch_size 11
# python loss_landscape_analysis.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/proposed/bot --patch_size 11

# # transtype = 13
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 13 --patch_size 11
# python hessian_matrics_version_sota.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 13 --weights ./checkpoints/proposed/bot --patch_size 11
# python loss_landscape_analysis.py --model proposed --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 13 --weights ./checkpoints/proposed/bot --patch_size 11


# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.04 --trans_type 0 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.04 --trans_type 1 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.04 --trans_type 2 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.04 --trans_type 7 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.04 --trans_type 8 --patch_size 11


# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.12 --trans_type 0 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.12 --trans_type 1 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.12 --trans_type 2 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.12 --trans_type 7 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.12 --trans_type 8 --patch_size 11



# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.28 --trans_type 0 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.28 --trans_type 1 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.28 --trans_type 2 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.28 --trans_type 7 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.28 --trans_type 8 --patch_size 11



# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.36 --trans_type 0 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.36 --trans_type 1 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.36 --trans_type 2 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.36 --trans_type 7 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.36 --trans_type 8 --patch_size 11


# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.44 --trans_type 0 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.44 --trans_type 1 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.44 --trans_type 2 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.44 --trans_type 7 --patch_size 11
# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.44 --trans_type 8 --patch_size 11



# # python main.py --model cnn3d --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8 --patch_size 5
# python hessian_matrics_version_sota.py --model cnn3d --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/cnn3d/bot --patch_size 5
# python loss_landscape_analysis.py --model cnn3d --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/cnn3d/bot --patch_size 5


# python main.py --model dffn --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8 --patch_size 5
# python hessian_matrics_version_sota.py --model dffn --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/dffn/bot --patch_size 5
# python loss_landscape_analysis.py --model dffn --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/dffn/bot --patch_size 5

# python main.py --model m3ddcnn --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8 --patch_size 5
# python hessian_matrics_version_sota.py --model m3ddcnn --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/m3ddcnn/bot --patch_size 5
# python loss_landscape_analysis.py --model m3ddcnn --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/m3ddcnn/bot --patch_size 5

# python main.py --model rssan --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8 --patch_size 5
# python hessian_matrics_version_sota.py --model rssan --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/rssan/bot --patch_size 5
# python loss_landscape_analysis.py --model rssan --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/rssan/bot --patch_size 5

# # python main.py --model speformer --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8 --patch_size 5
# python hessian_matrics_version_sota.py --model speformer --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/speformer/bot --patch_size 5
# python loss_landscape_analysis.py --model speformer --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/speformer/bot --patch_size 5

# # python main.py --model group_transformer --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8 --patch_size 5 
# python hessian_matrics_version_sota.py --model group_transformer --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/group_transformer/bot --patch_size 5
# python loss_landscape_analysis.py --model group_transformer --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/group_transformer/bot --patch_size 5

# python main.py --model ssftt --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 8 --patch_size 5
# python hessian_matrics_version_sota.py --model ssftt --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/ssftt/bot --patch_size 5
# python loss_landscape_analysis.py --model ssftt --dataset_name bot --epoch 300 --bs 64 --ratio 0.2 --trans_type 8 --weights ./checkpoints/ssftt/bot --patch_size 5

# python main.py --model proposed --dataset_name bot --epoch 300 --bs 64 --device 0 --ratio 0.2 --trans_type 1 --patch_size 5

# ## sa dataset 0.1
# python main.py --model proposed --dataset_name ksc --epoch 300 --bs 64 --device 0 --ratio 0.1 --trans_type 0 --patch_size 5 
# python main.py --model proposed --dataset_name ksc --epoch 300 --bs 64 --device 0 --ratio 0.1 --trans_type 1 --patch_size 5 
# python main.py --model proposed --dataset_name ksc --epoch 300 --bs 64 --device 0 --ratio 0.1 --trans_type 2 --patch_size 5 
# python main.py --model proposed --dataset_name ksc --epoch 300 --bs 64 --device 0 --ratio 0.1 --trans_type 7 --patch_size 5 
# python main.py --model proposed --dataset_name ksc --epoch 300 --bs 64 --device 0 --ratio 0.1 --trans_type 8 --patch_size 5 
# python main.py --model m3ddcnn --dataset_name ksc --epoch 300 --bs 64 --device 0 --ratio 0.1 --trans_type 8 --patch_size 5 
# python main.py --model cnn3d --dataset_name ksc --epoch 300 --bs 64 --device 0 --ratio 0.1 --trans_type 8 --patch_size 5 
# python main.py --model rssan --dataset_name ksc --epoch 300 --bs 64 --device 0 --ratio 0.1 --trans_type 8 --patch_size 5 
# python main.py --model dffn --dataset_name ksc --epoch 300 --bs 64 --device 0 --ratio 0.1 --trans_type 8 --patch_size 5 
# python main.py --model speformer --dataset_name ksc --epoch 300 --bs 64 --device 0 --ratio 0.1 --trans_type 8 --patch_size 5 
# python main.py --model ssftt --dataset_name ksc --epoch 300 --bs 64 --device 0 --ratio 0.1 --patch_size 5 
# python main.py --model group_transformer --dataset_name ksc --epoch 300 --bs 64 --device 0 --ratio 0.1 --trans_type 8 --patch_size 5 
