# written by Bin Sun
# email: sun.bi@northeastern.edu

echo "starting script"
mkdir train_videos
mkdir val_videos
######change path_to_train_videos to your real path for training videos#####################
mv "C:/Users/bolua/Desktop/University of Kansas/Senior Year/data_test/train" train_videos/
######change path_to_val_videos to your real path for val videos#####################
mv "C:/Users/bolua/Desktop/University of Kansas/Senior Year/data_test/val" val_videos/

cd data_process
python wholepose_features_extraction.py --video_path ../train_videos/ --feature_path ../data/train_features --istrain True
python wholepose_features_extraction.py --video_path ../val_videos/ --feature_path ../data/train_features
cd ..
# if you want to delete videos, un common the following command
#rm -rf train_videos
#rm -rf val_videos

####### training #############################
echo "training start"
python train_parallel.py --dataset_path "C:/Users/bolua/Desktop/University of Kansas/Senior Year/Sign-Language-Translator/src/SSTCN/data/train_features"  --batch_size 160
echo "done training"
###### testing ###########################
echo "testing start"
python test.py --dataset_path "C:/Users/bolua/Desktop/University of Kansas/Senior Year/Sign-Language-Translator/src/SSTCN/data/train_features"
echo "done testing"
#python test.py --checkpoint_model model_checkpoints/your model

echo "hit enter to close"
read
