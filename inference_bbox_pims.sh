start=$1
end=$2
python exp/inference/inference_bbox_pims.py --loadmodel $HOME/Graphonomy/data/pretrained_model/inference.pth --BASE_DIR $HOME/modidatasets/VoxCeleb2/ --START $start --END $end 

