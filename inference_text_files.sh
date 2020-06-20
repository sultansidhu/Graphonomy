vid_list=$1
python exp/inference/inference_text_files.py --loadmodel $HOME/Graphonomy/data/pretrained_model/inference.pth --BASE_DIR $HOME/modidatasets/VoxCeleb2/ --video_list $vid_list 
