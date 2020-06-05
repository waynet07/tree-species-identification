==========Creatation==========
Create two folders: "src" and "label" folders
    Run:  python right_mask.py

==========Training==========
Initialization requirements:
1. a "src" folder and a "label" folder
2. implement generate_val.py to generate val folder 
    Run: python generate_val.py
3. train model
    Run: python train.py -m segnet.h5

==========Pre-processing for prediction==========
Initialization requirements:
1. a trained model file named "segnet.h5"
2. put the files you want identified in "val" folder
    Run: python predict.py

==========Output the most possible classification results==========
Initialization requirement:
1. calculate each file in "val" folder and output the most possible classification results
    Run:  python calc_pixel_sort.py



