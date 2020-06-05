1、11、8、10、7、2、4 編號的 取出來 訓練七類
3、5、6、9、12、13、14

del *_3.jpg
del *_5.jpg
del *_6.jpg
del *_9.jpg
del *_12.jpg
del *_13.jpg
del *_14.jpg

python -m venv tensorflow_2.0_V1


conda info --envs

conda create --name tensorflow_2.0_V1 python=3.6

conda create --name tensorflow1 python=3.6

conda activate aaa
conda activate tensorflow1 

conda activate tensorflow_2.0_V1

============= 產生 src label 兩個資料夾 =============
  python right_mask.py

=================train=================
初始條件: 
1.需有 src label 兩個資料夾  
2.執行 generate_val.py 產生 val 資料夾
  python generate_val.py

使用方式:
  python train.py -m segnet.h5
  python train.py -m "segnet_7+1.h5"

================predict================
初始條件: 
1.需有 已訓練 segnet.h5 
2.需有 val 資料夾內含多個資料夾

使用方式:
  python predict.py

================val================
初始條件:
1.預測出來的each_result

使用方式:
  python calc_pixel_sort.py