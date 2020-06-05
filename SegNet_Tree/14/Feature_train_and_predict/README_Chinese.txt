============= 產生 src label 兩個資料夾 =============
使用產生方式:  
 python right_mask.py

=================訓練=================
初始條件: 
1.需有 src label 兩個資料夾  
2.執行 generate_val.py 產生 val 資料夾
  python generate_val.py

使用訓練方式:
  python train.py -m segnet.h5

================預測前置處理動作================
初始條件: 
1.需有已訓練過後的檔案，檔名如右: segnet.h5 
2. 將要辨識的檔案，放置於 val 資料夾內

使用預測辨識方式:
  python predict.py

================輸出最有可能的分類結果================
初始條件:
1. 針對 val 資料夾，計算當中的每一個檔案，輸出最有可能的分類結果。

輸出最有可能的分類結果的方法:
  python calc_pixel_sort.py