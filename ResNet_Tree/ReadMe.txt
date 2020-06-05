train ==> 
python CNNtrainAndTest.py -type 0 -folder \14\Original  -network resnet -weight Resnet_Original_14.hdf5
python CNNtrainAndTest.py -type 0 -folder \14\Feature -network resnet -weight Resnet_Feature_14.hdf5
python CNNtrainAndTest.py -type 0 -folder \14\PCA -network resnet -weight Resnet_PCA_14.hdf5
python CNNtrainAndTest.py -type 0 -folder \7\Original  -network resnet -weight Resnet_Original_7.hdf5
python CNNtrainAndTest.py -type 0 -folder \7\Feature -network resnet -weight Resnet_Feature_7.hdf5
python CNNtrainAndTest.py -type 0 -folder \7\PCA -network resnet -weight Resnet_PCA_7.hdf5

test ==> 
python CNNtrainAndTest.py -type 1 -folder \14\Original  -network resnet -weight Resnet_Original_14.hdf5
python CNNtrainAndTest.py -type 1 -folder \14\Feature -network resnet -weight Resnet_Feature_14.hdf5
python CNNtrainAndTest.py -type 1 -folder \14\PCA -network resnet -weight Resnet_PCA_14.hdf5
python CNNtrainAndTest.py -type 1 -folder \7\Original  -network resnet -weight Resnet_Original_7.hdf5
python CNNtrainAndTest.py -type 1 -folder \7\Feature -network resnet -weight Resnet_Feature_7.hdf5
python CNNtrainAndTest.py -type 1 -folder \7\PCA -network resnet -weight Resnet_PCA_7.hdf5

getLayers.py : 抓取每層圖片的特徵圖

python getLayers.py -image XXX.jpg -network resnet

-image: 指定輸入圖片
-network: 指定網路架構 resnet vgg segnet

CNNtrainAndTest.py : 訓練和測試訓練結果

train ==> 
python CNNtrainAndTest.py -type 0 -folder Original -network resnet -weight Resnet_Original_14.hdf5

test ==> 
python CNNtrainAndTest.py -type 1 -folder Original -network resnet -weight Resnet_Original_14.hdf5

python CNNtrainAndTest.py -type X -folder XXX -network XXX -weight XXX.hdf5

-type: 訓練為0 測試為1
-folder: 指定訓練資料夾(會自己連接資料夾內的"train"和"vali"， 如果是測試則是"test")

-network: 指定訓練網路(vgg or resnet)

-weight: 載入測試資料夾，如果是測試不可為空