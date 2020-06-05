how to run example: 

python test.py -image 24-23_6.jpg -weight resnet50_ori.hdf5

python test.py -image 10-63_1.jpg -weight resnet50_ori.hdf5

***************************************
getLayers.py : 抓取每層圖片的特徵圖

python getLayers.py -image 10-63_1.jpg -network resnet

python getLayers.py -image 10-63_1.jpg -network resnet -weight Resnet_Original_14.hdf5