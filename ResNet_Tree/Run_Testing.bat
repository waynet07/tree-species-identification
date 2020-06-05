python CNNtrainAndTest.py -type 1 -folder \14\Original -network resnet -weight Resnet_Original_14.hdf5 -size 14 -name Original
python CNNtrainAndTest.py -type 1 -folder \14\Feature -network resnet -weight Resnet_Feature_14.hdf5 -size 14 -name CS-based
python CNNtrainAndTest.py -type 1 -folder \14\PCA -network resnet -weight Resnet_PCA_14.hdf5 -size 14 -name PCA-based
python CNNtrainAndTest.py -type 1 -folder \7\Original -network resnet -weight Resnet_Original_7.hdf5 -size 7 -name Original
python CNNtrainAndTest.py -type 1 -folder \7\Feature -network resnet -weight Resnet_Feature_7.hdf5 -size 7 -name CS-based
python CNNtrainAndTest.py -type 1 -folder \7\PCA -network resnet -weight Resnet_PCA_7.hdf5 -size 7 -name PCA-based