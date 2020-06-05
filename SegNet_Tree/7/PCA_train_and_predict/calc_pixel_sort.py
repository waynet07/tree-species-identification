import cv2
import numpy as np
import math,time
import sys,os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

y_true = []
y_pred = []

# 1、2、4、7、8、10、11 編號的 取出來 訓練七類

COLOR = {
    1 :(128,   0,   0	),
    2 :(255, 204,   0	),
    4 :(  0, 255, 128	),
    7 :( 75,   0, 128	),
    8 :(255, 140, 105	),
    10:(218, 112, 214	),
    11:(255, 192, 203	)
}

def calc_and_sort(img,p_name):

    #wct = math.ceil(img.shape[1]/256)  
    #hct = math.ceil(img.shape[0]/256) 
   ## print(img.shape)
    for a in range(0,img.shape[0]):
        for b in range(0,img.shape[1]):
                x = tuple(img[a,b].tolist())  #numpy.ndarray  to lsit to tuple   = =
                
                #去color 找起來 for aa in range(1,len(COLOR)):
                for aa in COLOR.keys(): 
                    if( x == COLOR[aa]):
                        dict_count[str(aa)] = dict_count[str(aa)]  + 1

    '''
    yee = []
    yee = [ a for a in sorted(dict_count.values(),reverse=True)]
    print("yee",yee)
    print()
    '''
    #calc dict_count each value total
    di_ct_v = []
    di_ct_v = dict_count.values()
    total = 0
    for a in di_ct_v:
        total = total + a
   ## print("total=",total)

    list_vk = []
    for k,v in dict_count.items():
        list_vk.append((v,k))
    #print(list_vk)
    sorted_vk  = sorted(list_vk,reverse=True)
    ## print("sorted_vk===",sorted_vk)   

    ans = []
    ans2 = []         
    for a in range(0,1):
        ans.append( [ sorted_vk[a][1] ,  sorted_vk[a][0] ] )
        ans2.append(  [ p_name, sorted_vk[a][1] , round((sorted_vk[a][0]/total),4) ] )
  #  print(ans)
  #  print(ans2)

    find_unline = re.search(r"_\d\d?",p_name)
    find_unline = find_unline.group(0)
    find_unline = re.search(r"\d\d?",find_unline)
    # print(find_unline.group(0))
    # print(type(find_unline.group(0)))
    
    ans_predict_class = sorted_vk[a][1]
  #  print("label,predict===================>",find_unline.group(0),",",ans_predict_class)
    y_pred.append(ans_predict_class)
    y_true.append(find_unline.group(0))


    if(ans_predict_class == find_unline.group(0)):
        global suc_count
        global all_count
        suc_count += 1
   ##     print("cong~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",all_count)
   ## else:
   ##     print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",all_count)

    
    with open("./ranking.txt","a") as yee:
        yee.write(str(ans2)+"\n") 
        all_count += 1

    confmatrix1 = confusion_matrix(y_true, y_pred)
    # print (confmatrix1)
  #  target_names = ['class 1', 'class 2', 'class 3','class 4','class 5','class 6',
  #  'class 7','class 8','class 9','class 10','class 11','class 12','class 13','class 14']
  #  print(classification_report(y_true, y_pred, target_names=target_names))
   # ConfusionMatrixNew(confmatrix1)
	
def ConfusionMatrixPlot(confmatrix):    
    #pd.DataFrame(confmatrix).to_csv('confusion_matrix.csv')
    clsnames = np.arange(0, 7)
    tick_marks = np.arange(len(clsnames))
    plt.figure(figsize=(10, 10))
    plt.title('Confusion matrix of SegNet for PCA-based images',fontsize=15,pad=10)
    iters = np.reshape([[[i, j] for j in range(len(clsnames))] for i in range(len(clsnames))], (confmatrix.size, 2))
    for i, j in iters:
        plt.text(j, i, format(confmatrix[i, j]), fontsize=15, va='center', ha='center')  # 显示对应的数字

    plt.gca().set_xticks(tick_marks + 0.5, minor=True)
    plt.gca().set_yticks(tick_marks + 0.5, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')

    plt.imshow(confmatrix, interpolation='nearest', cmap='cool')  # 按照像素显示出矩阵
    plt.xticks(tick_marks,['1','2','4','7','8','10','11'])#7分类 横纵坐标类别名
    plt.yticks(tick_marks,['1','2','4','7','8','10','11'])
   
    plt.ylabel('Actual Species',labelpad=-5,fontsize=15)
    plt.xlabel('Predicted Species',labelpad=10,fontsize=15)
    plt.ylim([len(clsnames) - 0.5, -0.5])
    plt.tight_layout()

    cb=plt. colorbar()#热力图
   #cb.set_label('Numbers of Predict',fontsize = 15)
    plt.savefig('./Results/Confusion Matrix.png')

#main =================

print("strat the timer WWWWWWWWWWWWWWWWW")
t_strat = time.time()

if ("ranking.txt" in os.listdir("./")):
    os.remove("./ranking.txt")



photo_name_all = os.listdir("./each_result")


all_count = 0 
suc_count = 0
for p_name in photo_name_all:
  ##  print("photo_name=",p_name)
    img = cv2.imread("./each_result/"+p_name)

    dict_count={}
    for a in range(0,15+1):
        dict_count[str(a)]=0
    #print(dict_count)

    calc_and_sort(img,p_name)

confmatrix1 = confusion_matrix(y_true, y_pred)
print (confmatrix1)

matrixInput = np.array(confmatrix1)

PercentageInput = (matrixInput.T / matrixInput.astype(np.float).sum(axis=1)).T

#print (PercentageInput)

AroundPercentageInput = np.around(PercentageInput, decimals=3)

print (AroundPercentageInput)

ConfusionMatrixPlot(AroundPercentageInput)

t_end = time.time()
percentage = suc_count/ all_count

target_names = ["1","2","4","7","8","10","11"]
print(classification_report(y_true, y_pred, target_names=target_names))

f = open("./Results/pred_results.txt", "w")
f.write(classification_report(y_true, y_pred, target_names=target_names))
#f.write("suc_count,all_coun,=",suc_count,"/",all_count)
#f.write("percentage=",percentage)
f.write("end the timer  MMMMMMMMMMMMMMMMMM"+"\n")
f.write("total testing time =" +  str(t_end - t_strat) + " sec" )
f.close()

print("suc_count,all_coun,=",suc_count,"/",all_count)

print("percentage=",percentage)

print("end the timer  MMMMMMMMMMMMMMMMMM")
print("total testing time =" +  str(t_end - t_strat) + " sec" )



