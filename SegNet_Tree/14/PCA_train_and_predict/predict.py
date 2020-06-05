import cv2
import random
import numpy as np
import os,string,sys
import math

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder  

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

COLOR = {
	0 :(  0,   0,   0   ),
    1 :(128,   0,   0	),
    2 :(255, 204,   0	),
    3 :(128, 128,   0	),
    4 :(  0, 255, 128	),
    5 :(  0, 128, 128	),
    6 :(  0,   0, 128	),
    7 :( 75,   0, 128	),
    8 :(255, 140, 105	),
    9 :(230, 230, 250	),
    10:(218, 112, 214	),
    11:(255, 192, 203	),
    12:(255, 255,   0	),
    13:(204, 255,   0	),
    14:(  0, 255, 255	)
}


image_size = 256

classes = [1.,  2.,   3.  , 4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.]  
  
labelencoder = LabelEncoder()  
labelencoder.fit(classes) 



def cut_img(img):
        #path = "D:\\UAV_Images\191015_all_in\tmp"        
        x = [[ None for i in range(wct)] for j in range(hct)]  
        print(x)
        count = 0
        back_ground = np.zeros((hct*256,wct*256,3),dtype=np.uint8) 
        print("back_ground.shape=",back_ground.shape)
        back_ground[0:img.shape[0],0:img.shape[1]] = img               
        for b in range(0,hct):
                for a in range(0,wct):
                        count = count + 1
                        yee = np.zeros((256,256,3),dtype=np.uint8)                 
                        yee = back_ground[0+b*256:256+b*256,0+a*256:256+a*256]    
                        x[b][a] = yee  
                        #cv2.imshow("["+ str(b) +"]"+  "["+ str(a) +"]" +"photo c"+str(count), yee)
                        #cv2.imwrite(os.path.join(path, "["+ str(b) +"]"+  "["+ str(a) +"]" +"photo_c"+str(count)+".bmp"), yee)                      
                        cv2.imwrite("./tmp/"+str(count)+".bmp", yee)                      
                        

        return (x,wct,hct)  
        
def merge_img(plist,wct,hct):   
        count = 0
        
        x_list = []  
        y_list = []
        merged_tmp = []
        for a in range(0,hct):
                x_list = []
                for b in range(0,wct):
                        count = count + 1 
                        #cv2.imshow("["+ str(a) +"]"+  "["+ str(b) +"]" +"photo_c"+str(count), plist[a][b])
                        x_list.append(plist[a][b])
                        
                merged_tmp = np.hstack(tuple(x_list))
                y_list.append(merged_tmp)


        merged_y = np.vstack(tuple(y_list))
        #cv2.imshow("suc========",merged_y)
        cv2.imwrite("merge_result.bmp",merged_y)                

def key_make(size=5, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
def color(img):
	tmp = "./tmp/"+key_make()+".png"
	cv2.imwrite(tmp,img)
	a = cv2.imread(tmp)
	for b in a :
		for c in b:
			
			mask = c[0]
			c[0] = COLOR[mask][0]
			c[1] = COLOR[mask][1]
			c[2] = COLOR[mask][2]
			#print (c)


	os.remove(tmp)
	return a #cv2.imwrite(IMG_PATH+"ss.png",a)

model = load_model("./segnet.h5")

def predict(image):#cv2.imread('./test/' + path)

    
    stride = image_size 
    #model = load_model("./segnet.h5")
    h,w,_ = image.shape
    padding_h = (h//stride + 1) * stride 	
    padding_w = (w//stride + 1) * stride
    padding_img = np.zeros((padding_h,padding_w,3),dtype=np.uint8)
    padding_img[0:h,0:w,:] = image[:,:,:]
    padding_img = padding_img.astype("float") / 255.0
    padding_img = img_to_array(padding_img)

    mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
    for i in range(padding_h//stride):
        for j in range(padding_w//stride):
            crop = padding_img[i*stride:i*stride+image_size,j*stride:j*stride+image_size,:3]
            ch,cw,_ = crop.shape

            if ch != 256 or cw != 256:

                continue                   
            crop = np.expand_dims(crop, axis=0)

            pred = model.predict_classes(crop,verbose=2)
            pred = labelencoder.inverse_transform(pred[0])  
            #print (np.unique(pred))  
            pred = pred.reshape((256,256)).astype(np.uint8)
            #print 'pred:',pred.shape
            mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size] = pred[:,:]

    #cv2.imwrite('aa.png',mask_whole[0:h,0:w])
    return color(mask_whole[0:h,0:w]) #cv2.imwrite('./predict/pre'+str(n+1)+'.png',mask_whole[0:h,0:w])

# main         

each_result_path = "./each_result"
tmp_p = "./tmp"

if not os.path.isdir(each_result_path)  :
    os.mkdir("./each_result")
if not os.path.isdir(tmp_p):
    os.mkdir("./tmp")


print(os.path.isdir(each_result_path))

print(os.path.isdir(tmp_p))


all_count = 0
file_1 = os.listdir("./val")
print("///////",file_1)
for a in file_1:       
    #for b in os.listdir("./val/"+a) :
    photo_name = a
    print("photo_name=",photo_name)
    
    #photo_name = sys.argv[1]
    img = np.zeros(1)
    img = cv2.imread("./val/" + photo_name)
    wct = math.ceil(img.shape[1]/256)  
    hct = math.ceil(img.shape[0]/256) 
    #cv2.imshow('My Image', img)
    print("=========wvt,hct="+str(wct)+","+str(hct))
    

    #for c in range(1,wct*hct+1):
        #imgpath = "./tmp/" + str(c) + ".bmp"        
    try:
    #if True:            
        img = predict(img)
        cv2.imwrite("./each_result/new" + str(photo_name) + ".bmp",img)
        print("fin===>" + str(photo_name))
    except Exception as e:
        #os.remove(img)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("ERROR:",end="")
        print(exc_type, exc_obj, exc_tb.tb_lineno)

    
    
    print("====================================suc",all_count)
    all_count+=1
