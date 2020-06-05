# -*- coding:utf-8 -*-
import cv2,os
import numpy as np #1

from PIL import Image
COLOR = {
    1:(128, 0, 0	),
    2:(255, 204, 0	),
    3:(128, 128, 0	),
    4:(0, 255, 128	),
    5:(0, 128, 128	),
    6:(0, 0, 128	),
    7:(75, 0, 128	),
    8:(255, 140, 105	),
    9:(230, 230, 250	),
    10:(218, 112, 214	),
    11:(255, 192, 203	),
    12:(255, 255, 0	),
    13:(204, 255, 0	),
    14:(0, 255, 255	),
    15:(139, 0, 255	)
}
try:
    os.mkdir("src")
    os.mkdir("label")
except:
    pass
Image.MAX_IMAGE_PIXELS = 999999999999999
global count 
count = 0
class BIG():
    def __init__(self):
        #self.canvas = np.zeros((256, 256, 3), dtype="uint8")
        self.mask = canvas = np.zeros((256, 256, 3), dtype="uint8") #3
        self.first_flag = True
        self.wight = 0
        self.hight = 0
        self.pre_wight = 15
        self.pre_hight = 15
        self.class_num = []
        self.howbig = 256
    def isClass(self,label_num):
        if label_num in self.class_num:
            return True
        self.class_num.append(label_num)
        return False
    def returnpic(self):
        return self.canvas,self.mask
    def add(self,pic,labal_num):
        if (self.first_flag):
            self.isClass(labal_num)
            self.first_flag = False
            self.canvas = pic
            contours = np.array( [ [0,0], [0,self.howbig], [self.howbig, self.howbig], [self.howbig,0] ] )
            cv2.fillPoly(self.mask,pts =[contours], color=(labal_num,labal_num,labal_num))
            self.wight += self.pre_wight
            self.hight += self.pre_hight
        else:
            if not self.isClass(labal_num):
                for i in range(len(pic)):
                    for j in range(len(pic[i])):
                        if not j < self.hight :
                            self.canvas[i][j]=pic[i][j]
                            self.mask[i][j]=labal_num
                
                        
                #little = pic[self.wight:self.hight,self.howbig-self.wight:self.howbig-self.hight] #截取第5行到89行的第500列到630列的区域
                #print(pic,little)
                #cv2.imshow("截取", little)
                #self.canvas[self.wight:self.hight,self.howbig-self.wight:self.howbig-self.hight]= little
                #contours = np.array( [ [self.hight,self.wight], [self.hight,self.howbig], [self.howbig, self.howbig], [self.howbig,self.wight] ] )
                #cv2.fillPoly(self.mask,pts =[contours], color=(labal_num,labal_num,labal_num))
                
                #cv2.imshow("原来", self.canvas)
                #cv2.imshow("原来", self.mask)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
        
                self.wight += self.pre_wight
                self.hight += self.pre_hight

MIX = BIG()
def cut(id,vx,vy):

    global count
    count+=1
    name1 = "org/"+id 
    name2 = "src/"+str(count) + "-"
    name4 = "label/"+str(count) + "-"
    import re
    
    num= int(re.search('([0-9]+)',i).group(1))
    #im =Image.open(name1)
    import tifffile as tiff
    im =  tiff.imread(name1)
    print(im.shape)
    im_y,im_x,trash = im.shape
    im = cv2.resize(im,(int(im_x/10),int(im_y/10)),interpolation=cv2.INTER_CUBIC) 
    im_y,im_x,trash = im.shape
    #im_x ,im_y = im.size
    #偏移量
    print(im_x,end="x")
    print(im_y)
    print(count,num,id)
    im = Image.fromarray(im.astype('uint8'), 'RGB')
    print(im.size)
    dx = 64
    dy = 64
    n = 1
    import time
    time.sleep(0.5)

    x1 = 0
    y1 = 0
    x2 = vx
    y2 = vy
    
   
    while x2 <= im_y:
     
        while y2 <= im_x:
            name3 = name2 + str(n) + "_" +str(num) + ".jpg"
           
            im2 = im.crop((y1, x1, y2, x2))
            im2.save(name3)
            y1 = y1 + dy
            y2 = y1 + vy
            
            name5 = name4 + str(n) + "_" +str(num) + ".jpg"
            canvas = np.zeros((256, 256, 3), dtype="uint8") #3
            contours = np.array( [ [0,0], [0,256], [256, 256], [256,0] ] )
            cv2.fillPoly(canvas,pts =[contours], color=(num,num,num))
            n = n + 1
            MIX.add(np.array(im2),num)

            print(canvas)
            print([num])
            
            import imageio
            imageio.imwrite(name5, canvas)

        x1 = x1 + dx
        x2 = x1 + vx
        y1 = 0
        y2 = vy

    print ("suc")
    return n-1


if __name__=="__main__":

    
    for i in os.listdir("org"):
        id = str(i)  
        res = cut(id,256,256)

        print (res)
    canvas,mask=MIX.returnpic()
    cv2.imwrite("src/0.jpg", canvas)
    
    cv2.imwrite("label/0.jpg",mask)
