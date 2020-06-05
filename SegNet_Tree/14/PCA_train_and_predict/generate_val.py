import re,os
import random
import cv2


try:
    p = "./src"

    if not ( os.path.isdir("./val") ):
        os.mkdir("./val")
    print()
    list_all = os.listdir(p)

    #for a in range(0,5):
        #img = imread("./scr/"+list_all[])

    list_catch = random.sample(list_all,500)
    print(len(list_catch),"move")
    for a in list_catch:
        img = cv2.imread("./src/"+a)
        cv2.imwrite("./val/"+a,img)  #(name,writePhoto)
        os.remove("./src/"+a)
        os.remove("./label/"+a)

    print("suc")
except:
    print("err")

'''
#check

list_src = os.listdir("./src")
list_label = os.listdir("./label")

s1 = set(list_src)
s2 = set(list_label)
print(s1.difference(s2))
'''