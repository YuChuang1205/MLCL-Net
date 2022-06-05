#coding=gbk
'''
Created on 2021年4月28日

@author: 余创
'''

from model import *
from utils import *
import os
import cv2


def soft_loss(y_true,y_pred):
    loss  = 1-(K.sum(y_true*y_pred)+1e-6)/(K.sum(y_true)+K.sum(y_pred)-K.sum(y_true*y_pred)+1e-6)
    return loss

def soft_acc(y_true, y_pred):
    acc_out = (K.sum(y_true*y_pred)+1e-6)/(K.sum(y_true)+K.sum(y_pred)-K.sum(y_true*y_pred)+1e-6)
    return acc_out


root_path = os.path.abspath('.')
pic_path = os.path.join(root_path,"data/test/image")
pic_out_path = os.path.join(root_path,"data/test_results")
make_dir(pic_out_path)

model_path = os.path.join(root_path,"logs/seg_model_best.hdf5")
model = load_model(model_path,custom_objects={'soft_loss':soft_loss,'soft_acc':soft_acc})


piclist = os.listdir(pic_path)
#piclist.sort(key= lambda x:int(x[:-4])) 
for n in range(len(piclist)):
#     print(image.shape)
    new_name = piclist[n].split('.')[0]+'.png'
    image = cv2.imread(os.path.join(pic_path,piclist[n]))
    #print(image.shape)
    #print(image)
    image=image/255
    h,w,c = image.shape
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image,verbose=1)
    print(np.max(pred))
    pred = np.where(pred>=0.5,255,0)
    pred = pred.reshape((h,w,1))
    print(np.shape(pred))
    
    cv2.imwrite(os.path.join(pic_out_path,new_name), pred)
    print(np.max(pred),np.min(pred))
print("Done!!!")


