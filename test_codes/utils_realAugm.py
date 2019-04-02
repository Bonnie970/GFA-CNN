from random import shuffle
import scipy.misc
import numpy as np


def image_resize(x, resize_w):
    return scipy.misc.imresize(x, [resize_w, resize_w])

def transform(image, is_resize, resize_w, aug):
    if(aug[:-1] == 'flip_ud'):
        image = image[::-1,:,:]
    if(aug[:-1] == 'flip_lr'):
        image = image[:,::-1,:]
    if is_resize:
        resize_image = image_resize(image, resize_w)
    else:
        resize_image = image
    return np.array(resize_image)/127.5 - 1.

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def get_tra_image(image_path, is_resize=True, resize_w=64, is_grayscale = False):
    [ipath, idLbl, anLbl, aug] = image_path.split(' ')
    return transform(imread(ipath, is_grayscale), is_resize, resize_w, aug)

def get_tst_image(image_path, is_resize=True, resize_w=64, is_grayscale = False):
    aug = 'none'
    [ipath, _, anLbl] = image_path.split(' ')
    return transform(imread(ipath, is_grayscale), is_resize, resize_w, aug)

def get_tra_antiLabel(image_path, n_class):
    [ipath, idLbl, anLbl, aug]= image_path.split(' ')
    return anLbl

def get_tra_idLabel(image_path, n_class):
    [ipath, idLbl, anLbl, aug]= image_path.split(' ')
    return idLbl

def get_tst_antiLabel(image_path, n_class):
    image_path = image_path.strip()
    [ipath, _, anLbl]= image_path.split(' ')
    return anLbl

def get_idxxx(image_path, n_class):
    xx = '0'
    return xx

def image_resize(x, resize_w):
    return scipy.misc.imresize(x, [resize_w, resize_w])

def realAugument(lists):
    fr = open('./realAugument_list.txt', 'w')
    for i in range(len(lists)):
        ss = lists[i].split(' ')
        if(int(ss[2]) == 0):
            fr.write(lists[i][:-1] +  ' ' + 'none' + '\n')
            fr.write(lists[i][:-1] +  ' ' + 'flip_ud' + '\n')        
            fr.write(lists[i][:-1] +  ' ' + 'flip_lr' + '\n')
        else:
            fr.write(lists[i][:-1] +  ' ' + 'none' + '\n')
    fr.close() 
    ff = open('./realAugument_list.txt')
    Ls = ff.readlines()
    return Ls

def assignIDlbl_realAugm(lists, lblDic):
    Ls = [ss.split(' ')[0]+' '+lblDic[ss.split(' ')[1]][:-1]+' '+ ss.split(' ')[2]+' '+ ss.split(' ')[3] for ss in lists]
    return Ls






