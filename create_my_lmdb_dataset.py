""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

# import fire
import os
import lmdb
import cv2

import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def COCOText_data_processing(datalist):
    def check(s):
        try:
            imagePath, label = s.strip('\n').split(',',1)
        except:
            return False
        if len(imagePath) == 7:
            for i in imagePath:
                if not i.isdigit():
                    return False
            return True
        else:
            return False
    new_datalist = []
    i = 0
    maxlen = len(datalist)
    while i<maxlen:
        s = datalist[i]
        assert s[-1] == '\n'
        while i<maxlen-1 and not check(datalist[i+1]):
            i = i+1
            s = s[:-1]+datalist[i]
            flag=True
        new_datalist.append(s)
        i = i+1
    return new_datalist

def createDataset(inputPath, gtFile, outputPath, checkValid=True, dataname='COCOText'):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()
        if dataname == 'COCOText':
            datalist = COCOText_data_processing(datalist)

    nSamples = len(datalist)
    for i in range(nSamples):
        if dataname in ('COCOText','SynthTextAdd'):
            try:
                imagePath, label = datalist[i].strip('\n').split(',',1)
            except Exception as E:
                print(datalist[i], len(datalist[i]))
                print('Exception:{}'.format(E))
                continue
            if dataname == 'SynthTextAdd':
                assert label[0] == label[-1] == '\"'
                label = label[1:-1]
            elif dataname == 'COCOText':
                if label[0] == '|' and label[-1] == '|':
                    label = label[1:-1]
                assert '|' not in label
                imagePath = '{}.jpg'.format(imagePath)
        else:
            imagePath, label = datalist[i].strip('\n').split('\t')
        imagePath = os.path.join(inputPath, imagePath)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

import os
def create_SynthTextAdd(root='/data1/chuxiaojie/Datasets/SynthText_Add/', target_root='/data1/chuxiaojie/Datasets/SynthTextAdd_lmdb'):
    for i in range(1,21):
        inputPath = os.path.join(root,'crop_img_{}'.format(i))
        gtFile = os.path.join(root,'annotationlist/gt_{}.txt'.format(i))
        outputPath = os.path.join(target_root,'crop{}'.format(i))
        createDataset(inputPath, gtFile, outputPath, dataname='SynthTextAdd')

def create_COCOText(root='/data1/chuxiaojie/Datasets/COCOText/', target_root='/data1/chuxiaojie/Datasets/COCOText_lmdb'):
    for name in ['train','val']:
        inputPath = os.path.join(root,'{}_words'.format(name))
        gtFile = os.path.join(root,'{}_words_gt.txt'.format(name))
        outputPath = os.path.join(target_root,'{}_words'.format(name))
        createDataset(inputPath, gtFile, outputPath, dataname='COCOText')


if __name__ == '__main__':
    # fire.Fire(createDataset)
    create_SynthTextAdd()
    create_COCOText()
