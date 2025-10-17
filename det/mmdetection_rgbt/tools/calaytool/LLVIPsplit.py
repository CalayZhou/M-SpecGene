import os.path as osp
import mmcv
import os
import pickle



pkl_filename = '/home/calay/DATASET4/object detection/LLVIP/AnnotationsCOCO/LLVIPcoco.json'
with open(pkl_filename, 'rb') as fid:
    pkl = pickle.load(fid, encoding='iso-8859-1')

pass