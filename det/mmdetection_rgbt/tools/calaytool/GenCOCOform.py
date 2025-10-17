'''
3.生成mmdetection所需的标签格式

例如针对 IOD-Video_COCO_S1,分别修改out_file、IOD_TEST_DST为
out_file = 'GOD_val_S1.json'
IOD_TEST_DST = "/home/calayzhou/Dataset/GOD-Video/IOD-Video_COCO_S1/val/"
与
out_file = 'GOD_train_S1.json'
IOD_TEST_DST = "/home/calayzhou/Dataset/GOD-Video/IOD-Video_COCO_S1/train/"

'''

import os.path as osp
import mmcv
import os
import pickle



pkl_filename = 'TrueLeakedGas.pkl'
with open(pkl_filename, 'rb') as fid:
    pkl = pickle.load(fid, encoding='iso-8859-1')

pkl_gttubes = {}
for key in pkl['gttubes']:
    new_key = key.split('/')[1][:3]
    value = pkl['gttubes'][key][0][0]
    pkl_gttubes.update({new_key: value})

out_file = 'GOD_val.json'
IOD_TEST_DST = "/home/calayzhou/zkl/Dataset/IOD-Video_COCO/val/"
# out_file = 'GOD_train.json'
# IOD_TEST_DST = "/home/calayzhou/zkl/Dataset/IOD-Video_COCO/train/"


annotations = []
images = []
obj_count = 0
count = 0
for image_sample in os.listdir(IOD_TEST_DST):
    # print(image_sample)
    # for idx, v in enumerate(mmcv.track_iter_progress(data_infos.values())):
    filename = image_sample #v['filename']
    img_path = osp.join(IOD_TEST_DST, filename)
    # height, width = mmcv.imread(img_path).shape[:2]
    height, width = 240, 320

    video_id = image_sample.split('.')[0].split('_')[0]
    frame_id = image_sample.split('.')[0].split('_')[1]
    idx = int(video_id) * 100000 + int(frame_id)

    images.append(dict(
        id=idx,
        file_name=filename,
        height=height,
        width=width))

    video_length = pkl_gttubes[video_id].shape[0]
    bbox = pkl_gttubes[video_id][int(frame_id) - 1][1:]
    bboxes = []
    labels = []
    masks = []
    # for _, obj in v['regions'].items():
    #     assert not obj['region_attributes']
    #     obj = obj['shape_attributes']
    #     px = obj['all_points_x']
    #     py = obj['all_points_y']
    #     poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
    #     poly = [p for x in poly for p in x]
    #
    #     x_min, y_min, x_max, y_max = (
    #         min(px), min(py), max(px), max(py))


    x_min = float(bbox[0])
    y_min = float(bbox[1])
    x_max = float(bbox[2])
    y_max =float(bbox[3])
    # if x_min < 0 or y_min < 0 or x_max < 0 or y_max <0 or \
    #         x_min > width or y_min > height or x_max > width or y_max >height :
    # if x_min < 0 :
    #     x_min = 0
    #     print(x_min, y_min, x_max, y_max)
    #     count = count + 1
    # if x_max > width:
    #     x_max = width
    #     print(x_min, y_min, x_max, y_max)
    #     count = count + 1
    # if y_min < 0 :
    #     y_min = 0
    #     print(x_min, y_min, x_max, y_max)
    #     count = count + 1
    # if x_min > x_max :
    #     print(x_min, y_min, x_max, y_max)
    # if  y_min > y_max:
    #     print(x_min, y_min, x_max, y_max)
    if x_max - x_min<1:
        print(x_min, y_min, x_max, y_max)
    if y_max - y_min<1:
        print(x_min, y_min, x_max, y_max)

    poly = [(x_min, y_min),(x_max, y_max)]
    poly = [p for x in poly for p in x]

    data_anno = dict(
        image_id=idx,
        id=obj_count,
        category_id=0,
        bbox= [x_min, y_min, x_max - x_min, y_max - y_min],
        area=(x_max - x_min) * (y_max - y_min),
        segmentation=[poly],
        video_length = video_length,
        iscrowd=0)
    annotations.append(data_anno)
    obj_count += 1

coco_format_json = dict(
    images=images,
    annotations=annotations,
    categories=[{'id':0, 'name': 'Gas'}])
mmcv.dump(coco_format_json, out_file)


print(count)


