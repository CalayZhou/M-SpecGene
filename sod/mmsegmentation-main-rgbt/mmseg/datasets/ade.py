# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ADE20KDataset(BaseSegDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        # classes=('Background', 'Car_Stop', 'Bike', 'Bicyclist',
        #          'Motorcycle', 'Motorcyclist', 'Car', 'Tricycle', 'Traffic_light',
        #          'Box', 'Pole', 'Curve', 'Person'),
        # palette=[[0, 0, 0], [0, 0, 142], [0, 60, 100], [0, 0, 230], [119, 11, 32],
        #          [255, 0, 0], [0, 139, 139], [255, 165, 150], [192, 64, 0], [211, 211, 211],
        #          [100, 33, 128], [117, 79, 86], [153, 153, 153]])
        classes=('Background', 'Saliency'),
        palette=[[0, 0, 0], [150, 150, 0]])
        # classes=('Background', 'Car', 'Bus', 'Motorcycle', 'Bicycle',
        #          'Pedestrian', 'Motorcyclist','Bicyclist', 'Cart', 'Bench',
        #          'Umbrella', 'Box', 'Pole', 'Street_lamp', 'Traffic_light',
        #          'Traffic_sign', 'Car_stop', 'Color_cone', 'Sky', 'Road',
        #          'Sidewalk', 'Curb', 'Vegetation', 'Terrain', 'Building',
        #          'Ground'),
        # palette=[[0, 0, 0], [0, 0, 142], [0, 60, 100], [0, 0, 230],[119, 11, 32],
        #          [255, 0, 0], [0, 139, 139], [255, 165, 150], [192, 64, 0],[211, 211, 211],
        #          [100, 33, 128], [117, 79, 86], [153, 153, 153], [190, 122, 222],[250, 170, 30],
        #          [220, 220, 0], [222, 142, 35], [205, 155, 155], [70, 130, 180],[128, 64, 128],
        #          [244, 35, 232], [0, 0, 70], [107, 142, 35], [152, 251, 152],[70, 70, 70],
        #          [110, 80, 100]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,#True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
