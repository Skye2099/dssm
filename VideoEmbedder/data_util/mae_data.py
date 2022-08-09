
import os
from torchvision import transforms
from .kinetics import VideoMAEClsDataset

def build_dataset(data_path,is_train,test_mode,nb_classes,target_dir=None):
    mode = None
    anno_path = None
    if is_train:
        mode = 'train'
        anno_path = os.path.join(data_path, 'train.csv')
    elif test_mode:
        mode = 'test'
        anno_path = os.path.join(data_path, 'test.csv')
    else:
        mode = 'val'
        anno_path = os.path.join(data_path, 'val.csv')
    dataset = VideoMAEClsDataset(
        anno_path=anno_path,
        data_path='/',
        mode=mode,
        clip_len = 16,
        frame_sample_rate=4,
        num_segment=1,
        test_num_segment=5,
        test_num_crop=3,
        num_crop=1 if not test_mode else 3,
        keep_aspect_ratio=True,
        crop_size=224,
        short_side_size=224,
        new_height=256,
        new_width=320,
        target_dir=target_dir)
    nb_classes = nb_classes
    return dataset,nb_classes
