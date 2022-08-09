import os
import numpy as np
import torch
import decord
import pickle
from PIL import Image
from torchvision import transforms
from .util.random_erasing import RandomErasing
import warnings
from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from .util import video_transforms as video_transforms
from .util import volume_transforms as volume_transforms
from tqdm import tqdm
from .generate_buffers import VideoMAEEmbedder
import torch

mae_pretrained_model = torch.jit.load('./result/vit_base_patch16_20_224_freeze.pt')

class VideoMAEClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,target_dir=None):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.aug = False
        self.rand_erase = False
        self.reprob = 0.25
        self.num_sample = 1
        self.target_dir = target_dir
        if self.mode in ['train']:
            self.aug = True
            if self.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        import pandas as pd
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=',')
        dataset_samples = list(cleaned.values[:, 0])
        self.dataset_samples = []
        if mode == 'train' and os.path.exists('/root/notebook/VideoMAE_Embedder/mae_finetune_train.pkl'):
            with open('/root/notebook/VideoMAE_Embedder/mae_finetune_train.pkl', 'rb') as f:
                data = pickle.load(f)
                self.dataset_samples = data['dataset_samples']
                self.label_array = data['label_array']
        elif mode=='val' and os.path.exists('/root/notebook/VideoMAE_Embedder/mae_finetune_val.pkl'):
            with open('/root/notebook/VideoMAE_Embedder/mae_finetune_val.pkl', 'rb') as f:
                data = pickle.load(f)
                self.dataset_samples = data['dataset_samples']
                self.label_array = data['label_array']
        elif mode=='test' and os.path.exists('/root/notebook/VideoMAE_Embedder/mae_finetune_test.pkl'):
            with open('/root/notebook/VideoMAE_Embedder/mae_finetune_test.pkl', 'rb') as f:
                data = pickle.load(f)
                self.dataset_samples = data['dataset_samples']
                self.label_array = data['label_array']
        elif mode=='train' and not os.path.exists('/root/notebook/VideoMAE_Embedder/mae_finetune_train.pkl'):
            for sample in tqdm(dataset_samples):
                video_name = sample.split('/')[-1].split('.')[0]
                video_path = sample
                if target_dir is None:
                    video_dir = os.path.dirname(sample)+'/'+video_name
                else:
                    video_dir = target_dir+'/'+video_name
                videos = [video_dir+'/'+f'{i}.mp4' for i in range(20)]
                embeddings = video_dir+'/'+'embeddings.pth'
                buffers = video_dir+'/'+'buffers.pth'
                self.dataset_samples.append((videos, embeddings, buffers,video_path))
            # self.label_array = list(cleaned.values[:, 1])
            # with open('/root/notebook/VideoMAE_Embedder/mae_finetune_train.pkl', 'wb') as f:
            #     data = {'dataset_samples': self.dataset_samples, 'label_array': self.label_array}
            #     pickle.dump(data, f)
        elif mode=='val' and not os.path.exists('/root/notebook/VideoMAE_Embedder/mae_finetune_val.pkl'):
            for sample in tqdm(dataset_samples):
                video_name = sample.split('/')[-1].split('.')[0]
                video_path = sample
                if target_dir is None:
                    video_dir = os.path.dirname(sample) + '/' + video_name
                else:
                    video_dir = target_dir + '/' + video_name
                videos = [video_dir+'/'+f'{i}.mp4' for i in range(20)]
                embeddings = video_dir + '/' + 'embeddings.pth'
                buffers = video_dir + '/' + 'buffers.pth'
                self.dataset_samples.append((videos, embeddings, buffers,video_path))
            # self.label_array = list(cleaned.values[:, 1])
            # with open('/root/notebook/VideoMAE_Embedder/mae_finetune_val.pkl', 'wb') as f:
            #     data = {'dataset_samples': self.dataset_samples, 'label_array': self.label_array}
            #     pickle.dump(data, f)
        elif mode=='test' and not os.path.exists('/root/notebook/VideoMAE_Embedder/mae_finetune_test.pkl'):
            for sample in tqdm(dataset_samples):
                video_name = sample.split('/')[-1].split('.')[0]
                video_path = sample
                if target_dir is None:
                    video_dir = os.path.dirname(sample) + '/' + video_name
                else:
                    video_dir = target_dir + '/' + video_name
                videos = [video_dir+'/'+f'{i}.mp4' for i in range(20)]
                embeddings = video_dir + '/' + 'embeddings.pth'
                buffers = video_dir + '/' + 'buffers.pth'
                self.dataset_samples.append((videos, embeddings, buffers,sample,video_path))
            # self.label_array = list(cleaned.values[:, 1])
            # with open('/root/notebook/VideoMAE_Embedder/mae_finetune_test.pkl', 'wb') as f:
            #     data = {'dataset_samples': self.dataset_samples, 'label_array': self.label_array}
            #     pickle.dump(data, f)


        if (mode == 'train'):
            pass

        elif (mode == 'val'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        if self.mode == 'train':
            scale_t = 1
            videos, embeddings, buffers,video_path = self.dataset_samples[index]
            if os.path.exists(embeddings):
                video_embedding = torch.load(embeddings)
            else:
                embedder = VideoMAEEmbedder(video_path, 20,self.target_dir)
                buffers = embedder.buffers
                buffers = buffers.unsqueeze(0).to('cuda')
                with torch.no_grad():
                    video_embedding = mae_pretrained_model(buffers)
            video_embedding = video_embedding.squeeze(0).cpu()
            torch.save(video_embedding, embeddings)
            return video_embedding

        elif self.mode == 'val':
            videos, embeddings, buffers,video_path = self.dataset_samples[index]
            if os.path.exists(embeddings):
                video_embedding = torch.load(embeddings)
            else:
                embedder = VideoMAEEmbedder(video_path, 20, self.target_dir)
                buffers = embedder.buffers
                buffers = buffers.unsqueeze(0).to('cuda')
                with torch.no_grad():
                    video_embedding = mae_pretrained_model(buffers)
            video_embedding = video_embedding.squeeze(0).cpu()
            torch.save(video_embedding, embeddings)
            return video_embedding


    def _aug_frame(
            self,
            buffer,
    ):

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.reprob,
                mode='pixel',
                max_count=1,
                num_splits=1,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            all_index = [x for x in range(0, len(vr), self.frame_sample_rate)]
            while len(all_index) < self.clip_len:
                all_index.append(all_index[-1])
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = len(vr) // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i * seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


class VideoMAE(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """
    def __init__(self,
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 lazy_init=False):

        super(VideoMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.lazy_init = lazy_init


        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                                   "Check your data directory (opt.data-dir)."))

    def build_preload_cache(self):
        for directory,target in tqdm(self.clips):
            video_dir = os.path.dirname(directory)
            video_basename = os.path.basename(directory).split('.')[0]
            video_name = directory
            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)
            segment_indices, skip_offsets = self._sample_train_indices(duration)
            images = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)
            process_data, mask = self.transform((images, None))  # T*C,H,W
            process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0, 1)
            process_data = process_data.numpy()
            np.savez(os.path.join(video_dir, f'{video_basename}_preloadcache.npz'), process_data=process_data, mask=mask)

    def _generate_preload_cache(self,directory):
        video_dir = os.path.dirname(directory)
        video_basename = os.path.basename(directory).split('.')[0]
        video_name = directory
        decord_vr = decord.VideoReader(video_name, num_threads=1)
        duration = len(decord_vr)
        segment_indices, skip_offsets = self._sample_train_indices(duration)
        images = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)
        process_data, mask = self.transform((images, None))  # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0, 1)
        process_data = process_data.numpy()
        np.savez(os.path.join(video_dir, f'{video_basename}_preloadcache.npz'), process_data=process_data, mask=mask)

    def _get_preload_cache(self, directory):
        video_dir = os.path.dirname(directory)
        video_basename = os.path.basename(directory).split('.')[0]
        video_name = directory
        if os.path.exists(os.path.join(video_dir, f'{video_basename}_preloadcache.npz')):
            cache = np.load(os.path.join(video_dir, f'{video_basename}_preloadcache.npz'))
            return cache['process_data'], cache['mask']
        else:
            return None, None

    def __getitem__(self, index):

        directory, target = self.clips[index]
        _cache_processed_data, _cache_mask = self._get_preload_cache(directory)
        if _cache_processed_data is not None:
            process_data = torch.from_numpy(_cache_processed_data)
            mask = _cache_mask
            return (process_data, mask)
        else:
            if self.video_loader:
                if '.' in directory.split('/')[-1]:
                    # data in the "setting" file already have extension, e.g., demo.mp4
                    video_name = directory
                else:
                    # data in the "setting" file do not have extension, e.g., demo
                    # So we need to provide extension (i.e., .mp4) to complete the file name.
                    video_name = '{}.{}'.format(directory, self.video_ext)

                decord_vr = decord.VideoReader(video_name, num_threads=1)
                duration = len(decord_vr)

            segment_indices, skip_offsets = self._sample_train_indices(duration)

            images = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)
            # for i in range(len(images)):
            #     images[i].save('{}.jpg'.format(i))

            process_data, mask = self.transform((images, None)) # T*C,H,W
            process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W

            return (process_data, mask)

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting):
        if not os.path.exists(setting):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(',')
                # line format: video_path, video_duration,
                if len(line_info) < 1:
                    raise(RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
                elif len(line_info) < 2:
                    clip_path = os.path.join(line_info[0])
                    target = 0
                    item = (clip_path, target)
                    clips.append(item)
                    continue
                else:
                    clip_path = os.path.join(line_info[0])
                    target = int(line_info[1])
                    item = (clip_path, target)
                    clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets


    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list
