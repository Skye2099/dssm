
import numpy as np
import torch
import glob
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from decord import VideoReader, cpu
import os
from pathlib import Path
from .util import video_transforms as video_transforms
from .util import volume_transforms as volume_transforms


input_size = 224
num_frames = 16
num_segments = 1
tubelet_size = 2
drop = 0.0
drop_path_rate = 0.1
attn_drop_rate = 0.0
drop_block_rate = None
init_scale = 0.001
num_classes = 128

class VideoMAEEmbedder:
    def __init__(self,video_path,num_patches,target_dir=None):
        self.keep_aspect_ratio = True
        self.new_height = 256
        self.new_width = 320
        self.clip_len = 16
        self.frame_sample_rate = 4
        self.num_patches = num_patches
        self.video_path = video_path
        self._split_video(self.video_path,target_dir)
        self.data_transform = video_transforms.Compose([
                video_transforms.Resize(224, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(224, 224)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        self.buffers = self._load_buffers()


    def _split_video(self,video_path,target_dir=None):
        splitter = VideoSplitter(video_path,self.num_patches)
        video_path = Path(video_path)
        video_name,video_dir,video_ext,video_path = video_path.stem,video_path.parent,video_path.suffix,str(video_path)
        if target_dir is None:
            self.patches_save_dir = video_dir/video_name
        else:
            self.patches_save_dir = Path(target_dir)/video_name
        self.patches_save_dir.mkdir(exist_ok=True)
        splitter.save_patches(self.patches_save_dir)
        print("Saved patches to %s" % str(self.patches_save_dir))


    def _load_buffers(self):
        samples = [i for i in self.patches_save_dir.iterdir() if str(i).endswith('.mp4')]
        buffers = [self.loadvideo_decord(str(i)) for i in samples]
        buffers = [self.data_transform(buffer) for buffer in buffers]
        res = torch.stack(buffers, dim=0)
        return res


    def _save_buffers(self):
        torch.save(self.buffers, self.patches_save_dir/'buffers.pth')

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

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = len(vr)

        all_index = []
        for i in range(1):
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

    def __repr__(self):
        return "VideoMAEEmbedder"

class VideoSplitter:
    def __init__(self,video_path,num_patches=20):
        self.video_path = video_path
        self.video_clip = VideoFileClip(video_path)
        self.video_duration = self.video_clip.duration
        self.num_patches = num_patches
        self.sub_clips = self._split_video()

    def _split_video(self):
        sub_clips = []
        clip_duration = self.video_duration/self.num_patches
        for i in range(self.num_patches):
            sub_clips.append(self.video_clip.subclip(i*clip_duration,(i+1)*clip_duration))
        return sub_clips

    def save_patches(self,save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        sub_clips = self._split_video()
        for i in tqdm(range(len(sub_clips))):
            sub_clips[i].write_videofile(os.path.join(save_dir,str(i)+'.mp4'))
            audio_clip = sub_clips[i].audio
            audio_clip.write_audiofile(os.path.join(save_dir,str(i)+'.wav'))
        self.video_clip.close()

if __name__ == '__main__':
    video_file_list = glob.glob('/opt/user-datasets/media/*.mp4')
    embedders = [VideoMAEEmbedder(i,20,target_dir='./test/') for i in video_file_list]
    for i in tqdm(embedders):
        i._save_buffers()

