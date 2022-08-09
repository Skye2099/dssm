import torch
from model import mae_model
from data_util.mae_data import build_dataset
from timm.models import create_model
# timm==0.4.12
model_name = 'vit_base_patch16_20_224'
batch_size = 1
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
target_dir = None

model = create_model(

            model_name,
            pretrained=False,
            num_classes=num_classes,
            all_frames=num_frames,
            tubelet_size=tubelet_size,
            drop_rate=drop,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=True,
            init_scale=init_scale,
        )
model.cuda()

def main(data_path,nb_classes):
    dataset_train,nbclasses = build_dataset(data_path,is_train=True,test_mode=False,nb_classes=nb_classes,target_dir=target_dir)
    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                    batch_size=batch_size,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    drop_last=True,
                                                    )
    for embedding in data_loader_train:
        embedding = embedding.cuda()
        print(model(embedding).shape)

if __name__ == '__main__':
    main('./input_data/',num_classes)
