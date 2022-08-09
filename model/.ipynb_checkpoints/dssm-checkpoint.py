"""
@Author: 1365677361@qq.com

@info:
2021.07.03: 加入取单塔功能
"""

from model.base_tower import BaseTower
from preprocessing.inputs import combined_dnn_input, compute_input_dim
from layers.core import DNN
from preprocessing.utils import Cosine_Similarity, get_qn_video_path
from timm.models import create_model
from layers.core import PredictionLayer
from VideoEmbedder.model import mae_model


class DSSM(BaseTower):
    """DSSM双塔模型"""
    def __init__(self, user_feature_columns, gamma=1, dnn_use_bn=True,
                 dnn_hidden_units=(300, 300, 128), dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-6,
                 dnn_dropout=0, init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
        # super(DSSM, self).__init__(user_dnn_feature_columns, item_dnn_feature_columns,
        #                             l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
        #                             device=device, gpus=gpus)  # for init embedding(not required here)
        super().__init__(user_feature_columns)
        self.video_embedding_path = "/opt/tml/tmp/dataphin-data/video/videoembedding/v1"
        self.user_dnn = DNN(len(user_feature_columns), dnn_hidden_units,
                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                            use_bn=dnn_use_bn, init_std=init_std, device=device)
        self.user_dnn_embedding = None

        self.video_dnn = create_model(
            'vit_base_patch16_20_224',
            pretrained=False,
            num_classes=128,
            all_frames=16,
            tubelet_size=2,
            drop_rate=0.0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_mean_pooling=True,
            init_scale=0.001,
        )
        self.video_dnn.cuda()
        self.item_dnn_embedding = None

        self.gamma = gamma
        self.l2_reg_embedding = l2_reg_embedding
        self.seed = seed
        self.task = task
        self.device = device
        self.gpus = gpus

    def forward(self, inputs):
        user_dnn_input = inputs[0].to(self.device).float()
        video_dnn_input = inputs[1].to(self.device).float()
        self.user_dnn_embedding = self.user_dnn(user_dnn_input)
#         print(inputs.shape)
#         print(type(video_dnn_input),video_dnn_input.shape)

        self.item_dnn_embedding = self.video_dnn(video_dnn_input)

  
        score = Cosine_Similarity(self.user_dnn_embedding, self.item_dnn_embedding, gamma=self.gamma)
        output = self.out(score,)
        return output

#         elif len(self.user_dnn_feature_columns) > 0:
#             return self.user_dnn_embedding

#         else:
#             raise Exception("input Error! user and item feature columns are empty.")

