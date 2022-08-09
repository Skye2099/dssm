import numpy as np
import torch
import requests

def get_qn_video_path(id):
    url = "http://creativestudio.apps03.ali-bj-prd01.shuheo.net/creativestudio/material/searchlist"
    parms = {"pageNum": 1, "pageSize": 10, "materialTypeList": ["VIDEO"], "materialCode": id}
    # parms = {"pageNum": 3, "pageSize": 30, "materialTypeList": ["VIDEO"], "scoreRange": 40}
    response = requests.post(url, json=parms)
    result = response.json()
    for i in range(1):
        video_info = result['data'][i]
        video_name = video_info['content'][0]['url']
#         print(video_name, video_info['modelScore'])
        return video_name

def get_video_embedding_from_vid(id):
    if isinstance(id,int):
        video_name = get_qn_video_path(id)
        embedding_name = video_name.split("/")[-1].split('.')[0]
        embedding_path = f"/opt/tml/tmp/dataphin-data/video/videoembedding/v1/{embedding_name}/embedding.pth"
    else:
        embedding_path = f"/opt/tml/tmp/dataphin-data/video/videoembedding/v1/{id}/embedding.pth"
    try:
        embedding_tensor = torch.load(embedding_path)
    except FileNotFoundError:
        print(embedding_path,' not found!')
        embedding_tensor = torch.randn(1,20,768).cuda()
    return embedding_tensor

def slice_arrays(arrays, start=None, stop=None):
    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if isinstance(start, list) and stop is not None:
        raise ValueError('The stop argument has to be None if the value of start '
                         'is a list.')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]


def Cosine_Similarity(query, candidate, gamma=1, dim=-1):
    query_norm = torch.norm(query, dim=dim)
    candidate_norm = torch.norm(candidate, dim=dim)
    cosine_score = torch.sum(torch.multiply(query, candidate), dim=-1)
    cosine_score = torch.div(cosine_score, query_norm*candidate_norm+1e-8)
    cosine_score = torch.clamp(cosine_score, -1, 1.0)*gamma
    return cosine_score