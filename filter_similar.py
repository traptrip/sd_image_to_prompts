import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import gc
from joblib import Parallel, delayed

embeddings = torch.load("embeddings.pth").numpy()


def get_ignoring_idxs(embeddings, batch_size=1024, thresh=0.95):
    bs = batch_size
    n_embeddings = len(embeddings)
    idxs_to_ignore = set()
    pbar = tqdm(total=n_embeddings)
    i = 0
    while i < n_embeddings:
        idxs = []
        while len(idxs) < bs and i < n_embeddings:
            if i not in idxs_to_ignore:
                idxs.append(i)
            i += 1
        cos_sim = cosine_similarity(embeddings[idxs], embeddings)
        np.fill_diagonal(cos_sim, 0)
        new_to_ignore = set(np.nonzero(cos_sim > thresh)[1])
        idxs_to_ignore = idxs_to_ignore.union(new_to_ignore)
        pbar.update(len(idxs))

    del embeddings
    gc.collect()

    return list(idxs_to_ignore)


IGNORING = get_ignoring_idxs(embeddings)
IGNORING = torch.tensor(IGNORING)
torch.save(IGNORING, "ignoring_idxs_resized224.pth")
