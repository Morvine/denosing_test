import logging

from model_unet import UNet
from model_vgg import VGG
import torch
import numpy as np
import glob

import os
import pandas as pd

import tqdm


if __name__ == "__main__":

    data_path = "./dataset"

    unet_path = './models/unet_2out.pth'

    seq_len = 160

    result_path = "result"
    result_path_denoise = "result/denoised"

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(result_path_denoise):
        os.mkdir(result_path_denoise)

    result_path_csv = "result/results.csv"

    result = pd.DataFrame(columns=['file_name', 'result', 'denoised_file'])

    data_paths = [image_path for image_path in glob.glob(data_path + '/**/*', recursive=True)
                           if image_path.split('.')[-1].lower() in ['npy']]

    unet = UNet()
    unet.load_state_dict(torch.load(unet_path))
    unet.eval()



    if torch.cuda.is_available():
        unet.cuda()

    for idx, path in tqdm.tqdm(enumerate(data_paths)):
        data = np.load(path).astype('float64').T
        arr_len = data.shape[1]
        count_part = arr_len // seq_len + 1 if arr_len % seq_len else arr_len // seq_len

        if arr_len < count_part * seq_len:
            data = np.pad(data, ((0, 0), (0, count_part * seq_len - arr_len)), mode='constant')

        data_parts = np.array_split(data, count_part, axis=1)

        data_s = torch.from_numpy(data_parts[0]).cuda().unsqueeze(0).unsqueeze(0).float()
        if torch.cuda.is_available():
            data_s = data_s.cuda()

        maper = {0: "clean", 1: "noise"}
        new_name = ""

        file_name = os.path.basename(path)

        prop = unet(data_s)[1]
        _, cls = prop.topk(1, dim=1)
        cls = int(cls.squeeze())
        if cls:
            clear = []
            for i in data_parts:
                data = torch.from_numpy(i).unsqueeze(0).unsqueeze(0).float()
                if torch.cuda.is_available():
                    data = data.cuda()
                clear_i = unet(data)[0].squeeze()
                clear.append(clear_i.detach().cpu().numpy())

            new_name = os.path.splitext(file_name)[0] + "_clean" + os.path.splitext(file_name)[1]

            data_clear = np.hstack(clear)[:, :arr_len].T
            np.save(os.path.join(result_path_denoise, new_name), data_clear)

        result.loc[idx] = [file_name, maper[cls], new_name]

    result.to_csv(result_path_csv, index=False)




