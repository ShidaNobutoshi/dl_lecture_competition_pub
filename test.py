import os

# set current dir
os.chdir('D:\\Work\\CUDA\Deep_Learning_Basic\\Final_Task\\GitHub\\dl_lecture_competition_pub')
os.getcwd()
os.listdir('./')

# Set environment variables
os.environ['HYDRA_FULL_ERROR'] = '1'    # for debug, HYDRA complete stack trace

import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from src.datasets import rec_train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import time

# print torch env
print(f'torch.__version__ : {torch.__version__}')

# print GPU env
## print(f'torch.version.cuda : {torch.version.cuda}')
## print(f'torch.cuda.device_count() : {torch.cuda.device_count()}')
## print(f'torch.cuda.current_device() : {torch.cuda.current_device()}')
## print(f'torch.cuda.get_device_name() : {torch.cuda.get_device_name()}')
## print(f'torch.cuda.get_device_capability() : {torch.cuda.get_device_capability()}')
## print(f'torch.cuda.get_arch_list() : {torch.cuda.get_arch_list()}')

# Set default GPU No.
## torch.cuda.set_device(torch.cuda.device_count() - 1)
## print(f'torch.cuda.current_device() : {torch.cuda.current_device()}')
## print(f'torch.cuda.get_device_name() : {torch.cuda.get_device_name()}')
## print(f'torch.cuda.get_device_capability() : {torch.cuda.get_device_capability()}')
## print(f'torch.cuda.get_arch_list() : {torch.cuda.get_arch_list()}')

# Set use device
## use_device = "cuda" if torch.cuda.is_available() else "cpu"
use_device = "cpu"      # force use cpu


class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_device == "cuda" :
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    '''
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    return epe

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    ## device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(use_device)
    '''
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        '''
    
    # ------------------
    #    Dataloader
    # ------------------
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4,
        visualize=False,
#        visualize=True,
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    collate_fn = train_collate
    rec_train_collate_fn = rec_train_collate
    train_data = DataLoader(train_set,
                                 batch_size=args.data_loader.train.batch_size,
                                 shuffle=args.data_loader.train.shuffle,
                                 collate_fn=rec_train_collate_fn,
                                 drop_last=False)
    test_data = DataLoader(test_set,
                                 batch_size=args.data_loader.test.batch_size,
                                 shuffle=args.data_loader.test.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)

    '''
    train data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
        Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
        Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない
    
    test data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    '''
    # ------------------
    #       Model
    # ------------------
    model = EVFlowNet(args.train).to(device)

    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)


##    current_time = time.strftime("%Y%m%d%H%M%S")
##    model_path = f"checkpoints/model_{current_time}.pth"
##    torch.save(model.state_dict(), model_path)
##    print(f"Model saved to {model_path}")

# baseline(batch_size=16)
#    model_path = f"checkpoints__Obsolete/model_20240626082556.pth"       # 2024/06/27, Omnicampus test EPE=9.16836, baseline epochs=1
#    model_path = f"checkpoints__Obsolete/model_20240626091718.pth"       # 2024/06/27, Omnicampus test EPE=4.12011, baseline epochs=2
#    model_path = f"checkpoints__Obsolete/model_20240626100851.pth"       # 2024/06/27, Omnicampus test EPE=3.78448, baseline epochs=3
#    model_path = f"checkpoints__Obsolete/model_20240626110037.pth"       # 2024/06/27, Omnicampus test EPE=3.71077, baseline epochs=4
#    model_path = f"checkpoints__Obsolete/model_20240626115203.pth"       # 2024/06/27, Omnicampus test EPE=3.99791, baseline epochs=5
#    model_path = f"checkpoints__Obsolete/model_20240626124346.pth"       # 2024/06/27, Omnicampus test EPE=3.31877, baseline epochs=6
#    model_path = f"checkpoints__Obsolete/model_20240626133536.pth"       # 2024/06/27, Omnicampus test EPE=3.41482, baseline epochs=7
#    model_path = f"checkpoints__Obsolete/model_20240626142715.pth"       # 2024/06/27, Omnicampus test EPE=2.69585, baseline epochs=8
#    model_path = f"checkpoints__Obsolete/model_20240626151902.pth"       # 2024/06/27, Omnicampus test EPE=3.01162, baseline epochs=9
#    model_path = f"checkpoints__Obsolete/model_20240626161044.pth"       # 2024/06/27, Omnicampus test EPE=3.09306, baseline epochs=10
#    model_path = f"checkpoints__Obsolete/model_20240626170208.pth"       # 2024/06/27, Omnicampus test EPE=2.53735, baseline epochs=11
#    model_path = f"checkpoints__Obsolete/model_20240626175302.pth"       # 2024/06/27, Omnicampus test EPE=3.24903, baseline epochs=12
#    model_path = f"checkpoints__Obsolete/model_20240626184400.pth"       # 2024/06/27, Omnicampus test EPE=2.57562, baseline epochs=13
#    model_path = f"checkpoints__Obsolete/model_20240626193500.pth"       # 2024/06/27, Omnicampus test EPE=2.47025, baseline epochs=14
#    model_path = f"checkpoints__Obsolete/model_20240626202551.pth"       # 2024/06/27, Omnicampus test EPE=2.75401, baseline epochs=15
#    model_path = f"checkpoints__Obsolete/model_20240626211650.pth"       # 2024/06/27, Omnicampus test EPE=2.44323, baseline epochs=16
#    model_path = f"checkpoints__Obsolete/model_20240626220736.pth"       # 2024/06/27, Omnicampus test EPE=2.3237 , baseline epochs=17
#    model_path = f"checkpoints__Obsolete/model_20240626225829.pth"       # 2024/06/27, Omnicampus test EPE=3.38431, baseline epochs=18
#    model_path = f"checkpoints__Obsolete/model_20240626234905.pth"       # 2024/06/27, Omnicampus test EPE=2.61247, baseline epochs=19
#    model_path = f"checkpoints__Obsolete/model_20240627003950.pth"       # 2024/06/27, Omnicampus test EPE=2.73177, baseline epochs=20

# SequenceRecurrent, Crop480x640, GradClip
#    model_path = f"checkpoints/model_20240717015317.pth"       # 2024/07/16, Omnicampus test EPE=9.33421, SeqRecurr, Crop480x640, GradClip, epochs=1
#    model_path = f"checkpoints/model_20240717024552.pth"       # 2024/07/16, Omnicampus test EPE=4.62714, SeqRecurr, Crop480x640, GradClip, epochs=2
#    model_path = f"checkpoints/model_20240717033820.pth"       # 2024/07/16, Omnicampus test EPE=2.77588, SeqRecurr, Crop480x640, GradClip, epochs=3
#    model_path = f"checkpoints/model_20240717043053.pth"       # 2024/07/16, Omnicampus test EPE=2.75733, SeqRecurr, Crop480x640, GradClip, epochs=4
#    model_path = f"checkpoints/model_20240717052328.pth"       # 2024/07/16, Omnicampus test EPE=5.01997, SeqRecurr, Crop480x640, GradClip, epochs=5
#    model_path = f"checkpoints/model_20240717061559.pth"       # 2024/07/16, Omnicampus test EPE=2.93726, SeqRecurr, Crop480x640, GradClip, epochs=6
#    model_path = f"checkpoints/model_20240717070859.pth"       # 2024/07/16, Omnicampus test EPE=2.93262, SeqRecurr, Crop480x640, GradClip, epochs=7
#    model_path = f"checkpoints/model_20240717080437.pth"       # 2024/07/16, Omnicampus test EPE=25.1431, SeqRecurr, Crop480x640, GradClip, epochs=8
#    model_path = f"checkpoints/model_20240717090552.pth"       # 2024/07/16, Omnicampus test EPE=3.90742, SeqRecurr, Crop480x640, GradClip, epochs=9
#    model_path = f"checkpoints/model_20240717102746.pth"       # 2024/07/16, Omnicampus test EPE=4.05273, SeqRecurr, Crop480x640, GradClip, epochs=10
#    model_path = f"checkpoints/model_20240717121824.pth"       # 2024/07/16, Omnicampus test EPE=4.11932, SeqRecurr, Crop480x640, GradClip, epochs=11
#    model_path = f"checkpoints/model_20240717144252.pth"       # 2024/07/16, Omnicampus test EPE=3.42704, SeqRecurr, Crop480x640, GradClip, epochs=12
#    model_path = f"checkpoints/model_20240717174603.pth"       # 2024/07/16, Omnicampus test EPE=2.40108, SeqRecurr, Crop480x640, GradClip, epochs=13
#    model_path = f"checkpoints/model_20240717211237.pth"       # 2024/07/16, Omnicampus test EPE=2.38351, SeqRecurr, Crop480x640, GradClip, epochs=14
#    model_path = f"checkpoints/model_20240718004350.pth"       # 2024/07/16, Omnicampus test EPE=2.53106, SeqRecurr, Crop480x640, GradClip, epochs=15
#    model_path = f"checkpoints/model_20240718043823.pth"       # 2024/07/16, Omnicampus test EPE=2.43061, SeqRecurr, Crop480x640, GradClip, epochs=16
#    model_path = f"checkpoints/model_20240718090829.pth"       # 2024/07/16, Omnicampus test EPE=2.53608, SeqRecurr, Crop480x640, GradClip, epochs=17
#    model_path = f"checkpoints/model_YYYYMMDDHHMMSS.pth"       # 2024/07/16, Omnicampus test EPE=x.xxx, SeqRecurr, Crop480x640, GradClip, epochs=X

# SequenceRecurrent, Crop480x640, H+V-Flip, GradClip
#    model_path = f"checkpoints/model_20240717104850.pth"       # 2024/07/17, Omnicampus test EPE=9.66682, SeqRecurr, Crop480x640, H+V-Flip, GradClip, epochs=1
#    model_path = f"checkpoints/model_20240717120817.pth"       # 2024/07/17, Omnicampus test EPE=56.8469, SeqRecurr, Crop480x640, H+V-Flip, GradClip, epochs=2
#    model_path = f"checkpoints/model_20240717132312.pth"       # 2024/07/17, Omnicampus test EPE=9.81483, SeqRecurr, Crop480x640, H+V-Flip, GradClip, epochs=3
#    model_path = f"checkpoints/model_20240717143757.pth"       # 2024/07/17, Omnicampus test EPE=16.2931, SeqRecurr, Crop480x640, H+V-Flip, GradClip, epochs=4
#    model_path = f"checkpoints/model_20240717155217.pth"       # 2024/07/17, Omnicampus test EPE=4.13064, SeqRecurr, Crop480x640, H+V-Flip, GradClip, epochs=5
#    model_path = f"checkpoints/model_20240717170340.pth"       # 2024/07/17, Omnicampus test EPE=4.67617, SeqRecurr, Crop480x640, H+V-Flip, GradClip, epochs=6
#    model_path = f"checkpoints/model_20240717181432.pth"       # 2024/07/17, Omnicampus test EPE=3.99518, SeqRecurr, Crop480x640, H+V-Flip, GradClip, epochs=7
#    model_path = f"checkpoints/model_20240717192630.pth"       # 2024/07/17, Omnicampus test EPE=2.68012, SeqRecurr, Crop480x640, H+V-Flip, GradClip, epochs=8
#    model_path = f"checkpoints/model_20240717204027.pth"       # 2024/07/17, Omnicampus test EPE=7.85379, SeqRecurr, Crop480x640, H+V-Flip, GradClip, epochs=9
#    model_path = f"checkpoints/model_20240717215844.pth"       # 2024/07/17, Omnicampus test EPE=10.6593, SeqRecurr, Crop480x640, H+V-Flip, GradClip, epochs=10
#    model_path = f"checkpoints/model_20240717234004.pth"       # 2024/07/17, Omnicampus test EPE=2.59421, SeqRecurr, Crop480x640, H+V-Flip, GradClip, epochs=11
#    model_path = f"checkpoints/model_20240718020752.pth"       # 2024/07/17, Omnicampus test EPE=3.78752, SeqRecurr, Crop480x640, H+V-Flip, GradClip, epochs=12
#    model_path = f"checkpoints/model_20240718052123.pth"       # 2024/07/17, Omnicampus test EPE=2.28978, SeqRecurr, Crop480x640, H+V-Flip, GradClip, epochs=13
#    model_path = f"checkpoints/model_20240718085926.pth"       # 2024/07/17, Omnicampus test EPE=358.902, SeqRecurr, Crop480x640, H+V-Flip, GradClip, epochs=14
#    model_path = f"checkpoints/model_YYYYMMDDHHMMSS.pth"       # 2024/07/17, Omnicampus test EPE=x.xxx, SeqRecurr, Crop480x640, H+V-Flip, GradClip, epochs=X

# SequenceRecurrent, Crop480x640, H+V-Flip, MidLayer-Loss, GradClip
#    model_path = f"checkpoints/model_20240717205851.pth"       # 2024/07/17, Omnicampus test EPE=16.6532, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=1
#    model_path = f"checkpoints/model_20240717214458.pth"       # 2024/07/17, Omnicampus test EPE=4.0828 , SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=2
#    model_path = f"checkpoints/model_20240717223132.pth"       # 2024/07/17, Omnicampus test EPE=317.568, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=3
#    model_path = f"checkpoints/model_20240717231757.pth"       # 2024/07/17, Omnicampus test EPE=97.9729, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=4
#    model_path = f"checkpoints/model_20240718000724.pth"       # 2024/07/17, Omnicampus test EPE=3.60742, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=5
#    model_path = f"checkpoints/model_20240718005313.pth"       # 2024/07/17, Omnicampus test EPE=2.48563, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=6
#    model_path = f"checkpoints/model_20240718013915.pth"       # 2024/07/17, Omnicampus test EPE=4.09661, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=7
#    model_path = f"checkpoints/model_20240718022510.pth"       # 2024/07/17, Omnicampus test EPE=2.83349, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=8
#    model_path = f"checkpoints/model_20240718031051.pth"       # 2024/07/17, Omnicampus test EPE=3.27819, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=9
#    model_path = f"checkpoints/model_20240718035619.pth"       # 2024/07/17, Omnicampus test EPE=14.856 , SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=10
#    model_path = f"checkpoints/model_20240718044157.pth"       # 2024/07/17, Omnicampus test EPE=2.58315, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=11
#    model_path = f"checkpoints/model_20240718052929.pth"       # 2024/07/17, Omnicampus test EPE=5.05879, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=12
#    model_path = f"checkpoints/model_20240718061605.pth"       # 2024/07/17, Omnicampus test EPE=16.9798, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=13
#    model_path = f"checkpoints/model_20240718070309.pth"       # 2024/07/17, Omnicampus test EPE=2.73436, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=14
#    model_path = f"checkpoints/model_20240718075029.pth"       # 2024/07/17, Omnicampus test EPE=2.38705, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=15
#    model_path = f"checkpoints/model_20240718083848.pth"       # 2024/07/17, Omnicampus test EPE=2.39299, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=16
#    model_path = f"checkpoints/model_20240718092602.pth"       # 2024/07/17, Omnicampus test EPE=2.41809, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=17
#    model_path = f"checkpoints/model_20240718101245.pth"       # 2024/07/17, Omnicampus test EPE=2.51126, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=18
#    model_path = f"checkpoints/model_20240718105853.pth"       # 2024/07/17, Omnicampus test EPE=24.2919, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=19
    model_path = f"checkpoints/model_20240718114713.pth"       # 2024/07/17, Omnicampus test EPE=2.31473, SeqRecurr, Crop480x640, H+V-Flip, MidLayer, GradClip, epochs=20


    print(f"Model loaded from {model_path}")

    # ------------------
    #   Start predicting
    # ------------------
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            batch_flow, batch_flow_dict = model(event_image) # [1, 2, 480, 640]
            flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
        print("test done")
    # ------------------
    #  save submission
    # ------------------
    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()
