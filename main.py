# 
# TODO: 
#   1. Data augmentation
#      + Running : Horizontal Flip, Vertical Flip
#      + TODO : Rotation
#   2. Suppress gradient vanishing and gradient divergence explosion
#      + TODO : Implement Gradient Clipping for model
#   3. Loss design
#      + TODO : Middle Layer loss evaluation for grad control
#   4. data processing
#      + TODO : 2-frame processing
#      + TODO : bi-directional flow like LSTM / multi-directional
#   5. Graph for algorithm quantitative evaluation -> skip
#      + TODO : Loss vs epoch/batch
# 

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
    #       Checkpoints
    # ------------------
    # Create the directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)

    # ------------------
    #   Start training
    # ------------------
    model.train()
##    for epoch in range(args.train.epochs):
##        total_loss = 0
##        print("on epoch: {}".format(epoch+1))
##        for i, batch in enumerate(tqdm(train_data)):
##            batch: Dict[str, Any]
##            event_image = batch["event_volume"].to(device) # [B, 4, 480, 640]
##            ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]
##            flow = model(event_image) # [B, 2, 480, 640]
##            loss: torch.Tensor = compute_epe_error(flow, ground_truth_flow)
##            print(f"batch {i} loss: {loss.item()}")
##            optimizer.zero_grad()
##
##            loss.backward()
##
##            # 勾配を絶対値1.0でクリッピングする(Gradient Clipping)
##            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
##
##            optimizer.step()
##
##            total_loss += loss.item()
##
##        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data)}')
##
##        # Save checkpoints
##        current_time = time.strftime("%Y%m%d%H%M%S")
##        model_path = f"checkpoints/model_{current_time}.pth"
##        torch.save(model.state_dict(), model_path)
##        print(f"Model saved to {model_path}")

    for epoch in range(args.train.epochs):
        total_loss = 0
        print("on epoch: {}".format(epoch+1))
        for j, seq_of_batch in enumerate(tqdm(train_data)):
            for i, batch in enumerate(seq_of_batch):
                batch: Dict[str, Any]
                event_image = batch["event_volume"].to(device) # [B, 4, 480, 640]
                ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]
                ground_truth_flow0 = batch["flow_gt0"].to(device) # [B, 2, 60, 80]
                ground_truth_flow1 = batch["flow_gt1"].to(device) # [B, 2, 120, 160]
                ground_truth_flow2 = batch["flow_gt2"].to(device) # [B, 2, 240, 320]

#                flow = model(event_image) # [B, 2, 480, 640]
#                loss: torch.Tensor = compute_epe_error(flow, ground_truth_flow)
                flow, flow_dict = model(event_image)
                loss: torch.Tensor = compute_epe_error(flow, ground_truth_flow) # [B, 2, 480, 640]
                + compute_epe_error(flow_dict['flow0'], ground_truth_flow0)     # [B, 2, 60, 80]
                + compute_epe_error(flow_dict['flow1'], ground_truth_flow1)     # [B, 2, 120, 160]
                + compute_epe_error(flow_dict['flow2'], ground_truth_flow2)     # [B, 2, 240, 320]

                print(f"batch {i} loss: {loss.item()}")

                optimizer.zero_grad()

                loss.backward()

                # 勾配を絶対値1.0でクリッピングする(Gradient Clipping)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data)}')

        # Save checkpoints
        current_time = time.strftime("%Y%m%d%H%M%S")
        model_path = f"checkpoints/model_{current_time}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


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
