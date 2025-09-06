import math
import os
import json
from dataclasses import dataclass, field

import random
import imageio
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTokenizer
import pickle
from mar3d import register
from mar3d.utils.base import Updateable
from mar3d.utils.config import parse_structured
from mar3d.utils.typing import *
from plyfile import PlyData, PlyElement
import pandas as pd
def save_ply_plyfile(points, filename):
    # 创建结构化数组
    vertex = np.zeros(len(points), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    
    # 创建PlyElement
    el = PlyElement.describe(vertex, 'vertex')
    
    # 写入PLY文件
    PlyData([el], text=True).write(filename)
def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def eul2rot(theta) :
    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])
    return R

@dataclass
class ObjaverseDataModuleConfig:
    root_dir: str = None
    data_type: str = "occupancy"         # occupancy or sdf
    n_samples: int = 4096                # number of points in input point cloud
    scale: float = 1.0                   # scale of the input point cloud and target supervision
    noise_sigma: float = 0.0             # noise level of the input point cloud
    
    load_supervision: bool = True        # whether to load supervision
    supervision_type: str = "occupancy"  # occupancy, sdf, tsdf, tsdf_w_surface

    n_supervision: int = 10000           # number of points in supervision
    
    load_image: bool = False             # whether to load images 
    image_data_path: str = ""            # path to the image data
    image_type: str = "rgb"              # rgb, normal
    background_color: Tuple[float, float, float] = field(
            default_factory=lambda: (1.0, 1.0, 1.0)
        )
    idx: Optional[List[int]] = None      # index of the image to load
    n_views: int = 1                     # number of views
    rotate: bool = False          # whether to rotate the input point cloud and the supervision

    load_caption: bool = False           # whether to load captions
    caption_type: str = "text"           # text, clip_embeds
    tokenizer_pretrained_model_name_or_path: str = ""

    batch_size: int = 32
    num_workers: int = 0


class ObjaverseDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        
        # make sure root_dir is list
        if isinstance(self.cfg.root_dir, str):
            self.cfg.root_dir = [self.cfg.root_dir]
            
        # cache file
        cache_file = ''
        if os.path.exists(cache_file):
           
            with open(cache_file, 'rb') as f:
                self.uids_og = pickle.load(f)
        else:
            self.uids_og = self._scan_files()
            with open(cache_file, 'wb') as f:
                pickle.dump(self.uids_og, f)

        self.background_color = torch.as_tensor(self.cfg.background_color)
        
        if self.cfg.load_image:
          
            mapping=''
     

            df = pd.read_csv(mapping, sep=',', header=None, names=['number', 'hash'], skiprows=1)

    
            self.mapping_dict = dict(zip(df['number'], df['hash']))
            self.uids= []
            for uid_idx, uid_tuple in enumerate(self.uids_og):
                # Extract the path and filename
                _, filename = uid_tuple
                # Get the part after the underscore
                if '_' in filename:
                    search_str = filename.split('_', 1)[1]

                    if search_str in self.mapping_dict.keys():

                        self.uids.append(uid_tuple)

        else:
            self.uids=self.uids_og
        print(f"Loaded {len(self.uids)} {split} usable uids")
    def _scan_files(self):
        uids = []
        total_files = []

        for root_dir in self.cfg.root_dir:
            files = os.listdir(root_dir)
            # 给每个文件添加对应的根目录信息
            files = [(root_dir, file) for file in files]
            total_files.extend(files)
            

        if self.split == 'train':
            total_files = total_files[150:]
        else:
            total_files = total_files[:150]
            
   
        return [
            (root_dir, file) for root_dir, file in total_files
            if os.path.exists(
                f'{root_dir}/{file}/xxx.npz'
            )
        ]

    def __len__(self):

        return len(self.uids)

    def _load_shape(self, index: int) -> Dict[str, Any]:

        if self.cfg.supervision_type == "sdf":

            sdfs3 = np.asarray(pointcloud['clean_surface_sdf'])
            ind3 = rng.choice(surface_og_n.shape[0], self.cfg.n_supervision//3, replace=False)

            rand_points3=surface_og[ind3]
            sdfs3 =sdfs3[ind3]
            normal3=surface_og_n[ind3]

            rand_points=np.concatenate((rand_points1,rand_points2,rand_points3),axis=0)
            sdfs=np.concatenate((sdfs1,sdfs2,sdfs3),axis=0)


        else:
            rand_points=np.concatenate((rand_points1,rand_points2),axis=0)
            sdfs=np.concatenate((sdfs1,sdfs2),axis=0)

        ret = {
            "uid": self.uids[index][1],
            "surface": surface.astype(np.float32),

        }

       
        ret["rand_points"] = rand_points.astype(np.float32)
        
        if self.cfg.supervision_type == "sdf":
            sdfs=np.nan_to_num(sdfs, nan=1.0, posinf=1.0, neginf=-1.0)
            # ret["sdf"] = sdfs.flatten().astype(np.float32).clip(-self.cfg.tsdf_threshold, self.cfg.tsdf_threshold) / self.cfg.tsdf_threshold
            # ret["sdf"] = sdfs.flatten().astype(np.float32).clip(-self.cfg.tsdf_threshold, self.cfg.tsdf_threshold)
            ret["sdf"] = sdfs.flatten().astype(np.float32)
            # ret["sdf"] = sdfs[ind2].flatten().astype(np.float32)
            ret['surface_normal']=normal3
        elif self.cfg.supervision_type == "occupancy":
            # ret["occupancies"] = np.where(sdfs[ind].flatten() < 1e-3, 0, 1).astype(np.float32)
            ret["occupancies"] = np.where(sdfs.flatten() < 0, 0, 1).astype(np.float32)
        else:
            raise NotImplementedError(f"Supervision type {self.cfg.supervision_type} not implemented")

        return ret
        
    def _load_image(self, index: int) -> Dict[str, Any]:
        name=self.uids[index][1].split('_')[1]
        file_path=self.mapping_dict[name]
        # image_paths=os.path.join(images_root,file_path,file_path,name)
        def _load_single_image(img_path):
            img = torch.from_numpy(
                np.asarray(
                    Image.fromarray(imageio.v2.imread(img_path))
                    .convert("RGBA")
                )
                / 255.0
            ).float()
            mask: Float[Tensor, "H W 1"] = img[:, :, -1:]
            image: Float[Tensor, "H W 3"] = img[:, :, :3] * mask + self.background_color[
                None, None, :
            ] * (1 - mask)
            return image
        ret = {}
        if self.cfg.image_type == "rgb" or self.cfg.image_type == "normal":
            assert self.cfg.n_views == 1, "Only single view is supported for single image"
            sel_idx = random.choice(self.cfg.idx)
            ret["sel_image_idx"] = sel_idx
   

            img_path=file_path
            ret["image"] = _load_single_image(img_path)
        else:
            raise NotImplementedError(f"Image type {self.cfg.image_type} not implemented")
        
        return ret

    def _load_caption(self, index: int, drop_text_embed: bool = False) -> Dict[str, Any]:
        ret = {}
        if self.cfg.caption_type == "text":
            caption = eval(json.load(open(f'{self.cfg.image_data_path}/' + "/".join(self.uids[index].split('/')[-2:]) + f'/annotation.json')))
            texts = [v for k, v in caption.items()]
            sel_idx = random.randint(0, len(texts) - 1)
            ret["sel_caption_idx"] = sel_idx
            ret['text_input_ids'] = self.tokenizer(
                texts[sel_idx] if not drop_text_embed else "",
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.detach()
        else:
            raise NotImplementedError(f"Caption type {self.cfg.caption_type} not implemented")
        
        return ret

    def get_data(self, index):
        # load shape
        ret = self._load_shape(index)

        
        if self.cfg.load_image:
            ret.update(self._load_image(index))
        return ret
        
    def __getitem__(self, index):
        try:
            return self.get_data(index)
        except Exception as e:
            print(f"Error in {self.uids[index]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))


    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        return batch



@register("objaverse-datamodule")
class ObjaverseDataModule(pl.LightningDataModule):
    cfg: ObjaverseDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ObjaverseDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = ObjaverseDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ObjaverseDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = ObjaverseDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None, num_workers=0) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn, num_workers=num_workers
        )
    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
            num_workers=self.cfg.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)