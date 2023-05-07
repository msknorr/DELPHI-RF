from torch.utils.data import DataLoader
from config import TrainGlobalConfig
from dataframe import define_targets, load_and_unshuffle_dataset
from dataset import TrainDataset
from model import MultiTargetCNN
from engine import Fitter

config = TrainGlobalConfig()


def train_loop(df, targets, modality):
    dataset = TrainDataset
    in_chans = 3 if modality == "face" else 6

    for t in range(config.num_folds):
        for v in range(config.num_folds):
            if t == v:
                continue
            print("Training", t, v)

            model = MultiTargetCNN(targets, in_chans=in_chans).to(config.device)
            train_ds = dataset(df, df[~df["fold"].isin([v, t])].index, targets, mode="train")
            val_ds = dataset(df, df[df["fold"] == v].index, targets, mode="val")
            test_ds = dataset(df, df[df["fold"] == t].index, targets, mode="val")

            train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                                      num_workers=config.num_workers,
                                      persistent_workers=config.persistent_workers)
            val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                                    num_workers=config.num_workers,
                                    persistent_workers=config.persistent_workers)

            fitter = Fitter(model, targets, config, modality=modality, fold=t)
            fitter.fit(train_loader, val_loader, subfold=v)

s
def train_face_model():
    df = load_and_unshuffle_dataset()
    df = df[df["face"].str.find("NaN") == -1].reset_index(drop=True)  # keep faces
    targets = define_targets()
    train_loop(df, targets, "face")


def train_body_model():
    df = load_and_unshuffle_dataset()
    df = df[(df["front"].str.find("NaN") == -1) | (df["back"].str.find("NaN") == -1)].reset_index(drop=True)  # keep body
    targets = define_targets()
    train_loop(df, targets, "body")


def train():
    train_face_model()
    train_body_model()
