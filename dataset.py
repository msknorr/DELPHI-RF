from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps
import albumentations as A

train_transform = A.Compose([
    A.Resize(320, 320),
    A.ShiftScaleRotate(shift_limit=0.15, rotate_limit=10, scale_limit=0.15, interpolation=1, border_mode=0, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.CoarseDropout(max_holes=5, max_height=24, max_width=24, p=0.5),
    A.HorizontalFlip(p=0.5),
],
)

val_transform = A.Compose([
    A.Resize(320, 320),
],)


def random_mask_face_area(faceim, face_points, facepart_list=None):
    if not isinstance(facepart_list, type(None)):
        if not isinstance(facepart_list, list):
            raise TypeError("facepart_list is not a list. Type: ", type(facepart_list))
    assert len(face_points) == 468
    drop_these = []
    if facepart_list is None:  # if None, generate parts to drop randomly
        how_many = np.random.choice([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7])
        for jj in range(how_many):
            facepart_list = np.random.choice(
                ["eyes", "nose", "mouth", "chin", "cheek_left", "cheek_right", "forehead", "rest", "rest", "rest"])
            if facepart_list not in drop_these:
                drop_these.append(facepart_list)
    else:
        drop_these = facepart_list
    for part in drop_these:
        faceim = mask_face_area(face_points, faceim, part)
    return faceim


def mask_face_area(face_points, faceim, facepart2drop: str):
    face_points = face_points.copy()
    face_points.T[0] *= faceim.shape[1]
    face_points.T[1] *= faceim.shape[0]
    faceim = faceim.astype(np.uint8)
    eye_right = [107, 66, 105, 63, 70, 156, 35, 31, 228, 229, 230, 231, 232, 128, 245, 193, 55]
    eye_left = [336, 296, 334, 293, 300, 383, 265, 261, 448, 449, 450, 451, 452, 357, 465, 417, 285]
    nose = [9, 336, 285, 417, 465, 343, 277, 355, 429, 358, 327, 326, 2, 97, 98, 129, 209, 126, 47, 114, 245, 193, 55,
            107]
    # old_mund = [0, 267, 269, 270, 409, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57, 185, 40, 39, 37]
    mouth = [164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57, 186, 92, 165, 167]
    chin = [57, 43, 106, 182, 83, 18, 313, 406, 335, 273, 287, 422, 431, 395, 378, 400, 377, 152, 148, 176, 149, 170,
            211, 202]
    cheek_left = [378, 395, 431, 422, 287, 410, 322, 391, 393, 164, 2, 326, 327, 358, 429, 355, 277, 343, 465, 357, 452,
                  451, 450, 449, 448, 261, 265, 383, 372, 345, 352, 376, 433, 367, 364, 379]
    cheek_right = [149, 170, 211, 202, 57, 186, 92, 165, 167, 164, 2, 97, 98, 129, 209, 126, 47, 114, 245, 128, 232,
                   231, 230, 229, 228, 31, 35, 156, 143, 116, 123, 147, 213, 138, 135, 150]
    forehead = [156, 71, 54, 103, 67, 109, 10, 338, 297, 332, 284, 301, 383, 300, 293, 334, 296, 336, 9, 107,
                66, 105, 63, 70]
    if facepart2drop == "eyes":
        contours = face_points[eye_right].astype(int)
        cv2.fillPoly(faceim, pts=[contours], color=(0, 0, 0))
        contours = face_points[eye_left].astype(int)
        cv2.fillPoly(faceim, pts=[contours], color=(0, 0, 0))
    elif facepart2drop == "nose":
        contours = face_points[nose].astype(int)
        cv2.fillPoly(faceim, pts=[contours], color=(0, 0, 0))
    elif facepart2drop == "mouth":
        contours = face_points[mouth].astype(int)
        cv2.fillPoly(faceim, pts=[contours], color=(0, 0, 0))
    elif facepart2drop == "chin":
        contours = face_points[chin].astype(int)
        cv2.fillPoly(faceim, pts=[contours], color=(0, 0, 0))
    elif facepart2drop == "cheek_left":
        contours = face_points[cheek_left].astype(int)
        cv2.fillPoly(faceim, pts=[contours], color=(0, 0, 0))
    elif facepart2drop == "cheek_right":
        contours = face_points[cheek_right].astype(int)
        cv2.fillPoly(faceim, pts=[contours], color=(0, 0, 0))
    elif facepart2drop == "forehead":
        contours = face_points[forehead].astype(int)
        cv2.fillPoly(faceim, pts=[contours], color=(0, 0, 0))
    elif facepart2drop == "rest":
        empty = np.ones(faceim.shape)
        for what in [eye_right, eye_left, nose, mouth, chin, cheek_left, cheek_right, forehead]:
            contours = face_points[what].astype(int)
            cv2.fillPoly(empty, pts=[contours], color=(0, 0, 0))
        faceim[empty == 1] = 0
    else:
        raise NotImplemented(f"Not implemented:", facepart2drop)  # probably this happened: [None] instead on None
    return faceim


class TrainDataset(Dataset):
    def __init__(self, df, indices, targets, mode="train", faceparts2drop: list = None):

        self.df = df[df.index.isin(indices)].reset_index(drop=True)
        self.transform = train_transform if mode == "train" else val_transform
        self.targets = targets
        self.mode = mode

        if not isinstance(faceparts2drop, type(None)):
            if not isinstance(faceparts2drop, list):
                raise TypeError("facepart_list is not a list. Type:", type(faceparts2drop))
        self.faceparts2drop = faceparts2drop

    def __len__(self):
        return len(self.df)

    def load_image(self, path):
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)  # this is for potentially bad rotated jpegs
        img = np.array(img)
        return img

    def get_targets_1hot(self, tmp):
        target_ = {}
        for key in self.targets.keys():
            if self.targets[key]["isClassification"]:
                empty = np.zeros(self.targets[key]["out_dim"])
                if tmp[key] == "NaN":
                    empty = -np.ones(self.targets[key]["out_dim"])  # set all to -1
                else:
                    empty[int(tmp[key])] = 1  # if error here: right nr. of output neurons chosen?
                target_[key] = empty
            else:
                if tmp[key] == "NaN":
                    target_[key] = -1
                else:
                    target_[key] = float(tmp[key])
        return target_

    def get_patient(self, idx):
        row = self.df.iloc[idx]
        targets = self.get_targets_1hot(row)

        if row["front"] != "NaN":
            front = self.load_image(row["base_dir"] + row["front"])
            front = self.transform(image=front)["image"]
            front = torch.tensor(front / 255).permute(2, 0, 1).float()
        else:
            front = torch.zeros((3, 320, 320)).float()
        if row["back"] != "NaN":
            back = self.load_image(row["base_dir"] + row["back"])
            back = self.transform(image=back)["image"]
            back = torch.tensor(back / 255).permute(2, 0, 1).float()
        else:
            back = torch.zeros((3, 320, 320)).float()

        if row["face"] != "NaN":
            face = self.load_image(row["base_dir"] + row["face"])


            if self.mode == "train" and np.random.random() > 0.2:
                face_loc = np.load(row["base_dir"] + row["face_locs"])
                face = random_mask_face_area(face, face_loc, facepart_list=None)

            if self.faceparts2drop is not None:
                face_loc = np.load(row["base_dir"] + row["face_locs"])
                face = random_mask_face_area(face, face_loc, facepart_list=self.faceparts2drop)

            face = self.transform(image=face)["image"]
            face = torch.tensor(face / 255).permute(2, 0, 1).float()
        else:
            face = torch.zeros((3, 320, 320)).float()

        images = {"face": face, "front": front, "back": back}
        return images, targets

    def __getitem__(self, idx):
        images, targets = self.get_patient(idx)
        meta = {"path": self.df.iloc[idx]["base_dir"]}
        return images, targets, meta
