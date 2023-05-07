from config import TrainGlobalConfig
from dataframe import define_targets, load_and_unshuffle_dataset
from dataset import TrainDataset#, TrainDatasetBody
from model import MultiTargetCNN, EnsembleInferenceModel
import torch
import glob
from tqdm.auto import tqdm
import numpy as np
import pickle
import torch.utils.mobile_optimizer as mobile_optimizer
import pandas as pd

config = TrainGlobalConfig()


def inference():
    df = load_and_unshuffle_dataset()
    targets = define_targets()

    body_result_dict, face_result_dict = inference_loop(df, targets, drop_what=None)

    with open(config.folder + '/face_result_dict.pickle', 'wb') as handle:
        pickle.dump(face_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(config.folder + '/body_result_dict.pickle', 'wb') as handle:
        pickle.dump(body_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def inference_extern():
    """
        For external validation on a different machine. Change the path below
    """

    df = pd.read_csv("../infdf.csv")  # set your path
    df = df.fillna("NaN")

    targets = define_targets()
    body_result_dict, face_result_dict = inference_loop(df, targets, drop_what=None, ext_val=True)
    with open(config.folder + '/face_result_dict_external.pickle', 'wb') as handle:
        pickle.dump(face_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(config.folder + '/body_result_dict_external.pickle', 'wb') as handle:
        pickle.dump(body_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def inference_facedrop():
    """
    Inference wrapper for running inference on parts of images, e.g., only cheeks
    """
    def invert_list(what):
        all_parts = ["eyes", "nose", "mouth", "chin", "cheek_left", "cheek_right", "forehead", "rest"]
        all_parts.remove(what)
        return all_parts

    for keep in [["eyes"], ["nose"], ["mouth"], ["chin"], ["cheek_left"], ["cheek_right"], ["forehead"]]:
        if keep is not None:
            print("Dropping everything but", keep)
            drop_what = [keep[0]] ##invert_list(keep[0])

        df = load_and_unshuffle_dataset()
        targets = define_targets()

        _, face_result_dict = inference_loop(df, targets, drop_what=drop_what)

        with open(config.folder + f'/facePic_drop{keep[0]}.pickle', 'wb') as handle:
            pickle.dump(face_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def trace_models():
    """
        JIT traces submodules for compatibility for different timm versions.
        This function loads weights from the folder and saves a smaller version.
    """
    targets = define_targets()
    for t in range(config.num_folds):

        fold_weights_face = glob.glob(config.folder + f"/face/{t}/best*.bin")
        for i in range(len(fold_weights_face)):
            print("loading", fold_weights_face[i])
            model = MultiTargetCNN(targets, in_chans=3)
            model.load_state_dict(torch.load(fold_weights_face[i])["model_state_dict"])
            model.to(torch.device("cpu"))

            model = torch.quantization.convert(model)
            model = torch.jit.script(model)
            model = mobile_optimizer.optimize_for_mobile(model)

            torch.jit.save(model, fold_weights_face[i][:-4] + ".pt")

        fold_weights_body = glob.glob(config.folder + f"/body/{t}/best*.bin")
        for i in range(len(fold_weights_body)):
            print("loading", fold_weights_body[i])
            model = MultiTargetCNN(targets, in_chans=6)
            model.load_state_dict(torch.load(fold_weights_body[i])["model_state_dict"])
            model.to(torch.device("cpu"))

            model = torch.quantization.convert(model)
            model = torch.jit.script(model)
            model = mobile_optimizer.optimize_for_mobile(model)

            torch.jit.save(model, fold_weights_body[i][:-4] + ".pt")


def inference_loop(df, targets, drop_what="", ext_val=False):
    """
    Runs two Models (Face-CNN, Body-CNN) on different folds
    :param df:
    :param targets:
    :param drop_what: Dropping everything except for Drop_what
    :param ext_val: If ext_val, run all folds on the same data
    :return: Two sets of model predictions (by face-CNN and body-CNN)
    """
    face_result_dict = {key: [] for key in targets.keys()}
    body_result_dict = {key: [] for key in targets.keys()}
    for t in range(config.num_folds):

        fold_weights_face = glob.glob(config.folder + f"/face/{t}/best*.bin")
        ensemble_face = EnsembleInferenceModel(fold_weights_face, targets, in_chans=3, use_quantized=config.use_quantized).to(config.device).eval()

        fold_weights_body = glob.glob(config.folder + f"/body/{t}/best*.bin")
        ensemble_body = EnsembleInferenceModel(fold_weights_body, targets, in_chans=6, use_quantized=config.use_quantized).to(config.device).eval()

        if not ext_val:
            test_ds = TrainDataset(df, df[df["fold"] == t].index, targets, mode="val", faceparts2drop=drop_what)
        else:
            test_ds = TrainDataset(df, df.index, targets, mode="val", faceparts2drop=drop_what)

        for i, (img, y, meta) in tqdm(enumerate(test_ds), total=len(test_ds)):

            hasFace, hasFront, hasBack = [(img[x].sum() > 0).item() for x in ["face", "front", "back"]]
            hasBody = sum([hasFront, hasBack]) > 0

            if hasFace:
                face = img["face"].unsqueeze(0).to(config.device, dtype=torch.float)
                with torch.no_grad():
                    out_dict_face, _ = ensemble_face(face)
                for key in out_dict_face.keys():
                    face_result_dict[key].extend(out_dict_face[key].detach().cpu().numpy())
            else:
                for key in targets.keys():
                    face_result_dict[key].extend([[np.nan, np.nan]] if targets[key]["isClassification"] else [[np.nan]])

            if hasBody:
                body = torch.cat([img["front"], img["back"]], dim=0).unsqueeze(0).to(config.device, dtype=torch.float)
                with torch.no_grad():
                    out_dict_body, _ = ensemble_body(body)
                for key in out_dict_body.keys():
                    body_result_dict[key].extend(out_dict_body[key].detach().cpu().numpy())
            else:
                for key in targets.keys():
                    body_result_dict[key].extend([[np.nan, np.nan]] if targets[key]["isClassification"] else [[np.nan]])
    face_result_dict = {key: np.array(value) for key, value in face_result_dict.items()}
    body_result_dict = {key: np.array(value) for key, value in body_result_dict.items()}
    return body_result_dict, face_result_dict


def inference_with_expert_weights():
    """
        The normal setup is one weight for all target variables.
        Expert weights refer to the best weight for the individual target variable.
    """
    df = load_and_unshuffle_dataset()
    targets = define_targets()

    face_result_dict = {key: [] for key in targets.keys()}
    body_result_dict = {key: [] for key in targets.keys()}
    for key in targets.keys():
        print("key", key)

        for t in range(config.num_folds):

            fold_weights_face = glob.glob(config.folder + f"/face/{t}/{key}*.bin")
            ensemble_face = EnsembleInferenceModel(fold_weights_face, targets, in_chans=3).to(config.device).eval()

            fold_weights_body = glob.glob(config.folder + f"/body/{t}/{key}*.bin")
            ensemble_body = EnsembleInferenceModel(fold_weights_body, targets, in_chans=6).to(config.device).eval()

            test_ds = TrainDataset(df, df[df["fold"] == t].index, targets, mode="val")

            for i, (img, y, meta) in tqdm(enumerate(test_ds)):

                hasFace, hasFront, hasBack = [(img[x].sum() > 0).item() for x in ["face", "front", "back"]]
                hasBody = sum([hasFront, hasBack]) > 0

                if hasFace:
                    face = img["face"].unsqueeze(0).to(config.device, dtype=torch.float)
                    with torch.no_grad():
                        out_dict_face, _ = ensemble_face(face)
                    face_result_dict[key].extend(out_dict_face[key].detach().cpu().numpy())
                else:
                    face_result_dict[key].extend([[np.nan, np.nan]] if targets[key]["isClassification"] else [[np.nan]])

                if hasBody:
                    body = torch.cat([img["front"], img["back"]], dim=0).unsqueeze(0).to(config.device, dtype=torch.float)
                    with torch.no_grad():
                        out_dict_body, _ = ensemble_body(body)
                    body_result_dict[key].extend(out_dict_body[key].detach().cpu().numpy())
                else:
                    body_result_dict[key].extend([[np.nan, np.nan]] if targets[key]["isClassification"] else [[np.nan]])

    face_result_dict = {key: np.array(value) for key, value in face_result_dict.items()}
    body_result_dict = {key: np.array(value) for key, value in body_result_dict.items()}

    with open(config.folder + '/face_result_dict_expert.pickle', 'wb') as handle:
        pickle.dump(face_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(config.folder + '/body_result_dict_expert.pickle', 'wb') as handle:
        pickle.dump(body_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


