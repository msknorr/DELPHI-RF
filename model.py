import timm
import torch
from torch import nn
import torch.utils.mobile_optimizer as mobile_optimizer


class MultiTargetCNN(nn.Module):
    def __init__(self, targets, in_chans=3, pretrained=True):
        super(MultiTargetCNN, self).__init__()
        model = timm.create_model('mobilenetv3_large_100', in_chans=in_chans, pretrained=pretrained)
        self.last_layer_shape = 1280  # model.fc.in_features
        self.backbone = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.output_layers = nn.ModuleList([nn.Linear(self.last_layer_shape, targets[key]["out_dim"]) for key in targets.keys()])

    def forward(self, img):
        latent = self.backbone(img)
        outs = [layer(latent) for layer in self.output_layers]
        return outs, latent

class MultiTargetCNNResnet50(nn.Module):
    def __init__(self, targets, in_chans=3, pretrained=True):
        super(MultiTargetCNN, self).__init__()
        model = timm.create_model('resnet50', in_chans=in_chans, pretrained=pretrained)
        self.last_layer_shape = 2048  # model.fc.in_features
        self.backbone = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.output_layers = nn.ModuleList([nn.Linear(self.last_layer_shape, targets[key]["out_dim"]) for key in targets.keys()])

    def forward(self, img):
        latent = self.backbone(img)
        outs = [layer(latent) for layer in self.output_layers]
        return outs, latent

class EnsembleInferenceModel(nn.Module):
    def __init__(self, weight_paths, targets, in_chans=3, use_quantized=False):
        super(EnsembleInferenceModel, self).__init__()

        models = []
        for i in range(len(weight_paths)):
            print("loading", weight_paths[i])

            if use_quantized == False:
                model = MultiTargetCNN(targets, in_chans=in_chans)
                model.load_state_dict(torch.load(weight_paths[i])["model_state_dict"])
            else:
                model = torch.jit.load(weight_paths[i][:-4] + ".pt")

            models.append(model)
        self.models = nn.ModuleList(models)
        self.targets = targets

    def forward(self, img):

        arr = []
        latents = []
        for i in range(len(self.models)):
            outp, latent = self.models[i](img)
            arr.append(outp)
            latents.append(latent)

        final = {}
        for t, key in enumerate(self.targets.keys()):
            new = torch.stack([a[t] for a in arr])
            if self.targets[key]["isClassification"]:
                new = new.softmax(2)
                new = new.mean(dim=0)
            else:
                new = new.mean(dim=0)
            final[key] = new

        return final, torch.mean(torch.stack(latents), dim=0)
