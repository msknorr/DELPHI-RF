import sys

from inference import inference, inference_facedrop, inference_extern, trace_models, inference_with_expert_weights
from train import train

if __name__ == '__main__':

    # train nested k-fold model
    if sys.argv[1] == "train":
        train()

    # run inference on oof data
    if sys.argv[1] == "test":
        inference()

    # inference on external PC. Please set external=True in config.py
    if sys.argv[1] == "ext":
        inference_extern()

    # save light version of model
    if sys.argv[1] == "trace":
        trace_models()

    # face drop analysis
    if sys.argv[1] == "face":
        inference_facedrop()