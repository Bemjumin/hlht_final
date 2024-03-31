import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from transformers import BertForSequenceClassification

def load_imagenet_model(model_name,device):
    model = None
    if model_name == 'vgg16':
        model = ptcv_get_model("vgg16", pretrained=False)
        checkpoint = torch.load("model_weights/vgg16-0865-5ca155da.pth", map_location=device)
        model.load_state_dict(checkpoint)
    elif model_name == 'resnet50':
        model = ptcv_get_model("resnet50", pretrained=False)
        checkpoint = torch.load("model_weights/resnet50-0633-b00d1c8e.pth", map_location=device)
        model.load_state_dict(checkpoint)
    elif model_name =='bert':
        BERT_MODEL_DIR = "./model_bert"
        model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    return model