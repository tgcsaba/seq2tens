
from ._classifiers import DeepLS2TClassifier, FCNClassifier, FCNLS2TClassifier, ResNetClassifier

_DICT_NAME_TO_CLASSIFIER_MAP = {
    'DEEPLS2T' : DeepLS2TClassifier,
    'FCN' : FCNClassifier,
    'FCNLS2T' : FCNLS2TClassifier,
    'RESNET' : ResNetClassifier
}

def get_classifier_by_name(model_name):
    model_name = model_name.replace('_', '').upper()
    if model_name in _DICT_NAME_TO_CLASSIFIER_MAP:
        return _DICT_NAME_TO_CLASSIFIER_MAP[model_name]
    else:
        raise ValueError(f'Error | get_classifier_by_name: Unknown model name \'{model_name}\'')

