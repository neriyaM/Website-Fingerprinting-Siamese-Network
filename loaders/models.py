from models.df_model import DF
from models.base_model import build_base_model
import utils
import itertools
from models.var_cnn_model import var_cnn


def load_base_models(name, input_size, embedding_size):
    if name == "basic":
        return load_basic_models(input_size, embedding_size)
    elif name == "DF":
        return load_DF_models(input_size, embedding_size)
    elif name == "var_cnn":
        return load_var_cnn_models(input_size, embedding_size)


def load_basic_models(input_size, embedding_size):
    base_models = []
    batchnorm = [True, False]
    dropout = [True, False]
    for values in itertools.product(batchnorm, dropout):
        base_model = build_base_model(input_size, values[0], values[1], embedding_size)
        base_models.append(utils.BaseModel(base_model, "basic", values[0], values[1], embedding_size))

    return base_models


def load_DF_models(input_size, embedding_size):
    return [utils.BaseModel(DF(input_size, embedding_size), "DF", False, True, embedding_size)]


def load_var_cnn_models(input_size, embedding_size):
    return [utils.BaseModel(var_cnn(input_size, embedding_size), "var_cnn", False, False, embedding_size)]
