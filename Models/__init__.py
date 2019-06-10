#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .model import uncertainty_net as model

model_dict = {'mod': model}

def allowed_models():
    return model_dict.keys()


def define_model(mod, **kwargs):
    if mod not in allowed_models():
        raise KeyError("The requested model: {} is not implemented".format(mod))
    else:
        return model_dict[mod](**kwargs)
