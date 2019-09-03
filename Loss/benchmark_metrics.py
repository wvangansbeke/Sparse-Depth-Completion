"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch

class Metrics(object):
    def __init__(self, max_depth=85.0, disp=False, normal=False):
        self.rmse, self.mae = 0, 0
        self.num = 0
        self.disp = disp
        self.max_depth = max_depth
        self.min_disp = 1.0/max_depth
        self.normal = normal

    def calculate(self, prediction, gt):
        valid_mask = (gt > 0).detach()

        self.num = valid_mask.sum().item()
        prediction = prediction[valid_mask]
        gt = gt[valid_mask]

        if self.disp:
            prediction = torch.clamp(prediction, min=self.min_disp)
            prediction = 1./prediction
            gt = 1./gt
        if self.normal:
            prediction = prediction * self.max_depth
            gt = gt * self.max_depth
        prediction = torch.clamp(prediction, min=0, max=self.max_depth)

        abs_diff = (prediction - gt).abs()
        self.rmse = torch.sqrt(torch.mean(torch.pow(abs_diff, 2))).item()
        self.mae = abs_diff.mean().item()

    def get_metric(self, metric_name):
        return self.__dict__[metric_name]


def allowed_metrics():
    return Metrics().__dict__.keys()
