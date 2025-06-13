import os
import torch
import logging
import datetime

from recstudio.utils import parser_yaml, get_logger, set_color, color_dict_normal
from rg2 import RG2
from rgx import RGX


dataset = "ml-10m" #"amazon-electronics","steam"

LOG_DIR = './logs'
model_class = RG2 #RGX
model_conf = parser_yaml('./config.yaml')

log_path = f"RG2/{dataset}/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.log"
logger = get_logger(log_path)
torch.set_num_threads(model_conf['train']['num_threads'])

logger.info("Log saved in {}.".format(os.path.abspath(os.path.join(LOG_DIR, log_path))))
model = model_class(model_conf)
dataset_class = model_class._get_dataset_class()

datasets = dataset_class(name=dataset, config="./ml-10m.yaml").build(**model_conf['data'])
logger.info(f"{datasets[0]}")
logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(model_conf, False))
val_result = model.fit(*datasets[:2])
test_result = model.evaluate(datasets[-1])
