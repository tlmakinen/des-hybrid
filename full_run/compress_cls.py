import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import optax

from tensorflow_probability.substrates.jax import distributions as tfd
from typing import Sequence, Any, Callable

import numpy as np
import flax.linen as nn
import matplotlib.pyplot as plt
import cloudpickle as pickle

import yaml,sys,os
from pathlib import Path
Array = Any

def load_config(config_path):
    with open(os.path.join(config_path)) as file:
        config = yaml.safe_load(file)
    return config

# load config file from command line
config = load_config(sys.argv[1])

# hack to ensure we can see all the utils code
sys.path.append(config["network_code_dir"])

from training_loop import *
from network.new_epe_code import *







print("loading data ...")

file = np.load(config["cls"]["cls_dataset"])

params_train = file["params_train"]
cls_train = file["cls_train"]

params_LFI = file["params_lfi"]
cls_lfi = file["cls_lfi"]

params_test = file["params_test"]
cls_test = file["cls_test"]


params_sys = file["params_sys"]
cls_sys = file["cls_sys"]

print(params_train[:, 2].min())


# TODO: SHOULD WE MOVE THIS TO SEPARATE MODULE ?

# calculate scaling for standardisation
S1_cls = cls_train.mean(0)
S2_cls = (cls_train**2).mean(0)

mean_cl = S1_cls
std_cl = np.sqrt(S2_cls - mean_cl**2)
cut_idx = config["cls"]["cut_idx"]


def slice_cls(cls):
    cls = (cls - mean_cl) / std_cl
    cls = cls[:, :, 0, 0::3, :cut_idx]
    return cls.reshape(cls.shape[0], -1)

def slice_cls_single(cls):
    cls = (cls - mean_cl) / std_cl
    cls = cls[:, 0, 0::3, :cut_idx]
    return cls.reshape(cls.shape[0], -1)



class Cls_MLP_Network(nn.Module):
    hidden_channels: list
    n_summaries: int
    act: Callable = nn.relu
    sigmoid_out: bool = False

    def setup(self):
        self.embed = nn.Dense(450)
        self.embed2 = nn.Dense(500)
        #self.layernorm = nn.LayerNorm()
        self.net = MLP(self.hidden_channels + (self.n_summaries,), act=self.act)

    def __call__(self, x):

        # cut down mass value
        x = slice_cls_single(x)
        x = x.reshape(-1)
        x = self.embed(x)
        # x = self.layernorm(x)
        #x = self.embed2(x)
        x = self.act(x)
        x = self.net(x)

        if self.sigmoid_out:
            x = nn.sigmoid(x)

        return x



class ClsModel(EPEModel, nn.Module):
    n_summaries: int
    n_params: int = 3
    n_components: int = 4
    n_hidden_mdn: int = 100
    

    def setup(self):
        self.mdn = MDN(
                        hidden_channels=[self.n_hidden_mdn],
                        n_components=self.n_components,
                        n_dimension=self.n_params,
                        act=nn.relu)
        
        self.mlp = Cls_MLP_Network(
                        hidden_channels=[256]*10,
                        n_summaries=self.n_summaries,
                        act=smooth_leaky,
                        sigmoid_out=False)
    
        #self.norm = nn.LayerNorm()

    def get_embed(self, x):
        return self.mlp(x)

    def log_prob(self, x, theta):
        x = self.mlp(x)
        return self.mdn(x, theta) 


    def __call__(self, x, theta):
        x = self.mlp(x)
        return self.mdn(x, theta)
    

print("training network ...")
print("\n extracting %d summaries"%(config["n_summaries_cls"]))

key = jr.PRNGKey(4)
cls_single_shape = (10, 2, 4, 28,)


model = ClsModel(n_summaries=int(config["n_summaries_cls"]),)
                #n_hidden_mdn=config["cls"]["n_hidden_mdn"])

w = model.init(key, cls_train[0], jnp.ones(3,), method=model.log_prob)





w, losses = run_training_loop(model, key, (cls_train, params_train),
                                    (cls_test, params_test), 
                                        learning_rate=1e-5,
                                        schedule=True,
                                        epochs=config["cls"]["epochs"],
                                        batch_size=128)


print("saving everything")

# save weights, losses, and config script to the output directory
outdir = os.path.join(config["project_dir"],  config["cls"]["net_dir"])
Path(outdir).mkdir(parents=True, exist_ok=True)

# save it all
save_obj(w, os.path.join(outdir, config["cls"]["w_filename"].split(".pkl")[0]))
save_obj(losses, os.path.join(outdir, "history"))
save_obj(config, os.path.join(outdir, "config_dict")) # save config just in case