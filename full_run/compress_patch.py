
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
from jax import numpy as jnp, random as jr
from sbijax import NPE
from sbijax.nn import make_maf, make_mdn
from tensorflow_probability.substrates.jax import distributions as tfd

import matplotlib.pyplot as plt
import numpy as np


import tensorflow as tf
import tensorflow_datasets as tfds
import sys,os,re,yaml,glob

def load_config(config_path):
    with open(os.path.join(config_path)) as file:
        config = yaml.safe_load(file)
    return config

# hack to ensure we can see all the utils code
sys.path.append(config["network_code_dir"])

from training_loop import *
from network.new_epe_code import *




base_path = "/data103/makinen/des_sims/Gower_street_SBI_tfrecords/"
use_noise_realisations =['0','1','2','3','10', '11', '12']#,'11','12','13','14','15']
cls_stats_file = jnp.load(config["cls"]["cls_normalisation_stats_filename"])
S2_cls=cls_stats_file["S2_cls"]
mean_cl=cls_stats_file["mean_cl"]
std_cl=cls_stats_file["std_cl"]
cut_idx=cls_stats_file["cut_idx"]

# ----- config stuff

# ----------------------------------------------------------------------------
def gaussian_noise_augmentation(x, y, cls, param_idx=None):
    x += tf.random.normal(
                        shape = [512, 512,8],
                        mean = 0,
                        stddev = 1e-3,
                        dtype = tf.float32
                        ) #Mask option?

    # add noise to cls
    cls += tf.random.normal(
                shape = [10, 2, 4, 28],
                mean = 0, 
                stddev =std_cl*1e-3,
                dtype = tf.float32
                )

    if param_idx is not None:
        y = tf.expand_dims(y[param_idx], 0)

    # data is now a dictionary
    return {"y": {"kappa": x, "cls": cls}, "theta": y}



def ret_scaled_param(data,param):
    s = 0.6
    if param == 'AIA':
        hi = 3
        lo = -3

    if param == "s8":
        hi = 1.32
        lo = 0.4

    if param == 'om':
        hi = 0.5
        lo = 0.13

    if param == "h":
        hi = 0.6
        lo = 0.8

    if param == 'w':
        hi = -0.33
        lo = -1.8
    if param == "ns":
        hi = 0.99
        lo = 0.94
    if param == 'ob':
        hi = 0.061
        lo = 0.037

    return (data[param] -lo)*s/(hi - lo) + (1 - s)*0.5


def param_proc(data, scale_params=True):
    Y = []
    for p in sel_params:
        # scale params if so desired
        if scale_params:
            if p=="S8":
                Y.append(data["s8"]*tf.math.sqrt(data["om"]/0.3)-0.25)
            else:
                Y.append(ret_scaled_param(data, p))
        
        # else just return the param --> move to S8
        else:
            if p=="S8":
                Y.append(data["s8"]*tf.math.sqrt(data["om"]/0.3))
            else:
                Y.append(data[p])
            
    Y = tf.convert_to_tensor( Y , dtype = tf.float32)
    return Y

def cls_proc(data):
    Cls = []
    for c in cl_modes:

        cls = tf.io.parse_tensor(data[c], out_type=tf.float32)
        cls = tf.cast(cls, tf.float32)
        
        Cls.append(cls)
    
    return Cls #tf.convert_to_tensor(Cls, dtype=tf.float32)


def return_train_test_lists(patch):
    files = glob.glob(base_path + "{}/".format(patch) +"shear_maps_*")
    print(base_path + "{}/".format(patch) +"shear_maps_*")

    # sort the files to match A to B to C !
    files.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

    
    print(len(files))
    train_file_list = []
    test_file_list = []
    lfi_file_list = []
    test_file_systematic_list = []
    for file in files:
        noiserel = file.split("_rel")[0].split('noiserel')[-1]
        if noiserel in use_noise_realisations:
            if noiserel == '3':
                test_file_list.append(file)
            elif noiserel == '10':
                lfi_file_list.append(file)
            elif noiserel == '12':
                test_file_systematic_list.append(file)
            else:
                train_file_list.append(file)
    return train_file_list, test_file_list, lfi_file_list, test_file_systematic_list

def parse_serialized_file(cereal_yum, scale_params=True, filter_w=True):

    features = {
            "kappa_patch":  tf.io.FixedLenFeature([], tf.string),
            #"CLS":  tf.io.FixedLenFeature([], tf.float32, default_value=0.0),

            # load the cls values
            "1_1": tf.io.FixedLenFeature([], tf.string),
            "1_2": tf.io.FixedLenFeature([], tf.string),
            "1_3": tf.io.FixedLenFeature([], tf.string),
            "1_4": tf.io.FixedLenFeature([], tf.string),
            "2_2": tf.io.FixedLenFeature([], tf.string),
            "2_3": tf.io.FixedLenFeature([], tf.string),
            "2_4": tf.io.FixedLenFeature([], tf.string),
            "3_3": tf.io.FixedLenFeature([], tf.string),
            "3_4": tf.io.FixedLenFeature([], tf.string),
            "4_4": tf.io.FixedLenFeature([], tf.string),

            # load parameters
            "s8": tf.io.FixedLenFeature([], tf.float32),
            "om": tf.io.FixedLenFeature([], tf.float32),
            "AIA": tf.io.FixedLenFeature([], tf.float32),
            "w": tf.io.FixedLenFeature([], tf.float32)
    }

    data = tf.io.parse_single_example(cereal_yum, features)


    # else:
    # proceed as usual to get everything
    Y = param_proc(data, scale_params=scale_params)
    CLS_DS = cls_proc(data)

    X_DS = tf.io.parse_tensor(data["kappa_patch"], out_type=tf.float32)
    X_DS = tf.cast(X_DS, tf.float32)
    X_DS = tf.reshape(X_DS, shape=(8,512,512))
    X_DS = tf.transpose(X_DS, perm = [1,2,0])
    
    return X_DS,Y,CLS_DS

# ----------------------------------------------------------------------------



# ----- dataset code -----
# ----------------------------------------------------------------------------
def get_tfdataset(files, 
                        batch_size=64, 
                        epochs=1000,
                        scale_params=True, 
                        param_idx=None,
                        to_numpy=True,
                        shuffle=True,
                        shuffle_buffer_size=100,
                        drop_remainder=True):
    
    num_files = len(files)
    print("num files: ", num_files)
    tfdataset = tf.data.Dataset.from_tensor_slices(files)

    if param_idx is not None:
        print("retrieving parameter %d"%(param_idx))
    
    tfdataset = tfdataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=n_readers,
        block_length=1,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    tfdataset = tfdataset.map(
        lambda serialized_example: parse_serialized_file(
            serialized_example,
            scale_params=scale_params    
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # control shuffle to keep interleaved datasets matched up
    if shuffle:
        tfdataset = tfdataset.shuffle(shuffle_buffer_size)

    
    tfdataset = tfdataset.map(lambda maps,vals,cls: \
                                gaussian_noise_augmentation(maps,vals,cls,param_idx=param_idx),\
                                num_parallel_calls=tf.data.AUTOTUNE)

    if to_numpy:
        tfdataset = tfdataset.as_numpy_iterator()
        tfdataset.num_batch_per_epoch = num_files // batch_size
        tfdataset.num_samples = (num_files // batch_size) * batch_size


    return tfdataset



def consolidate_dsets(A,B):
    # check that cosmo params are the same    
    return {"y": {"A": {"summaries": A[0], 
                        "theta_A": A[1]
                        },
            "B": {"kappa": B["y"]["kappa"], 
                "cls": B["y"]["cls"]},
            },
        "theta":B["theta"],  
            }


# LOAD IN SUMMARIES FROM PATCH A, ORDERED ACCORDINGLY
def stack_tfdatasets(summaries_A, params_A,
                        files_B,
                        epochs=1000,
                        batch_size=64, scale_params=True, to_numpy=True, shuffle=True,
                        shuffle_buffer_size=100, drop_remainder=True):

    num_files_A = len(summaries_A)
    num_files_B = len(files_B)
    # if num_files_A != num_files_B:
    #     raise Exception("number of files for patch A (%d) doesn't \n \
    #                             match number of files for patch B (%d)"%(num_files_A, num_files_B))
    
    tfdataset_A = tf.data.Dataset.from_tensor_slices((summaries_A, params_A)) # we probably don't need the params here
    tfdataset_B = get_tfdataset(files_B, batch_size=batch_size, scale_params=scale_params, 
                                to_numpy=False, shuffle=False, 
                                shuffle_buffer_size=shuffle_buffer_size, 
                                drop_remainder=drop_remainder)

    # just zip two datasets
    tfdataset =  tf.data.Dataset.zip((tfdataset_A,tfdataset_B))
    # now map the dictionary function to get in the right format for mdn
    tfdataset = tfdataset.map(lambda A,B: consolidate_dsets(A,B), num_parallel_calls=tf.data.AUTOTUNE,) 
    
    # THEN shuffle
    if shuffle:
        tfdataset = tfdataset.shuffle(shuffle_buffer_size)
    # batch here
    tfdataset = tfdataset.batch(batch_size, drop_remainder=drop_remainder) #.prefetch(tf.data.AUTOTUNE) 
    # repeat dataset
    tfdataset = tfdataset.repeat(epochs)

    if to_numpy:
        tfdataset = tfdataset.as_numpy_iterator()
        tfdataset.num_batch_per_epoch = num_files_A // batch_size
        tfdataset.num_samples = (num_files_A // batch_size) * batch_size

    return tfdataset
    
    # ----------------------------------------------------------------------------


    


# ----------------------------------------------------------------------------

from compress_cls import *
# these should have been imported from the Cls script

print("loading Cls network ...")

slice_cls = lambda d: slice_cls(d, cut_idx=cut_idx)
slice_cls_single = lambda d: slice_cls_single(d, cut_idx=cut_idx)

key = jax.random.PRNGKey(0) # pseudo-random key for Jax network.
cls_model = ClsModel(n_summaries=config["n_summaries"]["cls"])

cls_single_shape = (10, 2, 4, 28,)
clsdir = os.path.join(config["project_dir"],  config["cls"]["net_dir"])
w_cls_compress = load_obj(os.path.join(clsdir, config["cls"]["w_filename"]))

cls_compression = lambda d: cls_model.apply(w_cls_compress, d, method=cls_model.get_embed)



# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------

class simplePatchCNN(nn.Module):
    """CNN to extract extra information from WL field"""
    filters: Sequence[int]
    cls_compression: Callable
    n_extra: int 
    n_summaries_cls: int
    n_existing: int
    n_total: int
    summary_idxs: tuple
    act_cnn: Callable = nn.relu
    act_dense: Callable = smooth_leaky
    dtype: Any = jnp.float32


    @nn.compact
    def __call__(self, x):

        # unpack data
        inputs = x

        # RIGHT SO HERE WE HAVE EXISTING CLS NUMBERS AND OLD SUMMARIES
        # TAKE THE PATCH A SUMMARIES AND VARY THE CLS COMPRESSION UNDER NOISE
        existing_info = inputs["A"]["summaries"] #[:self.n_existing] # take everyting and then we'll write over with Cls
        # print("existing info", existing_info.shape)

        # pass the cls through the network again to increase variation in the optimisation
        cls_summs = self.cls_compression(inputs["B"]["cls"])
        
        x = inputs["B"]["kappa"]
        x = x.astype(self.dtype)

        filters = self.filters
        
        for i in range(8):
            x = nn.Conv(features=filters, kernel_size=(3,3), strides=(1,1), padding="SAME", dtype=self.dtype)(x)
            x = self.act_cnn(x)
            x = nn.avg_pool(x, (2,2), strides=(2,2))

        # dense net out
        x = x.reshape(-1)
        x = nn.Dense(20)(x)
        x = self.act_dense(x)
        x = nn.Dense(20)(x)
        x = self.act_dense(x)
        x = nn.Dense(10)(x)
        x = self.act_dense(x)
        x = nn.Dense(10)(x)
        x = self.act_dense(x)
        x = nn.Dense(5)(x)
        x = self.act_dense(x)
        x = nn.Dense(5)(x)
        x = self.act_dense(x)


        # vanilla code
        x = nn.Dense(self.n_extra, dtype=self.dtype)(x).reshape(-1)
        x = x.reshape(-1).astype(jnp.float32) # make sure output is float32
        x = nn.LayerNorm()(x)

        # concatenate all existing information
        #x = jnp.concatenate([cls_summs, existing_info.reshape(-1), x])

        # set all existing information in its right place
        # | Cls |     a|      b|     c|

        # stop1 = self.n_summaries_cls + self.n_existing
        #print("x", x.shape)
        #print("existing info", existing_info.shape)

        #outputs = jnp.zeros((self.n_total,))
        outputs = jnp.array(existing_info)
        # first existing information goes in (including old Cls information)
       # outputs = outputs.at[: self.n_existing].set(existing_info.reshape(-1))
        #print("outputs exist", outputs)
        # then Cls information from network application
        outputs = outputs.at[:self.n_summaries_cls].set(cls_summs)
        # then new information
        outputs = outputs.at[self.summary_idxs[0]:self.summary_idxs[1]].set(x)

        return outputs


# ----------------------------------------------------------------------------


if __name__ == "__main__":

    # load config file from command line
    config = load_config(sys.argv[1])

    # ---- config stuff
    patch = sys.argv[2]   # should probably arg to call
    print("beginning compression for patch %s"%(patch))

    with tf.device("CPU"):
        # could these move to config file ?
        sel_params = ["om","S8","w"]
        cl_modes = ["1_1","1_2","1_3","1_4","2_2","2_3","2_4","3_3","3_4", "4_4"]


        BATCH_SIZE = 128
        EPOCHS = 1000 # max epochs
        n_readers=1


        # smarter way to collect files ?
        train_files, test_files, lfi_files, test_sys_files = return_train_test_lists('{}'.format("A"))
        train_files_B, test_files_B, lfi_files_B, test_sys_files_B = return_train_test_lists('{}'.format("B"))

        # check to see if files are mismatched
        for i,t in enumerate(train_files):
            if t[55:] == train_files_B[i][55:]:
                pass
            else:
                print("alert ! file mismatch")

        SHUFFLE_BUFFER_SIZE = 100
        
        #train_files, test_files, lfi_files, test_sys_files = return_train_test_lists('{}'.format(patch))
        num_train_files = len(train_files)
        num_test_files = len(test_files)
        num_lfi_files = len(lfi_files)
        num_sys_files = len(test_sys_files)

        print("train files: ", num_train_files, "test files: ", num_test_files, "lfi files: ", num_lfi_files, "test sys files: ", num_sys_files,)


        # ----- dataset code -----

        print("collecting datasets ...")

        # ARGPARSE STUFF ?
        create_new_summary_file = bool(int(sys.argv[3]))

        # create empty summaries dataset if we're just dealing with patch A
        
        # summaries code
        # SOME IMPORTANT HYPERPARAMS --> FROM CONFIG
        N_PARAMS = config["n_params"]

        N_SUMMARIES_CLS = config["n_summaries"]["cls"]    # default 10
        N_SUMMARIES_A =   config["n_summaries"]["A"]
        N_SUMMARIES_B =   config["n_summaries"]["B"]
        N_SUMMARIES_C =   config["n_summaries"]["C"]

        N_TOTAL_SUMMARIES = N_SUMMARIES_CLS + N_SUMMARIES_A + N_SUMMARIES_B + N_SUMMARIES_C

        print("total summaries", N_TOTAL_SUMMARIES)

        idxs_A =  ((N_SUMMARIES_CLS), (N_SUMMARIES_CLS + N_SUMMARIES_A))
        idxs_B = ((N_SUMMARIES_CLS + N_SUMMARIES_A), (N_SUMMARIES_CLS + N_SUMMARIES_A + N_SUMMARIES_B))
        idxs_C = ((N_SUMMARIES_CLS + N_SUMMARIES_A + N_SUMMARIES_B), (N_SUMMARIES_CLS + N_SUMMARIES_A + N_SUMMARIES_B + N_SUMMARIES_C))
        
        if patch == "A":
            N_EXISTING = N_SUMMARIES_CLS
            idxs = idxs_A

        elif patch == "B":
            N_EXISTING = N_SUMMARIES_CLS + N_SUMMARIES_A
            idxs = idxs_B
        else:
            N_EXISTING =  N_SUMMARIES_CLS + N_SUMMARIES_A + N_SUMMARIES_B 
            idxs = idxs_C


        
        print("idxs", idxs)

        if create_new_summary_file:
            print("creating dummy summary file")
            summaries_A_file = dict(
                summaries_lfi=np.zeros((len(lfi_files), N_TOTAL_SUMMARIES)),
                params_lfi=np.zeros((len(lfi_files), N_PARAMS)),
                summaries_test=np.zeros((len(test_files), N_TOTAL_SUMMARIES)),
                params_test=np.zeros((len(test_files), N_PARAMS)),
                summaries_sys=np.zeros((len(test_sys_files), N_TOTAL_SUMMARIES)),
                params_sys=np.zeros((len(test_sys_files), N_PARAMS)),
                summaries_train=np.zeros((len(train_files), N_TOTAL_SUMMARIES)),
                params_train=np.zeros((len(train_files), N_PARAMS))
                )

        else:
            summary_file_path = config["summary_path"] + "patch_%s"%(patch) + ".npz"
            print("loading existing summary file from %s"%(summary_file_path))
            summaries_A_file = np.load(summary_file_path)

    

        #summaries_A_file = np.load("/data103/makinen/des_results/patch_nets/summaries_new_patchA_S8.npz")

        # train, test validation datasets
        train_dataset = stack_tfdatasets(summaries_A_file["summaries_train"], summaries_A_file["params_train"],
                                            train_files_B, batch_size=BATCH_SIZE, scale_params=True, to_numpy=True)
        
        test_dataset = stack_tfdatasets(summaries_A_file["summaries_test"], summaries_A_file["params_test"],
                    test_files_B, batch_size=BATCH_SIZE, scale_params=True, to_numpy=True, drop_remainder=False)

        
        lfi_dataset = stack_tfdatasets(summaries_A_file["summaries_lfi"], summaries_A_file["params_lfi"],
                    lfi_files_B, batch_size=BATCH_SIZE, scale_params=False, to_numpy=True, epochs=3)
        
        sys_dataset = stack_tfdatasets(summaries_A_file["summaries_sys"], summaries_A_file["params_sys"],
                    test_sys_files_B, batch_size=BATCH_SIZE, scale_params=False, to_numpy=True, epochs=3)
        

        # check to see that parameters line up along datasets
        if not create_new_summary_file:
            print('checking that dataset thetas line up')
            for i in range(2):
                data = next(iter(test_dataset))
                match = (data['y']['A']['theta_A'] == data['theta'])[0][0]

            assert not match, "datasets are not lined up !"

    
    # ----------------------------------------------------------------------------

    # MODEL --> from config
    print("testing network initialisation ...")

    model_key = jr.PRNGKey(int(config["model_key"]))

    patch_net = simplePatchCNN(
                    filters=32,
                    act_cnn=nn.relu,
                    act_dense=smooth_leaky,
                    cls_compression=cls_compression,
                    n_extra=4,
                    n_existing=N_EXISTING,
                    n_summaries_cls=N_SUMMARIES_CLS,
                    n_total=N_TOTAL_SUMMARIES,
                    summary_idxs = idxs,
                    dtype=jnp.float32,
    )


    wembed = patch_net.init(model_key, {"A":{"summaries": jnp.ones((N_TOTAL_SUMMARIES)),
                                        "cls": jnp.ones((10,2,4,28))},
                                        "B":{"kappa": jnp.ones((512,512,8)),
                                        "cls": jnp.ones((10,2,4,28))}
                                        })
    data = next(iter(train_dataset))
    #print(data['y']['A']['summaries'].shape)
    appl = lambda d: patch_net.apply(wembed, d)
    outs = jax.vmap(appl)(data['y'])
    print("output shape: ", outs.shape)
    print("example outs: ", outs[0])

    # FULL MODEL CLASS WITH MDN BITS
    
    class fullModel(EPEModel, nn.Module):
        n_extra: int
        n_params: int = 3
        n_components: int = 4

        def setup(self):
            self.mdn = MDN(
                            hidden_channels=[100],
                            n_components=self.n_components,
                            n_dimension=self.n_params,
                            act=nn.relu)
            
            self.embeding_net = simplePatchCNN(
                    filters=32,
                    act_cnn=nn.relu,
                    act_dense=smooth_leaky,
                    cls_compression=cls_compression,
                    n_extra=4,
                    n_existing=N_EXISTING,
                    n_summaries_cls=N_SUMMARIES_CLS,
                    n_total=N_TOTAL_SUMMARIES,
                    summary_idxs = idxs,
                    dtype=jnp.float32,
        )

        def get_embed(self, x):
            return self.embeding_net(x)

        def log_prob(self, x, y):
            x = self.embeding_net(x)
            return self.mdn(x, y) 
        
        def __call__(self, x, y):
            x = self.embeding_net(x)
            return self.mdn(x, y)

        

    print('initialising full model ...')

    model = fullModel(n_extra=config["n_summaries"][patch], 
                      n_params=config["n_params"], 
                      n_components=config['patch_net']['n_components'])

    wfull = model.init(model_key, {"A":{"summaries": jnp.ones((N_TOTAL_SUMMARIES)),
                                        "cls": jnp.ones((10,2,4,28))},
                                        "B":{"kappa": jnp.ones((512,512,8)),
                                        "cls": jnp.ones((10,2,4,28))}
                                        }, jnp.ones(3))


    # instatiate minimiser code
    epe_minimiser = EPE_minimiser(density_estimator=model)

    print('training compression ...')

    # Clip gradients at max value, and evt. apply weight decay
    transf = [optax.clip(2.0)]
    # transf.append(optax.add_decayed_weights(1e-4))
    optimizer = optax.chain(
        *transf,
        optax.adam(learning_rate=1e-5) # 8e-6
    )


    epochs = config["patch_net"]["epochs"]
    patience = config["patch_net"]["patience"]

    outdir = os.path.join(config["project_dir"], "patch_%s_net_%s/"%(patch, config["run_name"]))
    print("saving network weights and results to: ", outdir)

    w, losses = epe_minimiser.fit(jr.PRNGKey(2), 
                        data=None, 
                        batch_size=128,
                        n_iter=epochs,
                        n_early_stopping_patience=patience,  #20
                        train_dataset=train_dataset,
                        val_dataset=test_dataset,
                        optimizer=optimizer,
                        outdir=outdir,
                        #params=w,
                        )


    # save everything
    # save weights, losses, and config script to the output directory

    print('saving everything')
    save_obj(losses, os.path.join(outdir, "history"))
    save_obj(config, os.path.join(outdir, "config_dict")) # save config just in case



    # get all the summaries
