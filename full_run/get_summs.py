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
from tqdm import tqdm


def load_config(config_path):
    with open(os.path.join(config_path)) as file:
        config = yaml.safe_load(file)
    return config






if __name__ == "__main__":

    # load config file from command line
    config = load_config(sys.argv[1])

    # hack to ensure we can see all the utils code
    sys.path.append(config["network_code_dir"])

    from training_loop import *
    from network.new_epe_code import *
    from compress_patch import *


    # load config file from command line
    config = load_config(sys.argv[1])

    # ---- config stuff
    patch = sys.argv[2]   # should probably arg to call
    print("beginning compression for patch %s"%(patch))

    with tf.device("CPU"):
        # could these move to config file ?
        sel_params = ["om","S8","w"]
        cl_modes = ["1_1","1_2","1_3","1_4","2_2","2_3","2_4","3_3","3_4", "4_4"]


        BATCH_SIZE = config["patch_net"]["batch_size"]
        n_readers=1


        # smarter way to collect files ?
        train_files, test_files, lfi_files, test_sys_files = return_train_test_lists('{}'.format("A"))
        train_files_B, test_files_B, lfi_files_B, test_sys_files_B = return_train_test_lists('{}'.format(patch))

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


        
        print("idxs_B", idxs_B)
    # print("learning %d new summaries for patch compression"%(N_TOTAL_SUMMARIES))



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
    
    # pull in model setup

    print('initialising full model ...')

    model = fullModel(n_extra=4, n_params=3, n_components=4)
    wfull = model.init(model_key, {"A":{"summaries": jnp.ones((N_TOTAL_SUMMARIES)),
                                        "cls": jnp.ones((10,2,4,28))},
                                        "B":{"kappa": jnp.ones((512,512,8)),
                                        "cls": jnp.ones((10,2,4,28))}
                                        }, jnp.ones(3))

    # load in weights
    outdir = os.path.join(config["project_dir"], "patch_%s_net_%s/"%(patch, config["run_name"]))
    print("loading network weights and results from: ", outdir)

    w = load_obj(outdir + "best_params.pkl")

    def apply_embedding(input_data, w):
        fn = lambda d: model.apply(w, x=d, method=model.get_embed)
        return jax.vmap(fn)(input_data)


    # function for collecting summaries from each dataset -> could be made neater

    def get_unshuffled_summaries(
            outname,
            existing_summary_file,
                            ):


        # get datasets
        
        # train, test validation datasets
        train_dataset2 = stack_tfdatasets(summaries_A_file["summaries_train"], summaries_A_file["params_train"],
                                            train_files_B, batch_size=BATCH_SIZE, scale_params=True, to_numpy=True, shuffle=False)
        
        # scale params for training (default)
        test_dataset2 = stack_tfdatasets(summaries_A_file["summaries_test"], summaries_A_file["params_test"],
                    test_files_B, batch_size=BATCH_SIZE, scale_params=True, to_numpy=True, drop_remainder=False, shuffle=False)
        
        lfi_dataset = stack_tfdatasets(summaries_A_file["summaries_lfi"], summaries_A_file["params_lfi"],
                    lfi_files_B, batch_size=BATCH_SIZE, scale_params=False, to_numpy=True, epochs=3, shuffle=False)
        
        sys_dataset = stack_tfdatasets(summaries_A_file["summaries_sys"], summaries_A_file["params_sys"],
                    test_sys_files_B, batch_size=BATCH_SIZE, scale_params=False, to_numpy=True, epochs=3, shuffle=False)
        
        train_dataset_unscaled = stack_tfdatasets(summaries_A_file["summaries_train"], summaries_A_file["params_train"],
                                            train_files_B, batch_size=BATCH_SIZE, scale_params=False, to_numpy=True, shuffle=False)
        

        
        summaries_LFI = []
        params_Tru_LFI = []
        
        for i in tqdm(range(lfi_dataset.num_batch_per_epoch)):
        
            data = next(iter(lfi_dataset))
        
            X  = data['y']
            theta_true = data['theta']
            summs_out = apply_embedding(X)
            params_Tru_LFI.append(theta_true)
            summaries_LFI.append(summs_out)
        
        
        summaries_test = []
        params_Tru_test = []
        
        for i in tqdm(range(test_dataset2.num_batch_per_epoch)):
        
            data = next(iter(test_dataset2))
        
            X  = data['y']
            theta_true = data['theta']
            summs_out = apply_embedding(X)
            params_Tru_test.append(theta_true)
            summaries_test.append(summs_out)
        
        
        summaries_sys = []
        params_Tru_sys = []
        cls_sys = []
        
        for i in tqdm(range(sys_dataset.num_batch_per_epoch)):
        
            data = next(iter(sys_dataset))
        
            X  = data['y']
            cls = data['y']['B']['cls']
            theta_true = data['theta']
            summs_out = apply_embedding(X)
            params_Tru_sys.append(theta_true)
            summaries_sys.append(summs_out)
            cls_sys.append(cls)
        
        
        summaries_train = []
        params_Tru_train = []
        
        for i in tqdm(range(train_dataset2.num_batch_per_epoch)):
        
            data = next(iter(train_dataset2))
        
            X  = data['y']
            theta_true = data['theta']
            summs_out = apply_embedding(X)
            params_Tru_train.append(theta_true)
            summaries_train.append(summs_out)
        
        
        # save all summaries from patch A to pull in
        np.savez(outname,
                summaries_lfi=np.concatenate(summaries_LFI, 0),
                params_lfi=np.concatenate(params_Tru_LFI, 0),
                summaries_test=np.concatenate(summaries_test,0),
                params_test=np.concatenate(params_Tru_test,0),
                summaries_sys=np.concatenate(summaries_sys,0),
                params_sys=np.concatenate(params_Tru_sys,0),
                summaries_train=np.concatenate(summaries_train,0),
                params_train=np.concatenate(params_Tru_train,0),
                # save file lists as well
                train_files=train_files,
                test_files=test_files,
                lfi_files=lfi_files,
                sys_files=test_sys_files
                )

    # TODO: make this naming more clever
    outname = config["summary_path"] + config["run_name"] + "_" + patch

    get_unshuffled_summaries(outname, summaries_A_file)