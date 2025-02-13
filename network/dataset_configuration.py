import glob
import os
import re
import tensorflow as tf
import tensorflow_datasets as tfds


# TODO: put in config file
base_path = "/data103/makinen/des_sims/Gower_street_SBI_tfrecords/"
patch = "A"
use_noise_realisations =['0','1','2','3','10', '11', '12']#,'11','12','13','14','15']



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
                
        # else just return the param
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

# -------------------------




# TODO: I THINK THIS CODE SHOULD GO IN THE CALLS TO THE COMPRESSION 
# JUST LIKE WE DO IN THE NOTEBOOKS
# ----- dataset code -----


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
        
    #tfdataset = tfdataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE) 
    #tfdataset = tfdataset.repeat(epochs)
    

    if to_numpy:
        tfdataset = tfdataset.as_numpy_iterator()
        tfdataset.num_batch_per_epoch = num_files // batch_size
        tfdataset.num_samples = (num_files // batch_size) * batch_size


    return tfdataset

# is it better to consolidate the data in one dictionary or just pass a tuple to the network ?

# stack A and B patches
#train_dataset = tf.data.Dataset.zip((train_dataset,train_datasetB))

def consolidate_dsets(A,B,C=None):
    # check that cosmo params are the same    
    if C is not None:
        return {"y": {"A": {"kappa": A["y"]["kappa"], "cls": A["y"]["cls"]},
                        "B": {"kappa": B["y"]["kappa"], "cls": B["y"]["cls"]},
                        "C": {"kappa": C["y"]["kappa"], "cls": C["y"]["cls"]}
                        },
                "theta": A["theta"], 
                "theta_A": A["theta"], 
                "theta_B":B["theta"],  
                "theta_C":C["theta"],
        }
    else:
        return {"y": {"A": {"summaries": A[0], 
                            "theta_A": A[1]
                            },
                "B": {"kappa": B["y"]["kappa"], 
                    "cls": B["y"]["cls"]},
                },
            "theta":B["theta"],  
                }

# train_dataset = train_dataset.map(lambda A,B: consolidate_dsets(A,B))

# LOAD IN SUMMARIES FROM PATCH A, ORDERED ACCORDINGLY



def stack_tfdatasets(summaries_A, params_A,
                        files_B, files_C=None, 
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

    if files_C is not None:
        tfdataset_C = get_tfdataset(files_C, batch_size=batch_size, scale_params=True, 
                            to_numpy=False, shuffle=False, 
                            shuffle_buffer_size=shuffle_buffer_size, 
                            drop_remainder=drop_remainder)
        # zip all three
        tfdataset =  tf.data.Dataset.zip((tfdataset_A,tfdataset_B,tfdataset_C))
        tfdataset = tfdataset.map(lambda A,B,C: consolidate_dsets(A,B,C), num_parallel_calls=tf.data.AUTOTUNE,)
        
    else:
        # compress and cache dataset A
        #tfdataset_A = tfdataset_A.map(lambda d: compress_patch_A(d), num_parallel_calls=tf.data.AUTOTUNE).cache()
        #tfdataset_A = tfdataset_A.
        
        # just zip two datasets
        tfdataset =  tf.data.Dataset.zip((tfdataset_A,tfdataset_B))

        # do we need interleave here again ?
        # tfdataset = tfdataset.interleave(
        #         tf.data.TFRecordDataset,
        #         cycle_length=n_readers,
        #         block_length=1,
        #         num_parallel_calls=tf.data.AUTOTUNE,
        #         deterministic=True,
        #     )
        
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