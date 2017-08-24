from lib.rvae import *
from lib.vae import *
import numpy as np
import tensorflow as tf
import scipy.io
from lib.utils import *

np.random.seed(0)
tf.set_random_seed(0)
init_logging("experiment_rvae.log")

def load_citeulike_data():
  data = {}
  data_dir = "data/"
  variables = scipy.io.loadmat(data_dir + "mult_nor.mat")
  content = variables['X']
  links = load_links(data_dir + "citations.dat")
  train_links, test_links = data_split(links, 0.8)

  data["content"] = content
  data["train_links"] = train_links
  data["test_links"] = test_links

  return data

def load_links(path):
  links = []
  ind = 0
  for line in open(path):
      arr = line.strip().split()
      arr = [int(x) for x in arr]
      this_num_links = arr[0]
      if this_num_links == 0:
          links.append([])
          ind += 1
          continue
      links.append(arr[1:])
      # print links[ind]
      ind += 1
  return links

def data_split(links, ratio_train=0.8):
  # filtered = [i for i in range(len(links)) if len(links[i])>0]
  filtered = range(len(links))
  num_total = len(filtered)
  num_train = int(num_total*ratio_train)
  num_test = num_total - num_train
  perm_idx = np.random.permutation(filtered)
  train_idx = perm_idx[:num_train]
  test_idx = perm_idx[num_train:]
  
  train_links = [None] * num_total
  for i in range(num_total):
    train_links[i] = []
  num_train_links = 0
  for i in train_idx:
    this_link = links[i]
    # delete test items from training links  
    this_link = [x for x in this_link if x not in test_idx]
    # do the re-mapping
    train_links[i] = this_link
    num_train_links += len(train_links[i])
  num_train_links /= 2
  
  test_links = [None] * num_total
  for i in range(num_total):
    test_links[i] = []
  num_test_links = 0
  for i in test_idx:
    this_link = links[i]
    # keep only links to the training items
    this_link = [x for x in this_link if x in train_idx]
    # this_link = [x for x in this_link if x != i]
    test_links[i] = this_link
    num_test_links += len(test_links[i])

  logging.info("Dataset summary")
  logging.info("#%d training items with #%d training links" % (num_train, num_train_links))
  logging.info("#%d testing items with #%d testing links" % (num_test, num_test_links))
  return (train_links, test_links)

def pretrain(num_factors):
  logging.info('Pretrain')
  logging.info('loading data')
  variables = scipy.io.loadmat("data/mult_nor.mat")
  data = variables['X']
  idx = np.random.rand(data.shape[0]) < 0.8
  train_X = data[idx]
  test_X = data[~idx]
  logging.info('initializing vae model')
  model = VariationalAutoEncoder(input_dim=8000, dims=[200, 100], z_dim=num_factors, 
    activations=['sigmoid','sigmoid'], epoch=[50, 50], 
    noise='mask-0.3' ,loss='cross-entropy', lr=0.01, batch_size=128, print_step=1)
  logging.info('fitting data starts...')
  model.fit(train_X, test_X)
  weight_path = "model-exp/pretrain"

def experiment(num_factors):
  tf.reset_default_graph()
  params = Params()
  params.lambda_w = 1e-4
  params.lambda_v = 0.1
  params.lambda_n = 1
  params.lambda_e = 50
  params.lr = 0.01
  params.batch_size = 128
  params.n_epochs = 100

  logging.info('loading data')
  data = load_citeulike_data()
  logging.info('initializing rvae model')
  num_train = len(data["train_links"])
  model = RVAE(num_items=num_train, num_factors=num_factors, params=params, 
      input_dim=8000, dims=[200, 100], n_z=num_factors, activations=['sigmoid', 'sigmoid'], 
      loss_type='cross-entropy', random_seed=0, print_step=10, verbose=False)
  model.load_model(weight_path="model-exp/pretrain")
  model.run(data["content"], data["train_links"], data["test_links"])
  model.save_model(weight_path="model-exp/rvae", pmf_path="model-exp/pmf")

num_factors_list = [5, 10, 20, 40, 50]
num_repeat = 5
for num_factors in num_factors_list:
  logging.info("Experiment with num_factors=%d" % num_factors)
  pretrain(num_factors)
  for i in range(num_repeat):
    logging.info("Repeat index=%d" % i)
    experiment(num_factors)
