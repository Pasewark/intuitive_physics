# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
import ThreedUnet
from torchvision import transforms as ttransforms
import itertools

FLAGS = flags.FLAGS

_NUM_PROBE_TFRECORDS = 20
_NUM_FREEFORM_TRAIN_TFRECORDS = 100
_NUM_FREEFORM_TEST_TFRECORDS = 10

_FREEFORM_FEATURES = dict(
     image=tf.io.FixedLenFeature(dtype=tf.string, shape=()),                                                                                                                                             
     mask=tf.io.FixedLenFeature(dtype=tf.string, shape=()),                                                                                                                                              
     camera_pose=tf.io.FixedLenFeature(dtype=tf.float32, shape=(15, 6)),                                                                                                                              
)

_PROBE_FEATURES = dict(                                                                                                                                                                                                
    possible_image=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    possible_mask=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    possible_camera_pose=tf.io.FixedLenFeature(dtype=tf.float32,
                                               shape=(2, 15, 6)),
    impossible_image=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    impossible_mask=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    impossible_camera_pose=tf.io.FixedLenFeature(dtype=tf.float32,
                                                 shape=(2, 15, 6)),                                                                                                                            
)

def _parse_freeform_row(row):                                                                                                                                                                                             
  row = tf.io.parse_example(row, _FREEFORM_FEATURES)                                                                                                                                                                      
  row['image'] = tf.reshape(tf.io.decode_raw(row['image'], tf.uint8),
                            [15, 64, 64, 3])                                                                                                  

  row['mask'] = tf.reshape(tf.io.decode_raw(row['mask'], tf.uint8),
                           [15, 64, 64])                                                                                                       
  return row                                                                                                                                                                                                     
                                                                                                                                                                                                                 
def _parse_probe_row(row):                                                                                                                                                                                             
  row = tf.io.parse_example(row, _PROBE_FEATURES)                                                                                                                                                                      
  for prefix in ['possible', 'impossible']:                                                                                                                                                                      
    row[f'{prefix}_image'] = tf.reshape(
        tf.io.decode_raw(row[f'{prefix}_image'], tf.uint8),
        [2, 15, 64, 64, 3])                                                                                                  
    row[f'{prefix}_mask'] = tf.reshape(
        tf.io.decode_raw(row[f'{prefix}_mask'], tf.uint8),
        [2, 15, 64, 64])                                                                                                       
  return row                                                                                                                                                                                                     


def _make_tfrecord_paths(dir_name, subdir_name, num_records):
  root = f'gs://physical_concepts/{dir_name}/{subdir_name}/data.tfrecord'
  paths = [f'{root}-{i:05}-of-{num_records:05}' for i in range(num_records)]
  return paths

def make_freeform_tfrecord_dataset(is_train, shuffle=False):
  """Returns a TFRecordDataset for freeform data."""
  if is_train:
    subdir_str = 'train'
    num_records = _NUM_FREEFORM_TRAIN_TFRECORDS
  else:
    subdir_str = 'test'
    num_records = _NUM_FREEFORM_TEST_TFRECORDS

  tfrecord_paths = _make_tfrecord_paths('freeform', subdir_str, num_records)
  ds = tf.data.TFRecordDataset(tfrecord_paths, compression_type='GZIP')
  ds = ds.map(_parse_freeform_row)                        
  if shuffle:
    ds = ds.shuffle(buffer_size=50)                                                                                                                                                                
  return ds

def make_probe_tfrecord_dataset(concept_name, shuffle=False):
  """Returns a TFRecordDataset for probes data."""
  tfrecord_paths = _make_tfrecord_paths('probes', concept_name, 20)
  ds = tf.data.TFRecordDataset(tfrecord_paths, compression_type='GZIP')
  ds = ds.map(_parse_probe_row)
  if shuffle:
    ds = ds.shuffle(buffer_size=20)                                                                                                                                                                                         
  return ds

def concat_frames_horizontally(v):
  """Arrange a video as horizontally aligned frames."""
  num_frames = v.shape[0]
  # [F, H, W, C] --> [H, W*F, C].
  return np.concatenate([v[x] for x in range(num_frames)], axis=1)

def describe(d):
  """Describe the contents of a dict of np arrays."""
  for k, v in d.items():
    print(f'\'{k}\' has shape: {v.shape}')
    print(f'===================')
    print(f'min: {v.min()}, max: {v.max()}, type: {v.dtype}\n')

def colorize_mask(m):
  """Adds color channel to mask of unique object ids."""
  m = m[..., np.newaxis]
  min_val = np.max(m)
  max_val = np.min(m)
  # Use three different mappings into range [0-1] to form color.
  c1 = (m - min_val)/(max_val - min_val)
  c2 = np.abs((m - max_val)/(min_val - max_val))
  c3 = (c1+c2)/2.
  mask = np.concatenate([c1, c2, c3], axis=-1)
  return mask

def plot_video(v, name=''):
  """Plots something of the form [num_frames, height, width, channel]."""
  num_frames = v.shape[0]
  width = v.shape[2]
  v = concat_frames_horizontally(v)
  plt.figure(figsize=(30,5))
  plt.imshow(v)
  plt.xticks(ticks=[i*width+width/2 for i in range(num_frames)],
            labels=range(1,num_frames+1))
  plt.yticks([])
  plt.xlabel('Frame Number')
  plt.title(name)
  plt.show()

class MyDataset(torch.utils.data.IterableDataset):
    def __init__(self, data,batch_size,size):
        super(MyDataset).__init__()

        self.data=data
        transforms = [ttransforms.Resize((size, size))]
        frame_transform = ttransforms.Compose(transforms)
        self.frame_transform = frame_transform
        self.batch_size=batch_size

    def __iter__(self):
        while True:
            videos=[]
            for _ in range(self.batch_size):
                vid=next(self.data)['image'].astype(float)/255
                videos.append(self.frame_transform(torch.from_numpy(vid).permute(3,0,1,2)).type(torch.float32))
            yield torch.stack(videos,0)


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  #score_model = mutils.create_model(config)
  Image_size=32
  score_model= ThreedUnet.Unet3D(
    dim = Image_size,
    dim_mults = (1, 2,4,8)
  ).cuda()
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  #config.optim.lr=config.optim.lr/10
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  #train_ds, eval_ds, _ = datasets.get_dataset(config,
  #                                            uniform_dequantization=config.data.uniform_dequantization)
  #train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  #eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    
  train_ds = make_freeform_tfrecord_dataset(is_train=True, shuffle=True)
  torch_dataset=MyDataset(train_ds.as_numpy_iterator(),batch_size=8,size=Image_size)
  dataloader=torch.utils.data.DataLoader(torch_dataset,batch_size=1)
  train_iter=iter(dataloader)
  #train_iter = train_ds.as_numpy_iterator()
  #train_iter=train_ds.batch(8).as_numpy_iterator()
  eval_iter=train_ds.as_numpy_iterator()
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))

  #for step in range(initial_step, num_train_steps + 1):
  for epoch in range(100):
    losses_arr=[]
    print(epoch)
    train_ds = make_freeform_tfrecord_dataset(is_train=True, shuffle=True)
    torch_dataset=MyDataset(train_ds.as_numpy_iterator(),batch_size=32,size=Image_size)
    for step,data in enumerate(torch_dataset):
      # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
      #batch=next(train_iter)['image'].astype(float)/255
      #batch = torch.from_numpy(batch).to(config.device).float()
      #batch = batch.permute(0,4,1,2,3)
      batch=data.to(config.device).squeeze(0)
      batch = scaler(batch)
      if step==0:print('batch shape:',batch.shape,batch.dtype)
      # Execute one training step
      loss = train_step_fn(state, batch)
      losses_arr.append(loss.item())
      if step%50==0:
        print(step,'loss',np.mean(losses_arr))
        losses_arr=[]



def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Build data pipeline
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting)


  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                      uniform_dequantization=True, evaluation=True)
  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd
    bpd_num_repeats = 1
  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    bpd_num_repeats = 5
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
    while not tf.io.gfile.exists(ckpt_filename):
      if not waiting_message_printed:
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        waiting_message_printed = True
      time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      time.sleep(60)
      try:
        state = restore_checkpoint(ckpt_path, state, device=config.device)
      except:
        time.sleep(120)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      all_losses = []
      eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
      for i, batch in enumerate(eval_iter):
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        eval_loss = eval_step(state, eval_batch)
        all_losses.append(eval_loss.item())
        if (i + 1) % 1000 == 0:
          logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = np.asarray(all_losses)
      with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      for repeat in range(bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for batch_id in range(len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
          eval_batch = eval_batch.permute(0, 3, 1, 2)
          eval_batch = scaler(eval_batch)
          bpd = likelihood_fn(score_model, eval_batch)[0]
          bpd = bpd.detach().cpu().numpy().reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                              f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                 "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
      for r in range(num_sampling_rounds):
        logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

        # Directory to save samples. Different for each host to avoid writing conflicts
        this_sample_dir = os.path.join(
          eval_dir, f"ckpt_{ckpt}")
        tf.io.gfile.makedirs(this_sample_dir)
        samples, n = sampling_fn(score_model)
        samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
          (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())

        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()
        latents = evaluation.run_inception_distributed(samples, inception_model,
                                                       inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        gc.collect()
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(
            io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
          fout.write(io_buffer.getvalue())

      # Compute inception scores, FIDs and KIDs.
      # Load all statistics that have been previously computed and saved for each host
      all_logits = []
      all_pools = []
      this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
      stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
      for stat_file in stats:
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
          stat = np.load(fin)
          if not inceptionv3:
            all_logits.append(stat["logits"])
          all_pools.append(stat["pool_3"])

      if not inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
      all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

      # Load pre-computed dataset statistics.
      data_stats = evaluation.load_dataset_stats(config)
      data_pools = data_stats["pool_3"]

      # Compute FID/KID/IS on all samples together.
      if not inceptionv3:
        inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
      else:
        inception_score = -1

      fid = tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, all_pools)
      # Hack to get tfgan KID work for eager execution.
      tf_data_pools = tf.convert_to_tensor(data_pools)
      tf_all_pools = tf.convert_to_tensor(all_pools)
      kid = tfgan.eval.kernel_classifier_distance_from_activations(
        tf_data_pools, tf_all_pools).numpy()
      del tf_data_pools, tf_all_pools

      logging.info(
        "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
          ckpt, inception_score, fid, kid))

      with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                             "wb") as f:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
        f.write(io_buffer.getvalue())
