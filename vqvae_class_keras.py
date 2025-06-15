#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 13:40:21 2025

@author: boscolo
"""
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

#### VGG  #####################################################################
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

###############################################################################
# 1. Define the VGG16 feature extractor for perceptual loss
vgg = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3)) ### im. dimension i choose for my implementation
vgg.trainable = False  # Freeze VGG16 layers
feature_extractor = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)


class PerceptualLossClass:
    def __init__(self, layer_name='block3_conv3'):
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
        base_model.trainable = False
        self.feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

    def preprocess(self, images):
        images = images * 255.0
        images = images[..., ::-1]
        mean = tf.constant([103.939, 116.779, 123.68], shape=[1, 1, 1, 3])
        return images - mean

    def __call__(self, real, recon):
        real = self.preprocess(real)
        recon = self.preprocess(recon)
        real_features = self.feature_extractor(real)
        recon_features = self.feature_extractor(recon)
        return tf.reduce_mean(tf.square(real_features - recon_features))


################################################################################
def show_reconstructions(x_recon, x_orig, epoch, path, max_images=5):
    """
    Save a side-by-side comparison of original and reconstructed images.

    Args:
        x_recon (np.array or tf.Tensor): Reconstructed images.
        x_orig (np.array or tf.Tensor): Original images.
        epoch (int): Current epoch (for filename).
        path (str): Directory to save the image.
        max_images (int): Number of images to show.
    """
    if isinstance(x_recon, tf.Tensor):
        x_recon = x_recon.numpy()
    if isinstance(x_orig, tf.Tensor):
        x_orig = x_orig.numpy()

    os.makedirs(path, exist_ok=True)
    
    num_batch = min(max_images, len(x_recon))
    fig, axs = plt.subplots(2, num_batch, figsize=(num_batch * 2, 4))
    ### after selecting 5 random batch i get the first image for evry batch, this is just for test 
    for i in range(num_batch):
        axs[0, i].imshow(x_orig[i][0])
        axs[0, i].axis('off')
        if i == 0:
            axs[0, i].set_title("Original")

        axs[1, i].imshow(x_recon[i][0])
        axs[1, i].axis('off')
        if i == 0:
            axs[1, i].set_title("Reconstructed")

    plt.tight_layout()
    save_file = os.path.join(path, f"original_vs_reconstructions_epoch_{epoch:04d}.png")
    plt.savefig(save_file)
    plt.close(fig)

###############################################################################

class Encoder(tf.keras.Model):
    def __init__(self, IMAGE_H, IMAGE_W, latent_dim, encoder_filters, embedding_dim,**kwargs):
        super(Encoder, self).__init__(**kwargs)
        # Save the parameters as attributes
        self.IMAGE_H = IMAGE_H
        self.IMAGE_W = IMAGE_W
        self.latent_dim = latent_dim
        self.encoder_filters = encoder_filters
        self.embedding_dim = embedding_dim
        
        self.conv_layers = []
        for filters in encoder_filters:
              block = tf.keras.Sequential([
                  layers.Conv2D(filters, kernel_size=3, strides=2, padding='same', use_bias=False),
                  layers.BatchNormalization(),
                  layers.ReLU()
              ])
              self.conv_layers.append(block)


        self.conv3 = tf.keras.layers.Conv2D(embedding_dim, 1, strides=1, padding='same')
    def call(self, inputs, training=False):
        x = inputs # (batch, 48,48,3)
        #breakpoint()
        for layer in self.conv_layers:
            x = layer(x,  training=training) 
        #x = self.conv2(x)
        x = self.conv3(x)
        #if training:
        x = tf.nn.l2_normalize(x, axis=-1)
        return x
    
    def get_config(self):
        return {
            "IMAGE_H": self.IMAGE_H,
            "IMAGE_W": self.IMAGE_W,
            "latent_dim": self.latent_dim,
            "encoder_filters": self.encoder_filters,
            "embedding_dim": self.embedding_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
#########


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.block = tf.keras.Sequential([
            layers.Conv2D(filters, 3, padding='same'),
            layers.ReLU(),
            layers.Conv2D(filters, 3, padding='same')
        ])
        self.activation = layers.ReLU()

    def call(self, x):
        return self.activation(x + self.block(x))

class Decoder(tf.keras.Model):
    def __init__(self,IMAGE_H, IMAGE_W, latent_dim, decoder_filters,**kwargs):
        super(Decoder, self).__init__(**kwargs)
        
        # Save the parameters as attributes
        self.IMAGE_H = IMAGE_H
        self.IMAGE_W = IMAGE_W
        self.latent_dim = latent_dim
        self.decoder_filters = decoder_filters


        
        # self.decoder1 = tf.keras.Sequential([
        #  tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', use_bias=False),
        #  tf.keras.layers.BatchNormalization(),
        #  tf.keras.layers.ReLU()
        #  ])
        self.deconv_layers = []
        self.residual_blocks = []
        for filters in decoder_filters:
            block = tf.keras.Sequential([
                layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU()
            ])
            self.deconv_layers.append(block)
            self.residual_blocks.append(ResidualBlock(filters)) ### not very useful in case of shallow decoder ! may case is shallow


         # Further processing after upsampling
        # Keeps spatial size, changes channel count > mybe not necessary !!
        self.deconv2 =  tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=1,  padding='same', activation='relu', use_bias=False)
        #Final layer: output RGB image with 3 channels
        self.deconv3 = tf.keras.layers.Conv2DTranspose(3, 3, strides=1, padding='same', activation='sigmoid')

    def call(self,  inputs, training=False):
     
        x = inputs
        #x = self.decoder1(x,training=training)
        #for layer in self.deconv_layers: # #  →  # Each layer doubles spatial dims:  4x4 → 8x8 → 16x16 → 32x32
        for layer , res_block in zip(self.deconv_layers, self.residual_blocks):
            x = layer(x,training=training)
            x = res_block(x)  # Apply residual refinement
            
            
        x = self.deconv2(x,training=training)   # → # e.g., [B, 32, 32, F] → [B, 32, 32, 32]
        x = self.deconv3(x)   # → [64, 64, 3]
        return x
    
    def get_config(self):
        return {
            "IMAGE_H": self.IMAGE_H,
            "IMAGE_W": self.IMAGE_W,
            "latent_dim": self.latent_dim,
            "decoder_filters": self.decoder_filters,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

############
class VectorQuantizer(tf.keras.Model):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, epsilon=1e-5, **kwargs):
        super().__init__()
        
        
        self.embedding_dim = embedding_dim # # Each embedding vector has shape [embedding_dim]
        self.num_embeddings = num_embeddings  # Number of embeddings in the codebook (K)
        self.decay = decay  # EMA decay factor
        self.epsilon = epsilon #@ this is a small constant
        
        
        # Codebook matrix: shape [num_embeddings, embedding_dim] == [K, D]
        self.embeddings = self.add_weight(
        name="vq_embeddings",
        shape=(self.num_embeddings, self.embedding_dim),
        initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
        trainable=False ## switch to true if ema not used!
        )
        # Track how often each code is used
        self.code_usage = self.add_weight(
            name="code_usage",
            shape=(self.num_embeddings,),
            initializer="zeros",
            trainable=False
        )
        
        # EMA cluster counts (how often a code is selected): shape [K]
        self.ema_cluster_size = self.add_weight(
            name="ema_cluster_size",
            shape=(self.num_embeddings,),
            initializer="zeros",
            trainable=False
        )
        # EMA weighted sum of inputs per codebook index: shape [K, D]
        self.ema_dw = self.add_weight(
            name="ema_dw",
            shape=(self.num_embeddings, self.embedding_dim),
            initializer="zeros",
            trainable=False)

    def call(self, inputs,  training=False): ### input after convolution 
        # inputs: [batch, ..., embedding_dim]
        x = inputs["x"]  # input tensor shaped  # shape [B, H, W, D] batch, rows, col, Dim after conv layers
        current_cost = inputs["current_cost"]  # scalar tensor, this is commitment
        
        input_shape = tf.shape(x)
        
        # SPATIAL POSITION flat_inputs: [N, D] N = B*H*W   > GET A TENSOR OF SPATIAAL POSITION  (H×W per sample) CONTANING A VECTOR SIZE D
        flat_inputs = tf.reshape(x, [-1, self.embedding_dim])  
        # Compute squared distances to codebook vectors
        # Normalize codebook vectors  [K, D]
        normalized_embeddings = tf.nn.l2_normalize(self.embeddings, axis=1)
        
        ######### DISTANCE CALC ################################################
        # Compute L2 squared distances between each input vector and each embedding:
            # ||z - e||^2 = ||z||^2 - 2 z·e + ||e||^2
            
        emb_sq = tf.reduce_sum(normalized_embeddings ** 2, axis=1)  # shape [K] > SQUARE OF EMBEDDING 
        # RESHAPE IN row vector [1, K]
        emb_sq_row = tf.reshape(emb_sq, [1, self.num_embeddings])  # [1, K]
        
        #### SHAPEE : distances=[N,1]−2[N,K]+[1,K]⇒[N,K] > BROADCASTING !!!
        distances = (
            tf.reduce_sum(flat_inputs ** 2, axis=1, keepdims=True)   # [N, 1] > SQUARE OF INPUT AFTER FLAT ||z||^2

            - 2 * tf.matmul(flat_inputs, normalized_embeddings, transpose_b=True)  # [N, D]*[K,D]T = [N,K] () K IS THE NUMBER OF EMVEDDING
            + emb_sq_row #[1,K]
        )
        #######################################################################
        
      
        # NOT GET  nearest code index for each input vector: shape [N]
        encoding_indices = tf.argmin(distances, axis=1)  # [N] 
        

        unique_codes = tf.shape(tf.unique(encoding_indices).y)[0] ### number of unique code index used !
       
        # One-hot encodings: one or zero according to encoding indices -- shape [N, K]
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)

        # Reshape encodings to match input spatial dims: [..., K]
        spatial_dims = input_shape[:-1]
        encodings_3d = tf.reshape(encodings, tf.concat([spatial_dims, [self.num_embeddings]], axis=0))### [batck, H,W, n_embeddning]

        # Quantize: look up embeddings by indices
        quantized_flat = tf.matmul(encodings, normalized_embeddings)  # [N,K]*[K,D] = [N,D] TEKING ONLY EMB. VECTOR WHERE 1 IN ONE HOT MATRIX
        #quantized_flat = tf.matmul(encodings, self.embeddings)  # [N, D]
        quantized = tf.reshape(quantized_flat, input_shape)
        
        # EM  EMA updates for the codebook embeddings. AND Update code usage statistics
        if training:
           batch_usage = tf.reduce_sum(encodings, axis=0)
           self.code_usage.assign_add(batch_usage)
            

            #number of encoded vectors assigned to embedding K
           cluster_size = tf.reduce_sum(encodings, axis=0)  #  K > (num_embeddings,) 
           ## running count of how often each codebook vector is chosen (Count how many times each code was used)
           self.ema_cluster_size.assign(
               self.decay * self.ema_cluster_size + (1 - self.decay) * cluster_size
           )
           
           # sums of inputs VECTORS OR ENCODER OUTPUT(FLAT INPUT) assigned to each embedding K
           dw = tf.matmul(encodings, flat_inputs, transpose_a=True)  # [N,K]*[N,D]T = [K,D]  Get sum of encoder outputs assigned to each code
           #sum of inputs assigned to each embedding
           self.ema_dw.assign(
               self.decay * self.ema_dw + (1 - self.decay) * dw #### With decay = 0.99, 99% of the old value is kept !!
           )

           n = tf.reduce_sum(self.ema_cluster_size) #Total EMA usage count
           #rescales cluster sizes to avoid very small values 
           cluster_size_normalized = (
               (self.ema_cluster_size + self.epsilon)
               / (n + self.num_embeddings * self.epsilon) * n
           )
           ##### GOAL: Update each embedding vector with the average of the encoder outputs currently assigned to it !!!
           normalized_embeddings = self.ema_dw / tf.reshape(cluster_size_normalized + self.epsilon, [-1, 1]) ## [K,D]
           self.embeddings.assign(normalized_embeddings)


        
        # Compute losses
        e_loss = tf.reduce_mean(tf.square(tf.stop_gradient(quantized) - x))  # Codebook (EMA) loss
        q_loss = tf.reduce_mean(tf.square(quantized - tf.stop_gradient(x)))  # Commitment loss
        loss = q_loss + current_cost * e_loss  # Total loss with weighting
        
        # forward pass use the quantized values, but during the backward pass,  gradient of quantized is set to gradient of x (identity) !!
        quantized = x + tf.stop_gradient(quantized - x)
        
        # encoding_indices: shape [N], flat vector of codebook indices to use in pixelcnn !!!
        encoding_indices_flat = tf.reshape(encoding_indices, spatial_dims) 
        
        return quantized, loss, unique_codes, encodings_3d,encoding_indices_flat
    
    def get_config(self):
     return {
         "num_embeddings": self.num_embeddings,
         "embedding_dim": self.embedding_dim,
         "decay": self.decay,
         "epsilon": self.epsilon
     }

     @classmethod
     def from_config(cls, config):
         return cls(**config)
    

class ConditionalVariationalAutoencoder(tf.keras.Model):
    def __init__(self,
        IMAGE_H,
        IMAGE_W,
        label_dim,
        latent_dim,
        learning_rate,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        encoder_filters=(32,64, 128),
        decoder_filters=(64, 32),
        checkpoint_dir='checkpoint' ,
        save_images_dir = 'saved_images'

        ):
        super(ConditionalVariationalAutoencoder, self).__init__()
        
        self.IMAGE_H = IMAGE_H
        self.IMAGE_W = IMAGE_W
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        
        commitment_cost = tf.convert_to_tensor( commitment_cost, dtype=tf.float32)
        self.commitment_cost = tf.Variable(commitment_cost, trainable=False, dtype=tf.float32)
        #self.commitment_cost = commitment_cost
        self.num_embeddings   = num_embeddings
        self.embedding_dim  =embedding_dim
        # Build models
        self.encoder = Encoder(IMAGE_H, IMAGE_W, latent_dim, encoder_filters,embedding_dim)
        self.decoder = Decoder(IMAGE_H, IMAGE_W, latent_dim, decoder_filters)
        self.quantizer = VectorQuantizer( num_embeddings, embedding_dim)

        self.checkpoint_dir = checkpoint_dir 
        self.save_images_dir = save_images_dir
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")
        self.orth_loss_tracker = tf.keras.metrics.Mean(name="orth_loss")
        self.active_codes_tracker = tf.keras.metrics.Mean(name="active_codes")
        self.perc_loss_tracker = tf.keras.metrics.Mean(name="perc_loss")
        
        #self.revival_counter = 0
        
        # Initialize optimizer AFTER model components
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # Initialize checkpoint management in __init__
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            encoder=self.encoder,
            quantizer=self.quantizer,
            decoder=self.decoder,
            code_usage=self.quantizer.code_usage,
            commitment_cost=self.commitment_cost,
            #data_variance=self.data_variance
            #code_usage=self.code_usage
        )

        self.manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir, max_to_keep=1)
        
        ### perc loss
        self.perc_loss_fun= PerceptualLossClass()
        
        
    ### save and load 
    def save_h5_models(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.encoder.save(os.path.join(save_dir, 'encoder.keras'))
        #self.quantizer.save(os.path.join(save_dir, 'quantizer.h5'))
        
        self.quantizer.save(os.path.join(save_dir, 'quantizer.keras'))
        
        self.decoder.save(os.path.join(save_dir, 'decoder.keras'))
    # Save data_variance as numpy
    #np.save(os.path.join(save_dir, 'data_variance.npy'), self.data_variance.numpy() if isinstance(self.data_variance, tf.Variable) else self.data_variance)
    
    def load_h5_models(self, load_dir):
        if not os.path.exists(load_dir):
            print(f"Load directory {load_dir} does not exist.")
            return False
    

        try:
           # Register custom objects
           custom_objects = {
               "Encoder": Encoder,
               "VectorQuantizer": VectorQuantizer,
               "Decoder": Decoder
           }
           
           self.encoder = tf.keras.models.load_model(
               os.path.join(load_dir, 'encoder.keras'),
               custom_objects=custom_objects
           )
           self.quantizer = tf.keras.models.load_model(
               os.path.join(load_dir, 'quantizer.keras'),
               custom_objects=custom_objects
           )
           self.decoder = tf.keras.models.load_model(
               os.path.join(load_dir, 'decoder.keras'),
               custom_objects=custom_objects
           )
           
           print(f"Models successfully loaded from {load_dir}")
           return True
        except Exception as e:
           print(f"Error loading models: {e}")
           return False
    
    
    ###########################################################################
    @property
    def metrics(self):
        # list my useful `Metric`
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.vq_loss_tracker,
            self.orth_loss_tracker,
            self.perc_loss_tracker,
            self.active_codes_tracker,
            
        ]
    ###########################################################################

    def train_step(self, data):
        #batch_data, batch_labels = data
        batch_data= data
        #batch_labels = tf.cast(batch_labels, tf.float32)
        noisy_batch_data = batch_data + tf.random.normal(tf.shape(batch_data), stddev=0.05)
        
        with tf.GradientTape() as tape:
            # Forward pass
            z = self.encoder(noisy_batch_data, training=True)
            # quantized, vq_loss, _, encodings_3d = self.quantizer(
            #     z, current_cost=self.commitment_cost, training=True) #  {"x": x_tensor, "current_cost": current_cost}
            
            quantized, vq_loss, _, encodings_3d,_ = self.quantizer(
                {"x": z, "current_cost": self.commitment_cost}, training=True)
            
            x_recon = self.decoder(quantized, training=True)
          
            # Loss calculations
            recon_loss = tf.reduce_mean((batch_data - x_recon) ** 2) / self.data_variance
            
            ## orho loss > this is to encourage diversity in codebook selection
            codebook = self.quantizer.embeddings
            dot_products = tf.matmul(codebook, codebook, transpose_b=True)
            norms = tf.norm(codebook, axis=1, keepdims=True)
            cosine_sim = dot_products / (norms @ tf.transpose(norms) + 1e-8)
            orth_loss = tf.reduce_mean(tf.abs(cosine_sim - tf.eye(self.num_embeddings)))
            
            ### perc closs > try to improve a bit the reco using vgg model trained 
            perc_loss = self.perc_loss_fun(batch_data, x_recon) ## i m using the original data not noisy !
            
            total_loss = recon_loss + vq_loss + 0.05 * orth_loss + 0.000005 * perc_loss ## # Perceptual loss weighted low  minimizing possible artifacts introduced by VGG 
        
        # Gradient updates
        variables = (
            self.encoder.trainable_variables +
            self.quantizer.trainable_variables +
            self.decoder.trainable_variables
        )
        grads = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
        
        # Update code usage
        self.quantizer.code_usage.assign_add(tf.reduce_sum(encodings_3d, axis=[0, 1, 2]))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(vq_loss)
        self.orth_loss_tracker.update_state(orth_loss)
        self.perc_loss_tracker.update_state(perc_loss)
        
        # Calculate active codes  -> i m checking embeddings "active" (used more than 80% of the time).
        threshold = 0.8
        active_codes_count = tf.reduce_sum(
            tf.cast(self.quantizer.code_usage > threshold, tf.int32))
        self.active_codes_tracker.update_state(active_codes_count)
        
        # Adjust commitment dynamically
        self._adjust_commitment_cost(active_codes_count)
        
        # Revive dead codes 
        # if self.revival_counter % 5 == 0:
        #     self._revive_dead_codes()
        # self.revival_counter += 1
        
        # Decay code usage
        self.quantizer.code_usage.assign(self.quantizer.code_usage * 0.995)
        
        return {m.name: m.result() for m in self.metrics}

    ###########################################################################

    def test_step(self, data):
        #batch_data, batch_labels = data
        #batch_labels = tf.cast(batch_labels, dtype=tf.float32)
        batch_data= data
        # Forward pass
        z = self.encoder(batch_data, training=False)
        # quantized, vq_loss, _, _ = self.quantizer(
        #     z, current_cost=self.commitment_cost, training=False)
        
        quantized, vq_loss, _, encodings_3d,_ = self.quantizer(
            {"x": z, "current_cost": self.commitment_cost}, training=False)
        
        
        x_recon = self.decoder(quantized, training=False)
        
        # Loss calculations
        recon_loss = tf.reduce_mean((batch_data - x_recon) ** 2) / self.data_variance
        
        ## monito perc loss during val
        #perc_loss = self.perc_loss_fun(batch_data, x_recon)

        
        total_loss = recon_loss + vq_loss# + 0.1 * perc_loss ## 
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(vq_loss)
        #self.perc_loss_tracker.update_state(perc_loss)
        
        threshold = 0.8
        active_codes_count = tf.reduce_sum(
            tf.cast(self.quantizer.code_usage > threshold, tf.int32))
        self.active_codes_tracker.update_state(active_codes_count)
        
        
        return {m.name: m.result() for m in self.metrics}
    
    ###########################################################################

    def _adjust_commitment_cost(self, active_codes_count):
        min_commitment = 0.05
        max_commitment = 0.1
        target_active_codes = 0.5 * self.num_embeddings  # float
    
        active_codes_count = tf.cast(active_codes_count, tf.float32)
    
        def decrease_commitment():
            return tf.maximum(min_commitment, self.commitment_cost * 0.95) #### decay
    
        def increase_commitment():
            return tf.minimum(max_commitment, self.commitment_cost * 1.05)
    
        def slight_decay():
            #return self.commitment_cost * 0.99
            return tf.maximum(min_commitment, self.commitment_cost * 0.99)
    
        # Two conditions implemented 
        condition1 = active_codes_count < (target_active_codes * 0.5) ### decrease the commitment --> free to explore more code
        condition2 = active_codes_count > (target_active_codes * 1.5) ### increase the commitment --> less code explored
    
        new_commitment = tf.cond(condition1, decrease_commitment,
                            lambda: tf.cond(condition2, increase_commitment, slight_decay))
    
        self.commitment_cost.assign(new_commitment)
        
    ###########################################################################
    
    def _revive_dead_codes(self):
        threshold = 1.0
        max_noise_scale = 0.2
        epoch = getattr(self, "current_epoch", 0)
        noise_scale = max_noise_scale * max(0.1, (1.0 - epoch / 200.0))
    
        dead_mask = self.quantizer.code_usage < threshold ## code is marked as dead when its usage count falls below the threshold
        dead_indices = tf.where(dead_mask)
        num_dead = tf.shape(dead_indices)[0]
    
        def revive():
            active_codes = tf.boolean_mask(
                self.quantizer.embeddings, tf.logical_not(dead_mask))
            num_active = tf.shape(active_codes)[0]
    
            def handle_all_dead():
                new_embeddings = tf.random.normal(
                    shape=(self.num_embeddings, self.embedding_dim), stddev=0.1)
                self.quantizer.embeddings.assign(new_embeddings)
                return
    
            def revive_some():
                max_revivals_per_batch = 5
                num_to_replace = tf.minimum(
                    tf.minimum(num_dead, num_active), max_revivals_per_batch)
    
                sliced_dead_indices = dead_indices[:num_to_replace]
                replacements = tf.random.shuffle(active_codes)[:num_to_replace]
                replacements += tf.random.normal(
                    shape=tf.shape(replacements), stddev=noise_scale)
    
                updated_embeddings = tf.tensor_scatter_nd_update(
                    self.quantizer.embeddings, sliced_dead_indices, replacements)
                self.quantizer.embeddings.assign(updated_embeddings)
    
                usage_updates = tf.ones([num_to_replace], dtype=tf.float32) * threshold
                updated_usage = tf.tensor_scatter_nd_update(
                    self.quantizer.code_usage, sliced_dead_indices, usage_updates)
                self.quantizer.code_usage.assign(updated_usage)
    
            return tf.cond(num_active > 0, revive_some, handle_all_dead)
    
        tf.cond(num_dead > 0, revive, lambda: None)

    #### custom predict called outside ########################################
    def _predict(self, data):
            ## Generate reconstructions 
            #### input is an array from otside in thsi case ###################
            ## i m selecting 5 random indeces
            #random_indices = np.random.choice(len(data), size=5, replace=False)
            ### loop
            x_recon = []
            for idx in range(len(data)):
                data_item = data[idx]
                data_item_expanded = np.expand_dims(data_item, axis=0)  # need to add 
                z = self.encoder(data_item_expanded , training=False)
                
                # quantized, _, encoding_indices, _ = self.quantizer(
                #     z, current_cost=self.commitment_cost, training=False)
                
                quantized, _, _, _,_= self.quantizer(
                    {"x": z, "current_cost": self.commitment_cost}, training=False)
                
                
                x_recon_ = self.decoder(quantized, training=False)
                x_recon.append(x_recon_.numpy())
            
            return x_recon
        
    def _inference(self, batch_data):
            #Loading  checkpoint if exist !
            #loaded = self.load_checkpoint(self.checkpoint_dir)
            loaded = self.load_h5_models(self.checkpoint_dir)
        
            if loaded:
                x_recon = self._predict(batch_data)
                return x_recon
                print('Image Generated !!!')
            else:
                print('No checkpoint found - cannot perform inference')
                return None
    ########## VGG ###########################################################
    
    def prepare_for_vgg(self, images):
        images = images * 255.0  # Rescale to [0, 255] > VGG16 expects input images in the [0, 255
        images = images[..., ::-1]  # RGB to BGR
        mean = tf.constant([103.939, 116.779, 123.68], shape=[1, 1, 1, 3]) ## vgg trained with this mean subtracted input
        return images - mean  # Mean subtraction for VGG
    
    def perceptual_loss(y_true, y_pred):
        vgg = tf.keras.applications.VGG16(include_top=False, input_shape=(60, 60, 3), weights='imagenet')
        vgg.trainable = False
        features_true = vgg(y_true)
        features_pred = vgg(y_pred)
        return tf.reduce_mean(tf.square(features_true - features_pred))
    
    ###########################################################################
    def fit(self, 
                data,
                data_labels, 
                data_val=None,
                data_labels_val=None,
                data_variance=20,
                epochs=100,
                batch_size=16,
                patience=15,
                early_stopping_interval=10,
                min_epoch_to_start_saving=10, ## i m skiping the first 10 epochs
                **kwargs):
            
        
           #### custom fit() where i prepares the data and callbacks ##########
            # Initialize training variables 
            self.data_variance = data_variance
            self.revival_counter = 0
            self.best_loss = float('inf')
            self.wait = 0
            #self.epochs = epochs
            # Prepare datasets
            train_dataset = tf.data.Dataset.from_tensor_slices(data)#tf.data.Dataset.from_tensor_slices((data, data_labels))
            train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
            
            if data_val is not None:
                val_dataset = tf.data.Dataset.from_tensor_slices(data_val)#tf.data.Dataset.from_tensor_slices((data_val, data_labels_val))
                val_dataset = val_dataset.batch(batch_size)
            
            
            # Load checkpoint if available
            #loaded_chk=self.load_checkpoint(self.checkpoint_dir)
            loaded_chk = self.load_h5_models(self.checkpoint_dir)

            # Custom callback for validation and checkpoint save
            callbacks = [
                VQVAECallback(
                    self,
                    val_dataset=val_dataset if data_val is not None else None,
                    early_stopping_interval=early_stopping_interval,
                    patience=patience,
                    min_epoch_to_start_saving=min_epoch_to_start_saving,
                    checkpoint_dir=self.checkpoint_dir,
                    save_images_dir = self.save_images_dir,
                    checkpoint_loaded = loaded_chk,
                    epochs = epochs
                    
                )
            ]
            
            ########### ..AFTER.. #######
            
            # i m calling the paren class tf.keras.Model
            history = super().fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset if data_val is not None else None,
                callbacks=callbacks,
                #batch_size=batch_size, > batch sise is ignore because defined in datsset tensor_slices
                **kwargs
            )
            
            return history
        
    def call(self, inputs, training=False):
        z = self.encoder(inputs, training=training)
        # quantized, _, _, _ = self.quantizer(z, current_cost=self.commitment_cost, training=training)
        
        quantized, _ ,_, _ ,_= self.quantizer(
             {"x": z, "current_cost": self.commitment_cost}, training=training)
        
        return self.decoder(quantized, training=training)

###############################################################################

class VQVAECallback(tf.keras.callbacks.Callback):
    def __init__(self, 
                 vqvae_model, 
                 val_dataset=None, 
                 early_stopping_interval=10,
                 patience=15,
                 min_epoch_to_start_saving=25,
                 checkpoint_dir='checkpoint',
                 save_images_dir = 'saved_images',
                 checkpoint_loaded = False,
                 epochs = 1):
        super().__init__()
        self.vqvae_model = vqvae_model 
        self.val_dataset = val_dataset
        self.early_stopping_interval = early_stopping_interval
        self.patience = patience
        self.min_epoch_to_start_saving = min_epoch_to_start_saving
        self.checkpoint_dir = checkpoint_dir
        self.save_images_dir = save_images_dir 
        self.best_loss = float('inf')
        self.wait = 0
        self.epoch_count = 0
        self.initial_checkpoint_loaded = checkpoint_loaded 
        self.epochs = epochs
    
    def on_train_begin(self, logs=None): #  Called at the beginning of training
         # Initialize training state
         print("\nTraining starting...")
         if self.initial_checkpoint_loaded:
             print("Resuming from checkpoint - revival cooldown active for first xxxx batches")
         else:
             print("Starting new training run")
    
    ### updating learning rate !
    def on_epoch_begin(self, epoch, logs=None): # Called at the start of epoch
        self.model.current_epoch = epoch
        #Adjusting learning rate 
        self.epoch_count += 1
        if epoch > 0 and epoch % 20 == 0:
            min_learning_rate = 1e-5
            new_lr = max(min_learning_rate, 
                         self.vqvae_model.optimizer.learning_rate * (1 - epoch /  self.epochs )) 
            self.vqvae_model.optimizer.learning_rate.assign(new_lr)
            print(f"\nEpoch {epoch + 1}: Learning rate adjusted to {new_lr:.8f}")

    def on_train_batch_end(self, batch, logs=None): ### Called after batch
         if batch % 10 == 0:
             #print('Revived Dead Code Applied!')
             self.vqvae_model._revive_dead_codes()
    
    def on_epoch_end(self, epoch, logs=None): # Called at the end of each epoch
        # Print additional metrics
        print(f"\nEpoch {epoch + 1}:")
        print(f"  Active codes: {logs['active_codes']}/{self.vqvae_model.num_embeddings}")
        print(f"  Commitment cost: {self.vqvae_model.commitment_cost.numpy():.4f}")
        
        # Validation and checkpoint eval !!
        if (self.val_dataset is not None and 
            (epoch + 1) % self.early_stopping_interval == 0 and 
            epoch >= self.min_epoch_to_start_saving):
            
            #val_logs = self.model._evaluate(self.val_dataset)
            #val_loss = val_logs['total_loss']
            val_loss = logs['val_recon_loss'] ### i want to priorize the recostruction loss  not val_total_loss recon_loss
         
            if val_loss < self.best_loss:
                print(f"  Validation loss improved from {self.best_loss:.4f} to {val_loss:.4f}")
                self.best_loss = val_loss
                self.wait = 0
                #self.vqvae_model.manager.save() hgrhrt
                #breakpoint()
                self.vqvae_model.save_h5_models(self.checkpoint_dir)
                print(f"  Model saved to {self.checkpoint_dir}")
                
                ###### visualization during training
                dataset_list = list(self.val_dataset) 
                indices = tf.random.shuffle(tf.range(len(dataset_list)))[:5]
                sampled_data = [dataset_list[i.numpy()] for i in indices]

                x_recon = []
                for batch_data in sampled_data:
                    z = self.vqvae_model.encoder(batch_data, training=False)

                    
                    quantized, _, _, encodings_3d,_ = self.vqvae_model.quantizer(
                        {"x": z, "current_cost": self.vqvae_model.commitment_cost}, training=False)

                    
                    recon = self.vqvae_model.decoder(quantized, training=False)
                    x_recon.append(recon)
              
                path = self.save_images_dir
                show_reconstructions(x_recon,sampled_data,epoch,  path)
                
                
            else:
                self.wait += 1
                print(f"  No improvement in validation loss for {self.wait}/{self.patience} epochs")
                print(f"  Validation loss not improved from {self.best_loss:.4f} to {val_loss:.4f}")
                if self.wait >= self.patience:
                    self.model.stop_training = True
                    print("  Early stopping triggered")
    
    def on_train_end(self, logs=None): # calling training end
        ### ENDING !!!!!
        print("\nTraining completed!")
        if self.wait >= self.patience:
            print(f"Early stopped after {self.epoch_count} epochs")
        else:
            print(f"Completed all {self.epoch_count} epochs")


 