# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 10:00:49 2025

@author: Enrico
"""

from pathlib import Path
import pandas as pd
import os 
import numpy as np
import imageio               # For reading image files
from PIL import Image         # For resizing images
import logging
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import json
import vqvae_class_keras
from sklearn.model_selection import train_test_split
from PIL import ImageEnhance, ImageFilter 
import random


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

###############################################################################
# GPU distribution setup ######################################################
def setup_strategy():
    gpus = tf.config.list_physical_devices('GPU')  # Avoid using the deprecated experimental API

    if gpus:
        logger.info(f"Detected {len(gpus)} GPU(s). Setting up distribution strategy.")
        try:
            
            # tf.config.set_visible_devices(gpus[0], 'GPU')
            # #tf.config.set_visible_devices([], 'GPU')  # Hide all GPUs
            # logger.info("Forced to use CPU only. All GPUs are hidden from TensorFlow.")
            
            # # Set up the strategy (single GPU in this case)
            # strategy = tf.distribute.get_strategy()  # Single-device strategy
            
            # Only make GPU 1 visible
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            
            # Use OneDeviceStrategy explicitly
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")  # GPU 1 becomes GPU 0 internally
            logger.info("Using GPU 1 (logical GPU:0 after setting visibility).")
            
        except RuntimeError as e:
            logger.error(f"Error configuring GPU: {e}")
            strategy = tf.distribute.get_strategy()  # Fallback to default strategy
    else:
        logger.info("No GPUs detected. Using default strategy.")
        strategy = tf.distribute.get_strategy()  # Single-device strategy
    
    return strategy


###############################################################################

def augment_image(img_pil):
    """Apply multiple attribute-preserving augmentations to a PIL image."""
    aug_imgs = []

    # Horizontal flip
    flipped = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    aug_imgs.append(flipped)

    # Brightness
    enhancer = ImageEnhance.Brightness(img_pil)
    aug_imgs.append(enhancer.enhance(1.2))  # brighter
    aug_imgs.append(enhancer.enhance(0.8))  # darker

    # Contrast
    contrast = ImageEnhance.Contrast(img_pil)
    aug_imgs.append(contrast.enhance(1.3))  # more contrast
    aug_imgs.append(contrast.enhance(0.7))  # less contrast

    # Sharpness
    sharpness = ImageEnhance.Sharpness(img_pil)
    aug_imgs.append(sharpness.enhance(1.5))
    aug_imgs.append(sharpness.enhance(0.5))
    
    # Saturation (Color)
    color = ImageEnhance.Color(img_pil)
    aug_imgs.append(color.enhance(1.2))  # more saturation
    aug_imgs.append(color.enhance(0.8))  # less saturation

    # Slight rotation (±5 degrees)
    for angle in [-5, 5]:
        rotated = img_pil.rotate(angle, resample=Image.BILINEAR)
        aug_imgs.append(rotated)

    # Add Gaussian noise
    def add_noise(pil_img, noise_level=10):
        arr = np.array(pil_img).astype(np.float32)
        noise = np.random.normal(0, noise_level, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    # Translation (shifting image slightly)
    def translate(pil_img, max_shift=4):
       return pil_img.transform(
           pil_img.size,
           Image.AFFINE,
           (1, 0, random.randint(-max_shift, max_shift),
            0, 1, random.randint(-max_shift, max_shift)),
           resample=Image.BILINEAR
       )
    aug_imgs.append(translate(img_pil))
    
    # Blur
    aug_imgs.append(img_pil.filter(ImageFilter.GaussianBlur(radius=1)))
    
    # Zoom in
    def zoom_in(pil_img, zoom=0.9):
       w, h = pil_img.size
       crop_w, crop_h = int(w * zoom), int(h * zoom)
       left = (w - crop_w) // 2
       top = (h - crop_h) // 2
       cropped = pil_img.crop((left, top, left + crop_w, top + crop_h))
       return cropped.resize((w, h), Image.BILINEAR)
    aug_imgs.append(zoom_in(img_pil))


    aug_imgs.append(add_noise(img_pil, noise_level=5))


    return aug_imgs




def fetch_dataset(dx=80, dy=80, dimx=48, dimy=48, DATASET_PATH='', TXT_FILE_ATTR='')-> np.ndarray:
    """
    Fetches and processes images from the specified dataset directory.
    
    Parameters:
    dx (int): Number of pixels to crop from the left and right sides of the image.
    dy (int): Number of pixels to crop from the top and bottom of the image.
    dimx (int): The width to resize the image to after cropping.
    dimy (int): The height to resize the image to after cropping.
    DATASET_PATH (str): Path to the directory containing the dataset of images.
    
    Returns:
    np.ndarray: An array of processed images after resizing and cropping.
    """
    
    
    df_attrs = pd.read_csv(TXT_FILE_ATTR, sep='\t', skiprows=1,) 
    df_attrs = pd.DataFrame(df_attrs.iloc[:,:-1].values, columns = df_attrs.columns[1:])
    photo_ids_ = []; all_photos = []; all_attrs=[]
    seen_paths = set()  # Set to track already seen file paths
   
    for dirpath, dirnames, filenames in os.walk(str(DATASET_PATH)):
        for fname in filenames:
            if fname.endswith(".jpg"):
                fpath = os.path.join(dirpath,fname)
                # Check if the path has already been seen
                if fpath in seen_paths:
                    continue  # Skip if already processed
                # Add the path to the seen set
                seen_paths.add(fpath)
                photo_id = fname[:-4].replace('_',' ').split()
                person_id = ' '.join(photo_id[:-1])
                photo_number = int(photo_id[-1])
                photo_ids_.append({
                            'person':person_id,
                            'imagenum':photo_number,
                            'photo_path':fpath})
               
    df_photo_ids = pd.DataFrame(photo_ids_)
    df = pd.merge(df_photo_ids,df_attrs,on=('person','imagenum'), how='inner')
    ### some data are missing --> duplicated the images when needed !
    assert len(df)==len(df_attrs),"lost some data when merging dataframes"
    
    
    all_augmented_imgs = []
    all_augmented_attrs = []
    
    for i, row in df.iterrows():
        img = imageio.imread(row['photo_path'])
        img = img[dy:-dy, dx:-dx]
        img_pil = Image.fromarray(img).resize([dimx, dimy])

        # Add original image
        all_augmented_imgs.append(np.array(img_pil))
        #all_augmented_attrs.append(row.drop(['photo_path']).values.astype(np.float32))
        all_augmented_attrs.append(row)

        # Add augmented images
        for aug_pil in augment_image(img_pil):
            all_augmented_imgs.append(np.array(aug_pil))
            #all_augmented_attrs.append(row.drop(['photo_path','imagenum','person']).values.astype(np.float32))
            all_augmented_attrs.append(row)
    
        ### test image incremented
        # img = Image.fromarray(all_augmented_imgs[0])
        # img.save('test0.png')
    # Stack and convert
    all_photos = np.stack(all_augmented_imgs).astype('uint8')
    all_attrs = (pd.concat(all_augmented_attrs, axis=1).T).reset_index(drop=True)#np.stack(all_augmented_attrs)

    output_folder = 'images_saved'
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist
    
    # Select 4 random images
    num_images = len(all_photos)
    random_indices = np.random.choice(num_images, size=4, replace=False)  # Select 4 unique random indices
    
    # Save selected images
    for i, idx in enumerate(random_indices, start=1):
        image = all_photos[idx]  # Select image at random index
        file_name = os.path.join(output_folder, f'processed_original_{i}.png')  # Create file path
    
        # Save image as PNG
        img = Image.fromarray(image)
        img.save(file_name)
    
        print(f"Saved: {file_name}")
    
    return all_photos,all_attrs


###############################################################################
def runner(experiments,
           data_img,
           df_attrs ,
           data_val,
           data_attrs_val ,
           data_variance,
           enable_trainig,
           path_save_checkpoint,
           path_save_images
           ):
    
    logger.info('Starting train with experiments..')
    for exp_name, params in experiments.items():
        logger.info(f"Processing c_vae experiment '{exp_name}' with parameters: {params}")
        
        logger.info(f"Starting vqvae experiment: {exp_name}")
        IMAGE_H = data_img.shape[1]
        IMAGE_W = data_img.shape[2]
        
        label_dim = df_attrs.shape[1]
        if(enable_trainig): # start run
            logger.info("Starting training.....")
            history  = run_vqvae(
                data_img,
                df_attrs,
                data_val,
                data_attrs_val ,
                data_variance,
                IMAGE_H,
                IMAGE_W ,
                label_dim ,
                checkpoint_dir=path_save_checkpoint /f"{exp_name}_vqvae_best_weights/", 
                path_save_images = path_save_images ,
                **params ) # Unpack parameters
            # Handle the case where weights_file is None
            logger.info(f"Training Ended !")
        else:
            strategy = setup_strategy()
            logger.info("Loading VQVAE model...")
 
           
            pattern = str(path_save_checkpoint /f"{exp_name}_vqvae_best_weights/")
            # Search for the directory matching the pattern in the given directory
            weights_file = os.path.isdir(pattern)
            
            if weights_file:
                # Extract configuration for the model
                latent_dim = params['latent_dim']
                learning_rate = params['learning_rate']
                num_embeddings = params['num_embeddings']
                embedding_dim = params['embedding_dim']
                commitment_cost = params['commitment_cost']
                encoder_filters = tuple(params['encoder_filters'])
                decoder_filters = tuple(params['decoder_filters'])
                min_epoch_to_start_saving = params['min_epoch_to_start_saving']
                chk = checkpoint_dir=path_save_checkpoint /f"{exp_name}_vqvae_best_weights/"
                with strategy.scope():
                    vqvae = vqvae_class_keras.ConditionalVariationalAutoencoder(IMAGE_H,
                                                                              IMAGE_W,
                                                                              label_dim, 
                                                                              latent_dim,
                                                                              learning_rate,
                                                                              num_embeddings,
                                                                              embedding_dim,
                                                                              commitment_cost,
                                                                              encoder_filters,
                                                                              decoder_filters, chk )

               
                # Load checkpoint before training
                logger.info(f"Successfully loaded weights for VQVAE {exp_name}")
                num_samples = 10
                idx = np.random.choice(len(data_img), num_samples) ## random index
                test_images = data_img[idx]
                output_folder = 'images_saved'
                #path_out =  os.path.join(output_folder, f'originan_vs_generated.png')
                
                rec = vqvae._inference(test_images )
                rec_conc = np.concatenate(rec) #[5, Row, Col, Ch]
                
                if rec is not None:
                    os.makedirs(output_folder , exist_ok=True)
                    
                    fig, axs = plt.subplots(2, num_samples, figsize=(num_samples* 2, 4))
                    ### after selecting 5 random batch i get the first image for evry batch, this is just for test 
                    for i in range(num_samples):
                        axs[0, i].imshow(test_images[i])
                        axs[0, i].axis('off')
                        if i == 0:
                            axs[0, i].set_title("Original")

                        axs[1, i].imshow(rec_conc [i])
                        axs[1, i].axis('off')
                        if i == 0:
                            axs[1, i].set_title("Reconstructed")


                    plt.tight_layout()
                    save_file = os.path.join(output_folder , f"inference.png")
                    plt.savefig(save_file)
                    plt.close(fig)
                

                
            else:
                # Handle the case where weights_file is None
                logger.error(f"No weights file found for VQVAE {exp_name} in chk folder")
                
    
###############################################################################
#### RUN DEF ##################################################################
def run_vqvae(data, 
             data_label,
             data_val,
             data_label_val ,
             data_variance,
             IMAGE_H,
             IMAGE_W , 
             label_dim,
             latent_dim,
             num_embeddings,
             embedding_dim,
             commitment_cost,
             checkpoint_dir='checkpoints',
             save_images_dir = 'saved_images',
             learning_rate=0.001,
             encoder_filters=(32, 16),
             decoder_filters=(16, 32),
             epochs=100,
             batch_size=16, 
             patience=20,
             early_stopping_interval=10,
             min_epoch_to_start_saving=1):
        
        # Set up strategy based on available GPUs
        strategy = setup_strategy()
        
        # Initialize the VQVAE within the strategy scope
        with strategy.scope():
 
            vqvae = vqvae_class_keras.ConditionalVariationalAutoencoder(IMAGE_H,
                                                                      IMAGE_W,
                                                                      label_dim, 
                                                                      latent_dim,
                                                                      learning_rate,
                                                                      num_embeddings,
                                                                      embedding_dim,
                                                                      commitment_cost,
                                                                      encoder_filters,
                                                                      decoder_filters,
                                                                      checkpoint_dir,
                                                                      save_images_dir
                                                                      )
            vqvae.compile()
            history = vqvae.fit(
                        data,
                        data_label,
                        data_val,
                        data_label_val,
                        data_variance,
                        epochs=epochs, 
                        batch_size=batch_size,
                        patience=patience,
                        early_stopping_interval=early_stopping_interval,
                        min_epoch_to_start_saving = min_epoch_to_start_saving
                        )
        return history 

###############################################################################
    


###############################################################################

def main():
    
    ### ENABLING TRAINING OR LOADING MODEL PERFORMING RECOSTRUCTION CHECK
    enable_trainig = True### False
    
    ### PATH DEFINITION #######################################################
    path_root_data = Path(__file__).resolve().parent
    DATASET_PATH = path_root_data /'lfw-deepfunneled' 
    PICKLE_FILE_IMG = path_root_data/'all_photos.pkl'
    PICKLE_FILE_ATTRS = path_root_data/'all_attrs.pkl'
    TXT_FILE_ATTR = path_root_data/'lfw_attributes.txt'
    JSON_EXPERIMENT = path_root_data/'experiments.json'
    CHECKPOINT_PATH = path_root_data  / 'checkpoint/' # path for model chaekpoint
    IMAGE_PATH = path_root_data  /'images_saved/' ## h = '/home/boscolo/source/repos/VQVAE_PIXELCNN/C#
    ### #######################################################################
    dx=dy = 60 ## cropping
    dimx = dimy = 64 #128 # resizing

    
    ### IMAGE PROCESSING STEP ################################################
    # Check if the img pickle file exists
    if os.path.exists(PICKLE_FILE_IMG) and os.path.exists(PICKLE_FILE_ATTRS):
        # Load processed images from the pickle file
        with open(PICKLE_FILE_IMG, 'rb') as f:
            all_photos = pickle.load(f)
        
        with open(PICKLE_FILE_ATTRS, 'rb') as f:
            df_attrs = pickle.load(f)
            
        logger.info("Loaded images from pickle file")
    
    else:
        logger.info("Starting images processing")
        
        all_photos, df_attrs = fetch_dataset(dx=dx,dy=dy,
                                   dimx=dimx,dimy=dimy,
                                   DATASET_PATH=DATASET_PATH,
                                   TXT_FILE_ATTR=TXT_FILE_ATTR)
        
        with open(PICKLE_FILE_IMG, 'wb') as f:
            pickle.dump(all_photos, f)
        with open(PICKLE_FILE_ATTRS, 'wb') as f_a:
             pickle.dump(df_attrs, f_a)
        logger.info("Saved images to pickle file")
    
    # Load the JSON file with vae experiments
    with open(JSON_EXPERIMENT, "r") as json_file:
        experiments = json.load(json_file)
    
    ###########################################################################
    ### GET IMAGES WITH SOME INTERESTING ATTRIBUTES --> SMILING AND BALD
  
    #breakpoint()
    ## RESCALING ####
    data = np.array(all_photos / 255, dtype='float32')#- 0.5
    data_variance = np.var(all_photos/ 255.0)

    #data = np.array((all_photos / 127.5) - 1.0, dtype='float32') # scale the all_photos from the range [0, 255] to the range [-1, 1],
    ### conditioning vars  if somebody wan to use #############################
    
    # Min-Max scaling for each column separately
    df_cond = df_attrs[['Sunglasses','Smiling']]
    df_cond_scaled = df_cond.copy() 
    # Scale each column separately
    for column in df_cond_scaled.columns:
        min_val = df_cond_scaled[column].min()
        max_val = df_cond_scaled[column].max()
        df_cond_scaled[column] = (df_cond_scaled[column] - min_val) / (max_val - min_val)
    
    #Round the values to 3 decimal places before converting to NumPy array
    df_cond_scaled_float = df_cond_scaled.astype(np.float32)
    df_cond_scaled_rounded = df_cond_scaled_float.round(3)
    data_cond_scaled = df_cond_scaled_rounded.values
    
    ###########################################################################
    ### TRAIND A VALIDAITON DATASET
    test_size = 0.2  # 20% for validation

    # Split the data into training and validation sets
    data_train, data_val, data_cond_train, data_cond_val = train_test_split(
        data, data_cond_scaled, test_size=test_size, random_state=42
    )
    
    ## start runner
    _ = runner(
                experiments, 
                data_train,
                data_cond_train,
                data_val,
                data_cond_val,
                data_variance,
                enable_trainig = enable_trainig,
                path_save_checkpoint=CHECKPOINT_PATH,
                path_save_images = IMAGE_PATH
                )
    
    
###############################################################################
if __name__ == "__main__":
   
    main()