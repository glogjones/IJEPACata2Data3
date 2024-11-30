# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
import time

import numpy as np

from logging import getLogger

import torch
import torchvision

from cata2data import CataData
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord

_GLOBAL_SEED = 0
logger = getLogger()

image_path = '/content/drive/MyDrive/im_18k4as.deeper.DI.int.restored.fits'
catlog_path = '/content/drive/MyDrive/ijepa_logs/catalogue.txt'
imagesize=224


# Step 3: Open the FITS file to determine the RA/DEC range using WCS
with fits.open(image_path) as hdul:
    # Limit WCS to the last 2 dimensions for RA/DEC
    wcs = WCS(hdul[0].header, naxis=2)
    # Use only the 2D spatial dimensions of the image data
    image_shape = hdul[0].data.shape[-2:]  # Get the last two dimensions (e.g., 17325, 17325)
    print("Image dimensions (height, width):", image_shape)

    # Convert the corners of the image to RA/DEC to establish boundaries
    bottom_left = SkyCoord.from_pixel(0, 0, wcs=wcs)
    top_right = SkyCoord.from_pixel(image_shape[1] - 1, image_shape[0] - 1, wcs=wcs)
    ra_min, dec_min = bottom_left.ra.deg, bottom_left.dec.deg
    ra_max, dec_max = top_right.ra.deg, top_right.dec.deg

print(f"RA range: {ra_min} to {ra_max}")
print(f"DEC range: {dec_min} to {dec_max}")

# Step 4: Generate Random RA/DEC within the Image Bounds
num_cutouts = 10000  # Number of cutouts
ra_values = np.random.uniform(ra_min, ra_max, num_cutouts)
dec_values = np.random.uniform(dec_min, dec_max, num_cutouts)

# Step 5: Create the Catalogue DataFrame
df = pd.DataFrame({
    "RA_host": ra_values,
    "DEC_host": dec_values,
    "COSMOS": np.arange(1, num_cutouts + 1)  # Example additional field
})

# Step 6: Save the Catalogue with Commented Header Format
with open(catlog_path, 'w') as f:
    f.write("# RA_host DEC_host COSMOS\n")
    df.to_csv(f, sep=' ', index=False, header=False)


# NO transforms or pre-processing:

meerklass_data = CataData(
    catalogue_paths=[catlog_path],
    image_paths=[image_path],
    field_names=['COSMOS'],
    cutout_shape=(imagesize,imagesize)
)

meerklass_data.df.rename(mapper={"RA_host":"ra", "DEC_host":"dec"}, axis="columns", inplace=True)

# Check the dataset
print(f"Number of entries in dataset: {len(meerklass_data)}")
first_cutout = meerklass_data[0]  # Access first cutout
print(f"Shape of first cutout: {first_cutout[0].shape}")  # Confirm dimensions

def make_cutouts(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None
):
    dataset = meerklass_data(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training,
        copy_data=copy_data,
        index_targets=False
    )
    logger.info('Fits dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('Fits unsupervised data loader created')

    return dataset, data_loader, dist_sampler
