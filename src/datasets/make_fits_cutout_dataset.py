# Step 1: Import Libraries
import os
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from cata2data import CataData
from logging import getLogger
import torch
from torch.utils.data import DataLoader, distributed

logger = getLogger()

# Step 2: Define Paths and Constants
image_path = '/content/drive/MyDrive/im_18k4as.deeper.DI.int.restored.fits'
catalogue_path = '/content/drive/MyDrive/ijepa_logs/catalogue_cutouts.txt'
num_cutouts = 10000  # Number of cutouts
cutout_shape = 224  # Shape of each cutout in pixels
batch_size = 64  # Batch size for DataLoader
num_workers = 8  # Number of workers for DataLoader

# Step 3: Open the FITS File and Determine RA/DEC Range
logger.info("Opening FITS file and extracting RA/DEC range.")
with fits.open(image_path) as hdul:
    wcs = WCS(hdul[0].header, naxis=2)
    image_shape = hdul[0].data.shape[-2:]  # Height and width of the image

    bottom_left = SkyCoord.from_pixel(0, 0, wcs=wcs)
    top_right = SkyCoord.from_pixel(image_shape[1] - 1, image_shape[0] - 1, wcs=wcs)
    ra_min, dec_min = bottom_left.ra.deg, bottom_left.dec.deg
    ra_max, dec_max = top_right.ra.deg, top_right.dec.deg

logger.info(f"RA range: {ra_min} to {ra_max}")
logger.info(f"DEC range: {dec_min} to {dec_max}")

# Step 4: Generate Random RA/DEC within the Image Bounds
logger.info(f"Generating {num_cutouts} random RA/DEC values within the image bounds.")
ra_values = np.random.uniform(ra_min, ra_max, num_cutouts)
dec_values = np.random.uniform(dec_min, dec_max, num_cutouts)

# Step 5: Create the Catalogue DataFrame
df = pd.DataFrame({
    "RA_host": ra_values,
    "DEC_host": dec_values,
    "ID": np.arange(1, num_cutouts + 1)  # Unique IDs for each cutout
})

# Save the Catalogue
with open(catalogue_path, 'w') as f:
    f.write("# RA_host DEC_host ID\n")
    df.to_csv(f, sep=' ', index=False, header=False)
logger.info(f"Catalogue saved to {catalogue_path}.")

# Step 6: Load the Catalogue and FITS Image with CataData
logger.info("Loading catalogue and FITS image using CataData.")
meerklass_data = CataData(
    catalogue_paths=[catalogue_path],
    image_paths=[image_path],
    field_names=['ID'],  # Optional: Use additional columns like 'ID'
    cutout_shape=cutout_shape,
    catalogue_kwargs={
        'format': 'commented_header',
        'delimiter': ' '
    }
)

# Rename columns to match CataData expectations
if 'RA_host' in meerklass_data.df.columns and 'DEC_host' in meerklass_data.df.columns:
    meerklass_data.df.rename(columns={"RA_host": "ra", "DEC_host": "dec"}, inplace=True)

logger.info(f"Renamed columns: {meerklass_data.df.columns}")

# Step 7: Define a PyTorch Dataset Wrapper for CataData
class CutoutDataset(torch.utils.data.Dataset):
    def __init__(self, cata_data):
        self.cata_data = cata_data

    def __len__(self):
        return len(self.cata_data.df)

    def __getitem__(self, idx):
        cutout, metadata = self.cata_data[idx]  # Extract cutout and metadata
        cutout = torch.tensor(cutout, dtype=torch.float32)  # Convert to tensor
        return cutout, metadata['ID']  # Return cutout and ID

logger.info("Dataset wrapper for PyTorch defined.")

# Step 8: Create the DataLoader
logger.info("Creating PyTorch DataLoader with distributed sampler.")

def make_fits_cutout_dataset(
    transform=None,
    batch_size=batch_size,
    collator=None,
    pin_mem=True,
    num_workers=num_workers,
    world_size=1,
    rank=0,
    shuffle=True,
    drop_last=True
):
    dataset = CutoutDataset(meerklass_data)

    # Create a distributed sampler
    sampler = distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )

    # Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        sampler=sampler,
        pin_memory=pin_mem,
        num_workers=num_workers,
        drop_last=drop_last
    )

    return dataset, dataloader, sampler

# Example Usage (for Debugging)
if __name__ == "__main__":
    _, dataloader, _ = make_fits_cutout_dataset()
    for batch_idx, (images, ids) in enumerate(dataloader):
        logger.info(f"Batch {batch_idx + 1}")
        logger.info(f"Images shape: {images.shape}")  # Expect (batch_size, cutout_shape, cutout_shape)
        logger.info(f"IDs: {ids}")
        if batch_idx == 2:  # Example: Limit to first 3 batches
            break
