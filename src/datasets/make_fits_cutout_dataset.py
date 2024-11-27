import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, distributed
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from cata2data import CataData

logger = logging.getLogger(__name__)

def make_cata2data(
    batch_size=128,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    fits_file_path=None,
    catalogue_path=None,
    num_cutouts=10000,
    cutout_size=224,
    transform=None,
    collator=None,
    drop_last=True,
):
    """
    Creates a DataLoader for random cutouts from a .fits file using CataData.

    Args:
        batch_size: Number of samples per batch.
        pin_mem: Whether to pin memory for faster data transfer to GPU.
        num_workers: Number of worker threads for DataLoader.
        world_size: Total number of processes for distributed training.
        rank: Rank of the current process.
        fits_file_path: Path to the .fits file.
        catalogue_path: Path to save the generated catalogue file.
        num_cutouts: Total number of random cutouts to generate.
        cutout_size: Size of each cutout in pixels.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        dataset: The dataset object.
        data_loader: The DataLoader object.
        dist_sampler: The DistributedSampler object.
    """
    # Create the dataset
    dataset = CataDataDataset(
        fits_file_path=fits_file_path,
        catalogue_path=catalogue_path,
        num_cutouts=num_cutouts,
        cutout_size=cutout_size,
    )
    logger.info('CataData dataset created')

    # Create a distributed sampler for multi-process training
    dist_sampler = distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
    )

    # Create the DataLoader
    data_loader = DataLoader(
        dataset,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    logger.info('CataData unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class CataDataDataset(Dataset):
    """
    PyTorch Dataset for loading random cutouts from a .fits file using CataData.

    Attributes:
        fits_file_path (str): Path to the .fits file.
        catalogue_path (str): Path to the generated catalogue file.
        num_cutouts (int): Number of random cutouts to generate.
        cutout_size (int): Size of each cutout in pixels.
    """
    def __init__(
        self,
        fits_file_path,
        catalogue_path,
        num_cutouts=10000,
        cutout_size=224,
    ):
        self.fits_file_path = fits_file_path
        self.catalogue_path = catalogue_path
        self.num_cutouts = num_cutouts
        self.cutout_size = cutout_size

        # Generate the catalogue with random RA/DEC positions
        self.generate_random_catalogue()

        # Create CataData object
        self.cata_data = CataData(
            catalogue_paths=[self.catalogue_path],
            image_paths=[self.fits_file_path],
            field_names=['ID'],  # Specify the metadata field(s) present in the catalogue
            cutout_shape=self.cutout_size,
            catalogue_kwargs={
                'delimiter': ' ',  # Explicitly set the delimiter as space
                'format': 'ascii.comment_header'  # Explicitly set the format to match Script 2
            }
        )

    def generate_random_catalogue(self):
        """
        Generates a catalogue of random RA/DEC positions within the .fits image bounds.
        """
        with fits.open(self.fits_file_path) as hdul:
            wcs = WCS(hdul[0].header, naxis=2)
            image_shape = hdul[0].data.shape[-2:]
    
            # Get RA/DEC bounds
            bottom_left = SkyCoord.from_pixel(0, 0, wcs=wcs)
            top_right = SkyCoord.from_pixel(image_shape[1]-1, image_shape[0]-1, wcs=wcs)
            ra_min, dec_min = bottom_left.ra.deg, bottom_left.dec.deg
            ra_max, dec_max = top_right.ra.deg, top_right.dec.deg

        # Generate random RA/DEC within bounds
        ra_values = np.random.uniform(min(ra_min, ra_max), max(ra_min, ra_max), self.num_cutouts)
        dec_values = np.random.uniform(min(dec_min, dec_max), max(dec_min, dec_max), self.num_cutouts)

        # Create DataFrame with correct column names
        df = pd.DataFrame({
            'ra': ra_values,  # Rename to 'ra'
            'dec': dec_values,  # Rename to 'dec'
            'ID': np.arange(1, self.num_cutouts + 1)
        })

        # Save the catalogue with a commented header
        with open(self.catalogue_path, 'w') as f:
            f.write("# ra dec ID\n")  # Write commented header with correct names
            df.to_csv(f, sep=' ', index=False, header=False)  # Use space as delimiter

    def __len__(self):
        return len(self.cata_data.df)

    def __getitem__(self, idx):
        # Get cutout from CataData
        cutout = self.cata_data[idx]

        # Ensure the cutout is in [3, H, W] format


        # Convert to PyTorch tensor
        cutout_tensor = torch.tensor(cutout, dtype=torch.float32)

        return cutout_tensor, idx
