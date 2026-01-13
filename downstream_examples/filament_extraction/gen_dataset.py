import datasets
import netCDF4 as nc
from astropy.io import fits
from pathlib import Path
from datasets import Dataset
import numpy as np


def paired_nc_fits_generator(nc_dir: str, fits_dir: str):
    """
    Simple generator that yields paired NC and FITS files.
    
    Args:
        nc_dir: Directory containing .nc files
        fits_dir: Directory containing .fits files
    
    Yields:
        Dictionary with paired file info/data
    """
    nc_dir = Path(nc_dir)
    fits_dir = Path(fits_dir)

    nc_files = {f.stem: f for f in nc_dir.glob("*.nc")}
    fits_files = {f.stem[14:27]: f for f in fits_dir.glob("*.fits")}

    common_stems = set(nc_files.keys()) & set(fits_files.keys())
    common_stems = list(common_stems)

    for stem in sorted(common_stems):
        nc_path = nc_files[stem]
        print(nc_path)
        fits_path = fits_files[stem]
        print(fits_path)
        exit(0)

        record = {"filename_stem": stem}

        # Load NetCDF data
        with nc.Dataset(nc_path, 'r') as nc_dataset:
            for var in nc_dataset.variables:
                record[var] = np.array(nc_dataset.variables[var][:].astype(np.float16))

        # Load FITS data
        with fits.open(fits_path) as hdul:
            fits_data = [hdu.data.astype(np.bool) for hdu in hdul if hdu.data is not None][0]

        record["fits_data"] = fits_data

        yield record

if __name__ == "__main__":
    df = paired_nc_fits_generator(nc_dir="/home/michaelsmith/data/data/", fits_dir="/shared/huggingface_data/filaments/")
    ds = Dataset.from_generator(
        paired_nc_fits_generator,
        gen_kwargs={"nc_dir": "/home/michaelsmith/data/data/", "fits_dir": "/shared/huggingface_data/filaments/"}
    )

    ds.push_to_hub("Smith42/surya_filament_dataset", max_shard_size="5GB")
