import datasets
import netCDF4 as nc
from astropy.io import fits
from pathlib import Path

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

    for stem in sorted(common_stems):
        nc_path = nc_files[stem]
        fits_path = fits_files[stem]

        # Load NetCDF data
        with nc.Dataset(nc_path, 'r') as nc_dataset:
            nc_data = {
                var: nc_dataset.variables[var][:]
                for var in nc_dataset.variables
            }

        # Load FITS data
        with fits.open(fits_path) as hdul:
            fits_data = [hdu.data for hdu in hdul if hdu.data is not None]

        yield {
            "filename_stem": stem,
            "nc_data": nc_data,
            "fits_data": fits_data,
        }

if __name__ == "__main__":
    df = paired_nc_fits_generator(nc_dir="/home/michaelsmith/data/data/", fits_dir="/shared/huggingface_data/filaments/")
    dic = next(df)
    print(dic["nc_data"])
