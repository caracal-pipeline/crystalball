# -*- coding: utf-8 -*-

import argparse

import psutil


def _num_workers(nworkers_str):
    if nworkers_str == "default":
        return psutil.cpu_count()

    try:
        return int(nworkers_str)
    except ValueError as e:
        raise argparse.ArgumentError("%s is not a valid integer" % nworkers_str)


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms",
                   help="Input .MS file.")
    p.add_argument("-sm", "--sky-model", default="sky-model.txt",
                   help="Name of file containing the sky model. "
                        "Default is 'sky-model.txt'")
    p.add_argument("-o", "--output-column", default="MODEL_DATA",
                   help="Output visibility column. Default is '%(default)s'")
    p.add_argument("-f", "--fields", type=str,
                   help="Comma-separated list of Field names or ids "
                        "which should be predicted. "
                        "All fields are predicted by default.")
    p.add_argument("-rc", "--row-chunks", type=int, default=0,
                   help="Number of rows of input .MS that are processed in "
                        "a single chunk. If 0 it will be set automatically. "
                        "Default is 0.")
    p.add_argument("-mc", "--model-chunks", type=int, default=0,
                   help="Number of sky model components that are processed in "
                        "a single chunk. If 0 it wil be set automatically. "
                        "Default is 0.")
    p.add_argument("--exp-sign-convention", choices=['casa', 'thompson'],
                   default='casa',
                   help="Sign convention to use for the complex exponential. "
                        "'casa' specifies the e^(2.pi.I) convention while "
                        "'thompson' specifies the e^(-2.pi.I) convention in "
                        "the white book and Fourier analysis literature. "
                        "Defaults to '%(default)s'")
    p.add_argument("-sp", "--spectra", action="store_true",
                   help="Optional. Model sources as non-flat spectra. "
                        "The spectral coefficients and reference frequency "
                        "must be present in the sky model.")
    p.add_argument("-w", "--within", type=str,
                   help="Optional. Give JS9 region file. Only sources within "
                        "those regions will be included.")
    p.add_argument("-po", "--points-only", action="store_true",
                   help="Select only point-type sources.")
    p.add_argument("-ns", "--num-sources", type=int, default=0, metavar="N",
                   help="Select only N brightest sources.")
    p.add_argument("-j", "--num-workers", type=_num_workers,
                   default="default",
                   help="Explicitly set the number of worker threads.")
    p.add_argument("-mf", "--memory-fraction", type=float, default=0.5,
                   help="Fraction of system RAM that can be used. "
                        "Used when setting automatically the "
                        "chunk size. Default in 0.5.")

    return p
