#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import Crystalball.crystalball as predictor

def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("ms",
                   help="Input .MS file.")
    p.add_argument("-sm", "--sky-model", default="sky-model.txt",
                   help="Name of file containing the sky model. Default is 'sky-model.txt'")
    p.add_argument("-rc", "--row-chunks", type=int, default=10000,
                   help="Number of rows of input .MS that are processed in a single chunk. "
                        "Default is 10000.")
    p.add_argument("-mc", "--model-chunks", type=int, default=10,
                   help="Number of sky model components that are processed in a single chunk. "
                        "Default is 10.")
    p.add_argument("-iuvw", "--invert-uvw", action="store_true",
                   help="Optional. Invert UVW coordinates. Useful if we want "
                        "compare our visibilities against MeqTrees.")
    p.add_argument("-sp", "--spectra", action="store_true",
                   help="Optional. Model sources as non-flat spectra. The spectral "
                        "coefficients and reference frequency must be present in the sky model.")

    predictor.predict( p.parse_args() )
