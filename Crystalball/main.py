#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import Crystalball.crystalball as predictor

def main(argv):
    p = predictor.create_parser()

    predictor.predict( p.parse_args() )
