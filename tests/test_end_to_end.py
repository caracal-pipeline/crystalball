from unittest.mock import patch
import sys

from crystalball.crystalball import predict

import pytest

def test_end_to_end(tart_ms_tarfile, sky_model):
  with patch.object(sys, "argv", ["crystalball", "--sky-model", sky_model, tart_ms_tarfile]):
    predict(
      ms=tart_ms_tarfile,
      sky_model=sky_model,
    )
