"""Environment check: package versions, JSBSim load, trim metadata."""
import sys
import numpy as np
import scipy
import pandas as pd
import jsbsim

from ecqad_uav.plant.jsbsim_plant import JSBSimPlant
from ecqad_uav.utils.config import load_yaml

if __name__ == "__main__":
    print("Python:", sys.version)
    print("NumPy:", np.__version__)
    print("SciPy:", scipy.__version__)
    print("pandas:", pd.__version__)
    print("JSBSim:", getattr(jsbsim, "__version__", "version attribute unavailable"))
    plant = JSBSimPlant(load_yaml("aircraft.yaml"))
    info = plant.initialize(trim=True)
    print("Loaded model:", plant.model)
    print("Trim info:", info)
