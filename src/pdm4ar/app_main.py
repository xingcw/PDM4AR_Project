import contracts
import numpy as np
import pandas as pd
from pdm4ar.app import exercise_without_compmake

if __name__ == "__main__":
    contracts.disable_all()
    evals = exercise_without_compmake("final21")