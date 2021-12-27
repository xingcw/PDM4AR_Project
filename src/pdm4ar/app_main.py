import contracts
import numpy as np
from pdm4ar.app import exercise_without_compmake

if __name__ == "__main__":
    contracts.disable_all()
    for i, seed in enumerate(np.random.randint(0, 1000, size=(100,))):
        print(f"{'='*20} Current Test Number:{i} {'='*20}")
        exercise_without_compmake("final21", seed=seed)
