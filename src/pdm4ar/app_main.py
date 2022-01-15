import contracts
import numpy as np
import pandas as pd
from pdm4ar.app import exercise_without_compmake

if __name__ == "__main__":
    contracts.disable_all()
    static_records = []
    columns = ["seed", "goal", "collision", "distance", "time"]
    seed = 0
    # for i, seed in enumerate([805, 293, 721, 413]):
    #     print(f"{'=' * 20} Current Test Number:{i}, Seed: {seed} {'=' * 20}")
        # try:
    evals = exercise_without_compmake("final21", seed=seed)
        #     static_records.append([seed, evals[0].goal_reached, evals[0].has_collided,
        #                            evals[0].distance_travelled, evals[0].episode_duration])
        #     s_records = pd.DataFrame(static_records, columns=columns)
        #     s_stats = [0, np.sum(s_records["goal"]), np.sum(s_records["collision"]),
        #                np.sum(np.multiply(s_records["goal"], s_records["distance"])) / (
        #                            np.sum(s_records["goal"]) + 1e-16),
        #                np.sum(np.multiply(s_records["goal"], s_records["time"])) / (np.sum(s_records["goal"]) + 1e-16)]
        #     s_records = pd.concat([s_records, pd.DataFrame([s_stats], columns=columns)])
        #     s_records.to_csv(path_or_buf="../../out/dynamic.csv")
        # except:
        #     print("Errors!")
        #     continue