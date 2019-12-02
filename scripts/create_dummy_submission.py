import os
import glob
import pickle
import numpy as np
from skeleton.utils import compute_score


def main():
    test_files = glob.glob("../data/test/*.png")
    test_file_ids = [os.path.basename(path)[:4] for path in test_files]

    dummy_results = glob.glob("../data/train/*.pkl")
    submission = {}
    for id in test_file_ids:
        dummy_result_path = np.random.choice(dummy_results)
        with open(dummy_result_path, "rb") as f:
            dummy_result = pickle.load(f)
        submission[id] = dummy_result
    with open("../data/dummy_submission.pkl", "wb") as f:
        pickle.dump(submission, f)

    compute_score("../data/dummy_submission.pkl", "../data/dummy_submission.pkl")


if __name__ == "__main__":
    main()
