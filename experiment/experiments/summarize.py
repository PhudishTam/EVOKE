import collections
from fuzzywuzzy import fuzz

import json
from pprint import pprint
from typing import List, Optional
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import numpy as np
from scipy.stats import hmean

from util.globals import *

stemmer = PorterStemmer()


def main(
    dir_name,
    runs: Optional[List],
    first_n_cases=None,
    get_uncompressed=False,
    abs_path=False,
):  # runs = None -> all runs
    summaries = []
    uncompressed = []

    portability_tot = 0
    portability_type = None

    for run_dir in (RESULTS_DIR / dir_name if not abs_path else dir_name).iterdir():
        # Skip if we're not interested
        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        # Iterate through all case files
        cur_sum = collections.defaultdict(lambda: [])
        portability_cur_sum = collections.defaultdict(lambda: [])

        files = list(run_dir.glob("*case_*.json"))
        files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
        for case_file in files:
            try:
                with open(case_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Could not decode {case_file} due to format error; skipping.")

            case_id = data["case_id"]
            if first_n_cases is not None and case_id >= first_n_cases:
                break

            portability_type = data["port_type"]

            if ("portability" in data):
                if data["portability"] != {}:
                    x = data["portability"]["cf_metric"]
                    port_key = f"post_{portability_type}"

                    portability_cur_sum[f"{port_key}_EOS"].append(
                        np.mean([x["target_true"] < x["target_new"]])
                    )  # NLL: correct ans < overfit target = Prob: true>new,
                    portability_cur_sum[f"{port_key}_overfit_target_prob"].append(
                        np.exp(-x["target_new"]))
                    portability_cur_sum[f"{port_key}_correct_ans_prob"].append(
                        np.exp(-x["target_true"]))

                    if portability_type == "multihop":
                        portability_cur_sum[f"{port_key}_AMS"].append(
                            np.mean([(x["target_orig"] > x["target_true"])])
                        )
                        portability_cur_sum[f"{port_key}_ori_ans_prob"].append(
                            np.exp(-x["target_orig"]))

                    portability_tot = portability_tot + 1

            if "time" in data:
                cur_sum["time"].append(data["time"])

            for prefix in ["post"]:
                # Probability metrics for which new should be lower (better) than true
                for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
                    if prefix not in data or key not in data[prefix]:
                        continue

                    sum_key_discrete = f"{prefix}_{key.split('_')[0]}_AMS"
                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] > x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )

                # Probability metrics for which true should be lower (better) than new
                for key in ["distraction_prompts_probs", "neighborhood_prompts_probs"]:
                    if prefix not in data or key not in data[prefix]:
                        continue

                    task_name = key.split('_')[0]
                    task_name = task_name if task_name != 'neighborhood' else 'rel_spec'

                    sum_key_discrete = f"{prefix}_{task_name}_EOS"
                    sum_key_target_true_value = f"{prefix}_{task_name}_correct_ans_prob"
                    sum_key_target_new_value = f"{prefix}_{task_name}_overfit_target_prob"

                    if prefix in data and key in data[prefix]:
                        cur_sum[sum_key_discrete].append(
                            np.mean(
                                [
                                    x["target_true"] < x["target_new"]
                                    for x in data[prefix][key]
                                ]
                            )
                        )
                        cur_sum[sum_key_target_new_value].append(
                            np.mean(
                                [
                                    np.exp(-x["target_new"])
                                    for x in data[prefix][key]
                                ]
                            )
                        )
                        cur_sum[sum_key_target_true_value].append(
                            np.mean(
                                [
                                    np.exp(-x["target_true"])
                                    for x in data[prefix][key]
                                ]
                            )
                        )

                # Generation metrics that can be directly averaged
                for key in ["ngram_entropy", "reference_score", "essence_score"]:
                    if prefix in data and key in data[prefix]:
                        cur_sum[f"{prefix}_{key}"].append(data[prefix][key])

        if len(cur_sum) == 0:
            continue

        num_items = len(cur_sum[next(iter(cur_sum.keys()))])
        metadata = {
            "run_dir": str(run_dir),
            "num_cases": num_items,
            "port_tot": portability_tot,
            "port_task_type": portability_type
        }

        uncompressed.append(dict(cur_sum, **metadata))

        cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}
        for k, v in cur_sum.items():
            if all(exclude not in k for exclude in ["essence_score", "time"]):
                # Constant multiplication scales linearly with mean and stddev
                cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)

        portability_cur_sum = {k: (np.mean(v), np.std(v))
                               for k, v in portability_cur_sum.items()}
        for k, v in portability_cur_sum.items():
            if all(exclude not in k for exclude in ["essence_score", "time"]):
                # Constant multiplication scales linearly with mean and stddev
                portability_cur_sum[k] = tuple(
                    np.around(z * 100, 2) for z in v)

        # for prefix in ["pre", "post"]:
        #     for k_efficacy, k_generalization, k_specificity in [
        #         (
        #             f"{prefix}_rewrite_success",
        #             f"{prefix}_paraphrase_success",
        #             f"{prefix}_neighborhood_success",
        #         ),
        #     ]:
        #         if all(k in cur_sum for k in [k_efficacy, k_generalization, k_specificity]):
        #             hmean_list = [
        #                 cur_sum[k_efficacy][0],
        #                 cur_sum[k_generalization][0],
        #                 cur_sum[k_specificity][0],
        #             ]

        #             cur_sum[f"{prefix}_score"] = (hmean(hmean_list), np.nan)
        #             break

        cur_sum.update(metadata)
        cur_sum.update(portability_cur_sum)

        pprint(cur_sum)
        summaries.append(cur_sum)

    return uncompressed if get_uncompressed else summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs."
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="By default, summarizes each run in <dir_name>. "
        "If runs are specified, only evaluates those specific runs.",
    )
    parser.add_argument(
        "--first_n_cases",
        type=int,
        default=None,
        help="Restricts evaluation to first n cases in dataset. "
        "Useful for comparing different in-progress runs on the same slice of data.",
    )
    args = parser.parse_args()

    main(
        args.dir_name,
        None if args.runs is None else args.runs.split(","),
        args.first_n_cases,
    )
