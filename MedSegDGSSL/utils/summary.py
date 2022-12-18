"""
Summary the evaluation result after the training
Generate the csv file and the latex table (For simple evaluation)
"""

import os
import json
import argparse
import copy
import pandas as pd
import numpy as np

out_metrics = ['Dice', 'Hausdorff Distance 95']


def generate_summary_crossdomain(folder, method, metrics=out_metrics):
    folder_dir = os.path.join(folder, method)
    domain_list = [name for name in os.listdir(folder_dir) \
                    if not os.path.isfile(os.path.join(folder_dir, name))]
    
    domain_list = [name for name in os.listdir(folder_dir) \
                    if not os.path.isfile(os.path.join(folder_dir, name))]

    # Plus one for avaerage evaludation
    results_domain = copy.deepcopy(domain_list)
    results_domain.append("Average")
    results_domain_dict = {}

    for domain in domain_list:
        temp_summary_dir = os.path.join(folder_dir, domain, "summary")
        with open(os.path.join(temp_summary_dir, domain+"_detail_result.json")) as f:
            temp_summary = json.load(f)['mean_level']

        results_domain_dict[domain] = temp_summary

    label_names = list(temp_summary.keys())
    label_names.remove('Background')

    out_dict = {}

    for label_name in label_names:
        for metric in metrics:
            if "Distance" in metric:
                indent = 2
            else:
                indent = 4

            temp_metric_list  = []
            for domain in domain_list:
                temp_result = results_domain_dict[domain][label_name][metric]
                temp_mean, temp_std = temp_result["mean"], temp_result["std"]
                out_dict[label_name+metric+domain] = [label_name, metric, domain,
                                        f"{np.round(temp_mean, indent)} " +"\u00B1" + f" {np.round(temp_std, indent)}"]
                temp_metric_list.append(temp_mean)
            
            out_dict[label_name+metric+"Average"] = [label_name, metric, "Average",
                        f"{np.round(np.mean(temp_metric_list), indent)} " +"\u00B1" + f" {np.round(np.std(temp_metric_list), indent)}"]

    index_list = ['name', 'metric', "domain", method]
    df = pd.DataFrame(out_dict, index=index_list)
    return df
    

def generate_summary_interdomain(folder, method, metrics=out_metrics):
    folder_dir = os.path.join(folder, method)
    fold_list = [name for name in os.listdir(folder_dir) \
                    if not os.path.isfile(os.path.join(folder_dir, name))]
    
    domain_list = [name.split('_')[0] for name in os.listdir(os.path.join(folder_dir, fold_list[0],"summary")) \
                    if name.endswith('.json')]

    # Plus one for avaerage evaludation
    results_domain = copy.deepcopy(domain_list)
    results_domain.append("Average")
    results_domain_dict = {}

    for fold in fold_list:
        for domain in domain_list:
            if domain not in results_domain_dict.keys():
                results_domain_dict[domain] = {}
            
            temp_summary_dir = os.path.join(folder_dir, fold, "summary")
            with open(os.path.join(temp_summary_dir, domain+"_detail_result.json")) as f:
                temp_summary = json.load(f)['mean_level']

            results_domain_dict[domain][fold] = temp_summary

    label_names = list(temp_summary.keys())
    label_names.remove('Background')

    out_dict = {}

    for label_name in label_names:
        for metric in metrics:
            if "Distance" in metric:
                indent = 2
            else:
                indent = 4

            temp_metric_list  = []
            for domain in domain_list:
                cross_fold_metric_mean = []
                cross_fold_metric_std = []
                for fold in fold_list:
                    cross_fold_metric_mean.append(results_domain_dict[domain][fold][label_name][metric]["mean"])
                    cross_fold_metric_std.append(results_domain_dict[domain][fold][label_name][metric]["std"])
                
                temp_mean, temp_std = np.mean(cross_fold_metric_mean), np.sqrt(np.mean(np.array(cross_fold_metric_std)**2))
                out_dict[label_name+metric+domain] = [label_name, metric, domain,
                                        f"{np.round(temp_mean, indent)} " +"\u00B1" + f" {np.round(temp_std, indent)}"]
                temp_metric_list.append(temp_mean)
            
            out_dict[label_name+metric+"Average"] = [label_name, metric, "Average",
                        f"{np.round(np.mean(temp_metric_list), indent)} " +"\u00B1" + f" {np.round(np.std(temp_metric_list), indent)}"]

    index_list = ['name', 'metric', "domain", method]
    df = pd.DataFrame(out_dict, index=index_list)
    return df


def generate_summary(folder, method_list=None, metrics=out_metrics):
    if method_list is None:
        method_list = [name for name in os.listdir(folder) \
                    if not os.path.isfile(os.path.join(folder, name))]
    
    df_list = [generate_summary_crossdomain(folder, method, metrics=metrics) if not method=="intra_domain" \
                else generate_summary_interdomain(folder, method, metrics=metrics) for method in method_list]
    # print(df_list[0])
    final_df = pd.concat(df_list).drop_duplicates()
    # Get latex if needed
    # final_df.style.to_latex()
    return final_df

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='', help='path to result folder')
    parser.add_argument('--methods', '--names-list', nargs='+',default=None, help='method for the final result')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parse()
    folder = args.folder
    methods = args.methods
    print(generate_summary(folder, methods))