import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

SOME_SAMPLES = {
    "annotation_path": "/home/xinzhe/workspace/Qwen3-VL/qwen-vl-finetune/data_json/dataset_conversations.json",
    "data_path": "",  # Can be empty if paths are in annotations
}

TRAIN_JAN15_COM = {
    "annotation_path": "data_json/all_jan15/com_train.json",
    "data_path": "",  # Can be empty if paths are in annotations
}

TRAIN_JAN15_NONCOM = {
    "annotation_path": "data_json/all_jan15/noncom_train.json",
    "data_path": "",  # Can be empty if paths are in annotations
}

data_dict = {
    # "cambrian_737k": CAMBRIAN_737K,
    # "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    # "mp_doc": MP_DOC,
    # "clevr_mc": CLEVR_MC,
    # "videochatgpt": VIDEOCHATGPT,
    "samples": SOME_SAMPLES,
    "train_JAN15_com": TRAIN_JAN15_COM,
    "train_JAN15_noncom": TRAIN_JAN15_NONCOM

}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["samples"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
