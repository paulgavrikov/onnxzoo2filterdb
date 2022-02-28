import argparse
import logging
import pandas
import numpy as np
import os
from datetime import datetime
import functools
import h5py


def setup_logger(path) -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-4.4s]  %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(stream_handler)

    os.makedirs(os.path.join(path, "logs/"), exist_ok=True)
    
    file_handler = logging.FileHandler(os.path.join(path, f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_extract.log"))
    file_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(file_handler)


def get_dataframe_chunk_paths(dir: str) -> list:
    chunks = list()
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".pkl"):
                chunks.append(os.path.join(root, file))
    return chunks


def extract_from_chunk(chunk_path: str, meta_df: pandas.DataFrame, filter_shape: tuple, first_convs_only: bool) -> (
        np.array, list):
    logging.info(f"loading chunk {chunk_path}")
    df = pandas.read_pickle(chunk_path)
    logging.info(f"loaded ... processing")

    filter_infos = list()
    
    num_models = len(df)
    num_filters = 0
    for i in range(num_models):
        model_name = df["name"][i]
        num_filters += meta_df[meta_df.model == model_name].get(str(filter_shape) + " filters").fillna(0).astype(int).values[0]
    
    filters = np.zeros(shape=(num_filters, *filter_shape))
    
    idx = 0
    
    for i in range(num_models):
        model_name = df["name"][i]
        model_id = meta_df[meta_df.model == model_name].index.astype(int)[0]

        if first_convs_only:
            first_depth = sorted(list(df["conv_by_depth"][i].keys()))[0]
            iterator = [(first_depth, df["conv_by_depth"][i][first_depth])]
        else:
            iterator = df["conv_by_depth"][i].items()
            
        last_idx = idx
            
        depth_keys = sorted(list(df["conv_by_depth"][i].keys()))
        depth_len = len(depth_keys)
        layer_id = 0
        
        for depth, info_list in iterator:
            conv_depth = depth_keys.index(depth)
            for info_dict in info_list:
                filter_size = info_dict["w"].shape[-2:]
                in_channels = info_dict["w"].shape[1]
                out_channels = info_dict["w"].shape[0]
                weights = info_dict["w"].reshape(-1, *info_dict["w"].shape[-2:])  # infer first dim
                num_filters = weights.shape[0]
                if not filter_shape or filter_size == filter_shape:
                    filters[idx:idx + num_filters] = weights
                    filter_infos.append([idx, idx + num_filters, model_id, layer_id, depth, conv_depth, conv_depth / (depth_len - 1), in_channels, out_channels])
                    idx += num_filters
                    layer_id += 1
                    
        if idx == last_idx:
            logging.warn(f"{model_name} has no conv layers with desired shape")

    logging.info(f"finished chunk {chunk_path} with {num_models} models")
    
        
    
    del df
    return filters, filter_infos


def extract(args):
    logging.info(f"starting extraction {args}")
    all_filter_infos = list()
    all_filters = list()

    chunk_paths = sorted(get_dataframe_chunk_paths(args.dir))  # sorted for determinism

    meta_df = pandas.read_csv(args.meta_file)

    for filters, filter_infos in map(
            functools.partial(extract_from_chunk, meta_df=meta_df, filter_shape=args.filter_shape,
                              first_convs_only=args.first_convs_only),
            chunk_paths):
        all_filters.append(filters)
        all_filter_infos.extend(filter_infos)

    logging.info(f"writing filter info dataframe: {args.filter_info_output_file}")
    filter_infos_pd = pandas.DataFrame(all_filter_infos, columns=["filter_id_start", "filter_id_end", "model", "layer_id", "depth", "conv_depth", "conv_depth_norm", "in_channels", "out_channels"])
    filter_infos_pd.to_csv(args.filter_info_output_file, index=False)

    del all_filter_infos
    del filter_infos_pd

    logging.info(f"writing filters {args.filter_output_file}")
    if len(all_filters) > 1:
        na = np.vstack(all_filters)  # Note: vstack creates a new object and thus doubles RAM usage!
    else:
        na = all_filters[0]
        
    with h5py.File(os.path.join(args.filter_output_file, "dataset.h5"), "a") as f:
        if "filters" in f:
            del f["filters"]
        f["filters"] = na

    logging.info("finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dir",
                        help="Directory where the serialized dataframe chucks are stored. "
                             "Files are found using the '.pkl' suffix. "
                             "Make sure no other data has the same suffix in the directory or any subdirectories.")
    parser.add_argument("meta_file", type=argparse.FileType('r'),
                        help="Path to generated model meta csv.")
    parser.add_argument("filter_info_output_file",
                        help="Path to a csv file that associates every filter row with a model and depth. "
                             "The index corresponds with the index in the model info output file.")
    parser.add_argument("filter_output_file", help="Path to the h5 file to be created containing all 3x3 filters.")
    parser.add_argument("--filter_shape", default=(3, 3))
    parser.add_argument("--first_convs_only", default=False, action="store_true")

    _args = parser.parse_args()
    setup_logger(_args.filter_output_file)
    extract(_args)
