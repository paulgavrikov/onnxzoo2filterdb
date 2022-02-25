import argparse
import os
import onnx
import logging
import pandas
import sys
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from model_utils import ModelTopology
from onnx import numpy_helper
from multiprocessing import Pool
from datetime import datetime


def setup_logger(path) -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-4.4s]  %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    # logging.getLogger().addHandler(stream_handler)
    
    os.makedirs(os.path.join(path, "logs/"), exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(path, f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_create.log"))
    file_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(file_handler)


def load_meta_index(file) -> pandas.DataFrame:
    df = pandas.read_csv(file, header=0)
    return df


def get_model_paths(dir: str) -> set:
    paths = set()
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".onnx"):
                paths.add(os.path.join(root, file))
    return paths


def check_consistency(model_paths: list, meta_index_df: pandas.DataFrame) -> bool:
    consistent = True

    saved_and_indexed_df = meta_index_df[meta_index_df["Name"].notnull()]

    model_filenames = set(map(lambda p: p.split("/")[-1].replace(".onnx", ""), model_paths))
    models_in_index = set(saved_and_indexed_df["Name"].values)

    not_indexed_models = model_filenames - models_in_index
    if not_indexed_models:
        logging.warning(f"{len(not_indexed_models)} models are not indexed: {not_indexed_models}")
        consistent = False

    not_found_models = models_in_index - model_filenames
    if not_found_models:
        logging.warning(f"{len(not_found_models)} indexed models were not found: {not_found_models}")
        consistent = False

    for key in ["Paper", "Framework", "Visual Category", "Precision", "Conv Weight Initializer", "Optimizer", "Task"]:
        count = saved_and_indexed_df[key].isna().values.sum()
        if count:
            logging.warning(f"{count} models have no value for field '{key}'")

    return consistent


def print_info(df: pandas.DataFrame) -> None:
    for key in ["Paper", "Framework", "Visual Category", "Precision", "Conv Weight Initializer", "Optimizer", "Task"]:
        count = df[key].isna().values.sum()
        if count:
            logging.warning(f"{count} *selected* models have no value for field '{key}'")

    print(f"{len(df)} selected models")
    print()
    print(df.groupby("Task").count()["Name"])
    print()
    print(df.groupby("Training").count()["Name"])
    print()
    print(df.groupby("Datatype").count()["Name"])


def process_model(model_path: str) -> (str, bool, dict, dict):
    try:
        logging.info(f"processing {model_path}")
        model = onnx.load_model(model_path)
        # Check that the IR is well formed
        onnx.checker.check_model(model)

        topology = ModelTopology(model)
        layers_by_depth = topology.get_layers_by_depth()

        producer = f"{model.producer_name} {model.producer_version}"
        opset = model.opset_import[0].version
        total_depth = len(layers_by_depth.keys())
        filters_by_depth = defaultdict(list)  # depth:int -> filter_list:list -> dict (keys: name, id, w, b)

        row = {
            "model": model_path.split("/")[-1].replace(".onnx", ""),
            "path": model_path,
            "producer": producer,
            "op_set": opset,
        }

        for depth, node_list in layers_by_depth.items():
            for node in node_list:
                row[node.op_type] = row.get(node.op_type, 0) + 1
                if node.op_type == "Conv":  # TODO: does this cover all conv layers?

                    tensors = list(topology.get_node_input_tensors(node))
                    assert len(tensors) >= 1, "Convolutional layer has no trained parameters. " \
                                              "Were the weights exported?"
                    w = numpy_helper.to_array(tensors[0])
                    
                    if len(w.shape) == 4:
                        info = dict()
                        info["name"] = node.name
                        info["id"] = id(node)
                        info["w"] = w
                        info["b"] = numpy_helper.to_array(tensors[1]) if len(tensors) >= 2 else np.zeros(
                            info["w"].shape[0])

                        # calc number of each filter shape
                        weights = w.reshape(-1, *w.shape[-2:])  # infer first dim
                        shape_key = f"{weights.shape[-2:]} filters"
                        row[shape_key] = row.get(shape_key, 0) + weights.shape[0]

                        filters_by_depth[depth].append(info)
                        # logging.info(f"{depth} {node.name} {len(tensors)} {[tensor.dims for tensor in tensors]}")
                    else:
                         logging.info(f"skipping non 2D-Conv {depth} {node.name} with {len(w.shape)} weight dimensions")

        row["depth"] = total_depth
        
        if row.get("Conv", 0) == 0:
            logging.warning(f"{model_path} has no convolutions")

        return model_path, True, row, {"name" : row["model"], "conv_by_depth" : filters_by_depth}
    except:  # catch all
        logging.error(f"{model_path} raised an Exception: {sys.exc_info()}")
        return model_path, False, None, None


def process(args: argparse.Namespace) -> None:
    logging.debug(f"Arguments: {args}")

    model_paths = sorted(list(get_model_paths(args.onnx_zoo_dir)))
    meta_index_df = load_meta_index(args.meta_index_file)
    is_consistent = check_consistency(model_paths, meta_index_df)
    if not args.force:
        assert is_consistent, "Meta index and model zoo are out of sync. Run this script with --force to ignore"

    selected_model_paths = model_paths
    if args.only_selected:
        selected_df = meta_index_df[meta_index_df["Name"].notna() & meta_index_df["Selection"] == 1]
        selected_df = selected_df.fillna("?")
        selected_model_paths = [model_path for model_path in model_paths if model_path.split("/")[-1]
                                .replace(".onnx", "") in selected_df["Name"].values]
    
    logging.info(f"processing {selected_model_paths}")

    progress = tqdm(total=len(selected_model_paths))
    meta_rows = list()

    for i in range(0, len(selected_model_paths), args.split_after):
        failed_models = set()
        output_list = list()  # list that we will convert to a DF later. Redundant but avoids (slow) resizing of DF.
        start_idx = i
        end_idx = i + args.split_after
        with Pool(32) as pool:
            for model_path, has_completed, meta_row, conv_row in pool.imap_unordered(process_model,
                                                                           selected_model_paths[start_idx:end_idx]):
                progress.update()
                if has_completed:
                    output_list.append(conv_row)
                    meta_rows.append(meta_row)
                else:
                    failed_models.add(model_path)

        if len(failed_models):
            logging.error(f"{len(failed_models)} model could not be processed. Affected files: {failed_models}")

        output_df = pandas.DataFrame(output_list)
        output_df.to_pickle(os.path.join(args.output_file, f"datastorage.{start_idx:04d}.pkl"))

        del output_df
    progress.close()

    meta_df = pandas.DataFrame(meta_rows)
    meta_df.to_csv(os.path.join(args.output_file, "datastorage.meta.csv"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_zoo_dir",
                        help="Directory where the onnx models are stored")
    parser.add_argument("meta_index_file", type=argparse.FileType('r'),
                        help="Path to the index file which stores meta information about the models")
    parser.add_argument("output_file", default="./", type=str,
                        help="Path where the dataframe chunks to be created.")
    parser.add_argument("--split_after", default=10, type=int,
                        help="Splits the final dataframe into multiple chunks of given number of elements"
                             " to avoid OOM errors.")
    parser.add_argument("--force", action="store_true", help="Force processing even if index and zoo are not in sync")
    parser.add_argument("--only_selected", action="store_true", help="Only process models marked as selected in the meta table.")
    _args = parser.parse_args()
    setup_logger(_args.output_file)
    process(_args)
