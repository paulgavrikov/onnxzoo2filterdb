import onnx
from collections import defaultdict
import logging


class ModelTopology:

    def __init__(self, model: onnx.ModelProto):
        self.model = model

        # build up inverted indices
        self.node_by_id = dict()
        self.nodes_by_tensor_input = defaultdict(list)
        self.nodes_by_tensor_output = defaultdict(list)

        for node in model.graph.node:
            self.node_by_id[id(node)] = node
            for input_tensor in node.input:
                self.nodes_by_tensor_input[input_tensor].append(id(node))
            for output_tensor in node.output:
                self.nodes_by_tensor_output[output_tensor].append(id(node))

        self.tensors_by_name = dict()

        for tensor in model.graph.initializer:
            self.tensors_by_name[tensor.name] = tensor

    def get_node_input_tensors(self, node: onnx.NodeProto):
        for tensor in node.input:
            if tensor in self.tensors_by_name:
                yield self.tensors_by_name[tensor]
            else:
                pass
                #  logging.warning(
                #     f"{tensor} not found in index. Can be ignored if this is an input or a wrongly traced constant.")

    def get_node_output_ids(self, node: onnx.NodeProto):
        for output_tensor in node.output:
            for connected_node_id in self.nodes_by_tensor_input[output_tensor]:
                yield connected_node_id

    def get_node_input_ids(self, node: onnx.NodeProto):
        for input_tensor in node.input:
            for connected_node_id in self.nodes_by_tensor_output[input_tensor]:
                yield connected_node_id

    def get_model_input_node_ids(self) -> set:

        input_all = [node.name for node in self.model.graph.input]
        input_initializer = [node.name for node in self.model.graph.initializer]
        net_feed_input = set(input_all) - set(input_initializer)

        input_nodes = set()
        for input_tensor_name in net_feed_input:
            for node_id in self.nodes_by_tensor_input[input_tensor_name]:
                input_nodes.add(node_id)

        return input_nodes

    def get_layers_by_depth(self) -> defaultdict:
        layers_by_depth = defaultdict(list)  # depth:int -> nodes:list
        nodes_depth = defaultdict(lambda: -1)  # node_id:str -> depth:int mapping

        # BF-search for each input. We intentionally visit every node multiple times as we are interested in finding the
        # longest path for each node
        for input_node in self.get_model_input_node_ids():
            level = 0
            candidates = set()
            candidates.add(input_node)
            while len(candidates) > 0:
                # logging.debug(f"Traversing graph from {candidates}")
                new_candidates = set()
                for candidate_id in candidates:
                    nodes_depth[candidate_id] = max(level, nodes_depth[candidate_id])
                    for connected_node in self.get_node_output_ids(self.node_by_id[candidate_id]):
                        new_candidates.add(connected_node)

                candidates = new_candidates
                level += 1

        # Now that we have the max depth for each node, simply put into the dict
        for node_id, depth in nodes_depth.items():
            layers_by_depth[depth].append(self.node_by_id[node_id])

        logging.info(f"{len(layers_by_depth.keys())} depths layers found")

        all_nodes_names = set(map(id, self.model.graph.node))
        all_found = set(nodes_depth.keys())

        diff = set(all_nodes_names) - all_found

        if len(diff):
            logging.warning(
                f"The following nodes were skipped, since they have constant inputs: {[f'{self.node_by_id[node_id].name} ({self.node_by_id[node_id].op_type})' for node_id in diff]}. "
                f"This warning can be ignored if it does not contain Conv layers.")
        assert len(nodes_depth.keys()) == len(
            set(nodes_depth.keys())), "Duplicates were found in list. Does the graph have cycles?"

        return layers_by_depth
