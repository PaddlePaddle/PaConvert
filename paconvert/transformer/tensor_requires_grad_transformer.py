# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast

from paconvert.base import BaseTransformer


class TensorRequiresGradTransformer(BaseTransformer):
    """
    process torch.requires_grad attribute left value
    -  *.requires_grad = value -> *.stop_gradient = not value
    -  a, *.requires_grad = value ->
                        a, temp = value
                        *.stop_gradient = not temp
    """

    def __init__(
        self, root, file, imports_map, logger, all_api_map=None, unsupport_api_map=None
    ):
        super(TensorRequiresGradTransformer, self).__init__(
            root, file, imports_map, logger, all_api_map, unsupport_api_map
        )
        self.insert_nodes_list = []

    @property
    def parent_node(self):
        return self.node_stack[-2]

    def visit_Assign(self, node):
        # left value
        if isinstance(node, (ast.Assign)):
            if isinstance(node.targets[0], ast.Attribute):
                if node.targets[0].attr == "requires_grad":
                    node.targets[0].attr = "stop_gradient"
                    node = ast.Assign(
                        targets=[node.targets[0]],
                        value=ast.UnaryOp(ast.Not(), operand=node.value),
                    )

            elif isinstance(node.targets[0], ast.Tuple):
                for j in range(len(node.targets[0].elts)):
                    if isinstance(node.targets[0].elts[j], ast.Attribute) and (
                        node.targets[0].elts[j].attr == "requires_grad"
                    ):
                        new_node = ast.Name(id="temp", ctx=ast.Load())
                        node.targets[0].elts[j].attr = "stop_gradient"
                        assign_node = ast.Assign(
                            targets=[node.targets[0].elts[j]],
                            value=ast.UnaryOp(ast.Not(), operand=new_node),
                        )
                        node.targets[0].elts[j] = new_node
                        index = self.parent_node.body.index(node)
                        self.insert_nodes_list.append(
                            (self.parent_node, index, assign_node)
                        )

        return node

    def insert_assign_node(self):
        for parent_node, index, node in self.insert_nodes_list[::-1]:
            parent_node.body.insert(index + 1, node)

    def transform(self):
        self.visit(self.root)
        self.insert_assign_node()
