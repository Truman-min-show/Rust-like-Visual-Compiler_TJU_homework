import os
import subprocess
from tkinter import messagebox
from graphviz import Digraph

class ASTGraphvizDrawer:
    # ... (ASTGraphvizDrawer 代码来自之前的提示 - 保持不变) ...
    # (确保它包含了您 AST 节点的所有 visit 方法)
    def __init__(self):
        self.graph = Digraph('AST', node_attr={'shape': 'box', 'fontname': 'monospace'})
        self.node_id = 0

    def add_node(self, label):
        node_name = f"node{self.node_id}"
        self.graph.node(node_name, label)
        self.node_id += 1
        return node_name

    def visit_program(self, node):
        program_node = self.add_node("Program")
        for stmt in node.statements:
            child = stmt.accept(self)
            if child:  # 确保子节点有效
                self.graph.edge(program_node, child)
        return program_node

    def visit_let_statement(self, node):
        label = f"Let: {node.name.value}"
        if node.mutable: label = f"Let mut: {node.name.value}"
        let_node = self.add_node(label)
        if node.value:
            value_node = node.value.accept(self)
            if value_node: self.graph.edge(let_node, value_node, label="value")
        if node.type_info:
            type_node = node.type_info.accept(self)
            if type_node: self.graph.edge(let_node, type_node, label="type")
        return let_node

    def visit_return_statement(self, node):
        ret_node = self.add_node("Return")
        if node.return_value:
            value_node = node.return_value.accept(self)
            if value_node: self.graph.edge(ret_node, value_node)
        return ret_node

    def visit_expression_statement(self, node):
        expr_node = self.add_node("ExpressionStmt")
        if node.expression:
            child_node = node.expression.accept(self)
            if child_node: self.graph.edge(expr_node, child_node)
        return expr_node

    def visit_block_statement(self, node):
        block_node = self.add_node("Block")
        for stmt in node.statements:
            child = stmt.accept(self)
            if child: self.graph.edge(block_node, child)
        return block_node

    def visit_function_declaration(self, node):
        func_label = f"Function: {node.name.value}"
        func_node = self.add_node(func_label)
        for param in node.parameters:
            param_node = param.accept(self)
            if param_node: self.graph.edge(func_node, param_node, label="param")
        if node.body:
            body_node = node.body.accept(self)
            if body_node: self.graph.edge(func_node, body_node, label="body")
        if node.return_type:
            return_type_node = node.return_type.accept(self)
            if return_type_node: self.graph.edge(func_node, return_type_node, label="return_type")
        return func_node

    def visit_parameter_node(self, node):
        label = f"Param: {node.name.value}"
        if node.mutable: label = f"Param mut: {node.name.value}"
        param_node = self.add_node(label)
        if node.type_info:
            type_node = node.type_info.accept(self)
            if type_node: self.graph.edge(param_node, type_node, label="type")
        return param_node

    def visit_if_statement(self, node):
        if_node = self.add_node("IfStatement")
        cond_node = node.condition.accept(self)
        if cond_node: self.graph.edge(if_node, cond_node, label="condition")
        cons_node = node.consequence.accept(self)
        if cons_node: self.graph.edge(if_node, cons_node, label="then")
        if node.alternative:
            alt_node = node.alternative.accept(self)
            if alt_node: self.graph.edge(if_node, alt_node, label="else")
        return if_node

    def visit_identifier(self, node):
        return self.add_node(f"Identifier: {node.value}")

    def visit_integer_literal(self, node):
        return self.add_node(f"Integer: {node.value}")

    def visit_boolean_literal(self, node):
        return self.add_node(f"Boolean: {str(node.value)}")

    def visit_prefix_expression(self, node):
        label = f"Prefix: {node.operator}"
        prefix_node = self.add_node(label)
        right_node = node.right.accept(self)
        if right_node: self.graph.edge(prefix_node, right_node, label="right")
        return prefix_node

    def visit_infix_expression(self, node):
        label = f"Infix: {node.operator}"
        infix_node = self.add_node(label)
        left_node = node.left.accept(self)
        right_node = node.right.accept(self)
        if left_node: self.graph.edge(infix_node, left_node, label="left")
        if right_node: self.graph.edge(infix_node, right_node, label="right")
        return infix_node

    def visit_call_expression(self, node):
        call_node = self.add_node("Call")
        func_node = node.function.accept(self)
        if func_node: self.graph.edge(call_node, func_node, label="function")
        for i, arg in enumerate(node.arguments):
            arg_node = arg.accept(self)
            if arg_node: self.graph.edge(call_node, arg_node, label=f"arg{i}")
        return call_node

    def visit_type_node(self, node):
        return self.add_node(f"Type: {node.name}")

    def visit_while_statement(self, node):
        while_node = self.add_node("While")
        cond_node = node.condition.accept(self)
        if cond_node: self.graph.edge(while_node, cond_node, label="condition")
        body_node = node.body.accept(self)
        if body_node: self.graph.edge(while_node, body_node, label="body")
        return while_node

    def visit_for_statement(self, node):
        for_node = self.add_node(f"For: {node.variable.value}")
        iterator_node = node.iterator.accept(self)
        if iterator_node: self.graph.edge(for_node, iterator_node, label="in")
        body_node = node.body.accept(self)
        if body_node: self.graph.edge(for_node, body_node, label="body")
        return for_node

    def visit_assignment_statement(self, node):
        assign_node = self.add_node("Assign")
        target_node = node.target.accept(self)
        if target_node: self.graph.edge(assign_node, target_node, label="target")
        value_node = node.value.accept(self)
        if value_node: self.graph.edge(assign_node, value_node, label="value")
        return assign_node

    def visit_loop_statement(self, node):
        loop_node = self.add_node("Loop")
        if node.body:
            body_node = node.body.accept(self)
            if body_node: self.graph.edge(loop_node, body_node, label="body")
        return loop_node

    def visit_break_statement(self, node):
        return self.add_node("Break")

    def visit_continue_statement(self, node):
        return self.add_node("Continue")

    def render_to_memory(self, base_dir):
        try:
            # 渲染图到二进制数据（而不是文件）
            tmp_filepath = os.path.join(base_dir, "ast_tmp_output")
            tmp_filepath_dot = tmp_filepath + ".dot"
            tmp_filepath_png = tmp_filepath + ".png"

            self.graph.save(tmp_filepath_dot)

            dot_exe_path = os.path.join(base_dir, 'graphviz_bin', 'dot.exe')
            subprocess.run([
                dot_exe_path,
                "-Tpng",
                tmp_filepath_dot,
                "-o",
                tmp_filepath_png
            ], check=True)

            with open(tmp_filepath_png, "rb") as f:
                img_bytes = f.read()

            os.remove(tmp_filepath_dot)
            os.remove(tmp_filepath_png)

            return img_bytes
        except Exception as e:
            print(f"AST图像错误: {e}")
            # Potentially return a placeholder image or raise the error
            # For GUI, it's better to show an error message to the user
            messagebox.showerror("AST图像错误", f"Graphviz渲染失败: {e}\n请确保Graphviz已正确安装并添加到系统PATH。")
            return None
