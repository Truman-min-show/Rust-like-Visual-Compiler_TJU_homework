# code/assembly_generator.py

from ast_nodes import (
    Visitor, Program, FunctionDeclarationStatement, LetStatement,
    AssignmentStatement, ReturnStatement, IfStatement, WhileStatement,
    BlockStatement, InfixExpression, PrefixExpression, CallExpression,
    Identifier, IntegerLiteral, BooleanLiteral, LoopStatement, BreakStatement,
    ContinueStatement, ExpressionStatement, ParameterNode
)
from symbol_table import SymbolTable, FunctionType, TYPE_VOID


class AssemblyGenerator(Visitor):
    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table
        self.assembly_code = []
        self.current_function_name = None
        self.label_count = 0
        # 用于跟踪每个函数内局部变量和参数在栈上的偏移量
        self.local_var_offsets = {}
        # 用于循环的标签栈
        self.loop_end_labels = []  # 'break' 跳转的目标
        self.loop_start_labels = []  # 'continue' 跳转的目标

    def _add_line(self, line, indent=True):
        """添加一行汇编代码。"""
        if indent:
            self.assembly_code.append(f"    {line}")
        else:
            self.assembly_code.append(line)

    def _new_label(self, hint=""):
        """生成一个唯一的标签。"""
        label = f".L_{hint}_{self.label_count}"
        self.label_count += 1
        return label

    def get_assembly(self):
        """获取所有生成的汇编代码。"""
        return "\n".join(self.assembly_code)

    def visit_program(self, node: Program):
        self._add_line("global main", indent=False)
        self._add_line("\nsection .text", indent=False)
        for stmt in node.statements:
            if isinstance(stmt, FunctionDeclarationStatement):
                stmt.accept(self)

    def visit_function_declaration(self, node: FunctionDeclarationStatement):
        self.current_function_name = node.name.value
        self.local_var_offsets[self.current_function_name] = {}
        offset_counter = 0
        arg_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]

        # 1. 为参数分配栈空间
        for param_node in node.parameters:
            offset_counter -= 8  # 每个参数/变量占用8字节 (64位)
            self.local_var_offsets[self.current_function_name][param_node.name.value] = offset_counter

        # 2. 递归遍历函数体，为所有局部变量计算栈偏移量
        def collect_vars(block_node):
            nonlocal offset_counter
            for stmt in block_node.statements:
                if isinstance(stmt, LetStatement):
                    var_name = stmt.name.value
                    if var_name not in self.local_var_offsets[self.current_function_name]:
                        offset_counter -= 8
                        self.local_var_offsets[self.current_function_name][var_name] = offset_counter
                elif isinstance(stmt, (IfStatement, WhileStatement, LoopStatement)):
                    if hasattr(stmt, 'consequence') and stmt.consequence: collect_vars(stmt.consequence)
                    if hasattr(stmt, 'alternative') and stmt.alternative: collect_vars(stmt.alternative)
                    if hasattr(stmt, 'body') and stmt.body: collect_vars(stmt.body)

        collect_vars(node.body)
        total_stack_space = -offset_counter

        # --- 函数序言 (Prologue) ---
        self._add_line(f"{self.current_function_name}:", indent=False)
        self._add_line("push rbp")
        self._add_line("mov rbp, rsp")
        if total_stack_space > 0:
            # 确保栈大小是16字节对齐的
            aligned_space = (total_stack_space + 15) & ~15
            self._add_line(f"sub rsp, {aligned_space}")

        # 3. **关键修复点**：将参数从寄存器保存到栈上
        for i, param_node in enumerate(node.parameters):
            if i < len(arg_regs):
                param_name = param_node.name.value
                offset = self.local_var_offsets[self.current_function_name][param_name]
                self._add_line(f"mov [rbp {offset}], {arg_regs[i]}")

        # --- 函数体 ---
        node.body.accept(self)

        # --- 函数尾声 (Epilogue) - 关键修正部分 ---
        self._add_line(f".L_ret_{self.current_function_name}:", indent=False)

        # 从符号表获取函数的返回类型
        func_symbol = self.symbol_table.resolve(self.current_function_name)
        is_void_return = True
        if func_symbol and isinstance(func_symbol.type, FunctionType):
            # 假设 TYPE_VOID 是从 symbol_table 导入的
            is_void_return = (func_symbol.type.return_type == TYPE_VOID)

        # 检查最后一条指令是否是跳转（意味着有显式 return）
        last_instr_is_jmp = False
        if self.assembly_code:
            last_instr_is_jmp = self.assembly_code[-1].strip().startswith("jmp")

        # 仅当函数声明为返回 void 且代码流可能“掉出”函数末尾时，
        # 才将 rax 设为 0 作为默认返回值。
        if is_void_return and not last_instr_is_jmp:
            self._add_line("mov rax, 0")

        self._add_line("mov rsp, rbp")
        self._add_line("pop rbp")
        self._add_line("ret")
        self._add_line("", indent=False)
        self.current_function_name = None

    def visit_block_statement(self, node: BlockStatement):
        for stmt in node.statements:
            stmt.accept(self)

    def visit_expression_statement(self, node: ExpressionStatement):
        if node.expression:
            node.expression.accept(self)  # 结果在 rax 中，但被丢弃

    def visit_let_statement(self, node: LetStatement):
        if node.value:
            node.value.accept(self)  # 结果在 rax 中
            var_name = node.name.value
            offset = self.local_var_offsets[self.current_function_name].get(var_name)
            if offset is not None:
                self._add_line(f"mov [rbp {offset}], rax")

    def visit_assignment_statement(self, node: AssignmentStatement):
        if isinstance(node.target, Identifier):
            var_name = node.target.value
            offset = self.local_var_offsets[self.current_function_name].get(var_name)
            if offset is not None:
                node.value.accept(self)
                self._add_line(f"mov [rbp {offset}], rax")

    def visit_return_statement(self, node: ReturnStatement):
        if node.return_value:
            node.return_value.accept(self)  # 结果在 rax 中
        self._add_line(f"jmp .L_ret_{self.current_function_name}")

    def visit_if_statement(self, node: IfStatement):
        else_label = self._new_label("else")
        end_if_label = self._new_label("endif")
        node.condition.accept(self)
        self._add_line("cmp rax, 0")
        self._add_line(f"je {else_label}")
        node.consequence.accept(self)
        self._add_line(f"jmp {end_if_label}")
        self._add_line(f"{else_label}:", indent=False)
        if node.alternative:
            node.alternative.accept(self)
        self._add_line(f"{end_if_label}:", indent=False)

    def visit_while_statement(self, node: WhileStatement):
        start_label = self._new_label("while_start")
        end_label = self._new_label("while_end")
        self.loop_start_labels.append(start_label)
        self.loop_end_labels.append(end_label)
        self._add_line(f"{start_label}:", indent=False)
        node.condition.accept(self)
        self._add_line("cmp rax, 0")
        self._add_line(f"je {end_label}")
        node.body.accept(self)
        self._add_line(f"jmp {start_label}")
        self._add_line(f"{end_label}:", indent=False)
        self.loop_start_labels.pop()
        self.loop_end_labels.pop()

    def visit_loop_statement(self, node: LoopStatement):
        start_label = self._new_label("loop_start")
        end_label = self._new_label("loop_end")
        self.loop_start_labels.append(start_label)
        self.loop_end_labels.append(end_label)
        self._add_line(f"{start_label}:", indent=False)
        node.body.accept(self)
        self._add_line(f"jmp {start_label}")
        self._add_line(f"{end_label}:", indent=False)
        self.loop_start_labels.pop()
        self.loop_end_labels.pop()

    def visit_break_statement(self, node: BreakStatement):
        if self.loop_end_labels:
            self._add_line(f"jmp {self.loop_end_labels[-1]}")

    def visit_continue_statement(self, node: ContinueStatement):
        if self.loop_start_labels:
            self._add_line(f"jmp {self.loop_start_labels[-1]}")

    def visit_parameter_node(self, node: ParameterNode):
        pass

    def visit_infix_expression(self, node: InfixExpression):
        node.right.accept(self)
        self._add_line("push rax")  # 将右操作数的值保存到栈
        node.left.accept(self)  # 左操作数的值在 rax
        self._add_line("pop rcx")  # 将右操作数的值弹到 rcx (避免使用 rdi)
        op = node.operator
        if op == '+':
            self._add_line("add rax, rcx")
        elif op == '-':
            self._add_line("sub rax, rcx")
        elif op == '*':
            self._add_line("imul rax, rcx")
        elif op == '/':
            self._add_line("cqo")
            self._add_line("idiv rcx")
        elif op in ['==', '!=', '<', '<=', '>', '>=']:
            self._add_line("cmp rax, rcx")
            op_map = {'==': 'sete', '!=': 'setne', '<': 'setl', '<=': 'setle', '>': 'setg', '>=': 'setge'}
            self._add_line(f"{op_map[op]} al")
            self._add_line("movzx rax, al")

    def visit_prefix_expression(self, node: PrefixExpression):
        node.right.accept(self)
        if node.operator == '-':
            self._add_line("neg rax")
        elif node.operator == '!':
            self._add_line("cmp rax, 0")
            self._add_line("sete al")
            self._add_line("movzx rax, al")

    def visit_call_expression(self, node: CallExpression):
        arg_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
        if len(node.arguments) > len(arg_regs):
            return

        for arg in node.arguments:
            arg.accept(self)
            self._add_line("push rax")

        for i in range(len(node.arguments) - 1, -1, -1):
            self._add_line(f"pop {arg_regs[i]}")

        # 确保栈在调用前是16字节对齐的
        self._add_line("sub rsp, 8  ; Align stack for call")
        self._add_line(f"call {node.function.value}")
        self._add_line("add rsp, 8  ; De-align stack after call")

    def visit_identifier(self, node: Identifier):
        offset = self.local_var_offsets[self.current_function_name].get(node.value)
        if offset is not None:
            self._add_line(f"mov rax, [rbp {offset}]")
        # else: an error should have been caught during semantic analysis

    def visit_integer_literal(self, node: IntegerLiteral):
        self._add_line(f"mov rax, {node.value}")

    def visit_boolean_literal(self, node: BooleanLiteral):
        self._add_line(f"mov rax, {1 if node.value else 0}")
