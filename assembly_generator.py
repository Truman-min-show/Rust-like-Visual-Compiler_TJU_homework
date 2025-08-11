# code/assembly_generator.py

from ast_nodes import (
    Visitor, Program, FunctionDeclarationStatement, LetStatement,
    AssignmentStatement, ReturnStatement, IfStatement, WhileStatement,
    BlockStatement, InfixExpression, PrefixExpression, CallExpression,
    Identifier, IntegerLiteral, BooleanLiteral, LoopStatement, BreakStatement,
    ContinueStatement, ExpressionStatement, ParameterNode, ForStatement
)
from symbol_table import SymbolTable, FunctionType


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
                offset_counter -= 8
                self.local_var_offsets[self.current_function_name][param_node.name.value] = offset_counter

            # 2. 递归遍历函数体，为所有局部变量计算栈偏移量
            def collect_vars(block_node):
                nonlocal offset_counter
                if not block_node: return # 安全检查
                for stmt in block_node.statements:
                    if isinstance(stmt, LetStatement):
                        var_name = stmt.name.value
                        if var_name not in self.local_var_offsets[self.current_function_name]:
                            offset_counter -= 8
                            self.local_var_offsets[self.current_function_name][var_name] = offset_counter
                    elif isinstance(stmt, ForStatement):
                        var_name = stmt.variable.value
                        if var_name not in self.local_var_offsets[self.current_function_name]:
                            offset_counter -= 8
                            self.local_var_offsets[self.current_function_name][var_name] = offset_counter
                        if hasattr(stmt, 'body') and stmt.body:
                            collect_vars(stmt.body)
                    elif isinstance(stmt, (IfStatement, WhileStatement, LoopStatement)):
                        if hasattr(stmt, 'consequence') and stmt.consequence: collect_vars(stmt.consequence)
                        if hasattr(stmt, 'alternative') and stmt.alternative: collect_vars(stmt.alternative)
                        if hasattr(stmt, 'body') and stmt.body: collect_vars(stmt.body)
            
            # --- 关键调用点 1 ---
            collect_vars(node.body)
            total_stack_space = -offset_counter

            # --- 函数序言 (Prologue) ---
            self._add_line(f"{self.current_function_name}:", indent=False)
            self._add_line("push rbp")
            self._add_line("mov rbp, rsp")
            if total_stack_space > 0:
                aligned_space = (total_stack_space + 15) & ~15
                self._add_line(f"sub rsp, {aligned_space}")

            # 3. 将参数从寄存器保存到栈上
            for i, param_node in enumerate(node.parameters):
                if i < len(arg_regs):
                    param_name = param_node.name.value
                    offset = self.local_var_offsets[self.current_function_name][param_name]
                    self._add_line(f"mov [rbp {offset}], {arg_regs[i]}")

            # --- 函数体 ---
            # --- 关键调用点 2: 这一行必须存在！ ---
            if node.body:
                node.body.accept(self)

            # --- 处理隐式返回 (BUG 修复点) ---
            # 如果函数控制流可以“自然地”到达函数末尾（即最后一条指令不是跳转），
            # 我们需要在这里处理默认返回值。
            if not self.assembly_code or not self.assembly_code[-1].strip().startswith("jmp"):
                # 检查是否是 main 函数，如果是，则默认返回 0。
                # 这是 C 语言和许多系统编程语言的惯例。
                if self.current_function_name == "main":
                    self._add_line("mov rax, 0")
                # 对于其他函数，如果你的语言支持隐式返回最后一个表达式的值，
                # 那么该表达式的结果应该已经留在 rax 中了，这里不需要做任何事。
                # 如果是 void 函数，也不需要做任何事。

            # --- 函数尾声 (Epilogue) ---
            # 这是所有 return 语句和函数体自然结束时跳转到的统一出口点。
            self._add_line(f".L_ret_{self.current_function_name}:", indent=False)
            # 这里不再有任何修改 rax 的逻辑。

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
        # --- 情况 A: *ptr = value ---
        if isinstance(node.target, PrefixExpression) and node.target.operator == '*':
            # 1. 计算右值 value，结果在 rax
            node.value.accept(self)
            # 2. 将 value 的结果压栈保存
            self._add_line("push rax")
            # 3. 计算左边的指针 ptr，它的值(一个地址)会放在 rax
            node.target.right.accept(self)
            # 4. 从栈上弹出 value 到 rcx
            self._add_line("pop rcx")
            # 5. 执行赋值: mov [ptr], value -> mov [rax], rcx
            # rax 存着地址，rcx 存着值
            self._add_line("mov [rax], rcx")
            return

        # --- 情况 B: var = value (已有逻辑) ---
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

    def visit_for_statement(self, node: ForStatement):
        start_label = self._new_label("for_start")
        end_label = self._new_label("for_end")
        continue_label = self._new_label("for_continue")

        self.loop_start_labels.append(continue_label)
        self.loop_end_labels.append(end_label)
        
        var_name = node.variable.value
        offset = self.local_var_offsets[self.current_function_name].get(var_name)
        if offset is None:
            print(f"Assembly Error: for loop variable '{var_name}' has no stack offset.")
            return

        # 初始化循环变量
        node.iterator.left.accept(self)
        self._add_line(f"mov [rbp {offset}], rax")

        # 计算结束值并保存
        node.iterator.right.accept(self)
        self._add_line("push rax") 
        
        self._add_line(f"{start_label}:", indent=False)
        
        # --- 修改点: 条件检查 ---
        self._add_line(f"mov rax, [rbp {offset}]") # 加载当前循环变量 i
        self._add_line("cmp rax, [rsp]")          # 与栈顶的结束值 end 比较
        
        if node.inclusive:
            # inclusive (i..=end) -> 循环条件是 i <= end，退出条件是 i > end
            self._add_line(f"jg {end_label}") # jg (jump if greater)
        else:
            # exclusive (i..end) -> 循环条件是 i < end，退出条件是 i >= end
            self._add_line(f"jge {end_label}") # jge (jump if greater or equal)
        
        # 循环体
        node.body.accept(self)
        
        self._add_line(f"{continue_label}:", indent=False)
        self._add_line(f"mov rax, [rbp {offset}]")
        self._add_line("inc rax")
        self._add_line(f"mov [rbp {offset}], rax")
        
        self._add_line(f"jmp {start_label}")
        
        self._add_line(f"{end_label}:", indent=False)
        self._add_line("add rsp, 8") # 清理栈
        
        self.loop_start_labels.pop()
        self.loop_end_labels.pop()

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
        op = node.operator

        # --- 创建引用: &var ---
        if op == '&' or op == '&mut':
            # 右边必须是 Identifier
            if isinstance(node.right, Identifier):
                var_name = node.right.value
                offset = self.local_var_offsets[self.current_function_name].get(var_name)
                if offset is not None:
                    # LEA: Load Effective Address
                    # 将 [rbp + offset] 这个地址本身加载到 rax
                    self._add_line(f"lea rax, [rbp {offset}]") 
                # else: 语义错误
            # else: 语义错误
            return

        # --- 解引用: *ptr (作为R-value, 读取值) ---
        if op == '*':
            # 先计算出指针 ptr 的值 (它本身是一个地址), 结果在 rax
            node.right.accept(self)
            # 现在 rax 存着一个地址，我们需要从这个地址加载值
            self._add_line(f"mov rax, [rax]")
            return

        # --- 原有逻辑 ---
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