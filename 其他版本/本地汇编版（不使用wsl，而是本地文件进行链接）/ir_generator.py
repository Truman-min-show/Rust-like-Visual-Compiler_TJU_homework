# ir_generator.py
from ast_nodes import (
    Visitor, Node, Program, LetStatement, ReturnStatement, ExpressionStatement,
    BlockStatement, FunctionDeclarationStatement, ParameterNode, IfStatement,
    Identifier, IntegerLiteral, PrefixExpression, InfixExpression,
    CallExpression, TypeNode, WhileStatement, LoopStatement, BreakStatement,
    ContinueStatement, AssignmentStatement, BooleanLiteral, ForStatement
)
# from symbol_table import TYPE_BOOL # 通常不需要在IR生成器中直接引用语义类型对象
from symbol_table import FunctionType
class Quadruple:
    def __init__(self, op, arg1, arg2, result):
        self.op = op        # 操作符 (字符串，如 '+', '-', 'assign', 'goto', 'if_false_goto', 'call', 'label')
        self.arg1 = arg1    # 操作数1 (IRValue 实例, 或 None)
        self.arg2 = arg2    # 操作数2 (IRValue 实例, 或 None)
        self.result = result # 结果 (IRValue 实例, 或 None)

    def __str__(self):
        # 为了更清晰的输出，处理 None 值
        s_arg1 = str(self.arg1) if self.arg1 is not None else '_'
        s_arg2 = str(self.arg2) if self.arg2 is not None else '_'
        s_result = str(self.result) if self.result is not None else '_'
        return f"({self.op}, {s_arg1}, {s_arg2}, {s_result})"

    def __repr__(self):
        return f"Quadruple(op='{self.op}', arg1={self.arg1!r}, arg2={self.arg2!r}, result={self.result!r})"

class IRValue:
    def __init__(self, value, is_temp=False, is_label=False, is_const=False, is_func_name=False, var_name=None):
        self.value = value # 实际值或名称 (如 t0, L1, 5, my_var, my_func)
        self.is_temp = is_temp
        self.is_label = is_label
        self.is_const = is_const
        self.is_func_name = is_func_name
        # var_name 字段可以用来存储原始变量名，即使value是处理过的（例如加了作用域信息）
        # 但对于简单IR，value本身通常就是变量名或临时变量名
        self.original_name = var_name if var_name else (str(value) if not (is_temp or is_label or is_func_name) else None)


    def __str__(self):
        return str(self.value)

    def __repr__(self):
        if self.is_temp: return f"Temp({self.value})"
        if self.is_label: return f"Label({self.value})"
        if self.is_const: return f"Const({self.value})"
        if self.is_func_name: return f"FuncName({self.value})"
        return f"Var({self.value})"


class IRGenerator(Visitor):
    def __init__(self):
        self.quads = []
        self.temp_count = 0
        self.label_count = 0
        self.current_function_name = None # 用于函数上下文
        self.loop_exit_labels = [] # break 跳转的目标标签栈
        self.loop_continue_labels = [] # continue 跳转的目标标签栈

    def _new_temp(self):
        temp_name = f"t{self.temp_count}"
        self.temp_count += 1
        return IRValue(temp_name, is_temp=True)

    def _new_label(self, hint="L"): # 允许为标签提供提示，如 "Else", "EndIf"
        label_name = f"{hint}{self.label_count}"
        self.label_count += 1
        return IRValue(label_name, is_label=True)

    def _emit(self, op, arg1=None, arg2=None, result=None):
        # 确保操作数是 IRValue 类型或 None
        def wrap(val):
            if val is None or isinstance(val, IRValue):
                return val
            if isinstance(val, (int, float)): # bool 会被 isinstance(int) 捕获
                return IRValue(val, is_const=True)
            if isinstance(val, bool): # 单独处理布尔值，通常转为 0 或 1
                 return IRValue(1 if val else 0, is_const=True)
            if isinstance(val, str): # 假设字符串是变量名，除非它是特殊标记
                return IRValue(val) # 作为普通变量名
            raise TypeError(f"Cannot directly emit value of type {type(val)} as IR operand. Wrap in IRValue or handle specifically.")

        self.quads.append(Quadruple(op, wrap(arg1), wrap(arg2), wrap(result)))

    def get_quadruples(self):
        return self.quads

    def visit_program(self, node: Program):
        self._emit("program_begin")
        for stmt in node.statements:
            stmt.accept(self)
        self._emit("program_end")
        return None

    def visit_let_statement(self, node: LetStatement):
        var_ir_value = IRValue(node.name.value) # 变量名
        # 可以在这里发射 'declare' 四元式，如果后端需要明确的声明信息
        # self._emit('declare', var_ir_value, str(node.name.eval_type)) # eval_type 是语义分析时赋予的

        if node.value:
            value_ir = node.value.accept(self)
            if value_ir is not None: # 确保表达式有结果
                self._emit("assign", value_ir, None, var_ir_value)
        return None

    def visit_assignment_statement(self, node: AssignmentStatement):
        value_ir = node.value.accept(self)
        
        # --- 情况 A: *ptr = val ---
        if isinstance(node.target, PrefixExpression) and node.target.operator == '*':
            ptr_ir_val = node.target.right.accept(self)
            self._emit("store", value_ir, None, ptr_ir_val) # STORE (存到地址)
            return None

        # --- 情况 B: var = val (已有逻辑) ---
        target_ir_value = node.target.accept(self)
        if target_ir_value is not None and value_ir is not None:
            self._emit("assign", value_ir, None, target_ir_value)
        return None

    def visit_identifier(self, node: Identifier):
        return IRValue(node.value)

    def visit_integer_literal(self, node: IntegerLiteral):
        return IRValue(node.value, is_const=True)

    def visit_boolean_literal(self, node: BooleanLiteral):
        # 在IR中，布尔值通常表示为 0 (false) 和 1 (true)
        return IRValue(1 if node.value else 0, is_const=True)

    def visit_infix_expression(self, node: InfixExpression):
        left_ir = node.left.accept(self)
        right_ir = node.right.accept(self)
        result_temp = self._new_temp()
        
        # 映射操作符 (AST的 '==' '!=' 等到 IR 的 'eq' 'ne' 'lt' 等)
        op_map = {
            '+': '+', '-': '-', '*': '*', '/': '/',
            '==': 'eq', '!=': 'ne',
            '<': 'lt', '<=': 'le',
            '>': 'gt', '>=': 'ge',
            # 如果有 '&&' '||', 它们通常需要转化为控制流，而不是简单的二元操作
        }
        ir_op = op_map.get(node.operator, node.operator) # 默认使用原操作符
        
        self._emit(ir_op, left_ir, right_ir, result_temp)
        return result_temp

    def visit_prefix_expression(self, node: PrefixExpression):
        op = node.operator

        if op == '&' or op == '&mut':
            # 右侧必须是Identifier
            if not isinstance(node.right, Identifier):
                # 语义分析应该已经捕获了这个错误
                return None
            
            var_ir_val = IRValue(node.right.value)
            result_temp = self._new_temp()
            self._emit("addr", var_ir_val, None, result_temp) # ADDR (取地址)
            return result_temp

        if op == '*':
            # 解引用
            ptr_ir_val = node.right.accept(self)
            result_temp = self._new_temp()
            self._emit("load", ptr_ir_val, None, result_temp) # LOAD (从地址加载)
            return result_temp
        
        # --- 原有逻辑 ---
        operand_ir = node.right.accept(self)
        result_temp = self._new_temp()
        op_map = {'-': 'uminus', '!': 'not'}
        ir_op = op_map.get(node.operator, f"unary_{node.operator}")
        self._emit(ir_op, operand_ir, None, result_temp)
        return result_temp
    
    def visit_expression_statement(self, node: ExpressionStatement):
        if node.expression:
            node.expression.accept(self) # 结果被丢弃，只关心副作用
        return None

    def visit_block_statement(self, node: BlockStatement):
        for stmt in node.statements:
            stmt.accept(self)
        # 如果块作为表达式，最后一个表达式的值可能是块的值
        # 但作为语句，它不返回值。这需要语义分析的 `eval_type` 来辅助判断。
        # 当前简单实现中，块语句本身不产生IR值。
        return None

    def visit_if_statement(self, node: IfStatement):
        cond_ir = node.condition.accept(self)
        
        else_label = self._new_label("Else")
        end_if_label = self._new_label("EndIf")
        
        # 如果条件为假 (0)，则跳转到 else_label
        self._emit("if_false_goto", cond_ir, else_label, None)
        
        # Then 部分
        node.consequence.accept(self)
        if node.alternative: # 只有在有 else 分支时才需要跳过它
            self._emit("goto", end_if_label, None, None)
            
        self._emit("label", else_label, None, None)
        if node.alternative:
            node.alternative.accept(self)
            
        self._emit("label", end_if_label, None, None)
        return None

    def visit_for_statement(self, node: ForStatement):
        # 脱糖为 while 循环的 IR
        
        # 1. 创建标签
        start_label = self._new_label("ForStart")
        end_label = self._new_label("ForEnd")
        
        self.loop_continue_labels.append(start_label) # continue 将跳到增量部分再跳到循环头
        self.loop_exit_labels.append(end_label)
        
        # 2. 初始化循环变量
        loop_var_ir = IRValue(node.variable.value)
        # 假设迭代器是 InfixExpression '..'
        start_val_ir = node.iterator.left.accept(self)
        end_val_ir = node.iterator.right.accept(self)
        
        self._emit("assign", start_val_ir, None, loop_var_ir)
        
        # 3. 循环开始和条件检查
        self._emit("label", start_label, None, None)
        cond_temp = self._new_temp()
        self._emit("lt", loop_var_ir, end_val_ir, cond_temp) # i < end
        self._emit("if_false_goto", cond_temp, end_label, None)
        
        # 4. 循环体
        node.body.accept(self)
        
        # 5. 循环变量自增
        const_one = IRValue(1, is_const=True)
        inc_temp = self._new_temp()
        self._emit("+", loop_var_ir, const_one, inc_temp)
        self._emit("assign", inc_temp, None, loop_var_ir)
        
        # 6. 跳回循环开始
        self._emit("goto", start_label, None, None)
        
        # 7. 循环结束
        self._emit("label", end_label, None, None)
        
        self.loop_continue_labels.pop()
        self.loop_exit_labels.pop()
        
        return None

    def visit_while_statement(self, node: WhileStatement):
        start_loop_label = self._new_label("WhileStart")
        # body_label = self._new_label("WhileBody") # 可以不需要，直接在条件后
        end_loop_label = self._new_label("WhileEnd")
        
        self.loop_exit_labels.append(end_loop_label)
        self.loop_continue_labels.append(start_loop_label) # continue 回到条件判断
        
        self._emit("label", start_loop_label, None, None)
        cond_ir = node.condition.accept(self)
        self._emit("if_false_goto", cond_ir, end_loop_label, None)
        
        # Loop body
        node.body.accept(self)
        self._emit("goto", start_loop_label, None, None) # 返回循环开始
        
        self._emit("label", end_loop_label, None, None)
        
        self.loop_exit_labels.pop()
        self.loop_continue_labels.pop()
        return None

    def visit_loop_statement(self, node: LoopStatement):
        start_label = self._new_label("LoopStart")
        end_label = self._new_label("LoopEnd")

        self.loop_exit_labels.append(end_label)
        self.loop_continue_labels.append(start_label) # continue 回到循环体开始

        self._emit("label", start_label, None, None)
        node.body.accept(self)
        self._emit("goto", start_label, None, None) # 无条件跳回循环开始

        self._emit("label", end_label, None, None) # break 跳到这里

        self.loop_exit_labels.pop()
        self.loop_continue_labels.pop()
        return None

    def visit_break_statement(self, node: BreakStatement):
        if not self.loop_exit_labels:
            # 语义分析应已捕获此错误
            print("IR Gen Error: 'break' outside loop (should have been caught by semantic analysis).")
            return None
        # 如果是 'break <expr>;' (loop作为表达式)，这里会更复杂
        self._emit("goto", self.loop_exit_labels[-1], None, None)
        return None

    def visit_continue_statement(self, node: ContinueStatement):
        if not self.loop_continue_labels:
            print("IR Gen Error: 'continue' outside loop (should have been caught by semantic analysis).")
            return None
        self._emit("goto", self.loop_continue_labels[-1], None, None)
        return None

    def visit_function_declaration(self, node: FunctionDeclarationStatement):
        func_name_ir = IRValue(node.name.value, is_func_name=True)
        self.current_function_name = func_name_ir
        
        # 获取参数数量和返回类型字符串（用于信息性输出）
        num_params = len(node.parameters)
        return_type_str = "void"
        if node.return_type and hasattr(node.return_type, 'name'): # 从 TypeNode 获取
            return_type_str = node.return_type.name
        elif hasattr(node.name, 'eval_type') and isinstance(node.name.eval_type, FunctionType): # 从语义分析结果获取
            return_type_str = str(node.name.eval_type.return_type)


        self._emit("func_begin", func_name_ir, IRValue(num_params, is_const=True), IRValue(return_type_str)) # 传递类型字符串

        # 为参数生成 "receive_param" 或类似指令（如果目标机器模型需要）
        for param_node in node.parameters:
            param_name_ir = IRValue(param_node.name.value)
            # param_type_str = str(param_node.eval_type) # eval_type 是语义分析时赋予的
            self._emit("receive_param", param_name_ir, None, None) # 标记参数

        if node.body:
            node.body.accept(self)

        # 确保非void函数总是有返回。void函数可以在末尾隐式返回。
        # 这一逻辑通常由语义分析保证，或在IR生成后进行检查/转换。
        # 如果函数声明了非void返回类型，但控制流可能不经过 return_val，则是个问题。
        # 简单的处理是，如果函数末尾没有显式的 return，且函数是void，则可省略。
        # 如果非void，语义分析应该强制所有路径都有return。

        self._emit("func_end", func_name_ir, None, None)
        self.current_function_name = None
        return None

    def visit_parameter_node(self, node: ParameterNode):
        # 当参数在函数体中作为表达式使用时，它解析为其名称
        return IRValue(node.name.value)

    def visit_return_statement(self, node: ReturnStatement):
        if node.return_value:
            value_ir = node.return_value.accept(self)
            self._emit("return_val", value_ir, None, None)
        else:
            self._emit("return_void", None, None, None) #  或简化为 "return"
        return None

    def visit_call_expression(self, node: CallExpression):
        # 1. 计算所有参数表达式，并生成 "param_push" 或 "arg" 指令
        # 通常参数从右到左压栈，或按顺序传递。这里我们按AST顺序生成 "arg" 指令。
        arg_irs = []
        for arg_expr in node.arguments:
            arg_ir = arg_expr.accept(self)
            arg_irs.append(arg_ir)
        
        for arg_ir_val in arg_irs: # 按参数列表顺序发射
            self._emit("arg", arg_ir_val, None, None) # 'arg' 或 'param_val'

        # 2. 获取函数名
        # node.function 可能是一个 Identifier，也可能是更复杂的表达式（如果支持函数指针/闭包）
        # 简单情况下，它是一个 Identifier
        func_target_ir = node.function.accept(self) # 应该返回 IRValue(func_name, is_func_name=True)

        # 3. 生成调用指令
        # 需要知道函数是否返回void，以决定是否需要一个临时变量来接收结果
        # 这个信息应该来自语义分析阶段，存储在 node.function.eval_type (一个 FunctionType 对象)
        result_temp = None
        is_void_call = True # 默认
        if hasattr(node, 'eval_type') and node.eval_type is not None: # eval_type 是整个调用表达式的类型
            if str(node.eval_type) != "void" and str(node.eval_type) != "TYPE_VOID":
                is_void_call = False
        
        if not is_void_call:
            result_temp = self._new_temp()
            self._emit("call", func_target_ir, IRValue(len(node.arguments), is_const=True), result_temp)
        else:
            self._emit("call_void", func_target_ir, IRValue(len(node.arguments), is_const=True), None)
            
        return result_temp # 如果非void，返回存储结果的临时变量；否则返回None

    def visit_type_node(self, node: TypeNode):
        # 类型节点在IR生成中通常不直接产生代码，它们的信息在语义分析时使用
        return None

    # 回退给未处理的AST节点
    def generic_visit(self, node: Node):
        # print(f"IRGenerator: Warning - No specific visit method for {type(node).__name__}")
        # 尝试访问子节点。对于表达式，通常需要其结果。
        # 这个通用访问可能不足以处理所有情况，最好为每个节点类型提供显式访问者。
        last_expr_val = None
        for attr_name in dir(node):
            if attr_name.startswith('_'): continue
            try:
                attr_value = getattr(node, attr_name)
                if isinstance(attr_value, Node):
                    res = attr_value.accept(self)
                    if isinstance(res, IRValue): # 如果子节点是表达式并返回值
                        last_expr_val = res
                elif isinstance(attr_value, list):
                    for item in attr_value:
                        if isinstance(item, Node):
                            res = item.accept(self)
                            if isinstance(res, IRValue):
                                last_expr_val = res
            except AttributeError:
                pass
        return last_expr_val # 返回最后一个表达式子节点的结果，或None