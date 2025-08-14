from ast_nodes import (
    Visitor, Node, Program, LetStatement, ReturnStatement, ExpressionStatement,
    BlockStatement, FunctionDeclarationStatement, ParameterNode, IfStatement,
    Identifier, IntegerLiteral, PrefixExpression, InfixExpression,
    CallExpression, TypeNode, WhileStatement, LoopStatement, BreakStatement,
    ContinueStatement, AssignmentStatement, BooleanLiteral
)
from symbol_table import (
    SymbolTable, Type, FunctionType,
    TYPE_I32, TYPE_BOOL, TYPE_VOID, TYPE_UNKNOWN, TYPE_ERROR
)


class SemanticAnalyzer(Visitor):
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.current_function_return_type = None
        self.loop_depth = 0
        self._is_read_context = False  # Internal flag to track if an identifier is being read

    def get_errors(self):
        return self.symbol_table.semantic_errors

    def _get_node_type(self, node_visited_by_accept):
        # Ensure eval_type exists, default to TYPE_ERROR if not (e.g., parse error prevented visit)
        return getattr(node_visited_by_accept, 'eval_type', TYPE_ERROR)

    def _process_expr_in_read_context(self, expr_node: Node | None) -> Type:
        """ Helper to visit an expression node ensuring it's treated as an R-value. """
        if expr_node is None:
            return TYPE_VOID  # Or TYPE_ERROR if None is unexpected here

        original_is_read_context = self._is_read_context
        self._is_read_context = True
        expr_node.accept(self)
        self._is_read_context = original_is_read_context
        return self._get_node_type(expr_node)

    def visit_program(self, node: Program):
        self.symbol_table.enter_scope()  # Global scope (level 0)
        for stmt in node.statements:
            stmt.accept(self)
        self.symbol_table.exit_scope()
        node.eval_type = TYPE_VOID

    def visit_let_statement(self, node: LetStatement):
        var_name = node.name.value
        is_mutable = node.mutable

        declared_type_obj = TYPE_UNKNOWN
        if node.type_info:
            node.type_info.accept(self)  # Sets node.type_info.eval_type
            declared_type_obj = self._get_node_type(node.type_info)

        value_type_from_expr = TYPE_VOID  # Type of the initializer expression
        r_value_is_concrete_and_error_free = False  # Can R-value be used for inference/initialization?

        if node.value:
            value_type_from_expr = self._process_expr_in_read_context(node.value)
            if value_type_from_expr not in [TYPE_ERROR, TYPE_VOID, TYPE_UNKNOWN]:
                r_value_is_concrete_and_error_free = True

        final_var_type = TYPE_ERROR  # The type to be stored for the variable
        is_initialized_by_this_let = False

        if declared_type_obj != TYPE_UNKNOWN and declared_type_obj != TYPE_ERROR:
            # Case 1: Explicit type annotation (e.g., let x: T)
            final_var_type = declared_type_obj
            if node.value:  # Has initializer (e.g., let x: T = val)
                if value_type_from_expr != TYPE_ERROR:  # Initializer expression is valid
                    if value_type_from_expr == final_var_type or \
                            value_type_from_expr == TYPE_UNKNOWN:  # Compatible or inferable
                        is_initialized_by_this_let = True
                    else:  # Type mismatch
                        self.symbol_table.add_error(
                            f"Type mismatch for variable '{var_name}'. Explicit type is '{final_var_type}', initializer is '{value_type_from_expr}'.",
                            node
                        )
                # else: initializer had an error, so not initialized by this let.
            # else: no initializer (e.g. let x: T;), not initialized by this let.
        elif node.value:
            # Case 2: No explicit type, but has initializer (e.g., let x = val) - Type Inference
            if r_value_is_concrete_and_error_free:
                final_var_type = value_type_from_expr
                is_initialized_by_this_let = True
            else:
                self.symbol_table.add_error(
                    f"Cannot infer type for variable '{var_name}' from initializer of type '{value_type_from_expr}'.",
                    node)
                # final_var_type remains TYPE_ERROR
        else:
            # Case 3: No explicit type, no initializer (e.g., let x;) - Error
            self.symbol_table.add_error(
                f"Variable '{var_name}' needs an explicit type or an initializer for type inference.", node)
            # final_var_type remains TYPE_ERROR

        # Define the symbol to handle shadowing, even if its type is problematic or an error occurred.
        self.symbol_table.define(var_name, final_var_type, "variable", is_mutable,
                                 is_initialized=is_initialized_by_this_let, node_for_error=node.name)

        node.eval_type = TYPE_VOID  # Let statement itself is void
        node.name.eval_type = final_var_type  # Store type on the Identifier node of the declaration

    def visit_identifier(self, node: Identifier):
        symbol = self.symbol_table.resolve(node.value)
        if not symbol:
            self.symbol_table.add_error(f"Identifier '{node.value}' not declared.", node)
            node.eval_type = TYPE_ERROR
            return

        node.eval_type = symbol.type  # Set the type of the AST identifier node

        # Check for use before initialization IF this identifier is being read.
        if self._is_read_context and symbol.kind == "variable" and not symbol.is_initialized:
            self.symbol_table.add_error(f"Variable '{node.value}' used before being initialized.", node)
            # Note: We don't set node.eval_type to TYPE_ERROR here based *only* on initialization.
            # The type itself might be known; the error is about its value state.
            # Subsequent operations might still fail if they depend on a valid value.

    def visit_assignment_statement(self, node: AssignmentStatement):
        # --- 1. Process Target (L-value) ---
        # Temporarily set read_context to False for the target itself
        original_is_read_context = self._is_read_context
        self._is_read_context = False
        node.target.accept(self)  # This resolves the target (e.g., calls visit_identifier)
        self._is_read_context = original_is_read_context

        target_ast_node_type = self._get_node_type(node.target)
        target_symbol = None

        if isinstance(node.target, Identifier):
            target_symbol = self.symbol_table.resolve(node.target.value)
            if not target_symbol:
                # Error should have been caught by node.target.accept() -> visit_identifier
                node.eval_type = TYPE_ERROR
                return
            if target_ast_node_type == TYPE_ERROR:  # If identifier resolution or type itself is an error
                node.eval_type = TYPE_ERROR
                return
        else:
            # Add support for other L-values like array indexing, field access later
            self.symbol_table.add_error(
                f"Target of assignment must be a simple variable for now. Got {type(node.target).__name__}.",
                node.target)
            node.eval_type = TYPE_ERROR
            return

        if not target_symbol.is_mutable:
            self.symbol_table.add_error(
                f"Cannot assign to immutable variable '{target_symbol.name}'. Variable must be declared 'mut'.",
                node.target)
            # Assignment is invalid due to immutability, but still check type of R-value.

        # --- 2. Process Value (R-value) ---
        value_type_from_expr = self._process_expr_in_read_context(node.value)

        if value_type_from_expr == TYPE_ERROR:  # Error in evaluating the R-value
            node.eval_type = TYPE_ERROR  # Assignment statement is erroneous
            return

        # --- 3. Type Check and Update Initialization ---
        if target_symbol.type != TYPE_ERROR and \
                value_type_from_expr != target_symbol.type and \
                value_type_from_expr != TYPE_UNKNOWN:  # Allow assigning 'unknown' for now, might tighten later
            self.symbol_table.add_error(
                f"Type mismatch in assignment to '{target_symbol.name}'. Expected '{target_symbol.type}', got '{value_type_from_expr}'.",
                node
            )
        else:  # Types are compatible (or target was error, or value was unknown)
            # Mark as initialized only if target is mutable and types are fine
            if target_symbol.type != TYPE_ERROR and target_symbol.is_mutable:
                target_symbol.is_initialized = True

        node.eval_type = TYPE_VOID  # Assignment statement itself is void

    def visit_integer_literal(self, node: IntegerLiteral):
        node.eval_type = TYPE_I32

    def visit_boolean_literal(self, node: BooleanLiteral):
        node.eval_type = TYPE_BOOL

    def visit_type_node(self, node: TypeNode):
        if node.name == "i32":
            node.eval_type = TYPE_I32
        elif node.name == "bool":
            node.eval_type = TYPE_BOOL
        else:
            self.symbol_table.add_error(f"Unknown type name '{node.name}'.", node)
            node.eval_type = TYPE_ERROR

    def visit_expression_statement(self, node: ExpressionStatement):
        if node.expression:
            self._process_expr_in_read_context(node.expression)  # Expression is evaluated
        node.eval_type = TYPE_VOID

    def visit_infix_expression(self, node: InfixExpression):
        left_type = self._process_expr_in_read_context(node.left)
        right_type = self._process_expr_in_read_context(node.right)
        op = node.operator
        result_type = TYPE_ERROR

        if left_type == TYPE_ERROR or right_type == TYPE_ERROR:
            node.eval_type = TYPE_ERROR
            return

        if op in ['+', '-', '*', '/']:
            if left_type == TYPE_I32 and right_type == TYPE_I32:
                result_type = TYPE_I32
            else:
                self.symbol_table.add_error(
                    f"Arithmetic operator '{op}' requires i32 operands, got '{left_type}' and '{right_type}'.", node)
        elif op in ['<', '<=', '>', '>=', '==', '!=']:
            if (left_type == TYPE_I32 and right_type == TYPE_I32) or \
                    (left_type == TYPE_BOOL and right_type == TYPE_BOOL and op in ['==', '!=']):
                result_type = TYPE_BOOL
            else:
                self.symbol_table.add_error(
                    f"Comparison operator '{op}' expects compatible types (i32-i32 or bool-bool for equality), got '{left_type}' and '{right_type}'.",
                    node)
        elif op in ['&&', '||']:  # Assuming these are added to parser
            if left_type == TYPE_BOOL and right_type == TYPE_BOOL:
                result_type = TYPE_BOOL
            else:
                self.symbol_table.add_error(
                    f"Logical operator '{op}' requires boolean operands, got '{left_type}' and '{right_type}'.", node)
        else:
            self.symbol_table.add_error(f"Unsupported infix operator '{op}'.", node)
        node.eval_type = result_type

    def visit_prefix_expression(self, node: PrefixExpression):
        right_type = self._process_expr_in_read_context(node.right)
        op = node.operator
        result_type = TYPE_ERROR

        if right_type == TYPE_ERROR:
            node.eval_type = TYPE_ERROR
            return

        if op == '-':
            if right_type == TYPE_I32:
                result_type = TYPE_I32
            else:
                self.symbol_table.add_error(f"Unary minus operator '-' requires i32 operand, got '{right_type}'.", node)
        elif op == '!':
            if right_type == TYPE_BOOL:
                result_type = TYPE_BOOL
            else:
                self.symbol_table.add_error(f"Logical NOT '!' requires boolean operand, got '{right_type}'.", node)
        else:
            self.symbol_table.add_error(f"Unsupported prefix operator '{op}'.", node)
        node.eval_type = result_type

    def visit_block_statement(self, node: BlockStatement):
        self.symbol_table.enter_scope()
        block_eval_type = TYPE_VOID
        for i, stmt in enumerate(node.statements):
            stmt.accept(self)
            # If block is an expression, its type is the type of the last expression statement
            if i == len(node.statements) - 1 and isinstance(stmt, ExpressionStatement):
                if stmt.expression:  # Check if there's an actual expression
                    block_eval_type = self._get_node_type(stmt.expression)

        node.eval_type = block_eval_type  # Could be TYPE_VOID or type of last expression
        self.symbol_table.exit_scope()

    def visit_function_declaration(self, node: FunctionDeclarationStatement):
        func_name = node.name.value

        param_type_objects = []
        for p_node in node.parameters:
            if p_node.type_info:
                p_node.type_info.accept(self)
                param_type = self._get_node_type(p_node.type_info)
            else:  # Should be caught by parser if types are mandatory
                self.symbol_table.add_error(
                    f"Parameter '{p_node.name.value}' in function '{func_name}' is missing a type.", p_node)
                param_type = TYPE_ERROR
            param_type_objects.append(param_type)

        func_return_type_obj = TYPE_VOID
        if node.return_type:
            node.return_type.accept(self)
            func_return_type_obj = self._get_node_type(node.return_type)

        fn_type_signature = FunctionType(param_type_objects, func_return_type_obj)

        # Define function in the current (outer) scope
        existing_symbol = self.symbol_table.scopes[self.symbol_table.current_scope_level].get(func_name)
        if existing_symbol:  # Rudimentary check for redefinition
            self.symbol_table.add_error(f"Identifier '{func_name}' (function) redefined in the same scope.", node.name)

        self.symbol_table.define(func_name, fn_type_signature, "function", is_mutable=False,
                                 is_initialized=True, node_for_error=node.name)  # Functions are "initialized"
        node.name.eval_type = fn_type_signature

        # --- Process function body in new scope ---
        old_return_type = self.current_function_return_type
        self.current_function_return_type = func_return_type_obj

        self.symbol_table.enter_scope()
        param_names = set()
        for i, param_node in enumerate(node.parameters):
            p_name = param_node.name.value
            p_type = param_type_objects[i]
            param_node.eval_type = p_type  # Type of ParameterNode itself
            param_node.name.eval_type = p_type  # Type of Identifier within ParameterNode

            if p_name in param_names:
                self.symbol_table.add_error(f"Duplicate parameter name '{p_name}' in function '{func_name}'.",
                                            param_node.name)
            else:
                param_names.add(p_name)

            if p_type != TYPE_ERROR:  # Only define if type is not erroneous
                self.symbol_table.define(p_name, p_type, "parameter", param_node.mutable,
                                         is_initialized=True, node_for_error=param_node.name)  # Params are initialized

        if node.body:
            node.body.accept(self)
            # Optional: Check if block's type matches return type if it's an expression block
            # body_block_type = self._get_node_type(node.body)
            # if func_return_type_obj != TYPE_VOID and body_block_type != TYPE_VOID and \
            #    body_block_type != func_return_type_obj and body_block_type != TYPE_ERROR:
            #    self.symbol_table.add_error(f"Function body's implicit return type '{body_block_type}' "
            #                                f"doesn't match declared return type '{func_return_type_obj}'.", node.body)

        self.symbol_table.exit_scope()
        self.current_function_return_type = old_return_type
        node.eval_type = TYPE_VOID  # Function declaration statement is void

    def visit_parameter_node(self, node: ParameterNode):
        # This node is mostly a container. Type resolution happens in visit_function_declaration.
        # Ensure its sub-nodes (type_info, name) are visited if not done elsewhere,
        # and eval_type is set on ParameterNode itself.
        if node.type_info:
            node.type_info.accept(self)  # Set node.type_info.eval_type
            node.eval_type = self._get_node_type(node.type_info)
            if node.name:  # Parameter name (Identifier)
                node.name.eval_type = node.eval_type
        else:
            node.eval_type = TYPE_ERROR  # Should have type_info
            if node.name:
                node.name.eval_type = TYPE_ERROR

    def visit_return_statement(self, node: ReturnStatement):
        if self.current_function_return_type is None:
            self.symbol_table.add_error("'return' statement found outside of a function.", node)
            node.eval_type = TYPE_VOID  # Or a "never" type
            return

        expected_ret_type = self.current_function_return_type
        actual_ret_type = TYPE_VOID

        if node.return_value:
            actual_ret_type = self._process_expr_in_read_context(node.return_value)

        if expected_ret_type != TYPE_ERROR and actual_ret_type != TYPE_ERROR:
            if expected_ret_type != actual_ret_type and actual_ret_type != TYPE_UNKNOWN:
                self.symbol_table.add_error(
                    f"Return type mismatch. Function expects '{expected_ret_type}', but found '{actual_ret_type}'.",
                    node.return_value if node.return_value else node
                )
        node.eval_type = TYPE_VOID

    def visit_call_expression(self, node: CallExpression):
        # The function part of the call is an expression that's "read"
        func_signature_type = self._process_expr_in_read_context(node.function)

        if not isinstance(func_signature_type, FunctionType):
            if func_signature_type != TYPE_ERROR:  # Avoid duplicate error if func itself was error
                func_name_str = node.function.value if isinstance(node.function, Identifier) else "Expression"
                self.symbol_table.add_error(f"'{func_name_str}' is not a function and cannot be called.", node.function)
            node.eval_type = TYPE_ERROR
            return

        expected_param_types = func_signature_type.param_types
        num_expected_args = len(expected_param_types)
        num_actual_args = len(node.arguments)

        if num_actual_args != num_expected_args:
            func_name_str = node.function.value if isinstance(node.function, Identifier) else "Function"
            self.symbol_table.add_error(
                f"{func_name_str} expects {num_expected_args} arguments, but {num_actual_args} were provided.", node
            )
            node.eval_type = TYPE_ERROR  # Mismatch in arg count is a call error
            return  # Stop further processing for this call

        arg_type_error_found = False
        for i in range(num_actual_args):
            arg_expr = node.arguments[i]
            actual_arg_type = self._process_expr_in_read_context(arg_expr)
            expected_arg_type = expected_param_types[i]

            if actual_arg_type != TYPE_ERROR and expected_arg_type != TYPE_ERROR:
                if actual_arg_type != expected_arg_type and actual_arg_type != TYPE_UNKNOWN:
                    self.symbol_table.add_error(
                        f"Type mismatch for argument {i + 1} in call to '{node.function.value if isinstance(node.function, Identifier) else 'function'}'. Expected '{expected_arg_type}', got '{actual_arg_type}'.",
                        arg_expr
                    )
                    arg_type_error_found = True
            elif actual_arg_type == TYPE_ERROR:  # Propagate error from argument expression
                arg_type_error_found = True

        if arg_type_error_found:
            node.eval_type = TYPE_ERROR
        else:
            node.eval_type = func_signature_type.return_type

    def visit_if_statement(self, node: IfStatement):
        cond_type = self._process_expr_in_read_context(node.condition)
        if cond_type != TYPE_BOOL and cond_type != TYPE_ERROR and cond_type != TYPE_UNKNOWN:
            self.symbol_table.add_error(f"If condition must evaluate to a boolean, got '{cond_type}'.", node.condition)

        node.consequence.accept(self)
        consequence_type = self._get_node_type(node.consequence)

        alternative_type = TYPE_VOID
        if node.alternative:
            node.alternative.accept(self)
            alternative_type = self._get_node_type(node.alternative)

        # Type of the if-(else) construct if used as an expression
        if node.alternative:  # if-else
            if consequence_type != TYPE_ERROR and alternative_type != TYPE_ERROR:
                if consequence_type == alternative_type:
                    node.eval_type = consequence_type
                # Rust's if/else can evaluate to () if branches are divergent or one is unit.
                # For simplicity, if types mismatch and neither is void, it's an error.
                # If one is void, the expression type could be void.
                elif consequence_type == TYPE_VOID or alternative_type == TYPE_VOID:
                    if consequence_type == TYPE_VOID and alternative_type == TYPE_VOID:
                        node.eval_type = TYPE_VOID
                    # else: one is void, other is not -> might be error or specific unit type logic
                    # For now, consider mismatch an error if not both void or same.
                    else:
                        self.symbol_table.add_error(
                            f"If-else branches have incompatible types for expression: 'then' is '{consequence_type}', 'else' is '{alternative_type}'.",
                            node
                        )
                        node.eval_type = TYPE_ERROR
                else:  # Mismatch, neither is void
                    self.symbol_table.add_error(
                        f"If-else branches have incompatible types: 'then' branch is '{consequence_type}', 'else' branch is '{alternative_type}'.",
                        node
                    )
                    node.eval_type = TYPE_ERROR
            else:  # One of the branches had an error
                node.eval_type = TYPE_ERROR
        else:  # if without else (evaluates to void or unit type)
            node.eval_type = TYPE_VOID

    def visit_while_statement(self, node: WhileStatement):
        self.loop_depth += 1
        cond_type = self._process_expr_in_read_context(node.condition)
        if cond_type != TYPE_BOOL and cond_type != TYPE_ERROR and cond_type != TYPE_UNKNOWN:
            self.symbol_table.add_error(f"While condition must be boolean, got '{cond_type}'.", node.condition)

        node.body.accept(self)
        self.loop_depth -= 1
        node.eval_type = TYPE_VOID

    def visit_loop_statement(self, node: LoopStatement):
        self.loop_depth += 1
        node.body.accept(self)
        self.loop_depth -= 1
        # loop can be an expression if `break value;` is used.
        # For now, as a statement, it's TYPE_VOID.
        # If `break value;` is implemented, this needs to be the LUB of break values or specified type.
        node.eval_type = TYPE_VOID

    def visit_break_statement(self, node: BreakStatement):
        if self.loop_depth == 0:
            self.symbol_table.add_error("'break' statement found outside of a loop.", node)
        # If `break value;` is implemented, type of node.value would be checked here
        # against the expected type of the loop expression.
        node.eval_type = TYPE_VOID  # or a "never" type conceptually

    def visit_continue_statement(self, node: ContinueStatement):
        if self.loop_depth == 0:
            self.symbol_table.add_error("'continue' statement found outside of a loop.", node)
        node.eval_type = TYPE_VOID

    def generic_visit(self, node: Node):
        # This is a fallback. Specific visitors should handle read contexts.
        # Here, we can't know if children are R-values without more info.
        # For safety, assume children might be read if not specifically handled.
        original_is_read_context = self._is_read_context
        # self._is_read_context = True # Tentatively set if unsure

        for attr_name in dir(node):
            if attr_name.startswith('_') or attr_name == 'token': continue
            try:
                attr_value = getattr(node, attr_name)
                if isinstance(attr_value, Node):
                    attr_value.accept(self)
                elif isinstance(attr_value, list):
                    for item in attr_value:
                        if isinstance(item, Node):
                            item.accept(self)
            except AttributeError:
                pass

        self._is_read_context = original_is_read_context  # Restore

        if not hasattr(node, 'eval_type'):  # Ensure eval_type is set
            node.eval_type = TYPE_UNKNOWN  # Or TYPE_ERROR if it's an unhandled construct