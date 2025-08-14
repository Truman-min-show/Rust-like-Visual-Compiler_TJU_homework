# ast_printer.py or add to parser.py
import io # Needed to capture print output for GUI
from ast_nodes import * # Import Node definitions and Visitor base

class ASTPrinter(Visitor):
    def __init__(self):
        self.indent_level = 0
        self._output = io.StringIO() # Capture print output

    def _print(self, *args, **kwargs):
        """Helper to print with indentation to the internal buffer."""
        print(f"{'  ' * self.indent_level}", end='', file=self._output)
        print(*args, **kwargs, file=self._output)

    def get_output(self) -> str:
        """Returns the captured print output."""
        return self._output.getvalue()

    def visit_program(self, node: Program):
        self._print("Program:")
        self.indent_level += 1
        for stmt in node.statements:
            stmt.accept(self)
        self.indent_level -= 1

    def visit_let_statement(self, node: LetStatement):
        mut_str = "mut " if node.mutable else ""
        type_str = f": {node.type_info.name}" if node.type_info else "" # Adjust if TypeNode is complex
        self._print(f"LetStatement(name='{node.name.value}', mutable={node.mutable}, type='{type_str}')")
        if node.value:
            self.indent_level += 1
            self._print("Value:")
            node.value.accept(self)
            self.indent_level -= 1

    def visit_return_statement(self, node: ReturnStatement):
        self._print("ReturnStatement:")
        if node.return_value:
            self.indent_level += 1
            node.return_value.accept(self)
            self.indent_level -= 1

    def visit_expression_statement(self, node: ExpressionStatement):
        self._print("ExpressionStatement:")
        self.indent_level += 1
        node.expression.accept(self)
        self.indent_level -= 1

    def visit_block_statement(self, node: BlockStatement):
        self._print("Block:")
        self.indent_level += 1
        for stmt in node.statements:
            stmt.accept(self)
        self.indent_level -= 1

    def visit_loop_statement(self, node: LoopStatement):
        self._print("LoopStatement:")
        self.indent_level += 1
        self._print("Body:")
        node.body.accept(self)
        self.indent_level -= 1

    def visit_break_statement(self, node: BreakStatement):
        self._print(f"BreakStatement(token='{node.token_literal()}')")

    def visit_continue_statement(self, node: ContinueStatement):
        self._print(f"ContinueStatement(token='{node.token_literal()}')")

    def visit_function_declaration(self, node: FunctionDeclarationStatement):
        ret_type = f" -> {node.return_type.name}" if node.return_type else ""
        self._print(f"FunctionDeclaration(name='{node.name.value}', returns='{ret_type}')")
        self.indent_level += 1
        if node.parameters:
             self._print("Parameters:")
             self.indent_level += 1
             for param in node.parameters:
                 param.accept(self)
             self.indent_level -= 1
        self._print("Body:")
        node.body.accept(self)
        self.indent_level -= 1

    def visit_parameter_node(self, node: ParameterNode):
         mut_str = "mut " if node.mutable else ""
         type_str = node.type_info.name if node.type_info else "<?>"
         self._print(f"Parameter(name='{node.name.value}', type='{type_str}', mutable={node.mutable})")

    def visit_if_statement(self, node: IfStatement):
        self._print("IfStatement:")
        self.indent_level += 1
        self._print("Condition:")
        node.condition.accept(self)
        self._print("Consequence:")
        node.consequence.accept(self)
        if node.alternative:
            self._print("Alternative:")
            # Alternative could be another IfStatement or a BlockStatement
            node.alternative.accept(self)
        self.indent_level -= 1

    def visit_identifier(self, node: Identifier):
        self._print(f"Identifier(value='{node.value}')")

    def visit_integer_literal(self, node: IntegerLiteral):
        self._print(f"IntegerLiteral(value={node.value})")

    def visit_prefix_expression(self, node: PrefixExpression):
        self._print(f"PrefixExpression(operator='{node.operator}')")
        self.indent_level += 1
        node.right.accept(self)
        self.indent_level -= 1

    def visit_infix_expression(self, node: InfixExpression):
        self._print(f"InfixExpression(operator='{node.operator}')")
        self.indent_level += 1
        self._print("Left:")
        node.left.accept(self)
        self._print("Right:")
        node.right.accept(self)
        self.indent_level -= 1

    # Implement visit methods for other nodes (Boolean, Call, TypeNode etc.) as needed
    def visit_type_node(self, node: TypeNode):
         # Usually not printed directly, but within context (Let, Param)
         pass

    def visit_while_statement(self, node: WhileStatement):
        self._print("WhileStatement:")
        self.indent_level += 1
        self._print("Condition:")
        node.condition.accept(self)
        self._print("Body:")
        node.body.accept(self)
        self.indent_level -= 1

    def visit_call_expression(self, node: CallExpression):
        self._print("CallExpression:")
        self.indent_level += 1
        self._print("Function:")
        node.function.accept(self)
        if node.arguments:
            self._print("Arguments:")
            self.indent_level += 1
            for arg in node.arguments:
                arg.accept(self)
            self.indent_level -= 1
        else:
            self._print("Arguments: []")
        self.indent_level -= 1

    def visit_assignment_statement(self, node: AssignmentStatement):
        # target 节点可以�? Identifier, IndexExpression, AccessExpression �?
        target_str = f"'{node.target.value}'" if isinstance(node.target, Identifier) else "[Complex Target]"
        self._print(f"AssignmentStatement(target={target_str})")
        self.indent_level += 1
        self._print("Target:")
        node.target.accept(self)  # Visit the target node
        self._print("Value:")
        node.value.accept(self)  # Visit the value node
        self.indent_level -= 1