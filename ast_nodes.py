# ast_nodes.py

from abc import ABC, abstractmethod
from token_defs import Token # Assuming token_defs.py has Token definition

# --- Base Node ---
class Node(ABC):
    def __init__(self, token: Token):
        self.token = token # Store first relevant token for location info

    @abstractmethod
    def accept(self, visitor):
        pass

    def token_literal(self):
        return self.token.literal if self.token else ""

# --- Statements ---
class Statement(Node):
    pass

class Expression(Node):
    pass

class Program(Node):
    
    def __init__(self, statements: list[Statement]):
        super().__init__(statements[0].token if statements else None) # Token from first stmt
        self.statements = statements

    def accept(self, visitor):
        return visitor.visit_program(self)

class LetStatement(Statement):
    def __init__(self, token: Token, name: 'Identifier', value: Expression, type_info=None, mutable=False):
        super().__init__(token) # 'let' token
        self.name = name # Identifier node
        self.value = value # Expression node (or None)
        self.type_info = type_info # Optional TypeNode (implement later)
        self.mutable = mutable

    def accept(self, visitor):
        return visitor.visit_let_statement(self)

class ReturnStatement(Statement):
    def __init__(self, token: Token, return_value: Expression):
        super().__init__(token) # 'return' token
        self.return_value = return_value # Expression node (or None)

    def accept(self, visitor):
        return visitor.visit_return_statement(self)

class ExpressionStatement(Statement):
    def __init__(self, token: Token, expression: Expression):
        super().__init__(token) # First token of the expression
        self.expression = expression

    def accept(self, visitor):
        return visitor.visit_expression_statement(self)

class BlockStatement(Statement):
    def __init__(self, token: Token, statements: list[Statement]):
        super().__init__(token) # '{' token
        self.statements = statements

    def accept(self, visitor):
        return visitor.visit_block_statement(self)

class WhileStatement(Statement):
    def __init__(self, token: Token, condition: Expression, body: BlockStatement):
        super().__init__(token) # 'while' token
        self.condition = condition
        self.body = body

    def accept(self, visitor):
        return visitor.visit_while_statement(self) # 需要添加 visit_while_statement 方法

class ForStatement(Statement):
    def __init__(self, token: Token, variable: 'Identifier', iterator: Expression, body: BlockStatement, mutable: bool = False, inclusive: bool = False):
        super().__init__(token) # 'for' token
        self.variable = variable # 循环变量 (Identifier node)
        self.iterator = iterator # 迭代器表达式 (如 1..10)
        self.body = body         # 循环体 (BlockStatement node)
        self.mutable = mutable   # 变量是否可变 -- 新增
        self.inclusive = inclusive # 范围是否是闭合的 (..) 还是 (..=) -- 新增

    def accept(self, visitor):
        return visitor.visit_for_statement(self)

class LoopStatement(Statement):
    def __init__(self, token: Token, body: BlockStatement):
        super().__init__(token) # 'loop' token
        self.body = body

    def accept(self, visitor):
        return visitor.visit_loop_statement(self)

class BreakStatement(Statement):
    def __init__(self, token: Token): # 'break' token
        super().__init__(token)
        # Placeholder for potential value if implementing `break <expr>;` later
        # self.value: Expression | None = None

    def accept(self, visitor):
        return visitor.visit_break_statement(self)

class ContinueStatement(Statement):
    def __init__(self, token: Token): # 'continue' token
        super().__init__(token)

    def accept(self, visitor):
        return visitor.visit_continue_statement(self)

# 添加 CallExpression (如果 BooleanLiteral 下面还没有)
class CallExpression(Expression):
    def __init__(self, token: Token, function: Expression, arguments: list[Expression]):
         super().__init__(token) # The '(' token or function name token
         self.function = function # Identifier or other expression evaluating to function
         self.arguments = arguments

    def accept(self, visitor):
        return visitor.visit_call_expression(self)

class AssignmentStatement(Statement):
    def __init__(self, token: Token, target: Expression, value: Expression):
        super().__init__(token) # Token of the target (e.g., Identifier token)
        self.target = target # The assignable expression node (Identifier, IndexExpression etc.)
        self.value = value # The expression on the right side of '='

    def accept(self, visitor):
        return visitor.visit_assignment_statement(self) # 需要添加 visit_assignment_statement 方法

class FunctionDeclarationStatement(Statement):
     # Assuming ParameterNode and TypeNode exist or will be added
     def __init__(self, token: Token, name: 'Identifier', params: list['ParameterNode'], body: BlockStatement, return_type=None):
         super().__init__(token) # 'fn' token
         self.name = name
         self.parameters = params
         self.body = body
         self.return_type = return_type # Optional TypeNode

     def accept(self, visitor):
         return visitor.visit_function_declaration(self)

class ParameterNode(Node): # Needs definition
     def __init__(self, token: Token, name: 'Identifier', type_info, mutable=False):
         super().__init__(token) # 'mut' or IDENT token
         self.name = name
         self.type_info = type_info # TypeNode
         self.mutable = mutable

     def accept(self, visitor):
        return visitor.visit_parameter_node(self)


class IfStatement(Statement): # Or IfExpression if 'if' is an expression
     def __init__(self, token: Token, condition: Expression, consequence: BlockStatement, alternative: BlockStatement = None):
         super().__init__(token) # 'if' token
         self.condition = condition
         self.consequence = consequence # Then block
         self.alternative = alternative # Else block (or None)

     def accept(self, visitor):
         return visitor.visit_if_statement(self)

# --- Expressions ---
class Identifier(Expression):
    def __init__(self, token: Token, value: str):
        super().__init__(token)
        self.value = value

    def accept(self, visitor):
        return visitor.visit_identifier(self)

class IntegerLiteral(Expression):
    def __init__(self, token: Token, value: int):
        super().__init__(token)
        self.value = value

    def accept(self, visitor):
        return visitor.visit_integer_literal(self)

class PrefixExpression(Expression):
    def __init__(self, token: Token, operator: str, right: Expression):
        super().__init__(token) # The operator token (e.g., '-')
        self.operator = operator
        self.right = right # Expression node

    def accept(self, visitor):
        return visitor.visit_prefix_expression(self)

class InfixExpression(Expression):
    def __init__(self, token: Token, left: Expression, operator: str, right: Expression):
        super().__init__(token) # The operator token (e.g., '+')
        self.left = left
        self.operator = operator
        self.right = right

    def accept(self, visitor):
        return visitor.visit_infix_expression(self)

class BooleanLiteral(Expression): # Example if needed
    def __init__(self, token: Token, value: bool):
        super().__init__(token)
        self.value = value

    def accept(self, visitor):
        return visitor.visit_boolean_literal(self)

class CallExpression(Expression): # Example if needed
    def __init__(self, token: Token, function: Expression, arguments: list[Expression]):
         super().__init__(token) # The '(' token maybe? Or function name token
         self.function = function # Identifier or other expression evaluating to function
         self.arguments = arguments

    def accept(self, visitor):
        return visitor.visit_call_expression(self)

class TypeNode(Node): # Basic TypeNode placeholder
    def __init__(self, token: Token, name: str):
        super().__init__(token)
        self.name = name

    def accept(self, visitor):
        return visitor.visit_type_node(self)

# --- Visitor Base ---
class Visitor(ABC):
    # Default visit methods can raise NotImplementedError or do nothing
    def visit(self, node: Node):
        method_name = f'visit_{type(node).__name__.lower()}'
        visitor_method = getattr(self, method_name, self.generic_visit)
        return visitor_method(node)

    def generic_visit(self, node: Node):
        print(f"Warning: No specific visit method for {type(node).__name__}")
        return None

    @abstractmethod
    def visit_program(self, node: Program): pass
    @abstractmethod
    def visit_let_statement(self, node: LetStatement): pass
    @abstractmethod
    def visit_return_statement(self, node: ReturnStatement): pass
    @abstractmethod
    def visit_expression_statement(self, node: ExpressionStatement): pass
    @abstractmethod
    def visit_block_statement(self, node: BlockStatement): pass
    @abstractmethod
    def visit_function_declaration(self, node: FunctionDeclarationStatement): pass
    @abstractmethod
    def visit_parameter_node(self, node: ParameterNode): pass # Need visit method
    @abstractmethod
    def visit_if_statement(self, node: IfStatement): pass
    @abstractmethod
    def visit_identifier(self, node: Identifier): pass
    @abstractmethod
    def visit_integer_literal(self, node: IntegerLiteral): pass
    @abstractmethod
    def visit_prefix_expression(self, node: PrefixExpression): pass
    @abstractmethod
    def visit_infix_expression(self, node: InfixExpression): pass
    # Add methods for BooleanLiteral, CallExpression, TypeNode etc. if defined/needed
    def visit_type_node(self, node: TypeNode): pass # Basic pass for now
    
    @abstractmethod
    def visit_for_statement(self, node: ForStatement): pass # --- ADDED ---
    @abstractmethod
    def visit_while_statement(self, node: WhileStatement): pass  # 添加 while 的抽象方法
    @abstractmethod
    def visit_loop_statement(self, node: LoopStatement): pass
    @abstractmethod
    def visit_break_statement(self, node: BreakStatement): pass
    @abstractmethod
    def visit_continue_statement(self, node: ContinueStatement): pass
    @abstractmethod
    def visit_call_expression(self, node: CallExpression): pass  # 添加 call 的抽象方法

    @abstractmethod
    def visit_assignment_statement(self, node: AssignmentStatement): pass  # 添加赋值语句的抽象方法