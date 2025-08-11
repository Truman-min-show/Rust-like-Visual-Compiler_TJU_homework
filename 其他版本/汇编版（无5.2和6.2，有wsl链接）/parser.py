# parser.py
from token_defs import TokenType, Token
from lexer import Lexer
# --- NEW: Import AST nodes ---
from ast_nodes import * # Import all node classes and Visitor base

# --- Operator Precedence Levels ---
# Lower numbers mean lower precedence
LOWEST = 0
EQUALS = 1         # ==, !=
LESSGREATER = 2    # <, >, <=, >=
SUM = 3            # +, -
PRODUCT = 4        # *, /
PREFIX = 5         # -X or !X (or &X, *X)
CALL = 6           # myFunction(X)
INDEX = 7          # array[index]
ACCESS = 8         # tuple.0 or struct.field

# Map token types to precedence levels
PRECEDENCES = {
    TokenType.EQ: EQUALS,
    TokenType.NOT_EQ: EQUALS,
    TokenType.LT: LESSGREATER,
    TokenType.GT: LESSGREATER,
    TokenType.LTE: LESSGREATER,
    TokenType.GTE: LESSGREATER,
    TokenType.PLUS: SUM,
    TokenType.MINUS: SUM,
    TokenType.SLASH: PRODUCT,
    TokenType.ASTERISK: PRODUCT,
    TokenType.LPAREN: CALL,    # For function calls
    TokenType.LBRACKET: INDEX, # For array indexing - Placeholder if parsing not added
    TokenType.DOT: ACCESS,     # For tuple/struct access - Placeholder if parsing not added
}


# parser.py
# coding=gbk
from token_defs import TokenType, Token
from lexer import Lexer
# --- NEW: Import AST nodes ---
from ast_nodes import * # Import all node classes and Visitor base

# --- Operator Precedence Levels (Keep As Is) ---
# ... (LOWEST, EQUALS, ..., PRECEDENCES dictionary) ...

class Parser:
    # --- __init__ and Registrations (Keep As Is from your working version) ---
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.errors = []
        self.current_token: Token = None
        self.peek_token: Token = None
        self._next_token()
        self._next_token()
        self.prefix_parse_fns = {}
        self.infix_parse_fns = {}
        self.register_prefix(TokenType.IDENT, self._parse_identifier)
        self.register_prefix(TokenType.INT, self._parse_integer_literal)
        self.register_prefix(TokenType.MINUS, self._parse_prefix_expression)
        self.register_prefix(TokenType.AMPERSAND, self._parse_prefix_expression)
        self.register_prefix(TokenType.ASTERISK, self._parse_prefix_expression)
        self.register_prefix(TokenType.LPAREN, self._parse_grouped_expression_or_tuple)
        #self.register_prefix(TokenType.LBRACKET, self._parse_array_literal) # Keep commented
        self.register_prefix(TokenType.IF, self._parse_if_expression) # Or _parse_if_statement
        #self.register_prefix(TokenType.LOOP, self._parse_loop_expression) # Keep commented
        self.register_infix(TokenType.PLUS, self._parse_infix_expression)
        self.register_infix(TokenType.MINUS, self._parse_infix_expression)
        self.register_infix(TokenType.SLASH, self._parse_infix_expression)
        self.register_infix(TokenType.ASTERISK, self._parse_infix_expression)
        self.register_infix(TokenType.EQ, self._parse_infix_expression)
        self.register_infix(TokenType.NOT_EQ, self._parse_infix_expression)
        self.register_infix(TokenType.LT, self._parse_infix_expression)
        self.register_infix(TokenType.GT, self._parse_infix_expression)
        self.register_infix(TokenType.LTE, self._parse_infix_expression)
        self.register_infix(TokenType.GTE, self._parse_infix_expression)
        self.register_infix(TokenType.LPAREN, self._parse_call_expression)
        #self.register_infix(TokenType.LBRACKET, self._parse_index_expression) # Keep commented
        #self.register_infix(TokenType.DOT, self._parse_access_expression) # Keep commented

    # --- Helper Methods (Keep As Is from your working version) ---
    def register_prefix(self, token_type: TokenType, fn): self.prefix_parse_fns[token_type] = fn
    def register_infix(self, token_type: TokenType, fn): self.infix_parse_fns[token_type] = fn
    def _next_token(self): self.current_token = self.peek_token; self.peek_token = self.lexer.next_token()
    def _peek_error(self, expected_type: TokenType): msg = f"语法错误 (L{self.peek_token.line} C{self.peek_token.col}): 期望 {expected_type.name}, 得到 {self.peek_token.type.name} ('{self.peek_token.literal}')"; self.errors.append(msg) # Keep original error format
    def _expect_peek(self, expected_type: TokenType) -> bool:
        if self.peek_token.type == expected_type: self._next_token(); return True
        else: self._peek_error(expected_type); return False
    def _current_token_is(self, t_type: TokenType) -> bool: return self.current_token.type == t_type
    def _peek_token_is(self, t_type: TokenType) -> bool: return self.peek_token.type == t_type
    def _peek_precedence(self) -> int: return PRECEDENCES.get(self.peek_token.type, LOWEST)
    def _current_precedence(self) -> int: return PRECEDENCES.get(self.current_token.type, LOWEST)
    def _no_prefix_parse_fn_error(self, t_type: TokenType): msg = f"语法错误 (L{self.current_token.line} C{self.current_token.col}): 无法解析 {self.current_token.type.name} ('{self.current_token.literal}'), 没有对应的前缀解析函数"; self.errors.append(msg) # Keep original error format

    # --- Main Parsing Method ---
    def parse_program(self) -> Program | None:
        statements = []
        while not self._current_token_is(TokenType.EOF):
            # Keep original top-level dispatch logic
            stmt = self._parse_statement() # Assume this handles top-level correctly in original
            if stmt:
                statements.append(stmt)
            elif self.errors:
                return None # Error occurred

        if self.errors: return None
        return Program(statements=statements)

    # --- Statement Parsing ---
    def _parse_statement(self) -> Statement | None:
        """
        Parses a single statement.
        It determines the type of statement based on the current token
         and dispatches to the appropriate parsing method.
        Each parsing method is responsible for consuming tokens
        belonging to that statement, including the final semicolon
        (unless it's a block expression etc.).
        """
        # print(f"DEBUG: _parse_statement: 开始, current={self.current_token}") # Debugging

        # --- Dispatch based on the current token ---

        if self._current_token_is(TokenType.LET):
            # 处理 let 声明语句
            return self._parse_let_statement()

        elif self._current_token_is(TokenType.RETURN):
            # 处理 return 语句
            return self._parse_return_statement()

        elif self._current_token_is(TokenType.FN):
            # 处理函数声明语句
            return self._parse_function_declaration_statement()

        elif self._current_token_is(TokenType.IF):
            # 处理 if 语句 (或 if 表达式)
            # 根据上下文判断是语句还是表达式是复杂的，这里简化处理。
            # 如果你的语法将 if 仅视为语句，那么这样分发是合理的。
            # 如果 if 可以是表达式，需要在 parse_expression 中处理。
            # 基于你的语法规则和AST节点 (IfStatement)，这里将其视为语句解析。
            return self._parse_if_statement()

        elif self._current_token_is(TokenType.WHILE):
            # 处理 while 循环语句
            return self._parse_while_statement()
        elif self._current_token_is(TokenType.LOOP): 
            return self._parse_loop_statement()
        # --- 新增：处理以 Identifier 开头，可能是赋值语句或表达式语句 ---
        # 需要预读下一个 Token 来区分 Identifier = ... 和 Identifier(...); 或 Identifier;
        # 注意：这个检查顺序很重要，要放在其他以 Identifier 开头的语句（如某些函数调用语法糖）之前
        elif self._current_token_is(TokenType.IDENT) and self._peek_token_is(TokenType.ASSIGN):
            # 处理赋值语句 Identifier = Expression;
            return self._parse_assignment_statement()
        # --- 新增结束 ---

        elif self._current_token_is(TokenType.LBRACE):
            # 处理块语句 { ... }
            # _parse_block_statement 会解析整个块，并将 current_token 停留在 '}' 上
            # _parse_statement 调用者 (如 _parse_block_statement 循环) 会在之后消耗 '}'
            return self._parse_block_statement()

        elif self._current_token_is(TokenType.SEMICOLON):
            # 处理空语句 ;
            self._next_token()  # 消耗 ';'
            # print(f"DEBUG: _parse_statement: 空语句消耗 ';' 后, current={self.current_token}") # Debugging
            return None  # 返回 None 表示一个有效的空结构

        # Add FOR, BREAK, CONTINUE here if they were in your original working version
        # elif self._current_token_is(TokenType.FOR): return self._parse_for_statement()
        elif self._current_token_is(TokenType.BREAK): 
            return self._parse_break_statement()
        elif self._current_token_is(TokenType.CONTINUE): 
            return self._parse_continue_statement()

        # 如果不匹配任何已知的语句开头，尝试将其解析为表达式语句
        # 表达式语句是 Expression ;
        # 或者在 Rust 中，块的最后一个 Expression 可以没有分号作为返回值
        else:
            # print(f"DEBUG: _parse_statement: 尝试作为表达式语句, current={self.current_token}") # Debugging
            return self._parse_expression_statement()

        # print(f"DEBUG: _parse_statement: 结束") # Debugging

    def _parse_let_statement(self) -> LetStatement | None:
        #print(f"DEBUG: _parse_let_statement: 开始, current={self.current_token}")
        let_token = self.current_token  # 'let'
        is_mut = False

        # 1. 处理 'mut' (如果存在)
        if self._peek_token_is(TokenType.MUT):
            is_mut = True
            self._next_token()  # 消耗 let, current_token 现在是 'mut'
            #print(f"DEBUG: _parse_let_statement: 'mut' 之前, current={self.current_token}")
            if not self._expect_peek(TokenType.IDENT): return None  # 消耗 'mut', current_token 现在是 IDENT
            #print(f"DEBUG: _parse_let_statement: 'mut' 之后, current={self.current_token} (应为 IDENT)")
        # 2. 处理 IDENT (如果 'mut' 不存在或已处理)
        elif self._peek_token_is(TokenType.IDENT):
            self._next_token()  # 消耗 'let', current_token 现在是 IDENT
            #print(f"DEBUG: _parse_let_statement: 'let' 之后, current={self.current_token} (应为 IDENT)")
        else:
            self._peek_error(TokenType.IDENT)
            return None

        # current_token 现在是 IDENT
        name_ident_token = self.current_token
        name_node = Identifier(token=name_ident_token, value=name_ident_token.literal)
        #print(f"DEBUG: _parse_let_statement: 获得 IDENT '{name_node.value}', current={self.current_token}")

        type_node = None
        # 3. 处理可选的类型注解 ': Type'
        if self._peek_token_is(TokenType.COLON):
            self._next_token()  # 消耗 IDENT, current_token 现在是 ':'
            #print(f"DEBUG: _parse_let_statement: 类型 ':' 之前, current={self.current_token}")
            self._next_token()  # 消耗 ':', current_token 现在是 Type 的开始
            #print(f"DEBUG: _parse_let_statement: 类型开始, current={self.current_token}")

            type_node = self._parse_type()
            if type_node is None: return None
            #print(f"DEBUG: _parse_let_statement: _parse_type 后, current={self.current_token} (应为类型最后Token)")

            self._next_token()  # 消耗 Type 的最后一个 token
            #print(f"DEBUG: _parse_let_statement: 消耗类型后, current={self.current_token}")
        else:
            self._next_token()  # 消耗 IDENT (如果没有类型注解)
            #print(f"DEBUG: _parse_let_statement: 无类型, 消耗 IDENT 后, current={self.current_token}")

        # 4. 处理可选的初始化器 '= Expression'
        value_node = None
        #print(f"DEBUG: _parse_let_statement: 检查赋值, current={self.current_token}")
        if self._current_token_is(TokenType.ASSIGN):
            assign_token = self.current_token
            self._next_token()  # 消耗 '=', current_token 现在是 Expression 的开始
            #print(f"DEBUG: _parse_let_statement: 赋值表达式开始, current={self.current_token}")

            value_node = self.parse_expression(LOWEST)
            if value_node is None: return None
            #print(f"DEBUG: _parse_let_statement: parse_expression 后, current={self.current_token} (应为表达式最后Token)")

            self._next_token()  # 消耗 Expression 的最后一个 token
            #print(f"DEBUG: _parse_let_statement: 消耗表达式后, current={self.current_token}")

        # 5. 期望并消耗分号 ';'
        #print(f"DEBUG: _parse_let_statement: 检查分号, current={self.current_token}")
        if not self._current_token_is(TokenType.SEMICOLON):
            msg = f"语法错误 (L{self.current_token.line} C{self.current_token.col}): 'let' 语句期望以 ';' 结束, 得到 {self.current_token.type.name} ('{self.current_token.literal}')"
            self.errors.append(msg)
            #print(f"DEBUG: _parse_let_statement: 分号错误!")
            return None

        self._next_token()  # 消耗 ';'
        #print(f"DEBUG: _parse_let_statement: 消耗分号后, current={self.current_token}")
        #print(f"DEBUG: _parse_let_statement: 结束")

        return LetStatement(token=let_token, name=name_node, value=value_node, type_info=type_node, mutable=is_mut)

    def _parse_return_statement(self) -> ReturnStatement | None:
        # Keep original token consumption
        return_token = self.current_token
        self._next_token()
        return_value_node = None
        if not self._current_token_is(TokenType.SEMICOLON):
            return_value_node = self.parse_expression(LOWEST) # Returns Node
            if return_value_node is None: return None
        if not self._current_token_is(TokenType.SEMICOLON):
             if not self._expect_peek(TokenType.SEMICOLON): return None
        else: self._next_token()
        return ReturnStatement(token=return_token, return_value=return_value_node)

    def _parse_expression_statement(self) -> ExpressionStatement | None:
        start_token = self.current_token
        expression_node = self.parse_expression(LOWEST)
        if expression_node is None:
            # 错误已由 parse_expression 或其子调用记录
            return None

        # parse_expression 将 current_token 停在表达式的最后一个 token。
        # 我们需要检查 *下一个* token 是否是分号。
        if not self._peek_token_is(TokenType.SEMICOLON):
            self._peek_error(TokenType.SEMICOLON)  # 记录错误：期望分号
            return None

        self._next_token()  # 消耗表达式的最后一个 token
        self._next_token()  # 消耗 ';' (现在 current_token 指向 ';' 之后)

        return ExpressionStatement(token=start_token, expression=expression_node)

    # parser.py
    def _parse_block_statement(self) -> BlockStatement | None:
        block_token = self.current_token  # '{' token
        if not self._current_token_is(TokenType.LBRACE):
            msg = f"内部错误: _parse_block_statement 在非 '{{' Token 调用 (L{self.current_token.line})"
            self.errors.append(msg)
            return None
        self._next_token()  # 消耗 '{', current_token 现在是块内第一个 token (或 '}')

        statements = []

        # 循环直到遇到 '}' 或 EOF
        while not self._current_token_is(TokenType.RBRACE) and not self._current_token_is(TokenType.EOF):
            # 在这里直接调用 _parse_statement
            # _parse_statement 会解析一个完整语句，并消耗包括分号在内的所有相关 token
            # 成功时，_parse_statement 会将 current_token 定位到下一个语句的开始 token
            # 如果发生错误，_parse_statement 会添加错误并返回 None
            stmt_node = self._parse_statement()

            if stmt_node:
                statements.append(stmt_node)
            elif self.errors:
                # 如果 _parse_statement 内部记录了错误，则传播错误
                return None
            # else: stmt_node is None and no errors.
            # This happens for empty statements ';'.
            # _parse_statement should have consumed the ';' and advanced current_token.
            # If current_token did NOT advance here (which should not happen if _parse_statement is correct),
            # the loop condition will prevent infinite loop if current_token is RBRACE or EOF,
            # otherwise it indicates a bug in _parse_statement where it didn't consume anything.

        # 循环结束时，current_token 应该是 '}' 或 EOF
        if not self._current_token_is(TokenType.RBRACE):
            # 如果不是 '}'，说明块没有正确闭合
            self._peek_error(TokenType.RBRACE)  # 报告期望 '}'
            return None

        # _parse_block_statement 在成功时，将 current_token 停留在 '}' 上。
        # 调用者 (_parse_while_statement, _parse_function_declaration_statement etc.)
        # 负责在调用 _parse_block_statement 之后检查并消耗这个 '}' Token。

        return BlockStatement(token=block_token, statements=statements)

    def _parse_while_statement(self) -> WhileStatement | None:
        """解析 while 语句: while Expression BlockStatement"""
        while_token = self.current_token  # 'while'
        self._next_token()  # 消耗 'while'

        # 解析条件表达式
        condition_node = self.parse_expression(LOWEST)
        if condition_node is None: return None

        # ---> **修正: 检查 LBRACE 的时机** <---
        # parse_expression 之后，current_token 应该是表达式的最后一个 token
        # 我们需要检查 *下一个* token (peek_token) 是否为 '{'
        if not self._peek_token_is(TokenType.LBRACE):
            self._peek_error(TokenType.LBRACE)
            return None
        self._next_token()  # 消耗表达式的最后一个 token，现在 current_token 是 '{'
        # ---> 修正结束 <---

        # 解析循环体
        body_node = self._parse_block_statement()  # 这个方法内部会消耗 '{' 和 '}'
        if body_node is None: return None

        # ---> **修正: 确保 block 消耗了 '}'** <---
        # _parse_block_statement 应该将 current_token 停留在 '}' 上
        # 我们需要在调用 _parse_block_statement 后消耗 '}'
        if not self._current_token_is(TokenType.RBRACE):
            # 如果 _parse_block_statement 实现正确，这里不应该发生
            msg = f"内部错误: _parse_block_statement 未在 '}}' 处停止 (L{self.current_token.line})"
            self.errors.append(msg)
            return None
        self._next_token()  # 消耗 '}'
        # ---> 修正结束 <---

        return WhileStatement(token=while_token, condition=condition_node, body=body_node)

    def _parse_assignment_statement(self) -> Statement | None:
        # 假设调用时 current_token 是 IDENT (Assignable 的开头)
        # 并且我们已经预读到下一个 Token 是 ASSIGN

        # 解析可赋值的目标 (目前只处理 Identifier)
        # current_token 已经是 IDENT
        target_node = self._parse_identifier()  # 返回 Identifier 节点
        if target_node is None:
            # 如果 _parse_identifier 返回 None，说明有错误
            return None

        # 期望并消耗 '='
        # _parse_identifier 应该没有消耗 IDENT，所以 current_token 还是 IDENT
        # 我们需要消耗 IDENT，然后期望 ASSIGN
        self._next_token()  # 消耗 IDENT，current_token 现在是 ASSIGN

        if not self._current_token_is(TokenType.ASSIGN):
            # 这个检查理论上不应该失败，因为 _parse_statement 已经预读过了
            msg = f"内部错误: _parse_assignment_statement 在非 '=' Token 调用 (L{self.current_token.line})"
            self.errors.append(msg)
            return None
        assign_token = self.current_token  # 保存 '=' token
        self._next_token()  # 消耗 '=', current_token 现在是 Expression 的开始

        # 解析赋值号右侧的表达式
        value_node = self.parse_expression(LOWEST)  # 解析 Expression，current_token 停在 Expression 最后
        if value_node is None: return None

        # 消耗表达式的最后一个 token
        self._next_token()  # current_token 现在是表达式之后

        # 期望并消耗分号 ';'
        if not self._current_token_is(TokenType.SEMICOLON):
            self._peek_error(TokenType.SEMICOLON)  # 报告期望分号
            return None
        self._next_token()  # 消耗 ';'

        # 返回 AssignmentStatement 节点
        # 你可能需要在 ast_nodes.py 中添加 AssignmentStatement 节点定义
        # class AssignmentStatement(Statement): ...
        # 并在 ast_printer.py 中添加对应的 visit 方法

        # For now, return a placeholder or a generic ExpressionStatement if AST node not added yet
        # Let's define a simple AssignmentStatement node for clarity

        return AssignmentStatement(token=target_node.token, target=target_node, value=value_node)  # 返回节点
        # 注意：如果 AssignmentStatement 节点没定义，这里会报错。

    def _parse_function_declaration_statement(self) -> FunctionDeclarationStatement | None:
        # Keep original token consumption carefully
        fn_token = self.current_token
        if not self._expect_peek(TokenType.IDENT): return None
        name_node = Identifier(token=self.current_token, value=self.current_token.literal)
        if not self._expect_peek(TokenType.LPAREN): return None
        params = self._parse_function_parameters() # Returns list[ParameterNode]
        if params is None: return None
        # Assume original _parse_function_parameters left current on ')'
        if not self._current_token_is(TokenType.RPAREN):
             self.errors.append(f"内部错误: _parse_function_parameters 未停留在 ')' L{self.current_token.line}")
             return None
        self._next_token() # Consume ')'
        return_type_node = None
        if self._current_token_is(TokenType.ARROW):
             self._next_token()
             return_type_node = self._parse_type() # Returns TypeNode
             if return_type_node is None: return None
             self._next_token() # Consume type's last token
        if not self._current_token_is(TokenType.LBRACE):
             msg = f"语法错误 (L{self.current_token.line} C{self.current_token.col}): 函数声明期望左花括号开始函数体, 得到 {self.current_token.type.name} ('{self.current_token.literal}')";
             self.errors.append(msg); return None
        body_node = self._parse_block_statement() # Returns BlockStatement
        if body_node is None: return None
        # Assume original consumed '}' after block here
        self._next_token() # Consume '}'
        return FunctionDeclarationStatement(token=fn_token, name=name_node, params=params, body=body_node, return_type=return_type_node)

    def _parse_loop_statement(self) -> LoopStatement | None:
        """Parses a loop statement: loop BlockStatement"""
        loop_token = self.current_token  # 'loop' token
        self._next_token() # Consume 'loop'. Current token should now be '{'.

        if not self._current_token_is(TokenType.LBRACE):
            msg = f"语法错误 (L{self.current_token.line} C{self.current_token.col}): 'loop' 语句体期望以 '{{' 开始, 得到 {self.current_token.type.name} ('{self.current_token.literal}')"
            self.errors.append(msg)
            return None

        # current_token is LBRACE, _parse_block_statement expects this
        body_node = self._parse_block_statement()
        if body_node is None:
            return None # Error already reported by _parse_block_statement or its children

        # _parse_block_statement leaves current_token on RBRACE
        # We need to consume it.
        self._next_token() # Consume RBRACE

        return LoopStatement(token=loop_token, body=body_node)

    def _parse_break_statement(self) -> BreakStatement | None:
        """Parses a break statement: break;"""
        break_token = self.current_token  # 'break' token

        # For `break;`, we just need to consume 'break' and expect a semicolon.
        self._next_token() # Consume 'break'

        if not self._current_token_is(TokenType.SEMICOLON):
            msg = f"语法错误 (L{self.current_token.line} C{self.current_token.col}): 'break' 语句期望以 ';' 结束, 得到 {self.current_token.type.name} ('{self.current_token.literal}')"
            self.errors.append(msg)
            return None
        
        self._next_token() # Consume ';'
        return BreakStatement(token=break_token)

    def _parse_continue_statement(self) -> ContinueStatement | None:
        """Parses a continue statement: continue;"""
        continue_token = self.current_token # 'continue' token

        self._next_token() # Consume 'continue'

        if not self._current_token_is(TokenType.SEMICOLON):
            msg = f"语法错误 (L{self.current_token.line} C{self.current_token.col}): 'continue' 语句期望以 ';' 结束, 得到 {self.current_token.type.name} ('{self.current_token.literal}')"
            self.errors.append(msg)
            return None

        self._next_token() # Consume ';'
        return ContinueStatement(token=continue_token)

    def _parse_function_parameters(self) -> list[ParameterNode] | None:
        # Keep original token consumption carefully, ensure it leaves current on ')' if successful
        params = []
        if self._peek_token_is(TokenType.RPAREN):
            # Original working version *must* have advanced here to make caller work
            self._next_token() # Consume '(', current becomes ')'
            return params
        self._next_token() # Consume '('
        while True:
            param_token = self.current_token; is_mut = False; name_token = None
            if self._current_token_is(TokenType.MUT):
                is_mut = True
                if not self._expect_peek(TokenType.IDENT): return None
                name_token = self.current_token
            elif self._current_token_is(TokenType.IDENT): name_token = self.current_token
            else: msg = f"语法错误 (L{self.current_token.line} C{self.current_token.col}): 期望参数名 (IDENT) 或 'mut', 得到 {self.current_token.type.name} ('{self.current_token.literal}')"; self.errors.append(msg); return None
            name_node = Identifier(token=name_token, value=name_token.literal)
            if not self._expect_peek(TokenType.COLON): return None
            self._next_token() # Consume ':'
            type_node = self._parse_type() # Returns TypeNode
            if type_node is None: return None
            params.append(ParameterNode(token=param_token, name=name_node, type_info=type_node, mutable=is_mut))
            # Keep original check logic for ',' or ')'
            if self._peek_token_is(TokenType.COMMA):
                 self._next_token(); self._next_token() # Consume type, consume ','
                 if self._current_token_is(TokenType.RPAREN): msg = f"语法错误 (L{self.current_token.line} C{self.current_token.col}): 参数列表中逗号后不能直接跟 ')', 期望参数"; self.errors.append(msg); return None
            elif self._peek_token_is(TokenType.RPAREN):
                 self._next_token() # Consume type's last token
                 break # current is now ')'
            else: self._peek_error(TokenType.RPAREN); return None
        return params

    def _parse_type(self) -> TypeNode | None:
        """Parses a type annotation. Returns TypeNode or None.
        Leaves current_token on the last token of the parsed type."""
        type_token = self.current_token
        if self._current_token_is(TokenType.I32):
            # Return node, DO NOT advance token here.
            return TypeNode(token=type_token, name="i32")
        # Add elif for other types (&, [], () etc.) ensuring they also
        # leave current_token on their respective last token.
        else:
            msg = f"语法错误 (L{type_token.line} C{type_token.col}): 期望类型名 (如 i32), 得到 {type_token.type.name} ('{type_token.literal}')"
            self.errors.append(msg)
            return None


    def _parse_if_statement(self) -> IfStatement | None:
        # Keep original token consumption carefully
        if_token = self.current_token
        self._next_token()
        condition = self.parse_expression(LOWEST) # Returns Node
        if condition is None: return None
        # Assume original logic advanced correctly before checking '{'
        if not self._peek_token_is(TokenType.LBRACE): self._peek_error(TokenType.LBRACE); return None
        self._next_token() # Consume condition last token
        # Current is '{'
        consequence = self._parse_block_statement() # Returns Node, leaves current on '}'
        if consequence is None: return None
        self._next_token() # Consume '}'
        alternative = None
        if self._current_token_is(TokenType.ELSE):
            self._next_token()
            if self._current_token_is(TokenType.IF): alternative = self._parse_if_statement() # Returns Node
            elif self._current_token_is(TokenType.LBRACE):
                 alternative = self._parse_block_statement() # Returns Node, leaves current on '}'
                 if alternative is None: return None
                 self._next_token() # Consume '}'
            else: msg = f"语法错误 (L{self.current_token.line} C{self.current_token.col}): 'else' 后期望 'if' 或 '{{', 得到 {self.current_token.type.name} ('{self.current_token.literal}')"; self.errors.append(msg); return None
            if alternative is None and self.errors: return None
        return IfStatement(token=if_token, condition=condition, consequence=consequence, alternative=alternative)


    # --- Expression Parsing (Pratt - Modify Returns Only) ---
    def parse_expression(self, precedence: int) -> Expression | None:
        # Keep original Pratt logic flow *exactly*
        prefix_fn = self.prefix_parse_fns.get(self.current_token.type)
        if prefix_fn is None: self._no_prefix_parse_fn_error(self.current_token.type); return None

        # Assume prefix_fn returns AST node, use original token consumption
        left_expr = prefix_fn() # Returns Node
        if left_expr is None: return None

        # Keep original loop condition and infix handling logic
        while not self._peek_token_is(TokenType.SEMICOLON) and precedence < self._peek_precedence(): # Review terminators if needed
            infix = self.infix_parse_fns.get(self.peek_token.type)
            if infix is None: return left_expr

            self._next_token() # Assume original loop advanced operator here

            left_expr = infix(left_expr) # infix returns updated node
            if left_expr is None: return None

        return left_expr


    # --- Prefix Parsing Functions (Modify Returns Only) ---
    def _parse_identifier(self) -> Identifier | None:
        return Identifier(token=self.current_token, value=self.current_token.literal)
    def _parse_integer_literal(self) -> IntegerLiteral | None:
        try: value = int(self.current_token.literal)
        except ValueError: msg = f"语法错误 (L{self.current_token.line} C{self.current_token.col}): 无法将 '{self.current_token.literal}' 转换为整数"; self.errors.append(msg); return None
        return IntegerLiteral(token=self.current_token, value=value)
    def _parse_prefix_expression(self) -> PrefixExpression | None:
        # Keep original token consumption
        operator_token = self.current_token; operator = operator_token.literal
        self._next_token() # Assume original consumed operator here
        right_node = self.parse_expression(PREFIX) # Returns Node
        if right_node is None: return None
        return PrefixExpression(token=operator_token, operator=operator, right=right_node)
    def _parse_grouped_expression_or_tuple(self) -> Expression | None:
         # Keep original token consumption
         self._next_token() # Assume consumes '('
         # TODO: Add tuple parsing if original had it
         expr_node = self.parse_expression(LOWEST) # Returns Node
         if expr_node is None: return None
         if not self._expect_peek(TokenType.RPAREN): return None # Assume consumes ')'
         return expr_node
    def _parse_if_expression(self) -> IfStatement | None: # Treat 'if' expr like statement for now
         return self._parse_if_statement()


    # --- Infix Parsing Functions (Modify Returns Only) ---
    def _parse_infix_expression(self, left_expr: Expression) -> InfixExpression | None:
        # Keep original token consumption
        # left_expr is node, current_token is operator
        infix_token = self.current_token; operator = infix_token.literal
        precedence = self._current_precedence()
        self._next_token() # Assume original consumed operator here
        right_expr = self.parse_expression(precedence) # Returns Node
        if right_expr is None: return None
        return InfixExpression(token=infix_token, left=left_expr, operator=operator, right=right_expr)

    def _parse_call_expression(self, function_expr: Expression) -> CallExpression | None:
        # function_expr 是函数名/表达式节点
        # current_token 是 '('
        call_lparen_token = self.current_token  # 保存 '(' token

        arguments = self._parse_call_arguments() # 这个方法现在负责消耗到 ')' (并停在 ')' 上)
        if arguments is None: return None

        # _parse_call_arguments 结束后，current_token 应该已经是 ')' 了
        if not self._current_token_is(TokenType.RPAREN):
            # 内部错误
            # ...
            return None

        # 返回 CallExpression 节点。current_token 保持在 ')' 上。
        # Pratt 解析器的主循环会在之后处理这个 ')' (比如，如果后面没有更高优先级的操作符，
        # parse_expression 就会返回，此时 current_token 依然是 ')')
        return CallExpression(token=call_lparen_token, function=function_expr, arguments=arguments)

    def _parse_call_arguments(self) -> list[Expression] | None:
        """Parses function call arguments between '(' and ')'.
        Assumes current_token is '('.
        Advances tokens past arguments and commas.
        Leaves current_token pointing AT the ')' upon successful return.
        Returns None on error.
        """
        args = []

        # current_token 应该是 '('
        if not self._current_token_is(TokenType.LPAREN):
             # ... 错误处理 ...
             return None

        # 处理空参数列表: call()
        if self._peek_token_is(TokenType.RPAREN):
            self._next_token()  # 消耗 '(', current_token 现在是 ')'
            return args # 成功, current_token 是 ')'

        self._next_token()  # 消耗 '(', current_token 现在是第一个参数的开始

        # 解析第一个参数
        first_arg_node = self.parse_expression(LOWEST)
        if first_arg_node is None: return None
        args.append(first_arg_node)
        # current_token 现在是第一个参数的最后一个 token

        # 循环处理后续参数
        while self._peek_token_is(TokenType.COMMA):
            self._next_token()  # 消耗上一个参数的最后一个 token
            self._next_token()  # 消耗 ','
            # ... 检查逗号后不能直接跟 ')' ...
            arg_node = self.parse_expression(LOWEST)
            if arg_node is None: return None
            args.append(arg_node)
            # current_token 现在是当前参数的最后一个 token

        # 循环结束后，current_token 停留在最后一个参数的最后一个 token
        # 期望下一个 token 是 ')'
        if not self._peek_token_is(TokenType.RPAREN):
            self._peek_error(TokenType.RPAREN)
            return None

        self._next_token()  # 消耗最后一个参数的最后一个 token，使得 current_token 现在是 ')'

        return args # 成功, current_token 是 ')'

    # --- Placeholders for unimplemented features (Return None) ---
    # ... (Keep placeholders as they were in the working version, just return None) ...
    def _parse_array_literal(self): self.errors.append("数组字面量解析未实现"); return None
    # def _parse_loop_expression(self): self.errors.append("Loop 表达式解析未实现"); return None
    def _parse_index_expression(self, left): self.errors.append("索引表达式解析未实现"); return None
    def _parse_access_expression(self, left): self.errors.append("访问表达式解析未实现"); return None
    # def _parse_if_statement(self): return self._parse_if_expression() # Reuse if treating as expr

    # def _parse_loop_statement(self): self.errors.append("Loop 语句解析未实现"); return None
    def _parse_for_statement(self): self.errors.append("For 语句解析未实现"); return None
    # def _parse_break_statement(self): self.errors.append("Break 语句解析未实现"); return None
    # def _parse_continue_statement(self): self.errors.append("Continue 语句解析未实现"); return None
    def _is_if_expression_context(self): return False # Keep original context logic
    def _is_loop_expression_context(self): return False # Keep original context logic