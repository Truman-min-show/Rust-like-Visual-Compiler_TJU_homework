
from token_defs import TokenType, Token, lookup_ident # 确保导入

class Lexer:
    def __init__(self, input_code: str):
        self.input = input_code
        self.position = 0          # 当前字符在 input 中的索引
        self.read_position = 0     # 下一个要读取的字符的索引 (position + 1)
        self.ch = ''               # 当前正在检查的字符
        self.line = 1              # 当前行号
        self.col = 0               # 当前列号 (相对于当前行)
        self._read_char()          # 初始化 self.ch, self.position, self.read_position

    def _read_char(self):
        """读取输入中的下一个字符，并更新位置指针"""
        if self.read_position >= len(self.input):
            self.ch = ''  # 表示文件结束 (EOF)
        else:
            self.ch = self.input[self.read_position]

        self.position = self.read_position
        self.read_position += 1
        self.col += 1

    def _peek_char(self) -> str:
        """查看下一个字符，但不移动指针 (用于预读)"""
        if self.read_position >= len(self.input):
            return ''
        else:
            return self.input[self.read_position]

    def _skip_whitespace_and_comments(self):
        """跳过空白字符和注释 (包括换行符和不同类型的注释)"""
        while True:
            if self.ch.isspace():
                if self.ch == '\n':
                    self.line += 1
                    self.col = 0 # 换行后重置列号
                self._read_char()
            elif self.ch == '/' and self._peek_char() == '/': # 处理单行注释 //
                # 一直读到行尾或文件尾
                while self.ch != '\n' and self.ch != '':
                    self._read_char()
                # _read_char 会停在换行符或EOF，继续循环以跳过换行符或处理下一个空白/注释
                continue # 非常重要，防止跳过换行符后直接退出循环
            elif self.ch == '/' and self._peek_char() == '*': # 处理多行注释 /* ... */
                self._read_char() # 消耗 /
                self._read_char() # 消耗 *
                start_line, start_col = self.line, self.col # 记录注释开始位置以防未闭合
                while not (self.ch == '*' and self._peek_char() == '/'):
                    if self.ch == '\n':
                        self.line += 1
                        self.col = 0
                    if self.ch == '': # 检测到未闭合的注释
                        print(f"警告: 在 {start_line}:{start_col} 开始的多行注释未闭合")
                        break
                    self._read_char()
                if self.ch == '*': # 消耗 *
                    self._read_char()
                if self.ch == '/': # 消耗 /
                     self._read_char()
                continue # 继续检查注释或空白后的内容
            else:
                # 如果不是空白或注释的开始，则停止跳过
                break

    def _read_identifier(self) -> str:
        """读取一个完整的标识符"""
        start_pos = self.position
        # 第一个字符必须是 is_letter (包括 '_')
        if not self.is_letter(self.ch):
             return "" # Should not happen if called correctly

        # 后续字符可以是 is_letter 或数字
        while self.is_letter(self.ch) or self.ch.isdigit():
            self._read_char()
        return self.input[start_pos:self.position]

    def _read_number(self) -> str:
        """读取一个完整的整数"""
        start_pos = self.position
        while self.ch.isdigit():
            self._read_char()
        return self.input[start_pos:self.position]

    def is_letter(self, char: str) -> bool:
        """判断字符是否是标识符的合法组成部分 (字母或下划线)"""
        return 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char == '_'

    def next_token(self) -> Token:
        """获取并返回下一个 Token"""
        self._skip_whitespace_and_comments()

        tok = None
        start_line, start_col = self.line, self.col # 记录 Token 的起始位置

        # --- 根据当前字符 self.ch 确定 Token 类型 ---
        match self.ch:
            case '=':
                if self._peek_char() == '=': # '=='
                    ch = self.ch
                    self._read_char() # 消耗第二个 '='
                    literal = ch + self.ch
                    tok = Token(TokenType.EQ, literal, start_line, start_col)
                else: # '='
                    tok = Token(TokenType.ASSIGN, self.ch, start_line, start_col)
            case '+':
                tok = Token(TokenType.PLUS, self.ch, start_line, start_col)
            case '-':
                 if self._peek_char() == '>': # '->'
                     ch = self.ch
                     self._read_char() # 消耗 '>'
                     literal = ch + self.ch
                     tok = Token(TokenType.ARROW, literal, start_line, start_col)
                 else: # '-'
                     tok = Token(TokenType.MINUS, self.ch, start_line, start_col)
            case '*':
                tok = Token(TokenType.ASTERISK, self.ch, start_line, start_col)
            case '/':
                # 注释已被 _skip_whitespace_and_comments 处理
                 tok = Token(TokenType.SLASH, self.ch, start_line, start_col)
            case '<':
                 if self._peek_char() == '=': # '<='
                     ch = self.ch
                     self._read_char() # 消耗 '='
                     literal = ch + self.ch
                     tok = Token(TokenType.LTE, literal, start_line, start_col)
                 else: # '<'
                    tok = Token(TokenType.LT, self.ch, start_line, start_col)
            case '>':
                 if self._peek_char() == '=': # '>='
                     ch = self.ch
                     self._read_char() # 消耗 '='
                     literal = ch + self.ch
                     tok = Token(TokenType.GTE, literal, start_line, start_col)
                 else: # '>'
                     tok = Token(TokenType.GT, self.ch, start_line, start_col)
            case '!':
                 if self._peek_char() == '=': # '!='
                     ch = self.ch
                     self._read_char() # 消耗 '='
                     literal = ch + self.ch
                     tok = Token(TokenType.NOT_EQ, literal, start_line, start_col)
                 else: # '!' 单独出现暂时视为非法，或定义为 BANG
                     tok = Token(TokenType.ILLEGAL, self.ch, start_line, start_col)
                     # tok = Token(TokenType.BANG, self.ch, start_line, start_col)
            case '&':
                tok = Token(TokenType.AMPERSAND, self.ch, start_line, start_col)
            case '.':
                 if self._peek_char() == '.': # '..'
                     ch = self.ch
                     self._read_char() # 消耗第二个 '.'
                     literal = ch + self.ch
                     if self._peek_char() == '=': # '..='
                         self._read_char() # 消耗第二个 '.'
                         literal += self.ch
                         tok = Token(TokenType.DOTDOT_EQ, literal, start_line, start_col)
                     else: # '..'
                         tok = Token(TokenType.DOTDOT, literal, start_line, start_col)
                 else: # '.'
                     tok = Token(TokenType.DOT, self.ch, start_line, start_col)
            case '(':
                tok = Token(TokenType.LPAREN, self.ch, start_line, start_col)
            case ')':
                tok = Token(TokenType.RPAREN, self.ch, start_line, start_col)
            case '{':
                tok = Token(TokenType.LBRACE, self.ch, start_line, start_col)
            case '}':
                tok = Token(TokenType.RBRACE, self.ch, start_line, start_col)
            case '[':
                tok = Token(TokenType.LBRACKET, self.ch, start_line, start_col)
            case ']':
                tok = Token(TokenType.RBRACKET, self.ch, start_line, start_col)
            case ',':
                tok = Token(TokenType.COMMA, self.ch, start_line, start_col)
            case ':':
                tok = Token(TokenType.COLON, self.ch, start_line, start_col)
            case ';':
                tok = Token(TokenType.SEMICOLON, self.ch, start_line, start_col)

            # --- 处理标识符和数字 ---
            case _ if self.is_letter(self.ch): # 检查是否是标识符的开头
                literal = self._read_identifier() # 读取完整的标识符
                token_type = lookup_ident(literal) # 判断是关键字还是普通标识符
                # 注意: _read_identifier 已经移动了指针，不需要再调用 _read_char()
                return Token(token_type, literal, start_line, start_col)
            case _ if self.ch.isdigit(): # 检查是否是数字开头
                literal = self._read_number() # 读取完整的数字
                # 注意: _read_number 已经移动了指针
                return Token(TokenType.INT, literal, start_line, start_col)

            # --- 处理文件结束 ---
            case '': # 文件结束符
                tok = Token(TokenType.EOF, "", start_line, start_col)

            # --- 处理无法识别的字符 ---
            case _:
                tok = Token(TokenType.ILLEGAL, self.ch, start_line, start_col)

        # 对于上面 case 中匹配到的单个或两个字符的 token，需要移动到下一个字符
        self._read_char()
        return tok