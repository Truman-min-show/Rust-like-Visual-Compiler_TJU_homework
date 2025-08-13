
import enum
from collections import namedtuple

class TokenType(enum.Enum):
    # Keywords
    LET = 'let'
    MUT = 'mut'
    FN = 'fn'
    RETURN = 'return'
    IF = 'if'
    ELSE = 'else'
    WHILE = 'while'
    FOR = 'for'
    IN = 'in'
    LOOP = 'loop'
    BREAK = 'break'
    CONTINUE = 'continue'
    I32 = 'i32' # Type keyword

    # Identifiers & Literals
    IDENT = 'IDENT' # Identifier
    INT = 'INT'     # Integer literal

    # Operators
    ASSIGN = '='
    PLUS = '+'
    MINUS = '-'
    ASTERISK = '*'
    SLASH = '/'
    EQ = '=='      # Equal
    NOT_EQ = '!='   # Not equal
    LT = '<'       # Less than
    LTE = '<='      # Less than or equal
    GT = '>'       # Greater than
    GTE = '>='      # Greater than or equal
    BANG = '!'      # Used in != (potentially future NOT)
    AMPERSAND = '&' # Reference/Borrow
    DOT = '.'       # Field access
    DOTDOT = '..'   # Range
    DOTDOT_EQ = '..=' # Inclusive Range
    ARROW = '->'    # Function return type

    # Delimiters
    LPAREN = '('
    RPAREN = ')'
    LBRACE = '{'
    RBRACE = '}'
    LBRACKET = '['
    RBRACKET = ']'

    # Separators
    COMMA = ','
    COLON = ':'
    SEMICOLON = ';'

    # End of File
    EOF = 'EOF'

    # Illegal token (for errors)
    ILLEGAL = 'ILLEGAL'

# 使用 namedtuple 存储 Token 信息
# 参数: type (TokenType), literal (原始字符串), line (行号), col (列号)
Token = namedtuple('Token', ['type', 'literal', 'line', 'col'])

# 关键字字典
KEYWORDS = {
    "let": TokenType.LET,
    "mut": TokenType.MUT,
    "fn": TokenType.FN,
    "return": TokenType.RETURN,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "in": TokenType.IN,
    "loop": TokenType.LOOP,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "i32": TokenType.I32,
}

def lookup_ident(ident: str) -> TokenType:
    """检查标识符是否是关键字"""
    return KEYWORDS.get(ident, TokenType.IDENT)