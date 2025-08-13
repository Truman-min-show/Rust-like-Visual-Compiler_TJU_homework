# symbol_table.py
class Type:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if not isinstance(other, Type):
            return False
        return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Type('{self.name}')"

class FunctionType(Type):
    def __init__(self, param_types, return_type):
        super().__init__("function")
        self.param_types = param_types
        self.return_type = return_type

    def __eq__(self, other):
        if not isinstance(other, FunctionType):
            return False
        return self.param_types == other.param_types and self.return_type == other.return_type

    def __str__(self):
        params_str = ", ".join(map(str, self.param_types))
        return f"fn({params_str}) -> {str(self.return_type)}"

    def __repr__(self):
        return f"FunctionType(param_types={self.param_types!r}, return_type={self.return_type!r})"

TYPE_I32 = Type("i32")
TYPE_BOOL = Type("bool")
TYPE_VOID = Type("void")
TYPE_UNKNOWN = Type("unknown")
TYPE_ERROR = Type("error")

class ReferenceType(Type):
    def __init__(self, referenced_type: Type, is_mutable_ref: bool):
        # 构建类型名称，如 &i32 或 &mut i32
        ref_name = f"&{'mut ' if is_mutable_ref else ''}{str(referenced_type)}"
        super().__init__(ref_name)
        self.referenced_type = referenced_type
        self.is_mutable_ref = is_mutable_ref

    def __eq__(self, other):
        if not isinstance(other, ReferenceType):
            return False
        return (self.referenced_type == other.referenced_type and
                self.is_mutable_ref == other.is_mutable_ref)

    def __repr__(self):
        return f"ReferenceType(referenced_type={self.referenced_type!r}, is_mutable_ref={self.is_mutable_ref})"

class Symbol:
    def __init__(self, name, sym_type: Type, kind: str, is_mutable: bool = False,
                 scope_level: int = 0, attributes=None, is_initialized: bool = False): # ADDED is_initialized
        self.name = name
        self.type = sym_type
        self.kind = kind
        self.is_mutable = is_mutable
        self.scope_level = scope_level
        self.attributes = attributes if attributes else {}
        self.is_initialized = is_initialized # ADDED

    def __str__(self):
        return (f"<Symbol(name='{self.name}', type='{str(self.type)}', kind='{self.kind}', "
                f"mut={self.is_mutable}, scope={self.scope_level}, init={self.is_initialized}, attr={self.attributes})>") # ADDED init

class SymbolTable:
    def __init__(self):
        self.scopes = [{}]
        self.current_scope_level = -1
        self.semantic_errors = []

    def enter_scope(self):
        self.current_scope_level += 1
        self.scopes.append({})

    def exit_scope(self):
        if self.current_scope_level >= 0:
            self.scopes.pop()
            self.current_scope_level -= 1
        else:
            self.add_error("Internal Error: Attempted to exit non-existent scope.")

    def define(self, name: str, sym_type: Type, kind: str, is_mutable: bool = False,
               attributes=None, node_for_error=None, is_initialized: bool = False): # ADDED is_initialized parameter
        if self.current_scope_level < 0:
             self.add_error("Internal Error: Attempting to define symbol outside any scope.", node_for_error)
             return None

        # For function redefinition check (optional, Rust allows some forms of overloading/shadowing but simple check is fine)
        # if kind == "function" and name in self.scopes[self.current_scope_level]:
        #     self.add_error(f"Function '{name}' redefined in the same scope.", node_for_error)
            # If strict no-redefinition for functions, you might return early or flag symbol.

        symbol = Symbol(name, sym_type, kind, is_mutable, self.current_scope_level, attributes, is_initialized) # PASS is_initialized
        self.scopes[self.current_scope_level][name] = symbol # This handles shadowing naturally
        return symbol

    def resolve(self, name: str):
        for i in range(self.current_scope_level, -1, -1):
            if name in self.scopes[i]:
                return self.scopes[i][name]
        return None

    def add_error(self, message: str, node=None):
        if node and hasattr(node, 'token') and node.token:
            line = getattr(node.token, 'line', '?')
            col = getattr(node.token, 'col', '?')
            self.semantic_errors.append(f"Semantic Error (L{line} C{col}): {message}")
        else:
            self.semantic_errors.append(f"Semantic Error: {message}")