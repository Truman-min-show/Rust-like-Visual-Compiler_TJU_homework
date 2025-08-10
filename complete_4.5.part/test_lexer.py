
from token_defs import TokenType, Token
from lexer import Lexer

# 放入你的 Rust 测试代码
TEST_CODE = """
fn factorial(n: i32) -> i32 {
    // This is a comment
    let mut result: i32 = 1;
    let mut i: i32 = 1; /* Another
       comment style */
    while i <= n {
        result = result * i; // calculation
        i = i + 1;
    }
    return result;
}

fn main() {
    let x: i32 = 5;
    let mut fact_x: i32;
    fact_x = factorial(x);
    if x >= 1 && x != 0 { // && and || not in grammar yet
        loop { break; }
    }
    let arr = [1, 2];
    let tup = (10,); // Tuple
    let y = tup.0;
    let z = arr[0];
    let range = 1..10;
}
"""

def run_lexer_test():
    print("--- Running Lexer Test ---")
    lexer = Lexer(TEST_CODE)
    token_count = 0
    while True:
        tok = lexer.next_token()
        print(f"Line {tok.line:<2} Col {tok.col:<3}: {tok.type.name:<10} | Literal: '{tok.literal}'")
        token_count += 1
        if tok.type == TokenType.EOF:
            break
        if tok.type == TokenType.ILLEGAL:
            print(f"*** ILLEGAL TOKEN FOUND: '{tok.literal}' ***")

    print(f"\n--- Lexing Finished. Total Tokens: {token_count} ---")

if __name__ == "__main__":
    run_lexer_test()