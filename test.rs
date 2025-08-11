// 规则 1.4, 1.5: 定义一个带输入和输出的函数
fn factorial(n: i32) -> i32 {
    // 规则 2.3: 变量声明并赋值
    let mut result: i32 = 1;
    
    // 规则 5.2: for 循环语句
    for i in 1..n+1 {
        // 规则 2.2: 赋值语句
        result = result * i;
    }
    
    // 规则 1.3, 1.5: 返回一个表达式的值
    return result;
}

// 规则 1.1: 程序主入口
fn main() -> i32 {
    // --- 1. 变量声明与函数调用 ---
    
    // 规则 6.1: 声明不可变变量
    let num: i32 = 4;

    // 规则 2.1: 声明一个可变变量
    let mut final_result: i32;
    
    // 规则 3.3 (函数调用) 和 2.2 (赋值)
    final_result = factorial(num); // final_result = 4! = 24

    // --- 2. while 循环 ---
    
    // 规则 2.3: 声明并初始化可变变量
    let mut i: i32 = 0;
    // 规则 5.1: while 循环
    while i < 3 {
        i = i + 1;
        // 规则 4.1 (if) 和 5.4 (continue)
        if i == 2 {
            continue;
        }
        final_result = final_result + i;
    }
    // 循环后: final_result = 24 + 1 + 3 = 28

    // --- 3. loop 循环 ---
    
    i = 0; // 重置计数器
    // 规则 5.3: loop 无限循环
    loop {
        i = i + 1;
        final_result = final_result - 1;
        // 规则 4.1 (if) 和 5.4 (break)
        if i == 5 {
            break;
        }
    }
    // 循环后: final_result = 28 - 5 = 23

    // --- 4. 引用和解引用 ---
    
    // 规则 6.2: 借用
    let ref_to_result = &final_result;
    // 规则 6.2: 解引用 (在表达式右侧)
    let temp_val = *ref_to_result + 10; // temp_val = 23 + 10 = 33

    // --- 5. if 语句和最终返回 ---

    // 规则 4.1: if 选择结构
    if temp_val > 30 {
        final_result = temp_val; // final_result 现在是 33
    }

    // ==========================================================
    // --- 已实现规则的错误分析 ---
    // 以下代码块用于测试编译器在遇到已支持语法中的
    // 语法或语义错误时，是否能给出正确的错误提示。
    // ==========================================================

    // --- 规则 1.5 / 1.3: 返回语句错误 ---
    // fn test_return_1() -> i32 {
    //     return; // 语义错误: 函数声明返回 i32，但返回了空类型 
    // }
    // fn test_return_2() {
    //     return 1; // 语义错误: 函数未声明返回类型，但返回了 i32 
    // }
    // fn test_return_3() {
    //     return 1 // 语法错误: <返回语句> 必须以分号结尾 
    // }

    
    // --- 规则 2.2 / 6.1: 赋值错误 ---
    // let immutable_var = 100;
    // immutable_var = 200; // 语义错误: 不可变变量不可二次赋值 
    // undeclared_var = 50; // 语义错误: 不能给未声明的变量赋值 
    

    // --- 规则 3.3: 函数调用错误 ---
    // factorial(); // 语义错误: 实参数量(0)与形参数量(1)不一致 
    //
    // fn no_return_func() {}
    // let val = no_return_func(); // 语义错误: 无返回值的函数不能作为右值 


    // --- 规则 4.1: if 语句语法错误 ---
    // if 1 > 0  // 语法错误: if 条件后必须跟一个 '{...}' 包围的语句块 
    //     final_result = 0;
    
    
    // --- 规则 5.4: break/continue 语义错误 ---
    // if 1 > 0 {
    //     break; // 语义错误: break 必须出现在循环体内 
    // }
    // continue; // 语义错误: continue 必须出现在循环体内 


    // --- 规则 6.2: 引用和解引用错误 ---
    // let non_ref_var = 10;
    // let deref_error = *non_ref_var; // 语义错误: 不允许对非引用类型进行解引用 
    //
    // let immutable_ref_source = 20;
    // let mut_ref_error = &mut immutable_ref_source; // 语义错误: 仅允许从可变变量创建可变引用 
    
    return final_result; // 预期最终返回 33
}