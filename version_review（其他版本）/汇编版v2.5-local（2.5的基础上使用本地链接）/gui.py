# gui.py
import tkinter as tk
from tkinter import scrolledtext, font as tkFont, PanedWindow, ttk, Toplevel, Label, messagebox
import io
from PIL import Image, ImageTk
import subprocess
import os
import sys
from datetime import datetime

try:
    from token_defs import TokenType, Token
    from lexer import Lexer
    from parser import Parser
    from ast_printer import ASTPrinter
    from ast_drawer import ASTGraphvizDrawer
    from symbol_table import SymbolTable, Type, FunctionType, TYPE_I32, TYPE_BOOL, TYPE_VOID, TYPE_UNKNOWN, TYPE_ERROR
    from semantic_analyzer import SemanticAnalyzer
    from assembly_generator import AssemblyGenerator
    from ir_generator import IRGenerator, Quadruple, IRValue # 假设 Quadruple 和 IRValue 在 ir_generator.py 中
except ImportError as e:
    print(f"错误：无法导入所需模块: {e}")
    print("请确保 token_defs.py, lexer.py, parser.py, ast_nodes.py, ast_printer.py, "
          "symbol_table.py, semantic_analyzer.py, ir_generator.py 文件存在且位于同一目录。")
    exit()

def get_base_path():
    """
    获取资源的基准路径。
    - 如果作为 .exe 运行，则返回 .exe 文件所在的目录。
    - 如果作为 .py 脚本运行，则返回脚本所在的目录。
    """
    if getattr(sys, 'frozen', False):
        # 程序被 PyInstaller 打包成了 exe
        return os.path.dirname(sys.executable)
    else:
        # 程序作为正常的 .py 脚本运行
        return os.path.dirname(os.path.abspath(__file__))



class LexerApp:
    def __init__(self, master):
        self.master = master
        master.title("Rust 编译器 (词法/语法/语义分析 + IR/汇编生成 + 自动链接并执行/AST图像生成)") # 更新标题
        master.geometry("1000x700")

        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=11)
        text_font = tkFont.Font(family="Consolas", size=11)

        self.m = PanedWindow(master, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
        self.m.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = tk.Frame(self.m, bd=2, relief=tk.SUNKEN)
        self.m.add(left_frame, width=450)

        input_label = tk.Label(left_frame, text="输入类 Rust 代码:", font=default_font)
        input_label.pack(pady=(0, 5), fill=tk.X)
        input_frame = tk.Frame(left_frame)
        input_frame.pack(fill=tk.BOTH, expand=True)

        input_scroll_x = tk.Scrollbar(input_frame, orient=tk.HORIZONTAL)
        input_scroll_y = tk.Scrollbar(input_frame, orient=tk.VERTICAL)

        self.input_text = tk.Text(input_frame, wrap=tk.NONE, undo=True, font=text_font,
                                  yscrollcommand=input_scroll_y.set,
                                  xscrollcommand=input_scroll_x.set, height=20)

        input_scroll_y.config(command=self.input_text.yview)
        input_scroll_x.config(command=self.input_text.xview)

        input_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        input_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        try:
            # 使用新的辅助函数来获取正确的基准路径
            base_dir = get_base_path()
            rs_file_path = os.path.join(base_dir, "test.rs")  # <-- 修改点

            with open(rs_file_path, "r", encoding="utf-8") as f:
                rust_code = f.read()
            self.input_text.insert(tk.END, rust_code)
        except FileNotFoundError:
            # 如果文件不存在，显示错误信息
            error_message = f"错误: 未在程序目录中找到 'test.rs' 文件。\n请创建一个 test.rs 文件并放入示例代码。"
            self.input_text.insert(tk.END, error_message)
            messagebox.showerror("文件未找到", error_message)
        except Exception as e:
            # 处理其他可能的读取错误
            error_message = f"读取 'test.rs' 时发生错误: {e}"
            self.input_text.insert(tk.END, error_message)
            messagebox.showerror("文件读取错误", error_message)

        
        # 按钮顺序已调整，并添加了语义分析按钮
        lex_button = tk.Button(left_frame, text="1. 进行词法分析", command=self.analyze_code, font=default_font, pady=5)
        lex_button.pack(pady=(2,1), fill=tk.X)
        
        parse_button = tk.Button(left_frame, text="2. 进行语法分析", command=self.parse_syntax, font=default_font, pady=5)
        parse_button.pack(pady=(2, 1), fill=tk.X)
        
        semantic_button = tk.Button(left_frame, text="3. 进行语义分析与四元式生成", command=self.perform_semantic_analysis_and_ir, font=default_font, pady=5)
        semantic_button.pack(pady=(2, 1), fill=tk.X)

        asm_button = tk.Button(left_frame, text="4. 生成汇编代码", command=self.perform_assembly_generation,font=default_font, pady=5)
        asm_button.pack(pady=(2, 1), fill=tk.X)

        run_button = tk.Button(left_frame, text="5. 汇编、链接并运行", command=self.assemble_link_and_run,font=default_font, pady=5)
        run_button.pack(pady=(2, 1), fill=tk.X)
        
        ast_button = tk.Button(left_frame, text="6. 生成 AST 图像", command=self.show_ast, font=default_font, pady=5)
        ast_button.pack(pady=(2, 1), fill=tk.X)




        right_notebook = ttk.Notebook(self.m)
        self.m.add(right_notebook, width=550)

        # 词法分析 Tab
        lexer_tab = tk.Frame(right_notebook)
        right_notebook.add(lexer_tab, text='词法分析结果')
        lexer_output_label = tk.Label(lexer_tab, text="词法分析结果 (Tokens):", font=default_font)
        lexer_output_label.pack(pady=(5, 5), fill=tk.X, padx=5)
        self.lexer_output_text = self.create_scrollable_text(lexer_tab, height=20, font=text_font)

        # 语法分析 Tab
        parser_tab = tk.Frame(right_notebook)
        right_notebook.add(parser_tab, text='语法分析结果')
        parser_error_label = tk.Label(parser_tab, text="语法分析错误:", font=default_font)
        parser_error_label.pack(pady=(5, 5), fill=tk.X, padx=5)
        self.parser_error_text = self.create_scrollable_text(parser_tab, height=7, font=text_font)
        parser_struct_label = tk.Label(parser_tab, text="语法结构:", font=default_font)
        parser_struct_label.pack(pady=(10, 5), fill=tk.X, padx=5)
        self.parser_struct_text = self.create_scrollable_text(parser_tab, height=23, font=text_font)

        # --- 新增: 语义分析与IR Tab ---
        semantic_ir_tab = tk.Frame(right_notebook) # Renamed tab for clarity
        right_notebook.add(semantic_ir_tab, text='语义分析与四元式')
        
        semantic_error_label = tk.Label(semantic_ir_tab, text="语义分析错误:", font=default_font)
        semantic_error_label.pack(pady=(5, 5), fill=tk.X, padx=5)
        self.semantic_error_text = self.create_scrollable_text(semantic_ir_tab, height=7, font=text_font)

        ir_label = tk.Label(semantic_ir_tab, text="中间代码 (四元式):", font=default_font)
        ir_label.pack(pady=(10, 5), fill=tk.X, padx=5)
        ir_table_frame = tk.Frame(semantic_ir_tab)
        ir_table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        self.ir_table = ttk.Treeview(ir_table_frame, height=20, columns=("addr", "quad"), show='headings')
        self.ir_table.heading("addr", text="地址")
        self.ir_table.heading("quad", text="四元式")
        self.ir_table.column("addr", width=60, anchor="center")
        self.ir_table.column("quad", width=400, anchor="w")

        vsb = tk.Scrollbar(ir_table_frame, orient="vertical", command=self.ir_table.yview)
        hsb = tk.Scrollbar(ir_table_frame, orient="horizontal", command=self.ir_table.xview)
        self.ir_table.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.ir_table.pack(fill=tk.BOTH, expand=True)

        # --- 新增汇编代码 Tab ---
        self.symbol_table_for_analysis = None
        assembly_tab = tk.Frame(right_notebook)
        right_notebook.add(assembly_tab, text='汇编代码')
        # 1. 创建一个独立的标签
        assembly_label = tk.Label(assembly_tab, text="x86-64 汇编代码 (NASM 语法):", font=default_font)
        assembly_label.pack(pady=(5, 5), fill=tk.X, padx=5)
        # 2. 使用您已有的方法创建可滚动的文本框
        self.assembly_text = self.create_scrollable_text(assembly_tab, font=text_font, wrap=tk.NONE)

        assembly_output_label = tk.Label(assembly_tab, text="链接汇编并执行:", font=default_font)
        assembly_output_label.pack(pady=(5, 5), fill=tk.X, padx=5)
        self.run_output_text = self.create_scrollable_text(assembly_tab, font=text_font)

        # AST 图像 Tab
        ast_tab = tk.Frame(right_notebook)
        right_notebook.add(ast_tab, text='AST 图像')
        ast_output_label = tk.Label(ast_tab, text="AST 图像:", font=default_font)
        ast_output_label.pack(pady=(5, 5), fill=tk.X, padx=5)
        ast_frame = tk.Frame(ast_tab)
        ast_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.ast_canvas = tk.Canvas(ast_frame, bg="white")
        self.ast_scrollbar_y = tk.Scrollbar(ast_frame, orient=tk.VERTICAL, command=self.ast_canvas.yview)
        self.ast_scrollbar_x = tk.Scrollbar(ast_frame, orient=tk.HORIZONTAL, command=self.ast_canvas.xview)
        self.ast_canvas.configure(yscrollcommand=self.ast_scrollbar_y.set, xscrollcommand=self.ast_scrollbar_x.set)
        self.ast_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.ast_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.ast_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # self.ast_image_label = tk.Label(self.ast_canvas, bg="white")
        # # 使用 'nw' (north-west) anchor 以便滚动区域计算正确
        # self.ast_canvas.create_window((0, 0), window=self.ast_image_label, anchor='nw')
        # self.ast_image_label.bind('<Configure>', lambda e: self.ast_canvas.configure(scrollregion=self.ast_canvas.bbox("all")))
        self.ast_canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.ast_canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.ast_canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        zoom_frame = tk.Frame(ast_tab)
        zoom_frame.pack(pady=(5,5))
        self.zoom_in_button = tk.Button(zoom_frame, text="放大 +", command=self.zoom_in_ast)
        self.zoom_in_button.pack(side=tk.LEFT, padx=5)
        self.zoom_out_button = tk.Button(zoom_frame, text="缩小 -", command=self.zoom_out_ast)
        self.zoom_out_button.pack(side=tk.LEFT, padx=5)
        self.ast_image_original = None
        self.ast_zoom_ratio = 1.0
        self.tk_ast_image = None # 防止 PhotoImage 被垃圾回收

        # --- 用于在阶段间传递AST的实例变量 ---
        self.ast_root_for_analysis = None

    def create_scrollable_text(self, tab, height=10, font=None, wrap=tk.NONE):
        frame = tk.Frame(tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        text_widget = tk.Text(frame, wrap=wrap, state=tk.DISABLED, height=height, font=font)
        y_scroll = tk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
        x_scroll = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=text_widget.xview)

        text_widget.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        return text_widget

    def analyze_code(self):
        source_code = self.input_text.get("1.0", tk.END)
        self.lexer_output_text.config(state=tk.NORMAL)
        self.lexer_output_text.delete("1.0", tk.END)
        if not source_code.strip():
            self.lexer_output_text.insert(tk.END, "请输入代码后再进行分析。\n")
            self.lexer_output_text.config(state=tk.DISABLED)
            return
        lexer = Lexer(source_code)
        output_lines = []
        error_found = False
        try:
            while True:
                tok = lexer.next_token()
                line_str = f"L{tok.line:<3} C{tok.col:<3}: {tok.type.name:<10s} | '{tok.literal}'"
                output_lines.append(line_str)
                if tok.type == TokenType.EOF: break
                if tok.type == TokenType.ILLEGAL:
                    output_lines.append(f"*** 错误: 非法字符 '{tok.literal}' ***"); error_found = True
        except Exception as e:
             output_lines.append(f"\n*** 词法分析时发生意外错误: {e} ***"); error_found = True
        self.lexer_output_text.insert(tk.END, "\n".join(output_lines))
        if error_found: self.lexer_output_text.insert(tk.END, "\n\n分析过程中检测到错误。")
        self.lexer_output_text.config(state=tk.DISABLED)
        # 如果重新进行词法分析，清除之前存储的AST
        self.ast_root_for_analysis = None


    def parse_syntax(self):
        source_code = self.input_text.get("1.0", tk.END)
        self.parser_error_text.config(state=tk.NORMAL)
        self.parser_error_text.delete("1.0", tk.END)
        self.parser_struct_text.config(state=tk.NORMAL)
        self.parser_struct_text.delete("1.0", tk.END)
        self.ast_root_for_analysis = None # 清除之前的AST

        if not source_code.strip():
            self.parser_error_text.insert(tk.END, "请输入代码后再进行分析。\n")
            self.parser_error_text.config(state=tk.DISABLED)
            self.parser_struct_text.config(state=tk.DISABLED)
            return

        lexer = Lexer(source_code)
        parser = Parser(lexer)
        ast_root = parser.parse_program()

        if parser.errors:
            self.parser_error_text.insert(tk.END, "语法分析失败！检测到以下错误:\n" + "-----------------------------------\n")
            for error in parser.errors:
                self.parser_error_text.insert(tk.END, f"- {error}\n")
            self.parser_error_text.insert(tk.END, "-----------------------------------\n")
            self.parser_struct_text.insert(tk.END, "由于语法错误，无法生成语法结构。\n")
        elif ast_root is None:
            self.parser_error_text.insert(tk.END, "语法分析未能生成有效的程序结构 (可能是空输入或内部错误)。\n")
            self.parser_struct_text.insert(tk.END, "无法生成语法结构。\n")
        else:
            self.parser_error_text.insert(tk.END, "语法分析成功！未检测到明显语法错误。\n")
            self.ast_root_for_analysis = ast_root # 存储AST以供后续阶段使用
            try:
                printer = ASTPrinter()
                ast_root.accept(printer)
                ast_string = printer.get_output()
                self.parser_struct_text.insert(tk.END, ast_string)
            except Exception as e:
                self.parser_struct_text.insert(tk.END, f"生成语法结构可视化时发生错误: {e}\n")
                import traceback
                self.parser_struct_text.insert(tk.END, traceback.format_exc())

        self.parser_error_text.config(state=tk.DISABLED)
        self.parser_struct_text.config(state=tk.DISABLED)

    # --- 更新: 语义分析与IR生成方法 ---
    def perform_semantic_analysis_and_ir(self):
        self.semantic_error_text.config(state=tk.NORMAL)
        self.semantic_error_text.delete("1.0", tk.END)

        for row in self.ir_table.get_children():
            self.ir_table.delete(row)

        if self.ast_root_for_analysis is None:
            self.semantic_error_text.insert(tk.END, "请先成功执行语法分析 (步骤 2) 以生成 AST。\n")
            messagebox.showinfo("语义分析与四元式", "未找到有效的语法结构。请先成功执行语法分析。")
            self.ir_table.insert('', 'end', values=('', '未进行语义分析，因此无法生成中间代码。\n'))
        else:
            # 阶段1: 语义分析
            semantic_errors_found = False
            try:
                analyzer = SemanticAnalyzer()
                # SymbolTable 错误现在会收集在 analyzer.symbol_table.semantic_errors
                # 或者 analyzer 直接收集
                analyzer.visit_program(self.ast_root_for_analysis) # 启动分析
                
                semantic_errors = analyzer.get_errors() # 获取收集到的错误
                if semantic_errors:
                    semantic_errors_found = True
                    self.symbol_table_for_analysis = None  # 清除旧的
                    self.semantic_error_text.insert(tk.END, "语义分析检测到以下错误:\n" + "---------------------------\n")
                    for error in semantic_errors:
                        self.semantic_error_text.insert(tk.END, f"- {error}\n")
                    self.semantic_error_text.insert(tk.END, "---------------------------\n")
                    self.ir_table.insert('', 'end', values=('', '由于语义错误，未生成中间代码。\n'))
                else:
                    self.symbol_table_for_analysis = analyzer.symbol_table  # <<--- 保存符号表
                    self.semantic_error_text.insert(tk.END, "语义分析成功！未检测到语义错误。\n")
            
            except Exception as e_sem:
                semantic_errors_found = True
                self.semantic_error_text.insert(tk.END, f"语义分析过程中发生意外错误: {e_sem}\n")
                import traceback
                self.semantic_error_text.insert(tk.END, traceback.format_exc())
                self.ir_table.insert('', 'end', values=('', '由于语义分析中的意外错误，未生成中间代码。\n'))


            # 阶段2: 中间代码生成 (仅当语义分析无误时)
            if not semantic_errors_found:
                try:
                    ir_gen = IRGenerator()
                    ir_gen.visit_program(self.ast_root_for_analysis) # 使用同一个AST
                    quadruples = ir_gen.get_quadruples()
                    
                    if quadruples:
                        for i, quad in enumerate(quadruples):
                            self.ir_table.insert('', 'end', values=(f"{i:03d}", str(quad)))
                    else:
                        self.ir_table.insert('', 'end', values=('', '未生成中间代码。(可能是空程序或IR生成器未对所有情况处理)。\n'))
                except Exception as e_ir:
                    self.ir_table.insert('', 'end', values=('', f"生成中间代码时发生错误: {e_ir}"))
                    import traceback
                    self.ir_table.insert('', 'end', values=('', traceback.format_exc()))

        self.semantic_error_text.config(state=tk.DISABLED)

    def perform_assembly_generation(self):
        self.assembly_text.config(state=tk.NORMAL)
        self.assembly_text.delete("1.0", tk.END)

        # 1. 检查前置步骤是否完成 (这部分保持不变)
        if self.ast_root_for_analysis is None:
            messagebox.showinfo("操作失败", "请先成功执行语法分析 (步骤 2)。")
            self.assembly_text.insert(tk.END, "错误: 未找到有效的 AST。\n")
            self.assembly_text.config(state=tk.DISABLED) # 别忘了在出错时禁用文本框
            return # 增加 return
        elif self.symbol_table_for_analysis is None:
            messagebox.showinfo("操作失败", "请先成功执行语义分析 (步骤 3)。")
            self.assembly_text.insert(tk.END, "错误: 未找到有效的符号表。请确保语义分析无误。\n")
            self.assembly_text.config(state=tk.DISABLED) # 别忘了在出错时禁用文本框
            return # 增加 return
        
        assembly_code = "" # 初始化为空字符串

        # 2. 调用汇编生成器 (这部分保持不变)
        try:
            asm_gen = AssemblyGenerator(self.symbol_table_for_analysis)
            asm_gen.visit_program(self.ast_root_for_analysis)
            assembly_code = asm_gen.get_assembly() # 将结果存入变量

            if assembly_code:
                self.assembly_text.insert(tk.END, assembly_code)
            else:
                self.assembly_text.insert(tk.END, "未能生成汇编代码。\n")
        except Exception as e:
            error_message = f"生成汇编代码时发生意外错误: {e}\n"
            import traceback
            error_message += traceback.format_exc()
            self.assembly_text.insert(tk.END, error_message)
            assembly_code = "" # 发生错误时，确保 code 为空，不进行保存

        self.assembly_text.config(state=tk.DISABLED)

    def assemble_link_and_run(self):
        # 清空之前的运行结果
        self.run_output_text.config(state=tk.NORMAL)
        self.run_output_text.delete("1.0", tk.END)

        # 1. 获取汇编代码
        asm_code = self.assembly_text.get("1.0", tk.END).strip()
        if not asm_code or "未能生成汇编代码" in asm_code:
            messagebox.showerror("错误", "没有可用的汇编代码。请先成功生成汇编代码。")
            self.run_output_text.config(state=tk.DISABLED)
            return

        try:
            # 定义项目根目录和工具/输出目录
            base_dir = get_base_path()
            tools_dir = os.path.join(base_dir, "compiler_tools")
            output_dir = os.path.join(base_dir, "assembly_result")
            os.makedirs(output_dir, exist_ok=True)

            # 定义工具和文件的完整路径
            nasm_path = os.path.join(tools_dir, "nasm.exe")
            # --- 修改点: 链接器现在是 GoLink.exe ---
            linker_path = os.path.join(tools_dir, "GoLink.exe")

            asm_path = os.path.join(output_dir, "output.asm")
            obj_path = os.path.join(output_dir, "output.obj")
            exe_path = os.path.join(output_dir, "program.exe")

            # 检查工具是否存在
            if not os.path.exists(nasm_path) or not os.path.exists(linker_path):
                messagebox.showerror("工具未找到",
                                     f"未在 {tools_dir} 文件夹中找到 nasm.exe 或 GoLink.exe。\n请确保轻量级工具已正确放置。")
                return

            # 写入汇编文件
            with open(asm_path, "w", encoding="utf-8") as f:
                f.write(asm_code)

            # 3. 构建并执行命令
            self.run_output_text.insert(tk.END, f"输出目录: {output_dir}\n\n")
            self.run_output_text.insert(tk.END, "1. 正在通过本地 NASM 汇编...\n")

            # 命令：nasm.exe -f win64 -o output.obj output.asm
            nasm_cmd = [nasm_path, "-f", "win64", "-o", obj_path, asm_path]
            nasm_proc = subprocess.run(nasm_cmd, capture_output=True, text=True, encoding='utf-8')

            if nasm_proc.returncode != 0:
                self.run_output_text.insert(tk.END, f"汇编失败！\n错误:\n{nasm_proc.stderr}")
                self.run_output_text.config(state=tk.DISABLED)
                return

            # --- 修改点: 使用 GoLink 进行链接 ---
            self.run_output_text.insert(tk.END, "2. 正在通过 GoLink 链接...\n")

            # GoLink 命令解释:
            # /console       -> 创建一个控制台程序
            # /entry main    -> 指定程序的入口点是 'main' 标签
            # /fo program.exe -> 指定输出文件名为 program.exe
            # output.obj     -> 输入的目标文件
            # kernel32.dll   -> 需要链接的系统库 (因为我们用了 ExitProcess)
            link_cmd = [
                linker_path, "/console", "/entry", "main",
                "/fo", exe_path, obj_path, "kernel32.dll"
            ]
            link_proc = subprocess.run(link_cmd, capture_output=True, text=True, encoding='utf-8')

            if link_proc.returncode != 0:
                # GoLink 的输出通常在 stdout
                error_output = link_proc.stdout + link_proc.stderr
                self.run_output_text.insert(tk.END, f"链接失败！\n错误:\n{error_output}")
                self.run_output_text.config(state=tk.DISABLED)
                return

            self.run_output_text.insert(tk.END, "3. 正在运行本地程序 (.exe)...\n\n")
            self.run_output_text.insert(tk.END, "--- 程序输出 ---\n")

            # 直接运行可执行文件
            run_proc = subprocess.run([exe_path], capture_output=True, text=True, encoding='utf-8')

            if run_proc.stdout:
                self.run_output_text.insert(tk.END, f"标准输出:\n{run_proc.stdout}\n")
            if run_proc.stderr:
                self.run_output_text.insert(tk.END, f"标准错误:\n{run_proc.stderr}\n")

            self.run_output_text.insert(tk.END, f"--- 结果 ---\n程序退出码: {run_proc.returncode}\n")

        except Exception as e:
            self.run_output_text.insert(tk.END, f"发生意外错误: {e}\n")
        finally:
            self.run_output_text.config(state=tk.DISABLED)

    def display_ast_image(self):
        if self.ast_image_original is None:
            self.ast_canvas.delete("all") # 清空画布
            self.tk_ast_image = None # 清除引用
            return

        width, height = self.ast_image_original.size
        # 为了防止缩放比例过小导致图像尺寸为0而出错
        new_width = max(1, int(width * self.ast_zoom_ratio))
        new_height = max(1, int(height * self.ast_zoom_ratio))

        resized_image = self.ast_image_original.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 必须将 PhotoImage 对象保存为实例属性，否则会被垃圾回收
        self.tk_ast_image = ImageTk.PhotoImage(resized_image)

        # 清空画布，然后直接在(0,0)位置创建图像
        self.ast_canvas.delete("all")
        self.ast_canvas.create_image(0, 0, anchor='nw', image=self.tk_ast_image)

        # 更新滚动区域以匹配新图像的大小
        self.ast_canvas.config(scrollregion=self.ast_canvas.bbox("all"))

        # # 避免尺寸过小导致错误
        # if new_width <= 0 or new_height <= 0:
        #      # 可以选择不显示警告，或者让缩放比例有个下限
        #      # messagebox.showwarning("缩放警告", "图像已缩至最小。")
        #      if self.ast_zoom_ratio < 0.1: self.ast_zoom_ratio = 0.1 # 最小缩放比例
        #      new_width = int(width * self.ast_zoom_ratio)
        #      new_height = int(height * self.ast_zoom_ratio)
        #      if new_width <= 0 or new_height <= 0: return # 再次检查
        #
        # try:
        #     resized_image = self.ast_image_original.resize((new_width, new_height), Image.Resampling.LANCZOS)
        #     self.tk_ast_image = ImageTk.PhotoImage(resized_image)
        #     self.ast_image_label.config(image=self.tk_ast_image)
        # except Exception as e:
        #     print(f"调整或显示AST图像时出错: {e}")
        #     messagebox.showerror("图像显示错误", f"调整或显示AST图像时出错: {e}")


    def zoom_in_ast(self):
        if self.ast_image_original:
            self.ast_zoom_ratio *= 1.2
            self.display_ast_image()
        else:
            messagebox.showinfo("缩放", "请先生成 AST 图像。")


    def zoom_out_ast(self):
        if self.ast_image_original:
            self.ast_zoom_ratio /= 1.2
            if self.ast_zoom_ratio < 0.05: # 设置一个最小缩放因子防止过小
                self.ast_zoom_ratio = 0.05
            self.display_ast_image()
        else:
            messagebox.showinfo("缩放", "请先生成 AST 图像。")

    def on_drag_start(self, event):
        """记录拖拽开始的位置"""
        self.ast_canvas.scan_mark(event.x, event.y)

    def on_drag_motion(self, event):
        """根据鼠标移动的距离来拖拽画布"""
        self.ast_canvas.scan_dragto(event.x, event.y, gain=1)

    def on_mouse_wheel(self, event):
        """处理鼠标滚轮事件以进行缩放"""
        # 在 Windows 和大多数 Linux 系统上，event.delta 是一个表示滚动幅度的值
        # 向上滚动是正值，向下滚动是负值
        if event.delta > 0:
            self.zoom_in_ast()
        elif event.delta < 0:
            self.zoom_out_ast()

    def show_ast(self):
        if self.ast_root_for_analysis is None:
            source_code_present = self.input_text.get("1.0", tk.END).strip()

            # 用下面这行代码来清空画布
            self.ast_canvas.delete("all")
            self.ast_image_original = None
            self.tk_ast_image = None

            if not source_code_present:
                messagebox.showinfo("AST生成", "请输入代码。")
                return
            
            self.parse_syntax() # 尝试解析
            if self.ast_root_for_analysis is None:
                messagebox.showerror("AST生成失败", "无法生成 AST。请先确保语法分析成功 (步骤2)。")
                return

        
        try:
            drawer = ASTGraphvizDrawer()
            self.ast_root_for_analysis.accept(drawer) # 使用已存储的AST
            img_bytes = drawer.render_to_memory(get_base_path())

            if img_bytes is None: # 如果 render_to_memory 返回 None (例如Graphviz错误)
                return # 错误消息已由 drawer 显示

            image = Image.open(io.BytesIO(img_bytes))
            self.ast_image_original = image
            self.ast_zoom_ratio = 1.0 # 重置缩放
            self.display_ast_image()
        except Exception as e:
            messagebox.showerror("AST生成失败", f"生成AST图形时发生错误:\n{e}")
            import traceback
            print(f"AST生成失败: {traceback.format_exc()}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LexerApp(root)
    root.mainloop()