# implementations/llvm-ir_impl.py

from ..interfaces.base import CodeAnalyzerInterface
    
class LLVMIRAnalyzer(CodeAnalyzerInterface):

    def get_ast(self, file_path: str) -> object:
        # Get AST
        from antlr4 import CommonTokenStream, FileStream
        from ..utils.llvmir_utils.antlr.gen.LLVMIRLexer import LLVMIRLexer
        from ..utils.llvmir_utils.antlr.gen.LLVMIRParser import LLVMIRParser

        stream = FileStream(file_path, encoding="utf8")
        lexer = LLVMIRLexer(stream)
        token_stream = CommonTokenStream(lexer)
        parser = LLVMIRParser(token_stream)
        parse_tree = parser.compilationUnit()
        #to process the tree
        #parse_tree.toStringTree()
        return parse_tree

    def get_cfg(self, file_path: str) -> object:
        # Run CFG analysis
        pass

        
    def get_pg(self, code: str) -> object:
        # Run PDG analysis
        import programl as pg

        G = pg.from_llvm_ir(code)
        dot_output = pg.to_dot(G)
        return dot_output
