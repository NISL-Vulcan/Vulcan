# implementations/cpp_impl.py

from ..interfaces.base import CodeAnalyzerInterface
    
class CppAnalyzer(CodeAnalyzerInterface):

    def get_ast(self, file_path: str) -> object:
        # Get AST
        from antlr4 import CommonTokenStream, FileStream
        from ..utils.c_utils.src.antlr.gen.CPP14_v2Lexer import CPP14_v2Lexer
        from ..utils.c_utils.src.antlr.gen.CPP14_v2Parser import CPP14_v2Parser

        stream = FileStream(file_path, encoding="utf8")
        lexer = CPP14_v2Lexer(stream)
        token_stream = CommonTokenStream(lexer)
        parser = CPP14_v2Parser(token_stream)
        parse_tree = parser.translationunit()
        #to process the tree
        #parse_tree.toStringTree()
        return parse_tree

    def get_cfg(self, file_path: str) -> object:
        # Run CFG analysis
        from antlr4 import CommonTokenStream, FileStream
        from ..utils.c_utils.src.antlr.gen.CPP14_v2Lexer import CPP14_v2Lexer
        from ..utils.c_utils.src.antlr.gen.CPP14_v2Parser import CPP14_v2Parser
        from ..utils.c_utils.src.cfg_extractor.cfg_extractor_visitor import CFGExtractorVisitor
        from ..utils.c_utils.src.graph.visual import draw_CFG

        stream = FileStream(file_path, encoding="utf8")
        lexer = CPP14_v2Lexer(stream)
        token_stream = CommonTokenStream(lexer)
        parser = CPP14_v2Parser(token_stream)
        parse_tree = parser.translationunit()
        cfg_extractor = CFGExtractorVisitor()
        cfg_extractor.visit(parse_tree)
        funcs = cfg_extractor.functions
        for i, g in enumerate(funcs.values()):
            draw_CFG(g, f"../test_output/temp{i}", token_stream, verbose=False)

        
    def get_pg(self, code: str) -> object:
        # Run PDG analysis
        import programl as pg

        G = pg.from_cpp(code)
        dot_output = pg.to_dot(G)
        return dot_output
