"""Factory for language-specific code analyzers."""


class CodeAnalyzerFactory:
    @staticmethod
    def create_analyzer(language: str):
        if language == "python":
            from .implementations.python_impl import PythonAnalyzer

            return PythonAnalyzer()
        elif language == "java":
            from .implementations.java_impl import JavaAnalyzer

            return JavaAnalyzer()
        elif language == "c":
            from .implementations.c_impl import CAnalyzer

            return CAnalyzer()
        elif language == "cpp":
            from .implementations.cpp_impl import CppAnalyzer

            return CppAnalyzer()
        elif language == "llvmir":
            from .implementations.llvm_ir_impl import LLVMIRAnalyzer

            return LLVMIRAnalyzer()
        else:
            raise ValueError(f"No analyzer available for language: {language}")
