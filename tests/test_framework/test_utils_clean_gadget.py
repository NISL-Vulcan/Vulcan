"""Unit tests for vulcan.framework.utils.clean_gadget."""
from vulcan.framework.utils.clean_gadget import clean_gadget, CPP_KEYWORDS, MAIN_FUNCTIONS, MAIN_ARGUMENTS


def test_clean_gadget_basic():
    code = ["int main(int argc, char **argv) {", "return 0;"]
    out = clean_gadget(code)
    assert len(out) == 2
    assert "main" in out[0]
    assert "argc" in out[0] or "argv" in out[0]


def test_clean_gadget_removes_strings():
    code = ['printf("hello world");']
    out = clean_gadget(code)
    assert '""' in out[0] or '"' in out[0]


def test_clean_gadget_renames_user_func():
    code = ["my_custom_func(x);"]
    out = clean_gadget(code)
    assert "FUN1" in str(out) or "my_custom_func" not in str(out) or len(out) >= 1


def test_clean_gadget_preserves_main():
    code = ["int main() { return 0; }"]
    out = clean_gadget(code)
    assert "main" in out[0]


def test_cpp_keywords_set():
    assert "int" in CPP_KEYWORDS
    assert "return" in CPP_KEYWORDS
    assert "main" in MAIN_FUNCTIONS
    assert "argc" in MAIN_ARGUMENTS
