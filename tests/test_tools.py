from tools import calculator, fetch_url, run_tool


def test_calculator_basic_arithmetic():
    assert calculator("2 + 2") == "4"
    assert calculator("47 * 12") == "564"
    assert calculator("2 ** 10") == "1024"


def test_calculator_division_by_zero():
    assert calculator("1 / 0") == "Error: division by zero"


def test_calculator_rejects_non_arithmetic():
    result = calculator("__import__('os').system('echo hi')")
    assert result.startswith("Error:")


def test_calculator_rejects_percent_sign_syntax():
    # '%' is modulo, not percent -- '15% * 240' is invalid Python syntax.
    result = calculator("15% * 240")
    assert result.startswith("Error:")


def test_fetch_url_success():
    text = fetch_url("https://example.com")
    assert "Example Domain" in text


def test_fetch_url_unreachable_host_returns_error_not_exception():
    result = fetch_url("http://this-domain-does-not-exist.invalid")
    assert result.startswith("Error")


def test_run_tool_unknown_name():
    assert run_tool("not_a_real_tool", {}) == "Error: unknown tool 'not_a_real_tool'"


def test_run_tool_dispatches_calculator():
    assert run_tool("calculator", {"expression": "3 + 4"}) == "7"
