from uma_geometry_optimizer.decorators import time_it


def test_time_it_decorator_prints(capsys):
    @time_it
    def add(a, b):
        return a + b

    out = add(2, 3)
    assert out == 5
    captured = capsys.readouterr().out
    assert "Function:'add'" in captured or "Function: 'add'" in captured or "Function:add" in captured

