import logging
from uma_geometry_optimizer.decorators import time_it


def test_time_it_decorator_logs_time(caplog):
    @time_it
    def add(a, b):
        return a + b

    with caplog.at_level(logging.WARNING):
        out = add(2, 3)
    assert out == 5
    assert any("Function:" in rec.message and "took:" in rec.message for rec in caplog.records)
