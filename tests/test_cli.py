import argparse
import pytest

import uma_geometry_optimizer.cli as cli


def test_setup_parser_returns_parser():
    parser = cli.setup_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_optimize_parser_has_charge_and_multiplicity():
    parser = cli.setup_parser()
    args = parser.parse_args(["optimize", "--xyz", "file.xyz", "--output", "out.xyz", "--charge", "-1", "--multiplicity", "2"])
    assert args.charge == -1
    assert args.multiplicity == 2


def test_batch_parser_has_charge_and_multiplicity():
    parser = cli.setup_parser()
    args = parser.parse_args(["batch", "--multi-xyz", "m.xyz", "--output", "out.xyz", "--charge", "1", "--multiplicity", "1"])
    assert args.charge == 1
    assert args.multiplicity == 1


def test_help_exits():
    parser = cli.setup_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
