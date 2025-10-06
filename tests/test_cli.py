import argparse
from types import SimpleNamespace

from uma_geometry_optimizer.config import Config
import uma_geometry_optimizer.cli as cli
from uma_geometry_optimizer.structure import Structure
import uma_geometry_optimizer.optimizer as optimizer


def test_setup_parser_returns_parser():
    parser = cli.setup_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_cmd_optimize_smiles(monkeypatch, tmp_path):
    calls = {}

    monkeypatch.setattr(cli, "smiles_to_xyz", lambda s: Structure(["H"], [[0.0,0.0,0.0]]))

    def fake_optimize(s, cfg):
        s.energy = -1.0
        return s
    monkeypatch.setattr(optimizer, "optimize_single_structure", fake_optimize)

    monkeypatch.setattr(cli, "save_xyz_file", lambda s, p: tmp_path.joinpath(p).write_text("ok"))

    args = SimpleNamespace(smiles="H", xyz=None, output=str(tmp_path / "out.xyz"), config=None)
    cfg = Config()
    cfg.optimization.verbose = False
    cli.cmd_optimize(args, cfg)


def test_cmd_batch_multi_xyz(monkeypatch, tmp_path):
    calls = {}
    structs = [Structure(["H"], [[0.0,0.0,0.0]])]
    monkeypatch.setattr(cli, "read_multi_xyz", lambda p: structs)
    monkeypatch.setattr(optimizer, "optimize_structure_batch", lambda s, c: s)
    monkeypatch.setattr(cli, "save_multi_xyz", lambda s, p, c: tmp_path.joinpath(p).write_text("ok"))

    args = SimpleNamespace(multi_xyz=str(tmp_path / "m.xyz"), xyz_dir=None, output=str(tmp_path / "out.xyz"), config=None)
    cfg = Config(); cfg.optimization.verbose = False
    cli.cmd_batch(args, cfg)


def test_cmd_ensemble(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "smiles_to_ensemble", lambda smi, num_conformers=None: [Structure(["H"],[ [0,0,0] ])])
    monkeypatch.setattr(optimizer, "optimize_structure_batch", lambda s, c: s)
    monkeypatch.setattr(cli, "save_multi_xyz", lambda s, p, c: tmp_path.joinpath(p).write_text("ok"))

    args = SimpleNamespace(smiles="H", conformers=1, output=str(tmp_path/"e.xyz"), config=None)
    cfg = Config(); cfg.optimization.verbose = False
    cli.cmd_ensemble(args, cfg)


def test_cmd_convert(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "smiles_to_xyz", lambda s: Structure(["H"], [[0.0,0.0,0.0]]))
    monkeypatch.setattr(cli, "save_xyz_file", lambda s, p: tmp_path.joinpath(p).write_text("ok"))

    args = SimpleNamespace(smiles="H", output=str(tmp_path/"c.xyz"))
    cfg = Config(); cfg.optimization.verbose = False
    cli.cmd_convert(args, cfg)


def test_cmd_generate(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "smiles_to_ensemble", lambda s, n: [Structure(["H"], [[0,0,0]])])
    monkeypatch.setattr(cli, "save_multi_xyz", lambda s, p, c: tmp_path.joinpath(p).write_text("ok"))

    args = SimpleNamespace(smiles="H", conformers=1, output=str(tmp_path/"g.xyz"))
    cfg = Config(); cfg.optimization.verbose = False
    cli.cmd_generate(args, cfg)


def test_cmd_config_create_and_validate(tmp_path):
    cfg_path = tmp_path / "config.json"
    args_create = SimpleNamespace(create=str(cfg_path), validate=None)
    cli.cmd_config(args_create, Config())

    args_validate = SimpleNamespace(create=None, validate=str(cfg_path))
    cli.cmd_config(args_validate, Config())
