"""Verify startup ordering without requiring CUDA or importing the app."""
import sys
from types import SimpleNamespace

from service import __main__ as launcher


def test_cuda_warmup_precedes_application_import(monkeypatch):
    events = []

    def ones(size, device):
        events.append(('allocate', size, device))
        return SimpleNamespace(item=lambda: 1.0)

    fake_torch = SimpleNamespace(
        ones=ones,
        cuda=SimpleNamespace(synchronize=lambda device: events.append(('sync', device))))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setattr(launcher.ServiceSettings, 'from_env',
                        lambda: SimpleNamespace(device='cuda:1'))

    def import_app(name, package):
        assert events == [('allocate', 1, 'cuda:1'), ('sync', 'cuda:1')]
        events.append('import_app')
        return SimpleNamespace(main=lambda: events.append('serve'))

    monkeypatch.setattr(launcher, 'import_module', import_app)
    launcher.main()
    assert events[-2:] == ['import_app', 'serve']


def test_cpu_does_not_initialize_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU startup must not allocate CUDA memory')

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(ones=forbidden))
    launcher.initialize_device('cpu')


def test_cuda_failure_does_not_start_service(monkeypatch):
    import pytest

    def fail(*args, **kwargs):
        raise RuntimeError('CUDA unavailable')

    monkeypatch.setattr(launcher.ServiceSettings, 'from_env',
                        lambda: SimpleNamespace(device='cuda:0'))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(ones=fail))
    monkeypatch.setattr(launcher, 'import_module',
                        lambda *args, **kwargs: pytest.fail('Service started after CUDA failure'))
    with pytest.raises(RuntimeError, match='CUDA unavailable'):
        launcher.main()
