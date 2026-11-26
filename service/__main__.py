"""Start with ``python -m service`` to initialize CUDA before native dependencies."""
from importlib import import_module

from .config import ServiceSettings


def initialize_device(device: str) -> None:
    """Match the early CUDA allocation verified on the Linux deployment.

    Do not import the HTTP application, geometry stack or model extensions
    until CUDA initialization has completed. CPU startup needs no warmup.
    This is a startup-order workaround, not a CUDA compatibility check.
    """
    if not device.lower().startswith('cuda'):
        return
    import torch

    print('Initializing device: ' + device, flush=True)
    probe = torch.ones(1, device=device)
    torch.cuda.synchronize(device)
    print('CUDA initialized: ' + str(probe.item()), flush=True)
    del probe


def main() -> None:
    settings = ServiceSettings.from_env()
    initialize_device(settings.device)
    # A late import is essential: app construction imports native geometry
    # libraries, and model startup subsequently imports MMCV extensions.
    application = import_module('.app', package='service')
    application.main()


if __name__ == '__main__':
    main()
