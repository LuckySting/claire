from pathlib import Path

import yaml

from claire.claire import Claire
from claire.settings import ClaireSettings


def load_config() -> ClaireSettings:
    """Load configuration from config.yaml file."""
    config_path = Path(__file__).parent / "config.yaml"

    with config_path.open() as f:
        config_data = yaml.safe_load(f)

    return ClaireSettings(**config_data)


def main() -> None:
    """Initialize Claire with config from YAML file."""
    settings = load_config()
    claire = Claire(settings)

    claire.index_merge_request_discussions(37212, ignore_authors={"sa-nebius-nebo"})
    print(claire.ask("Why did we drop CreatePayOutOfBandPayment method in v1alpha1?"))


if __name__ == "__main__":
    main()
