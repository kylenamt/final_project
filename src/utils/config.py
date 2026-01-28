import yaml
import argparse
from collections.abc import MutableMapping

def _deep_merge(d, u):
    """
    Recursively merges dictionary `u` into `d`.
    """
    for k, v in u.items():
        if isinstance(v, MutableMapping):
            d[k] = _deep_merge(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def _parse_override_str(override_str):
    """
    Parses an override string 'key1.key2=value' into a nested dict.
    """
    key_str, value = override_str.split('=', 1)
    keys = key_str.split('.')
    
    # Attempt to convert value to float or int
    try:
        value = float(value)
        if value.is_integer():
            value = int(value)
    except ValueError:
        # Keep as string if conversion fails
        pass

    override_dict = current = {}
    for i, key in enumerate(keys):
        if i == len(keys) - 1:
            current[key] = value
        else:
            current[key] = {}
            current = current[key]
    return override_dict

def load_config(config_path, overrides=None):
    """
    Loads a YAML config file and applies optional overrides.

    Args:
        config_path (str): Path to the base YAML configuration file.
        overrides (list of str, optional): A list of override strings, 
                                           e.g., ['train.batch_size=64', 'audio.sr=22050'].

    Returns:
        dict: The final configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if overrides:
        for override_str in overrides:
            override_dict = _parse_override_str(override_str)
            config = _deep_merge(config, override_dict)
            
    return config

def main():
    """
    Example usage:
    python src/utils/config.py --config configs/default.yaml --override train.batch_size=128 audio.sr=22050
    """
    parser = argparse.ArgumentParser(description="Load YAML config with overrides.")
    parser.add_argument("--config", required=True, help="Path to the base config file.")
    parser.add_argument("--override", nargs='*', default=[], help="Override config values, e.g., key.subkey=value")
    
    args = parser.parse_args()
    
    config = load_config(args.config, args.override)
    
    import json
    print("Loaded and merged configuration:")
    print(json.dumps(config, indent=2))

if __name__ == "__main__":
    main()
