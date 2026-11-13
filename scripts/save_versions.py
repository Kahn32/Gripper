"""Record package versions + platform to configs/versions.json."""
import json, platform, sys
from importlib.metadata import version
pkgs = ["numpy", "scipy", "pandas", "pyarrow", "transformers", "tokenizers", "huggingface_hub", "matplotlib"]
out = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
       "packages": {p: version(p) for p in pkgs}}
json.dump(out, open("configs/versions.json", "w"), indent=2)
print(json.dumps(out, indent=2))
