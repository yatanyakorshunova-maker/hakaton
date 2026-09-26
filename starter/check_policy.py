from __future__ import annotations

import argparse
from importlib.util import find_spec
from pathlib import Path


def validate_policy(path: Path) -> None:
    import onnx
    from arena.protocol import MAX_ONNX, _validate_onnx_proto, load_onnx_policy, read_artifact

    data = read_artifact(path, MAX_ONNX)
    _validate_onnx_proto(onnx.load_model_from_string(data), obs_dim=160)
    if find_spec("onnxruntime") is None:
        print(f"Структура модели проверена: {path} ({len(data)} байт). "
              "Исполнение на 48 заездах проверит сервер.")
        return
    load_onnx_policy(path, obs_dim=160, batch_size=48)
    print(f"Модель проверена в ONNX Runtime: {path} ({len(data)} байт)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=Path("/output/policy.onnx"))
    args = parser.parse_args()
    validate_policy(args.path)


if __name__ == "__main__":
    main()
