"""Runnable mock interface, requiring only Python."""

import argparse
import json

from .predictors import MockPredictor
from .schemas import DecisionRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision engine skeleton (MOCK ONLY)")
    parser.add_argument("--state", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--options", nargs="+", required=True)
    args = parser.parse_args()
    try:
        request = DecisionRequest(args.state, args.question, tuple(args.options))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(MockPredictor().predict(request).to_dict(), indent=2))


if __name__ == "__main__":
    main()
