import argparse
import json
from .io import load_problem, save_result
from .solver import RAIConfig, solve
from .adaptive import AdaptiveConfig, solve_adaptive
from .net import RAINetPolicy, solve_net
from .moment_solver import MomentConfig
from .moments import solve_moments, solve_moment_net
from .moment_net import MomentPrior


def main():
    parser = argparse.ArgumentParser(description="RAI-ILT research prototype: canonical .mat input/output")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--config", help="JSON file with RAIConfig keyword arguments")
    parser.add_argument("--method", choices=("components", "density", "adaptive", "net", "moments", "moment-net"),
                        default=None, help="Explicit method; absent retains the v0.3 API")
    parser.add_argument("--policy", help="Numeric NPZ RAI-Net policy (required for --method net)")
    parser.add_argument("--prior", help="Numeric positive-moment prior NPZ (required for --method moment-net)")
    args = parser.parse_args()
    settings = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            settings = json.load(f)
    method = args.method or settings.pop("representation", "components")
    if method not in ("components", "density", "adaptive", "net", "moments", "moment-net"):
        parser.error("Unknown representation")
    representation={"net":"adaptive","moment-net":"moments"}.get(method,method)
    if "representation" in settings and settings.pop("representation") != representation:
        parser.error("--method and configuration representation disagree")
    if method == "net" and not args.policy:
        parser.error("--method net requires --policy")
    if method != "net" and args.policy:
        parser.error("--policy only applies to --method net")
    if (method == "moment-net") != bool(args.prior):
        parser.error("--method moment-net requires --prior; --prior only applies to moment-net")
    config = (MomentConfig(**settings) if method in ("moments", "moment-net") else
              AdaptiveConfig(**settings) if method in ("adaptive", "net") else RAIConfig(representation=method, **settings))
    problem = load_problem(args.input)
    if method == "moment-net":
        result = solve_moment_net(problem, MomentPrior.load(args.prior), config)
    elif method == "net":
        result = solve_net(problem, RAINetPolicy.load(args.policy), config)
    else:
        result = solve(problem, config)
    save_result(args.output, result, problem)
    print(json.dumps(result.diagnostics, indent=2))
    if not result.diagnostics["optimizer"]["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
