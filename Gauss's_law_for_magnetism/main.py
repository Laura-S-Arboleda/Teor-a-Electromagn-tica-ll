from __future__ import annotations

import argparse

from simulations.gauss_magnetism import (
    run_on_axis_validation,
    run_divergence_analysis,
    run_convergence_analysis,
    run_visualization,
    run_animation,
    plot_on_axis_validation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulaciones electromagnéticas para las ecuaciones de Maxwell."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="validation",
        choices=[
            "validation",
            "divergence",
            "convergence",
            "visualization",
            "animation",
        ],
        help="Modo de ejecución.",
    )
    parser.add_argument(
        "--anim-mode",
        type=str,
        default="3d",
        choices=["2d", "3d"],
        help="Modo de animación cuando --mode animation.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "validation":
        z_values, Bz_num, Bz_an, rel_err = run_on_axis_validation()
        plot_on_axis_validation(z_values, Bz_num, Bz_an, rel_err)

    elif args.mode == "divergence":
        run_divergence_analysis()

    elif args.mode == "convergence":
        df = run_convergence_analysis()
        print(df)

    elif args.mode == "visualization":
        run_visualization()

    elif args.mode == "animation":
        run_animation(mode=args.anim_mode)


if __name__ == "__main__":
    main()