"""CLI entrypoint for aiinfra_e2e."""

import typer

app = typer.Typer(help="AIInfra E2E command line interface.")


@app.callback()
def main_callback() -> None:
    """Run the root CLI callback."""


def main() -> None:
    app()


if __name__ == "__main__":
    main()
