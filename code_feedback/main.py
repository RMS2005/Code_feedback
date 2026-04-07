import click
from pathlib import Path
from src.pipeline import Pipeline

@click.group()
def cli():
    """Autograder: A modular, AI-powered grading engine."""
    pass

@cli.command()
@click.option('--assignment-config', required=True, type=click.Path(exists=True), help='Path to the assignment config.json file.')
@click.option('--submissions-dir', required=True, type=click.Path(exists=True), help='Path to the root directory of student submissions.')
@click.option('--output-dir', default='./reports', help='Directory to save feedback and reports.')
@click.option(
    '--level',
    type=click.Choice(['dynamic', 'embedding', 'full'], case_sensitive=False),
    default='full',
    show_default=True,
    help='Set the analysis level: "dynamic" (tests only), "embedding" (tests + code embeddings), or "full" (all analysis stages).'
)
def grade(assignment_config, submissions_dir, output_dir, level):
    """Runs the grading pipeline for a given assignment."""
    click.echo(f"Starting Autograder with analysis level: {level.upper()}")

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize and run the pipeline, passing the execution level
    pipeline = Pipeline(assignment_config, submissions_dir, str(output_path), level)
    pipeline.run()

    click.echo(f"\nGrading complete. Check '{output_dir}' for reports.")

if __name__ == '__main__':
    cli()
