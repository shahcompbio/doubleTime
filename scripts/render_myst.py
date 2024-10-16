import json
import jinja2
import click
import logging
import sys

@click.command()
@click.argument('input_filename')
@click.argument('output_filename')
@click.argument('params_json')
def render_myst(
        input_filename,
        output_filename,
        params_json,
    ):

    params = json.loads(params_json)

    template = jinja2.Template(open(input_filename, 'r').read())

    output = template.render(**params)

    with open(output_filename, 'w') as f:
        f.write(output)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    render_myst()