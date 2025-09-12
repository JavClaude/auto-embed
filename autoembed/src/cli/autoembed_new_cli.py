import yaml

import typer


app = typer.Typer()

@app.command()
def train(project_name: str, train_yaml_path: str):
    with open(train_yaml_path, "r") as f:
        yaml_as_dict = yaml.load(f, Loader=yaml.FullLoader)


def main():
    app()