import yaml

import typer


app = typer.Typer()

@app.command()
def train(project_name: str, train_yaml_path: str):
    pass

@app.command()
def predict(project_name: str):
    pass

@app.command()
def serve(project_name: str, train_yaml_path: str):
    pass

if __name__ == "__main__":
    app()