# output into HTML

from nbconvert import HTMLExporter
import nbformat
import os


def convert_notebook_to_html(notebook_path, output_html_path=None):
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Convert the notebook to HTML
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(notebook_content)

    # Determine the output HTML path
    if output_html_path is None:
        output_html_path = os.path.splitext(notebook_path)[0] + '.html'

    # Write the HTML file
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(body)

    return output_html_path

if __name__ == '__main__':
    convert_notebook_to_html('/Task/ml/train_svm_grid.ipynb')
    convert_notebook_to_html('/Task/ml/train_rf_grid.ipynb')
    convert_notebook_to_html('/Task/ml/train_mlp_grid.ipynb')
