"""Validate schema and ensure versioned notebooks carry no execution output."""
from pathlib import Path
import ast
import nbformat

for path in sorted((Path(__file__).resolve().parents[1]/'notebooks').glob('*.ipynb')):
    notebook=nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    for i,cell in enumerate(notebook.cells):
        if cell.cell_type=='code':
            assert cell.execution_count is None, (path,i,'execution count')
            assert not cell.outputs, (path,i,'outputs')
            code='\n'.join(line for line in cell.source.splitlines() if not line.lstrip().startswith(('%','!')))
            ast.parse(code,filename=f'{path.name}:cell-{i}')
    print(f'{path.name}: schema, Python syntax, cleared outputs OK ({len(notebook.cells)} cells)')
