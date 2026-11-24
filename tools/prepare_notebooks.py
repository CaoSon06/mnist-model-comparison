"""Rebuild clean reference notebook and provenance from the supplied original.

Usage: python tools/prepare_notebooks.py path/to/AI-Final.ipynb
The workflow notebook is maintained separately.
"""
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import nbformat


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=args.source.read_bytes()
    notebook=nbformat.reads(source.decode('utf-8'),as_version=4)
    report=root/'reports'/'notebook'
    figures=report/'figures'
    figures.mkdir(parents=True,exist_ok=True)
    exports={10:'mnist_samples',87:'confusion_matrices',124:'integrated_gradients',128:'gradcam',134:'attention_rollout'}
    provenance={'source_filename':args.source.name,'source_sha256':hashlib.sha256(source).hexdigest(),
                'source_cell_count':len(notebook.cells),'result_type':'historical saved notebook outputs; not reproduced',
                'figure_cells':{},'metric_cells':{'knn':36,'svm':38,'random_forest':40,'mlp':61,'cnn':83,'resnet9':83,'tinyvit':83}}
    for index,name in exports.items():
        for output in notebook.cells[index].get('outputs',[]):
            encoded=output.get('data',{}).get('image/png')
            if encoded:
                if isinstance(encoded,list): encoded=''.join(encoded)
                (figures/(name+'.png')).write_bytes(base64.b64decode(encoded))
                provenance['figure_cells'][name+'.png']=index
                break
    (report/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    # Explicit transcription of the source's saved output, with distinct F1 averages.
    rows=[['knn',.9705,'',.9705,10000,'official test'],
          ['svm',.9596,'',.9596,10000,'official test'],
          ['random_forest',.9477,'',.9477,10000,'official test'],
          ['mlp',.9757,.9755,'',7000,'recombined then resplit; different test set'],
          ['cnn',.9943,.9943,'',10000,'official test'],
          ['resnet9',.9948,.9948,'',10000,'official test'],
          ['tinyvit',.9783,.9781,'',10000,'official test; original train-validation overlap']]
    with (report/'historical_metrics.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file)
        writer.writerow(['model','accuracy','f1_macro','f1_weighted','n_test','protocol_note'])
        writer.writerows(rows)
    intro=nbformat.v4.new_markdown_cell('''# AI-Final — bản lưu nghiên cứu gốc

Notebook này được lấy từ `AI-Final.ipynb` do người dùng cung cấp. Đã xóa output,
execution count và metadata giao diện; giữ nguyên thứ tự và nội dung code.
Kết quả/hình gốc được trích riêng tại `reports/notebook/` với SHA-256 nguồn.

**Đây là bản tham chiếu lịch sử.** MLP dùng test split khác; Tiny-ViT có
train/validation overlap và positional embedding cần được quản lý lại.
Xem `docs/methodology.md`. Notebook gốc có nhiều bước tuning/training tốn thời
gian và các lệnh lưu model tương đối theo thư mục hiện tại.

Để chạy quy trình mới với phân chia dữ liệu thống nhất, mở `01_workflow.ipynb`
hoặc dùng CLI trong README. File gốc ngoài repository không bị thay đổi.
''')
    for index,cell in enumerate(notebook.cells):
        cell.metadata={}
        if notebook.nbformat_minor>=5:
            cell.id=f'reference-{index:03}'
        else:
            cell.pop('id',None)
        if cell.cell_type=='code':
            cell.outputs=[]
            cell.execution_count=None
    notebook.cells.insert(0,intro)
    if notebook.nbformat_minor<5:
        intro.pop('id',None)
    notebook.metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'},
                       'language_info':{'name':'python'}}
    (root/'notebooks').mkdir(exist_ok=True)
    nbformat.write(notebook,root/'notebooks'/'AI-Final.ipynb')
    print(f'Preserved {len(notebook.cells)-1} source cells; exported {len(provenance["figure_cells"])} historical figures.')


if __name__=='__main__': main()
