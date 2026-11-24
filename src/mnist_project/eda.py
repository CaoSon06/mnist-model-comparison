"""Headless EDA artifacts, complementary to the original detailed notebook."""
from pathlib import Path
import numpy as np
from .experiments import write_json


def create_eda(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(output)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'{directory} already contains files.')
    directory.mkdir(parents=True,exist_ok=True)
    summary={}
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,name,x,y in zip(axes,['train','validation','test'],[data.x_train,data.x_val,data.x_test],
                           [data.y_train,data.y_val,data.y_test]):
        counts=np.bincount(y,minlength=10)
        summary[name]={'samples':len(y),'class_counts':counts.tolist(),
                       'mean_intensity':float(x.mean()),'background_ratio':float((x==0).mean())}
        ax.bar(range(10),counts); ax.set(title=name,xlabel='Digit',ylabel='Count',xticks=range(10))
    fig.tight_layout(); fig.savefig(directory/'class_distribution.png',dpi=140); plt.close(fig)
    fig,axes=plt.subplots(2,10,figsize=(16,4))
    for digit in range(10):
        values=data.x_train[data.y_train==digit]
        for row,values_to_show in enumerate([values[0] if len(values) else np.zeros((28,28,1)),
                                             values.mean(axis=0) if len(values) else np.zeros((28,28,1))]):
            axes[row,digit].imshow(values_to_show.squeeze(),cmap='gray')
            axes[row,digit].set_title(str(digit)); axes[row,digit].axis('off')
    fig.tight_layout(); fig.savefig(directory/'samples_and_means.png',dpi=140); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].hist(data.x_train[:min(500,len(data.x_train))].ravel(),bins=40)
    axes[0].set(title='Normalized pixel distribution',xlabel='Intensity')
    axes[1].hist((data.x_train==0).mean(axis=(1,2,3)),bins=30)
    axes[1].set(title='Image sparsity',xlabel='Fraction of background pixels')
    fig.tight_layout(); fig.savefig(directory/'pixels_and_sparsity.png',dpi=140); plt.close(fig)
    write_json(directory/'summary.json',summary)
    return summary
