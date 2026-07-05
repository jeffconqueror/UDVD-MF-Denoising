"""Confusion matrix + per-class metrics for the new best config: merged-real model + 8-fold TTA."""
import json, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from dataset import TEMParticleDataset, build_transforms

CLASSES=['Dh','FCC','Ih+Ih_to_Dh']
SP='/shared/jingchl6/material/lc-research/classifier_runs/splits_particle/splits.json'
CK='/shared/jingchl6/material/lc-research/classifier_runs/swin_3c_merged_warmstart/best.pt'
OUT='/shared/jingchl6/material/lc-research/classifier_runs/swin_3c_merged_warmstart'
dev=torch.device('cuda')
ck=torch.load(CK,map_location=dev,weights_only=False)
m=timm.create_model(ck['arch'],pretrained=False,num_classes=3); m.load_state_dict(ck['state_dict']); m=m.to(dev).eval()
d=json.load(open(SP)); val=[(p,2 if y==3 else y) for p,y in d['val']]; y=np.array([t for _,t in val])
ds=TEMParticleDataset(val,transform=build_transforms(224,train=False))
ld=DataLoader(ds,batch_size=64,shuffle=False,num_workers=8,pin_memory=True)
def dih(x,k):
    if k>=4: x=torch.flip(x,dims=[3])
    return torch.rot90(x,k%4,dims=[2,3])
P=[]
with torch.no_grad():
    for x,_ in ld:
        x=x.to(dev); acc=torch.zeros(x.size(0),3)
        for k in range(8): acc+=F.softmax(m(dih(x,k)),1).cpu()
        P.append(acc/8)
p=torch.cat(P).numpy(); pred=p.argmax(1); acc=(pred==y).mean()
cm=np.zeros((3,3),int)
for a,b in zip(y,pred): cm[a,b]+=1
print(f'BEST (merged-real + 8-fold TTA): val_acc={acc*100:.2f}%')
for ci,n in enumerate(CLASSES):
    tp=cm[ci,ci]; fn=cm[ci].sum()-tp; fp=cm[:,ci].sum()-tp
    pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1); f1=2*pr*rc/max(pr+rc,1e-9)
    print(f'  {n:<12} prec={pr:.3f} rec={rc:.3f} f1={f1:.3f} n={cm[ci].sum()}')
fig,ax=plt.subplots(figsize=(7,6)); cmn=cm/cm.sum(1,keepdims=True).clip(min=1)
ax.imshow(cmn,cmap='Blues',vmin=0,vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES,rotation=20); ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'merged-real + 8-fold TTA  val_acc={acc*100:.2f}%')
for i in range(3):
    for j in range(3): ax.text(j,i,f'{cm[i,j]}\n{cmn[i,j]*100:.1f}%',ha='center',va='center',color='white' if cmn[i,j]>0.5 else 'black',fontsize=10)
plt.tight_layout(); plt.savefig(f'{OUT}/confusion_matrix_TTA.png',dpi=120,bbox_inches='tight')
print(f'Saved: {OUT}/confusion_matrix_TTA.png')
