"""Plot held-out structural-label errors and development-only training history."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from P2.hover_bptt.io import read,write,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run)
    target=root/'prediction/model_errors.png'
    if target.exists():raise FileExistsError('Preserve previous prediction figure')
    fit=read(root/'residual/training_summary.json');test=read(root/'prediction/summary.json');b,r=test['groups'][0]['values']
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(2,2,figsize=(12,8),dpi=110);x=np.arange(3);w=.35
    for j,score in enumerate([b,r]):
        label=['标称模型','学习残差'][j]
        ax[0,0].bar(x+(j-.5)*w,score['wrench_rmse_n_nm'][:3],w,label=label)
        ax[0,1].bar(x+(j-.5)*w,np.array(score['wrench_rmse_n_nm'][3:])*1000,w,label=label)
    ax[0,0].set(xticks=x,xticklabels=['Fx','Fy','Fz'],ylabel='力残差预测RMSE / N')
    ax[0,1].set(xticks=x,xticklabels=['τx','τy','τz'],ylabel='力矩残差预测RMSE / mNm')
    ax[1,0].bar([0,1],[b['motor_response_rmse_radps'],r['motor_response_rmse_radps']],color=['tab:blue','tab:orange'])
    ax[1,0].set(xticks=[0,1],xticklabels=['标称模型','学习残差'],ylabel='2ms电机响应RMSE / (rad/s)',title='电机响应泛化未改善')
    history=fit['history'];ax[1,1].plot([h['step'] for h in history],[h['loss'] for h in history],marker='o')
    ax[1,1].axvline(fit['selected_step'],color='red',ls=':',label='开发集选择步数')
    ax[1,1].set(xlabel='仅残差MLP更新次数',ylabel='开发集综合损失',title=f'优化及开发评分{fit["wall_s"]:.2f}s；Actor更新0')
    for axis in ax.flat:axis.grid(alpha=.25,axis='y')
    for axis in [ax[0,0],ax[0,1],ax[1,1]]:axis.legend(fontsize=9)
    fig.suptitle('技术路线：BPTT｜P3冻结26次策略｜同一外部绳端力下的结构误差学习\n留出参数组合；预测误差与闭环收益分别评价',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(target);plt.close(fig)
    write(root/'prediction/model_errors.json',dict(source_sha256=digest(__file__),test_summary_sha256=digest(root/'prediction/summary.json'),
        training_summary_sha256=digest(root/'residual/training_summary.json'),new_training=False,test_used_for_selection=False,
        motor_generalization_improved=r['motor_response_rmse_radps']<b['motor_response_rmse_radps']))


if __name__=='__main__':main()
