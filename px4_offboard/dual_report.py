"""Render historical interface statistics after user-authorized raw-data cleanup."""
from pathlib import Path
from .dual_audit import read


def main():
    base=Path(__file__).resolve().parents[1]/'px4_offboard/acceleration_dual'
    archive=read(base/'history/development_20261116.json')
    lines=['# 早期PX4接口开发统计（历史）','','最新P3→P2完整闭环及动图见[observer_report.md](observer_report.md)，同内环P2基线见[diagnosis.md](diagnosis.md)。本页不是当前策略成绩。','',
        '按用户2026-11-16整理授权，full_01/full_02早期调试轨迹、ULog和旧动图已删除；删除前八例原始轨迹、完整120s失效计罚和四份动图通过核验。原始汇总、逐例审计、飞行摘要、接口配置和唯一源码快照合并到[历史统计](history/development_20261116.json)。保留审计中的原始哈希供来源追踪，不宣称已删除的轨迹仍能本地重新计算。','',
        '| 批次／接口／风m/s | 实录／计划s | 三维／高度／水平RMSE m | 安全／稳态 |',
        '| --- | ---: | ---: | --- |']
    for c in archive['cases']:
        a=c['audit'];values='/'.join(f'{v:.3f}' for v in a['rmse_3d_h_xy_m'])
        lines.append(f"| {c['batch']}／{a['interface']}／{a['wind_mps']:g} | {a['physical_post_duration_s']:.3f}/120 | {values} | {a['safe_completed']}／{a['settled']} |")
    lines+=['','full_01为MAVLink遥测输入，full_02为MAVROS输入，两者均实际MAVROS输出且P3关闭。弱风1m/s恒定，强风前5s由1升至5m/s；种子2026114501/4503。早期Iris增益与实际XML机体不匹配，后续诊断单列，不能混作新P3效果。','',
        '早期目标按夹具气压参考评分，节点内部目标当时未单独录制。XML实际2.369kg，旧config.actual的2.3kg为错误元数据；P2标准仍2.3kg。PX4仍有GPS，Actor没有水平导航真值。没有正式独立或实飞验收。','',
        '[最新代码、一键入口和输入边界](README.md)']
    (base/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Historical report saved')


if __name__=='__main__':main()
