import math
from .cases import training_cases

def decision(before,after):
    reasons=[];worst=[]
    try:
        specs=training_cases();n=len(specs);ids=range(n)
        fields=['attachment','wind_speed','direction_rad','wind_profile']
        metrics=['height_rmse_m','horizontal_rmse_m','position_rmse_m','maximum_true_yaw_error_deg']
        if any(len(s[k])!=n for s in [before,after]for k in fields+metrics+['completed']):
            raise ValueError('Incomplete physical course')
        if any(before[k]!=after[k]or before[k]!=[c[k]for c in specs]for k in fields):
            raise ValueError('Ordered physical cases differ')
        if any(not math.isfinite(s['cost'])or any(not math.isfinite(v)or v<0 for k in metrics for v in s[k])for s in [before,after]):
            raise ValueError('Finite actual scores required')
        if not all(after['completed']):reasons.append('unsafe')
        if after['cost']>=before['cost']-max(1e-7,abs(before['cost'])*1e-7):reasons.append('fullcost_not_lower')
        rms=lambda score,selected:math.sqrt(math.fsum(score['position_rmse_m'][i]**2 for i in selected)/len(selected))
        if rms(after,ids)>=rms(before,ids):reasons.append('overall_3D_not_lower')
        if max(after['maximum_true_yaw_error_deg'])>5.:reasons.append('heading_guard')
        for i in ids:
            b=before['position_rmse_m'][i];a=after['position_rmse_m'][i];speed=before['wind_speed'][i]
            if after['height_rmse_m'][i]>1.1*before['height_rmse_m'][i]+.01:reasons.append('individual_height_guard')
            if speed<=1.:
                if after['horizontal_rmse_m'][i]>1.1*before['horizontal_rmse_m'][i]+.01:
                    reasons.append('individual_weak_horizontal_guard')
                if b<=.5 and a>.5+1e-6:reasons.append('new_weak_target_failure')
            if speed==5.:
                if b<=1.5 and a>1.5+1e-6:reasons.append('new_strong_target_failure')
                if b>1.5 and a>b+.01+1e-9:reasons.append('bounded_failed_lane_regression')
        for geometry in [2,1]:
            weak=[i for i in ids if before['attachment'][i]==geometry and before['wind_speed'][i]<=1.]
            strong=[i for i in ids if before['attachment'][i]==geometry and before['wind_speed'][i]==5.]
            if not weak or not strong:raise ValueError('Both geometries require weak/strong coverage')
            if rms(after,weak)>max(.5,rms(before,weak))+1e-6:reasons.append('geometry_weak_group')
            bm=max(before['position_rmse_m'][i]for i in strong);am=max(after['position_rmse_m'][i]for i in strong)
            worst.append(dict(attachment=geometry,before_max_m=bm,after_max_m=am))
            if bm>1.5 and am>=bm-1e-6:reasons.append('strong_worst_lane_not_lower')
        return dict(accepted=not reasons,reasons=sorted(set(reasons)),trajectory_count=n,
            final_targets_unchanged=True,every_lane_guarded=True,
            baseline_safe_completed=sum(bool(v)for v in before['completed']),
            candidate_all_safe_required=True,strong_worst_lane_by_geometry=worst)
    except(KeyError,TypeError,ValueError,IndexError):
        return dict(accepted=False,reasons=['invalid_course_or_nonfinite_score'],final_targets_unchanged=True)
