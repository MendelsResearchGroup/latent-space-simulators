"""Post-fit thermal response: two time-OLS strain fits with intercepts.

Delegates to the existing frame_time_ols_p_ratio in the existing thermal engineering metric:
fixed frame-zero 10% boundary groups; normalize widths/heights by their first
values; p=-slope_y/slope_x; >=8 finite frames and >=1e-3 fitted axial strain.
"""
import numpy as np

METHOD='physical_strain_time_ols_intercept_min8_axial1e-3_fixed_sides_q0.10'

def fit_response(width, height, frame):
    """Adapt cached side dimensions to the existing thermal evaluator."""
    from thermal_metric import frame_time_ols_p_ratio
    width=np.asarray(width,dtype=float);height=np.asarray(height,dtype=float);frame=np.asarray(frame,dtype=float)
    assert width.shape==height.shape==frame.shape and width.ndim==1
    # The existing metric needs only these four side means, not interior nodes.
    positions=np.zeros((len(frame),4,2),dtype=float)
    positions[:,1,0]=width;positions[:,3,1]=height
    sides={'left':np.array([0]),'right':np.array([1]),'bottom':np.array([2]),'top':np.array([3])}
    result=frame_time_ols_p_ratio(positions,sides,frame_index=frame,min_frames=8,min_axial_total_strain=1e-3)
    keys=('p_ratio','valid_frames','total_frames','slope_x','slope_y','fit_r2_x','fit_r2_y','fitted_axial_total_strain')
    return {key:result[key] for key in keys}

def apply_mixed_fits(table):
    """Replace mixed response only; preserve endpoint columns for audit."""
    table=table.copy()
    table['true_p_ratio_endpoint']=table['true_p_ratio'];table['pred_p_ratio_endpoint']=table['pred_p_ratio']
    mixed=table.loc[table['source'].str.endswith('_mixed')]
    for _,group in mixed.groupby(['source','trajectory_id','mode'],sort=False):
        group=group.sort_values('frame')
        for count,index in enumerate(group.index,start=1):
            prefix=group.iloc[:count]
            for kind in ('true','pred'):
                fit=fit_response(prefix[kind+'_width'],prefix[kind+'_height'],prefix['frame'])
                table.loc[index,kind+'_p_ratio']=fit['p_ratio']
                for key,value in fit.items():
                    if key!='p_ratio': table.loc[index,kind+'_fit_'+key]=value
            table.loc[index,'p_ratio_method']=METHOD
    table['p_ratio_valid']=np.isfinite(table['true_p_ratio'])&np.isfinite(table['pred_p_ratio'])
    return table
