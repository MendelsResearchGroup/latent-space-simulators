"""Periodic spring contacts and continuous, linear-step feasibility checks."""
from itertools import product
import numpy as np


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def _point_distance(point, start, vector):
    fraction=np.clip(np.sum((point-start)*vector, axis=1)/np.sum(vector*vector, axis=1), 0., 1.)
    return np.linalg.norm(point-start-fraction[:, None]*vector, axis=1)


def _measures(p, v, q, w, tolerance):
    """Minimum distance and contacts away from an ordinary shared endpoint."""
    lv, lw=np.linalg.norm(v, axis=1), np.linalg.norm(w, axis=1)
    distance=np.minimum.reduce([_point_distance(p,q,w), _point_distance(p+v,q,w),
                                _point_distance(q,p,v), _point_distance(q+w,p,v)])
    denominator=_cross(v,w)
    parallel=np.abs(denominator)<=tolerance*np.maximum(lv,lw)
    t=np.divide(_cross(q-p,w), denominator, out=np.zeros(len(p)), where=~parallel)
    u=np.divide(_cross(q-p,v), denominator, out=np.zeros(len(p)), where=~parallel)
    et, eu=tolerance/lv, tolerance/lw
    hit=(~parallel)&(t>=-et)&(t<=1+et)&(u>=-eu)&(u<=1+eu)
    interior=hit&(((t>et)&(t<1-et))|((u>eu)&(u<1-eu)))
    distance[hit]=0.
    collinear=parallel&(np.abs(_cross(q-p,v))<=tolerance*lv)
    coordinate=np.sum((q-p)*v,axis=1)/lv
    end=coordinate+np.sum(w*v,axis=1)/lv
    overlap=np.minimum(lv,np.maximum(coordinate,end))-np.maximum(0.,np.minimum(coordinate,end))
    return distance, interior|(collinear&(overlap>tolerance))


class PlanarConstraints:
    """Preserve clearances and prohibit crossing anywhere along an Adam move.

    Nonincident spring clearance is min(fraction * median original length,
    half the original pair clearance). Shared joints are allowed; overlaps
    beyond a shared joint are not. Nine neighboring periodic images are used.
    """
    def __init__(self, position, edge_index, box, clearance_fraction: float):
        self.reference=np.asarray(position, dtype=float).copy()
        self.edges=np.unique(np.sort(np.asarray(edge_index, dtype=int), axis=0), axis=1).T
        self.box=np.asarray(box,dtype=float)
        self.i,self.j=np.triu_indices(len(self.edges),1)
        self.shared=np.any(self.edges[self.i,:,None]==self.edges[self.j,None,:],axis=(1,2))
        self.images=np.round((self.reference[self.edges[:,1]]-self.reference[self.edges[:,0]])/self.box)
        p,v=self.segments(self.reference)
        lengths=np.linalg.norm(v,axis=1)
        if np.any(lengths<=0):
            raise ValueError('Collapsed original spring')
        self.tolerance=1e-8*float(np.median(lengths))
        if not 0 < clearance_fraction < 1:
            raise ValueError('Spring clearance fraction must lie between zero and one')
        self.clearance_fraction=float(clearance_fraction)
        self.target_clearance=self.clearance_fraction*float(np.median(lengths))
        self.shifts=np.array(list(product((-1,0,1),repeat=2)))*self.box
        original_distance=np.full(len(self.i),2*self.target_clearance)
        for shift in self.shifts:
            ids=self.near_pairs(p,v,p,v,shift,2*self.target_clearance)
            a,b=self.i[ids],self.j[ids]
            d,_=_measures(p[a],v[a],p[b]+shift,v[b],self.tolerance)
            original_distance[ids]=np.minimum(original_distance[ids],d)
        self.clearance=np.maximum(self.tolerance,np.minimum(self.target_clearance,.5*original_distance))

    def segments(self, position):
        position=np.asarray(position,dtype=float)
        p=position[self.edges[:,0]]
        return p,position[self.edges[:,1]]-p-self.images*self.box

    def near_pairs(self,p,v,q,w,shift,margin):
        amin,amax=np.minimum(p,p+v),np.maximum(p,p+v)
        bmin,bmax=np.minimum(q,q+w)+shift,np.maximum(q,q+w)+shift
        close=np.all((amax[self.i]+margin>=bmin[self.j])&(bmax[self.j]+margin>=amin[self.i]),axis=1)
        return np.flatnonzero(close)

    def violating_pairs(self, position, *, enforce_clearance: bool=True):
        p,v=self.segments(position)
        bad=[]
        for shift in self.shifts:
            ids=self.near_pairs(p,v,p,v,shift,self.target_clearance if enforce_clearance else self.tolerance)
            a,b=self.i[ids],self.j[ids]
            distance,shared_contact=_measures(p[a],v[a],p[b]+shift,v[b],self.tolerance)
            limit=self.clearance[ids] if enforce_clearance else self.tolerance
            contact=np.where(self.shared[ids],shared_contact,distance<=limit)
            bad.extend(ids[contact])
        return np.unique(bad).astype(int)

    @staticmethod
    def orientation_roots(a, b, c, da, db, dc):
        v,w=b-a,c-a
        dv,dw=db-da,dc-da
        qa,qb,qc=_cross(dv,dw),_cross(dv,w)+_cross(v,dw),_cross(v,w)
        scale=np.maximum.reduce([np.abs(qa),np.abs(qb),np.abs(qc),np.full(len(a),1e-30)])
        quadratic=np.abs(qa)>1e-12*scale
        discriminant=qb*qb-4*qa*qc
        real=quadratic&(discriminant>=0)
        root=np.sqrt(np.maximum(discriminant,0))
        first=np.divide(-qb-root,2*qa,out=np.full(len(a),np.nan),where=real)
        second=np.divide(-qb+root,2*qa,out=np.full(len(a),np.nan),where=real)
        linear=(~quadratic)&(np.abs(qb)>1e-12*scale)
        first[linear]=-qc[linear]/qb[linear]
        return first,second

    def swept_pairs(self, previous, candidate):
        p,v=self.segments(previous); pn,vn=self.segments(candidate)
        dp,dv=pn-p,vn-v
        lo=np.minimum.reduce([p,p+v,pn,pn+vn]);hi=np.maximum.reduce([p,p+v,pn,pn+vn])
        bad=[]
        for shift in self.shifts:
            near=np.all((hi[self.i]+self.tolerance>=lo[self.j]+shift)&(hi[self.j]+shift+self.tolerance>=lo[self.i]),axis=1)
            ids=np.flatnonzero(near);a,b=self.i[ids],self.j[ids]
            aa,bb,cc,dd=p[a],p[a]+v[a],p[b]+shift,p[b]+v[b]+shift
            da,db,dc,ddelta=dp[a],dp[a]+dv[a],dp[b],dp[b]+dv[b]
            triples=((aa,bb,cc,da,db,dc),(aa,bb,dd,da,db,ddelta),
                     (cc,dd,aa,dc,ddelta,da),(cc,dd,bb,dc,ddelta,db))
            for triple in triples:
                for time in self.orientation_roots(*triple):
                    valid=np.isfinite(time)&(time>1e-10)&(time<1-1e-10)
                    selected=ids[valid];at,bt=a[valid],b[valid];t=time[valid,None]
                    distance,shared_contact=_measures(p[at]+t*dp[at],v[at]+t*dv[at],
                        p[bt]+shift+t*dp[bt],v[bt]+t*dv[bt],self.tolerance*10)
                    hit=np.where(self.shared[selected],shared_contact,distance<=self.tolerance*10)
                    bad.extend(selected[hit])
        return np.unique(bad).astype(int)

    def blocked_nodes(self, previous, candidate):
        blocked=np.zeros(len(self.reference),dtype=bool)
        candidate=np.asarray(candidate,dtype=float)
        winding=np.round((candidate[self.edges[:,1]]-candidate[self.edges[:,0]])/self.box)
        changed=np.any(winding!=self.images,axis=1)
        blocked[self.edges[changed].ravel()]=True
        _,old_vector=self.segments(previous);_,new_vector=self.segments(candidate)
        movement=new_vector-old_vector
        denominator=np.sum(movement*movement,axis=1)
        time=np.clip(np.divide(-np.sum(old_vector*movement,axis=1),denominator,out=np.zeros(len(movement)),where=denominator>0),0,1)
        collapsed=np.linalg.norm(old_vector+time[:,None]*movement,axis=1)<=self.tolerance
        blocked[self.edges[collapsed].ravel()]=True
        if blocked.any():
            return blocked
        pairs=np.union1d(self.violating_pairs(candidate),self.swept_pairs(previous,candidate))
        blocked[self.edges[self.i[pairs]].ravel()]=True
        blocked[self.edges[self.j[pairs]].ravel()]=True
        return blocked

    def summary(self):
        return dict(periodic=True,continuous_linear_step_check=True,target_clearance=self.target_clearance,
                    minimum_pair_clearance=float(self.clearance[~self.shared].min()),
                    clearance_fraction=self.clearance_fraction,clearance_rule='min(fraction * median original spring length, 0.5 * original pair distance); numerical tolerance floor',
                    shared_joints_allowed=True,collinear_overlap_forbidden=True)
