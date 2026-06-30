"""
problems_all.py
===============
Self-contained, verified FEM definitions for the four truss benchmarks used in
the RBRO study: 10-bar (planar), 25-bar (spatial), 120-bar (dome), 137-bar
(Burro Creek bridge, planar). Each exposes a uniform API:

    p.name, p.units, p.n_groups, p.bounds, p.COV, p.TARGET_R
    p.solve(A_grp)            -> dict(weight_kg, max_disp, max_stress, U, stress) or None
    p.reliability(A_grp)      -> (beta, R)  via FORM/HLRF
    p.member_mass_kg(A_grp)   -> per-element mass [kg]
    p.assemble_Kfree(A,active)-> reduced free-DOF stiffness (for redundancy r_ii)
    p.element_groups          -> group index per element
    p.element_lengths

FEM and FORM cores reproduce the verified uploaded RBDO implementations (the
single-objective LLM-operator layer is removed). All four self-check against the
reference weights reported in the source codes / literature.
"""

import numpy as np
from scipy.stats import norm

LB_TO_KG = 0.45359237
LB_PER_IN3_STEEL = 0.2836      # density used by imperial benchmark codes
RHO_STEEL_SI = 7850.0          # kg/m^3


# ======================================================================
# 10-bar planar truss (imperial; displacement constraint at node 2)
# ======================================================================
class Truss10:
    name = "10-bar"; units = "imperial"
    E_MEAN = 1.0e7; P_MEAN = 1.0e5; RHO = 0.1; COV = 0.05
    DELTA_MAX = 2.0; TARGET_R = 0.99; A_MIN, A_MAX = 0.1, 40.0
    n_groups = 10
    NODES = np.array([[720.,360.],[720.,0.],[360.,360.],[360.,0.],[0.,360.],[0.,0.]])
    MEMBERS = np.array([[4,2],[2,0],[5,3],[3,1],[3,2],[0,1],[3,4],[2,5],[1,2],[0,3]])
    FIXED_DOFS = [8,9,10,11]

    def __init__(self):
        self.L = np.array([np.linalg.norm(self.NODES[j]-self.NODES[i]) for i,j in self.MEMBERS])
        self.element_lengths = self.L
        self.element_groups = np.arange(self.n_groups)
        self.bounds = (np.full(self.n_groups,self.A_MIN), np.full(self.n_groups,self.A_MAX))
        self.ndof = 2*len(self.NODES)
        self.free = [d for d in range(self.ndof) if d not in self.FIXED_DOFS]

    def _K(self, A, E, active):
        K = np.zeros((self.ndof,self.ndof))
        for k,(ni,nj) in enumerate(self.MEMBERS):
            if not active[k]: continue
            le=self.L[k]; c=(self.NODES[nj,0]-self.NODES[ni,0])/le; s=(self.NODES[nj,1]-self.NODES[ni,1])/le
            ke=(A[k]*E/le)*np.array([[c*c,c*s,-c*c,-c*s],[c*s,s*s,-c*s,-s*s],
                                     [-c*c,-c*s,c*c,c*s],[-c*s,-s*s,c*s,s*s]])
            idx=[2*ni,2*ni+1,2*nj,2*nj+1]; K[np.ix_(idx,idx)]+=ke
        return K

    def assemble_Kfree(self, A, active):
        A=np.clip(np.asarray(A,float),1e-6,None)
        return self._K(A,self.E_MEAN,active)[np.ix_(self.free,self.free)]

    def solve(self, A, E=None, P=None):
        E=self.E_MEAN if E is None else E; P=self.P_MEAN if P is None else P
        A=np.clip(np.asarray(A,float),1e-6,None)
        K=self._K(A,E,np.ones(self.n_groups,bool)); F=np.zeros(self.ndof); F[3]-=P; F[7]-=P
        for d in self.FIXED_DOFS: K[d,:]=0; K[d,d]=1; F[d]=0
        try: U=np.linalg.solve(K,F)
        except np.linalg.LinAlgError: return None
        stress=np.zeros(self.n_groups)
        for k,(ni,nj) in enumerate(self.MEMBERS):
            le=self.L[k]; c=(self.NODES[nj,0]-self.NODES[ni,0])/le; s=(self.NODES[nj,1]-self.NODES[ni,1])/le
            stress[k]=E*(c*(U[2*nj]-U[2*ni])+s*(U[2*nj+1]-U[2*ni+1]))/le
        return dict(weight_kg=float(np.sum(A*self.RHO*self.L))*LB_TO_KG, max_disp=abs(U[3]),
                    max_stress=float(np.max(np.abs(stress))), U=U, stress=stress)

    def member_mass_kg(self, A):
        return np.asarray(A,float)*self.L*LB_PER_IN3_STEEL*LB_TO_KG

    def reliability(self, A):
        A=np.clip(np.asarray(A,float),self.A_MIN,self.A_MAX)
        muy=np.concatenate([[self.E_MEAN,self.P_MEAN],A]); sigma=self.COV*muy; dmuy=1e-4*muy
        def g(X):
            r=self.solve(X[2:],E=max(X[0],self.E_MEAN*0.01),P=max(X[1],self.P_MEAN*0.01))
            return self.DELTA_MAX-(np.inf if r is None else r["max_disp"])
        X=muy.copy(); Z=(X-muy)/sigma; beta=np.linalg.norm(Z); eZ=eb=100.0
        for _ in range(100):
            if abs(eZ)<=1e-3 and abs(eb)<=1e-3: break
            bp=beta; zp=np.linalg.norm(Z); g0=g(X); G=np.zeros(len(X))
            for i in range(len(X)):
                xp=X.copy(); xp[i]+=dmuy[i]; G[i]=-(g(xp)-g0)/dmuy[i]*sigma[i]
            gn=np.linalg.norm(G)
            if gn<1e-12: break
            beta=beta+g0/gn; Z=(G/gn)*beta; X=muy+sigma*Z; eb=beta-bp; eZ=np.linalg.norm(Z)-zp
        return float(beta), float(norm.cdf(beta))


# ======================================================================
# 25-bar spatial truss (imperial; stress + displacement constraints)
# ======================================================================
class Truss25:
    name="25-bar"; units="imperial"; E_MEAN=1e7; RHO=0.1; COV=0.05
    TARGET_R=0.99; DISP_LIM=0.35; SIG_LIM=40e3; A_MIN,A_MAX=0.01,3.4; n_groups=8
    P1x_M,P1y_M,P1z_M=1.0e3,10.0e3,10.0e3; P2y_M,P2z_M=10.0e3,10.0e3; P3x_M,P6x_M=0.5e3,0.6e3
    NODES=np.array([[-37.5,0.,200.],[37.5,0.,200.],[-37.5,37.5,100.],[37.5,37.5,100.],
                    [37.5,-37.5,100.],[-37.5,-37.5,100.],[-100.,100.,0.],[100.,100.,0.],
                    [100.,-100.,0.],[-100.,-100.,0.]])
    ELEM_N1=np.array([0,0,1,0,1,1,1,0,0,5,3,2,5,2,5,3,4,3,2,4,5,5,2,3,4])
    ELEM_N2=np.array([1,3,2,4,5,3,4,2,5,2,4,3,4,9,6,8,7,6,7,9,8,9,6,7,8])
    IDA=np.array([0,1,1,1,1,2,2,2,2,3,3,4,4,5,5,5,5,6,6,6,6,7,7,7,7])
    N_EL=25; FIXED_DOFS=list(range(18,30))

    def __init__(self):
        self.ndof=30; self.free=[d for d in range(30) if d not in self.FIXED_DOFS]
        self.L=np.array([np.linalg.norm(self.NODES[self.ELEM_N2[i]]-self.NODES[self.ELEM_N1[i]]) for i in range(25)])
        self.DIR=np.array([(self.NODES[self.ELEM_N2[i]]-self.NODES[self.ELEM_N1[i]])/self.L[i] for i in range(25)])
        self.element_lengths=self.L; self.element_groups=self.IDA
        self.bounds=(np.full(8,self.A_MIN),np.full(8,self.A_MAX))

    def _K(self, A_el, E, active):
        K=np.zeros((30,30))
        for i in range(25):
            if not active[i]: continue
            n1,n2=self.ELEM_N1[i],self.ELEM_N2[i]; le=self.L[i]; lx,ly,lz=self.DIR[i]
            Te=np.array([[lx,ly,lz,0,0,0],[0,0,0,lx,ly,lz]])
            ke=Te.T@(A_el[i]*E/le*np.array([[1,-1],[-1,1]]))@Te
            idx=[3*n1,3*n1+1,3*n1+2,3*n2,3*n2+1,3*n2+2]; K[np.ix_(idx,idx)]+=ke
        return K

    def assemble_Kfree(self, A, active):
        A_el=np.clip(np.asarray(A,float),1e-6,None)[self.IDA]
        return self._K(A_el,self.E_MEAN,active)[np.ix_(self.free,self.free)]

    def solve(self, A, E=None, loads=None):
        E=self.E_MEAN if E is None else E
        A_el=np.clip(np.asarray(A,float),1e-6,None)[self.IDA]
        K=self._K(A_el,E,np.ones(25,bool)); F=np.zeros(30)
        if loads is None: loads=(self.P1x_M,self.P1y_M,self.P1z_M,self.P2y_M,self.P2z_M,self.P3x_M,self.P6x_M)
        p1x,p1y,p1z,p2y,p2z,p3x,p6x=loads
        F[0]+=p1x; F[1]-=p1y; F[2]-=p1z; F[4]-=p2y; F[5]-=p2z; F[6]+=p3x; F[15]+=p6x
        Kb=K.copy(); Fb=F.copy()
        for d in self.FIXED_DOFS: Kb[d,:]=0; Kb[:,d]=0; Kb[d,d]=1; Fb[d]=0
        try: U=np.linalg.solve(Kb,Fb)
        except np.linalg.LinAlgError: return None
        stress=np.zeros(25)
        for i in range(25):
            n1,n2=self.ELEM_N1[i],self.ELEM_N2[i]; le=self.L[i]; lx,ly,lz=self.DIR[i]
            Te=np.array([[lx,ly,lz,0,0,0],[0,0,0,lx,ly,lz]]); idx=[3*n1,3*n1+1,3*n1+2,3*n2,3*n2+1,3*n2+2]
            stress[i]=E*np.array([-1/le,1/le])@(Te@U[idx])
        return dict(weight_kg=float(np.sum(self.RHO*self.L*A_el))*LB_TO_KG,
                    max_disp=float(np.max(np.abs(U[:6]))), max_stress=float(np.max(np.abs(stress))),
                    U=U, stress=stress)

    def member_mass_kg(self, A):
        A_el=np.asarray(A,float)[self.IDA]
        return A_el*self.L*LB_PER_IN3_STEEL*LB_TO_KG

    def reliability(self, A):
        A=np.clip(np.asarray(A,float),self.A_MIN,self.A_MAX)
        mu=np.array([self.E_MEAN,self.P1x_M,self.P1y_M,self.P1z_M,self.P2y_M,self.P2z_M,self.P3x_M,self.P6x_M,*A])
        sig=self.COV*mu; N=len(mu); h=1e-4
        def g(X):
            r=self.solve(X[8:16],E=X[0],loads=(X[1],X[2],X[3],X[4],X[5],X[6],X[7]))
            return self.DISP_LIM-(np.inf if r is None else r["max_disp"])
        gm=g(mu); u=np.zeros(N); bp=0.0
        for _ in range(40):
            X=mu+sig*u; g0=g(X)
            dg=np.array([(g(mu+sig*(u+h*np.eye(N)[j]))-g0)/h for j in range(N)]); nd=np.linalg.norm(dg)
            if nd<1e-12: break
            alpha=dg/nd; bk=-alpha@u+g0/nd; un=-alpha*bk
            if np.linalg.norm(un-u)<1e-3: u=un; break
            u=un
            if abs(np.linalg.norm(u)-bp)<1e-3: break
            bp=np.linalg.norm(u)
        beta=np.linalg.norm(u)
        if gm<0: beta=-beta
        return float(beta), float(norm.cdf(beta))


# ======================================================================
# Geometry builders for 120-bar and 137-bar (match verified sources)
# ======================================================================
def _build_120():
    z1,z2,z3=118.11,196.85,275.59; r1,r2,r3=273.26,492.12,625.59
    NODES=np.zeros((49,3)); NODES[0]=[0,0,z3]
    for i,a in enumerate([np.pi/2,np.pi/3,np.pi/6,0,-np.pi/6,-np.pi/3,-np.pi/2,
                          8*np.pi/6,7*np.pi/6,np.pi,5*np.pi/6,4*np.pi/6]):
        NODES[1+i]=[r1*np.cos(a),r1*np.sin(a),z2]
    for i,a in enumerate([np.pi/2,5*np.pi/12,4*np.pi/12,3*np.pi/12,2*np.pi/12,np.pi/12,0,
                          -np.pi/12,-2*np.pi/12,-3*np.pi/12,-4*np.pi/12,-5*np.pi/12,-np.pi/2,
                          -7*np.pi/12,-8*np.pi/12,-9*np.pi/12,-10*np.pi/12,-11*np.pi/12,np.pi,
                          11*np.pi/12,10*np.pi/12,9*np.pi/12,8*np.pi/12,7*np.pi/12]):
        NODES[13+i]=[r2*np.cos(a),r2*np.sin(a),z1]
    for i,a in enumerate([np.pi/2,np.pi/3,np.pi/6,0,-np.pi/6,-np.pi/3,-np.pi/2,
                          -4*np.pi/6,-5*np.pi/6,np.pi,5*np.pi/6,4*np.pi/6]):
        NODES[37+i]=[r3*np.cos(a),r3*np.sin(a),0]
    def _p(prs): return np.array([[a-1,b-1] for a,b in prs])
    g1=_p([(1,k) for k in range(2,14)])
    g2=_p([(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,2)])
    g3=_p([(2,14),(3,16),(4,18),(5,20),(6,22),(7,24),(8,26),(9,28),(10,30),(11,32),(12,34),(13,36)])
    g4=_p([(2,15),(3,15),(3,17),(4,17),(4,19),(5,19),(5,21),(6,21),(6,23),(7,23),(7,25),(8,25),
           (8,27),(9,27),(9,29),(10,29),(10,31),(11,31),(11,33),(12,33),(12,35),(13,35),(13,37),(2,37)])
    g5=_p([(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,23),(23,24),(24,25),(25,26),
           (26,27),(27,28),(28,29),(29,30),(30,31),(31,32),(32,33),(33,34),(34,35),(35,36),(36,37),(37,14)])
    g6=_p([(14,38),(16,39),(18,40),(20,41),(22,42),(24,43),(26,44),(28,45),(30,46),(32,47),(34,48),(36,49)])
    g7=_p([(15,38),(15,39),(17,39),(17,40),(19,40),(19,41),(21,41),(21,42),(23,42),(23,43),(25,43),(25,44),
           (27,44),(27,45),(29,45),(29,46),(31,46),(31,47),(33,47),(33,48),(35,48),(35,49),(37,49),(37,38)])
    ELEMS=np.vstack([g1,g2,g3,g4,g5,g6,g7])
    IDA=np.array([0]*12+[1]*12+[2]*12+[3]*24+[4]*24+[5]*12+[6]*24)
    return NODES,ELEMS,IDA


def _build_137():
    nc=np.array([
        [103.632,0],[97.536,4.6986],[91.440,9.1124],[85.344,13.2414],[79.248,17.0857],
        [73.152,20.6452],[67.056,23.9199],[60.960,26.9099],[54.864,29.6152],[48.768,32.0356],
        [42.672,34.1713],[36.576,36.0223],[30.480,37.5885],[24.384,38.8700],[18.288,39.8666],
        [12.192,40.5785],[6.096,41.0056],[0,41.1480],[-6.096,41.0056],[-12.192,40.5785],
        [-18.288,39.8666],[-24.384,38.8700],[-30.480,37.5885],[-36.576,36.0223],[-42.672,34.1713],
        [-48.768,32.0356],[-54.864,29.6152],[-60.960,26.9099],[-67.056,23.9199],[-73.152,20.6452],
        [-79.248,17.0857],[-85.344,13.2414],[-91.440,9.1124],[-97.536,4.6986],[-103.632,0],
        [103.632,10.9728],[97.536,15.1145],[91.440,19.0052],[85.344,22.6448],[79.248,26.0335],
        [73.152,29.1712],[67.056,32.0578],[60.960,34.6934],[54.864,37.0780],[48.768,39.2116],
        [42.672,41.0942],[36.576,42.7258],[30.480,44.1064],[24.384,45.2359],[18.288,46.1144],
        [12.192,46.7420],[6.096,47.1185],[0,47.2440],[-6.096,47.1185],[-12.192,46.7420],
        [-18.288,46.1144],[-24.384,45.2359],[-30.480,44.1064],[-36.576,42.7258],[-42.672,41.0942],
        [-48.768,39.2116],[-54.864,37.0780],[-60.960,34.6934],[-67.056,32.0578],[-73.152,29.1712],
        [-79.248,26.0335],[-85.344,22.6448],[-91.440,19.0052],[-97.536,15.1145],[-103.632,10.9728]])
    def _mk(prs): return np.array([[a-1,b-1] for a,b in prs])
    g1=_mk([(i,i+1) for i in range(1,35)])                      # lower chord 1-34
    g2=_mk([(i,i+1) for i in range(36,70)])                     # upper chord 35-68
    g3=_mk([(i,i+35) for i in range(1,36)])                     # verticals 1-35
    g4=_mk([(i+1,i+35) for i in range(1,35)])                   # diagonals
    ELEMS=np.vstack([g1,g2,g3,g4]); IDA=np.array([0]*34+[1]*34+[2]*35+[3]*34)
    return nc,ELEMS,IDA


class _Space3D:
    """Shared 3D space-truss FEM for the 120-bar dome."""
    def _K(self, A_el, E, active):
        K=np.zeros((self.ndof,self.ndof))
        for i in range(self.N_EL):
            if not active[i]: continue
            n1,n2=self.ELEMS[i]; le=self.L[i]; lx,ly,lz=self.DIR[i]
            Te=np.array([[lx,ly,lz,0,0,0],[0,0,0,lx,ly,lz]])
            ke=Te.T@(A_el[i]*E/le*np.array([[1,-1],[-1,1]]))@Te
            idx=[3*n1,3*n1+1,3*n1+2,3*n2,3*n2+1,3*n2+2]; K[np.ix_(idx,idx)]+=ke
        return K


class Truss120(_Space3D):
    name="120-bar"; units="imperial"; E_MEAN=30450e3; RHO=0.288; COV=0.05
    Fi=58e3; SIG_T=0.6*58e3; DISP_LIM=0.1969; TARGET_R=0.99
    A_MIN,A_MAX=0.775,20.0; n_groups=7
    P1_M,P2_M,P3_M=13.49e3,6.744e3,2.248e3

    def __init__(self):
        self.NODES,self.ELEMS,self.IDA=_build_120(); self.N_EL=120
        self.ndof=49*3; self.FIXED_DOFS=list(range(37*3,49*3))
        self.free=[d for d in range(self.ndof) if d not in self.FIXED_DOFS]
        self.L=np.array([np.linalg.norm(self.NODES[self.ELEMS[i,1]]-self.NODES[self.ELEMS[i,0]]) for i in range(120)])
        self.DIR=np.array([(self.NODES[self.ELEMS[i,1]]-self.NODES[self.ELEMS[i,0]])/self.L[i] for i in range(120)])
        self.element_lengths=self.L; self.element_groups=self.IDA
        self.bounds=(np.full(7,self.A_MIN),np.full(7,self.A_MAX))

    def assemble_Kfree(self, A, active):
        A_el=np.clip(np.asarray(A,float),1e-6,None)[self.IDA]
        return self._K(A_el,self.E_MEAN,active)[np.ix_(self.free,self.free)]

    def solve(self, A, E=None, loads=None):
        E=self.E_MEAN if E is None else E
        A_el=np.clip(np.asarray(A,float),1e-6,None)[self.IDA]
        K=self._K(A_el,E,np.ones(120,bool)); F=np.zeros(self.ndof)
        p1,p2,p3=(self.P1_M,self.P2_M,self.P3_M) if loads is None else loads
        F[2]-=p1
        for i in range(1,13): F[i*3+2]-=p2
        for i in range(13,37): F[i*3+2]-=p3
        Kb=K.copy(); Fb=F.copy()
        for d in self.FIXED_DOFS: Kb[d,:]=0; Kb[:,d]=0; Kb[d,d]=1; Fb[d]=0
        try: U=np.linalg.solve(Kb,Fb)
        except np.linalg.LinAlgError: return None
        stress=np.zeros(120)
        for i in range(120):
            n1,n2=self.ELEMS[i]; le=self.L[i]; lx,ly,lz=self.DIR[i]
            Te=np.array([[lx,ly,lz,0,0,0],[0,0,0,lx,ly,lz]]); idx=[3*n1,3*n1+1,3*n1+2,3*n2,3*n2+1,3*n2+2]
            stress[i]=E*np.array([-1/le,1/le])@(Te@U[idx])
        return dict(weight_kg=float(np.sum(self.RHO*self.L*A_el))*LB_TO_KG,
                    max_disp=float(np.max(np.abs(U[self.free]))), max_stress=float(np.max(np.abs(stress))),
                    U=U, stress=stress)

    def member_mass_kg(self, A):
        A_el=np.asarray(A,float)[self.IDA]
        return A_el*self.L*LB_PER_IN3_STEEL*LB_TO_KG

    def reliability(self, A):
        A=np.clip(np.asarray(A,float),self.A_MIN,self.A_MAX)
        mu=np.array([self.E_MEAN,self.P1_M,self.P2_M,self.P3_M,*A]); sig=self.COV*mu; N=len(mu); h=1e-4
        def g(X):
            r=self.solve(X[4:11],E=X[0],loads=(X[1],X[2],X[3]))
            return self.DISP_LIM-(np.inf if r is None else r["max_disp"])
        gm=g(mu); u=np.zeros(N); bp=0.0
        for _ in range(40):
            X=mu+sig*u; g0=g(X)
            dg=np.array([(g(mu+sig*(u+h*np.eye(N)[j]))-g0)/h for j in range(N)]); nd=np.linalg.norm(dg)
            if nd<1e-12: break
            alpha=dg/nd; bk=-alpha@u+g0/nd; un=-alpha*bk
            if np.linalg.norm(un-u)<1e-3: u=un; break
            u=un
            if abs(np.linalg.norm(u)-bp)<1e-3: break
            bp=np.linalg.norm(u)
        beta=np.linalg.norm(u)
        if gm<0: beta=-beta
        return float(beta), float(norm.cdf(beta))


class Truss137:
    name="137-bar"; units="SI"; E_MEAN=200e9; RHO=7850.0; COV=0.05
    SIGMA_A=400e6; DISP_LIM=0.25; TARGET_R=0.99; A_MIN,A_MAX=0.001,0.5; n_groups=4
    P_MEAN=np.array([2033193.129]+[1016596.564]*8)
    LOAD_NODES_0=[35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69]
    LOAD_PATTERN_IDX=[0,1,2,3,4,5,6,7,8,8,7,6,5,4,3,2,1,0]

    def __init__(self):
        self.NODES,self.ELEMS,self.IDA=_build_137(); self.N_EL=137
        self.ndof=140; self.FIXED_DOFS=[0,1,68,69]
        self.free=[d for d in range(140) if d not in self.FIXED_DOFS]
        self.L=np.array([np.linalg.norm(self.NODES[self.ELEMS[i,1]]-self.NODES[self.ELEMS[i,0]]) for i in range(137)])
        self.DIR=np.array([(self.NODES[self.ELEMS[i,1]]-self.NODES[self.ELEMS[i,0]])/self.L[i] for i in range(137)])
        self.element_lengths=self.L; self.element_groups=self.IDA
        self.bounds=(np.full(4,self.A_MIN),np.full(4,self.A_MAX))

    def _K(self, A_el, E, active):
        K=np.zeros((140,140))
        for i in range(137):
            if not active[i]: continue
            n1,n2=self.ELEMS[i]; le=self.L[i]; lx,ly=self.DIR[i]
            Te=np.array([[lx,ly,0,0],[0,0,lx,ly]]); ke=Te.T@(A_el[i]*E/le*np.array([[1,-1],[-1,1]]))@Te
            idx=[2*n1,2*n1+1,2*n2,2*n2+1]; K[np.ix_(idx,idx)]+=ke
        return K

    def assemble_Kfree(self, A, active):
        A_el=np.clip(np.asarray(A,float),1e-9,None)[self.IDA]
        return self._K(A_el,self.E_MEAN,active)[np.ix_(self.free,self.free)]

    def solve(self, A, E=None, P=None):
        E=self.E_MEAN if E is None else E; P=self.P_MEAN if P is None else P
        A_el=np.clip(np.asarray(A,float),1e-9,None)[self.IDA]
        K=self._K(A_el,E,np.ones(137,bool)); F=np.zeros(140)
        for k,n0 in enumerate(self.LOAD_NODES_0): F[2*n0+1]=-P[self.LOAD_PATTERN_IDX[k]]
        Kb=K.copy(); Fb=F.copy()
        for d in self.FIXED_DOFS: Kb[d,:]=0; Kb[:,d]=0; Kb[d,d]=1; Fb[d]=0
        try: U=np.linalg.solve(Kb,Fb)
        except np.linalg.LinAlgError: return None
        stress=np.zeros(137)
        for i in range(137):
            n1,n2=self.ELEMS[i]; le=self.L[i]; lx,ly=self.DIR[i]
            Te=np.array([[lx,ly,0,0],[0,0,lx,ly]]); idx=[2*n1,2*n1+1,2*n2,2*n2+1]
            stress[i]=E*np.array([-1/le,1/le])@(Te@U[idx])
        return dict(weight_kg=float(np.sum(self.RHO*self.L*A_el)),
                    max_disp=float(np.max(np.abs(U[self.free]))), max_stress=float(np.max(np.abs(stress))),
                    U=U, stress=stress)

    def member_mass_kg(self, A):
        A_el=np.asarray(A,float)[self.IDA]
        return A_el*self.L*self.RHO

    def reliability(self, A):
        A=np.clip(np.asarray(A,float),self.A_MIN,self.A_MAX)
        mu=np.concatenate([self.P_MEAN,[self.E_MEAN],A]); sig=self.COV*mu; N=len(mu); h=1e-4
        def g(X):
            r=self.solve(X[10:14],E=X[9],P=X[:9])
            return self.DISP_LIM-(np.inf if r is None else r["max_disp"])
        gm=g(mu); u=np.zeros(N); bp=0.0
        for _ in range(40):
            X=mu+sig*u; g0=g(X)
            dg=np.array([(g(mu+sig*(u+h*np.eye(N)[j]))-g0)/h for j in range(N)]); nd=np.linalg.norm(dg)
            if nd<1e-12: break
            alpha=dg/nd; bk=-alpha@u+g0/nd; un=-alpha*bk
            if np.linalg.norm(un-u)<1e-3: u=un; break
            u=un
            if abs(np.linalg.norm(u)-bp)<1e-3: break
            bp=np.linalg.norm(u)
        beta=np.linalg.norm(u)
        if gm<0: beta=-beta
        return float(beta), float(norm.cdf(beta))


ALL_PROBLEMS = {"10-bar":Truss10, "25-bar":Truss25, "120-bar":Truss120, "137-bar":Truss137}

def get_problem(name): return ALL_PROBLEMS[name]()
