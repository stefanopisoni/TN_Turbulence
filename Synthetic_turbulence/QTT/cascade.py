#This code uses torchtt (https://github.com/ion-g-ion/torchTT) for TN operations
import torchtt as tntt
import torch as tn
from tutils import *
import numpy as np
import numba as nb

def linear_i(mps, fscale, f_boundary, eps = 1e-15,epsilon = 0.1):
    """ Liner MPS interpolation
    """
    ni = len(mps.N)
    nc = fscale - ni 
    
    unif = X_qtt(ni)
    phi1 = tntt.TT(qtt_polynomial_cores([1,-1], nc))
    phi2 = tntt.TT(qtt_polynomial_cores([0,1], nc))
    sm = R_qtt( 2**ni -1, ni)
    id = I_qtt(ni)
    smy = zkron( id , sm)
    smx = zkron(sm,id)
    smxy = zkron(sm,sm)

    delta = dmps(2**ni -1,ni)

    if f_boundary[0] == 'gaussian':
        scale = ni
        base_scaling = 2.0 ** (-4.0 / 3.0)
        r = f_boundary[1]
        corner = 1 + (base_scaling**scale)*epsilon*np.random.randn()
        by = 1 + (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]).round(eps)
        bx = 1 +(base_scaling**scale) * epsilon *tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]).round(eps)
    
    elif f_boundary == 'zero':
        corner = 1
        by = tntt.ones([2]*ni)
        bx = tntt.ones([2]*ni)

    else:
        by = tntt.TT( f_boundary([ unif.full().flatten(), tn.ones(2**ni)]), [2]*ni)
        bx = tntt.TT( f_boundary([ tn.ones(2**ni), unif.full().flatten()]), [2]*ni)
        corner = f_boundary([tn.tensor(0),tn.tensor(0)])

    rmps = tntt.kron( mps, zkron( phi1 , phi1 ) ) + tntt.kron( smy @ mps  + zkron(by,delta) , zkron( phi1, phi2 ) )

    rmps += tntt.kron( smx @ mps + zkron(delta,bx), zkron( phi2 , phi1 ) ) + tntt.kron( smxy @ mps + zkron( sm @ by , delta ) + zkron( delta , sm @ bx ) + zkron( delta , delta)* corner, zkron( phi2 , phi2 ) )
   
    rmps = rmps.round(eps)
    return rmps


def cuadratic_i(mps, fscale, f_boundary, eps = 1e-15,epsilon = 0.1):
    # Quadratic 2d mps interpolation
    ni = len(mps.N)
    nc = fscale - ni + 1
    
    phi1 = tntt.TT(qtt_polynomial_cores([1,-3,2], nc))
    phi2 = tntt.TT(qtt_polynomial_cores([0,4,-4], nc))
    phi3 = tntt.TT(qtt_polynomial_cores([0,-1,2], nc))


    fttr00 = reduce(mps,0)
    fttr10 = reduce(mps,2)
    fttr01 = reduce(mps,1)
    fttr11 = reduce(mps,3)

    if f_boundary[0] == 'gaussian':
        scale = ni
        base_scaling = 2.0 ** (-4.0 / 3.0)
        r = f_boundary[1]
        corner = 1 + (base_scaling**scale)*epsilon*np.random.randn()
        by = 1 + (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]).round(eps)
        bx = 1 +(base_scaling**scale) * epsilon *tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]).round(eps)
        by0 = reduceg(by,0,-1)
        by1 = reduceg(by,1,-1)
        bx0 = reduceg(bx,0,-1)
        bx1 = reduceg(bx,1,-1)
    
    elif f_boundary == 'zero':
        corner = 1
        by = tntt.ones([2]*ni)
        bx = tntt.ones([2]*ni)
        by0 = tntt.ones([2]*(ni-1))
        by1 = tntt.ones([2]*(ni-1))
        bx0 = tntt.ones([2]*(ni-1))
        bx1 = tntt.ones([2]*(ni-1))

    else:
        unif = X_qtt(ni)
        fr = f_boundary[0]
        by = tntt.TT( tn.tensor(fr([ unif.full().flatten(), tn.ones(2**(ni))]),dtype=tn.float64), [2]*(ni),eps=eps)
        bx = tntt.TT( tn.tensor(fr([ tn.ones(2**(ni)), unif.full().flatten()]),dtype=tn.float64), [2]*(ni),eps=eps)
        by0 = reduceg(by,0,-1)
        by1 = reduceg(by,1,-1)
        bx0 = reduceg(bx,0,-1)
        bx1 = reduceg(bx,1,-1)
        corner = f_boundary([tn.tensor(0),tn.tensor(0)])

    # define shift matrix
    delta = dmps( 2**(ni-1) - 1 , ni - 1)
    sm = R_qtt( 2**(ni-1) - 1 , ni - 1)
    id = I_qtt( ni-1 )

    smy = zkron(  id , sm )
    smx = zkron( sm , id )
    smxy = zkron( sm , sm )

    mask2 = tntt.kron( fttr00 , zkron( phi1,phi1 ) ) + tntt.kron(fttr10 , zkron( phi2,phi1 ) ) + tntt.kron(fttr01 , zkron( phi1 ,phi2 ) ) + tntt.kron( fttr11, zkron( phi2 ,phi2 ) )
    mask2 += tntt.kron(smy@fttr00  + zkron(by0,delta) , zkron( phi1,phi3 ) ) + tntt.kron(smx@fttr00 + zkron(delta,bx0), zkron( phi3,phi1 ) ) 
    mask2 += tntt.kron(smy@fttr10  + zkron(by1,delta), zkron( phi2,phi3 ) ) + tntt.kron(smx@fttr01 + zkron(delta,bx1) , zkron( phi3,phi2 ) )
    mask2 += tntt.kron( smxy@fttr00  + smx@zkron(by0,delta) + smy@zkron(delta,bx0) + corner*zkron(delta,delta), zkron( phi3,phi3 ) )
    mask2 = mask2.round(eps)


    return mask2


def apply_local_op(mps,index,local_op):
    cores = mps.cores

    cores[index] = tn.einsum( 'abc, bd -> adc',cores[index],local_op)

    return tntt.TT(cores)

def kcubicspline_i(mps, fscale, f_boundary, eps = 1e-15,epsilon = 0.1):

    if not isinstance(f_boundary, tuple):
        f_boundary = [f_boundary, None]

    ni = len(mps.N)
    nc = fscale - ni 

    # 2D cubic kernel interpolation
    Mkc = tn.tensor([
        [0, 2, 0, 0],
        [-1, 0, 1, 0],
        [2, -5, 4, -1],
        [-1, 3, -3, 1]
    ], dtype=tn.float64) /2

    Mkct = Mkc.t()
    id = I_qtt(ni)

    sm1 = R_qtt( 2**ni -1, ni)
    sm2 = R_qtt( 2**ni -2, ni)
    smm1 = L_qtt( 1, ni)

    #minus operator
    Om1 = smm1 + 3*dmpo(0,0,ni) - 3*dmpo(0,1,ni) + dmpo(0,2,ni)
    #plus two operators
    O2 = sm2 - 3*dmpo(2**ni-1,2**ni-1,ni) + dmpo(2**ni-1,2**ni-2,ni)

    #boundary conditions
    delta = dmps(2**ni -1,ni)
    delta2 = dmps(2**ni -2,ni) + 3*dmps(2**ni -1,ni)

    if f_boundary[0] == 'gaussian':
        scale = ni
        base_scaling = 2.0 ** (-4.0 / 3.0)
        r = f_boundary[1]
        f11 = 1 + (base_scaling**scale)*epsilon*np.random.randn()
        by = (1 + (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1])) * (1-delta) + f11*delta
        bx = (1 + (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]) ) * (1-delta) + f11*delta

        by = by.round(1e-15)
        bx = by.round(1e-15)

        corner = f11*zkron(delta,delta)
        cornery = f11*zkron(delta , delta2)
        cornerx = f11*zkron(delta2 , delta)
        corner2 = f11*zkron(delta2 , delta2)

    
    elif f_boundary[0] == 'zero':
        f11 = 1
        by = tntt.ones([2]*ni)
        bx = tntt.ones([2]*ni)
        corner = f11*zkron(delta,delta)
        cornery = f11*zkron(delta , delta2)
        cornerx = f11*zkron(delta2 , delta)
        corner2 = f11*zkron(delta2 , delta2)


    else:
        unif = X_qtt(ni)
        fr = f_boundary[0]
        by = tntt.TT( tn.tensor(fr([ unif.full().flatten(), tn.ones(2**(ni))]),dtype=tn.float64), [2]*(ni),eps=eps).to(dtype=tn.float64)
        bx = tntt.TT( tn.tensor(fr([ tn.ones(2**(ni)), unif.full().flatten()]),dtype=tn.float64), [2]*(ni),eps=eps).to(dtype=tn.float64)
        corner = fr(1,1)*zkron(delta,delta)
        cornery = fr(1,1)*zkron(delta , delta2)
        cornerx = fr(1,1)*zkron(delta2 , delta)
        corner2 = fr(1,1) * zkron(delta2, delta2)

    mby = zkron( by , delta )
    mbx = zkron( delta , bx )
    mby2 = zkron( by , delta2 )
    mbx2 = zkron( delta2 , bx )
    
    
    I4 = tn.eye(4,dtype=tn.float64)
    t4 = [ tntt.TT(I4[i]) for i in range(4)]

    mpsck = tntt.kron( mps, zkron( t4[1], t4[1] ) ) + tntt.kron( zkron(id,sm1)@mps + mby , zkron( t4[1],t4[2] ) ) + tntt.kron( zkron(sm1,id)@mps + mbx, zkron(t4[2],t4[1])) +  tntt.kron( zkron(sm1,sm1)@mps + zkron(sm1,id)@mby + zkron(id,sm1)@mbx + corner, zkron(t4[2],t4[2]))
    #mpsck = mpsck.round(eps)
    mpsck += tntt.kron(zkron(Om1, Om1) @ mps, zkron(t4[0], t4[0])) + tntt.kron(zkron(Om1, id) @ mps, zkron(t4[0], t4[1])) + tntt.kron(zkron(Om1, sm1) @ mps + zkron(Om1, id) @ mby, zkron(t4[0], t4[2])) + tntt.kron(zkron(id, Om1) @ mps, zkron(t4[1], t4[0])) + tntt.kron(zkron(sm1, Om1) @ mps + zkron(id, Om1) @ mbx, zkron(t4[2], t4[0]))
    #mpsck = mpsck.round(eps)
    mpsck += tntt.kron(zkron(O2, Om1) @ mps + zkron(id, Om1) @ mbx2, zkron(t4[3], t4[0])) + tntt.kron(zkron(Om1, O2) @ mps + zkron(Om1, id) @ mby2, zkron(t4[0], t4[3])) + tntt.kron(zkron(O2, id) @ mps + mbx2, zkron(t4[3], t4[1])) + tntt.kron(zkron(id, O2) @ mps + mby2, zkron(t4[1], t4[3]))
    #mpsck = mpsck.round(eps)
    mpsck += tntt.kron( zkron(O2, sm1) @ mps + zkron(id, sm1) @ mbx2 + zkron(O2, id) @ mby + cornerx, zkron(t4[3], t4[2]) ) + tntt.kron( zkron(sm1, O2) @ mps + zkron(id, O2) @ mbx + cornery + zkron(sm1, id) @ mby2, zkron(t4[2], t4[3]))
    #mpsck = mpsck.round(eps)
    mpsck += tntt.kron( zkron(O2, O2) @ mps + zkron(id, O2) @ mbx2 + zkron(O2, id) @ mby2 + corner2, zkron(t4[3], t4[3]))
    mpsck = mpsck.round(eps)

    pols = [tntt.kron( t4[i], tntt.TT(qtt_polynomial_cores(Mkct[i], nc)) ) for i in range(4)]
    Ps = sum(pols).round(1e-15)
    Pz = zkron(Ps,Ps).round(1e-15)

    mpscks = connect(mpsck,Pz,pd=16).round(eps)

    return mpscks


def acubicspline_i(mps, fscale, f_boundary, eps = 1e-15,epsilon = 0.1):
    ni = len(mps.N)
    nc = fscale - ni 
    #Generate approximation
    M = tn.tensor([
        [1, 1, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 2],
        [0, 1, 0, 3]
    ], dtype=tn.float64)

    Mi = M.inverse()

    id = I_qtt(ni)
    delta = dmps(2**ni -1,ni)
    sm = R_qtt(2**(ni)-1, ni)

    smx = zkron(sm,id)
    smy = zkron( id , sm)
    smxy = zkron( sm , sm)

    Dop = R_qtt( 2**ni -1, ni,dtype = tn.float64) - L_qtt( 1, ni ,dtype = tn.float64) - 2*dmpo(0,0,ni) + dmpo(0,1,ni) +  2*dmpo(2**ni -1,2**ni -1,ni) - dmpo(2**ni -1,2**ni -2,ni)
    Dop = (Dop/2).round(1e-15)
    #Dop = R_qtt( 2**p -1, p,dtype = tn.float64) - id +  2*dmpo(2**p -1,2**p -1,p) - dmpo(2**p -1,2**p -2,p)
    #Dop = (Dop).round(1e-15)
    Dx = zkron( Dop,id)
    Dy = zkron( id,Dop)
    Dxy = zkron( Dop,Dop)

    if f_boundary[0] == 'gaussian':
        scale = ni
        base_scaling = 2.0 ** (-4.0 / 3.0)
        r = f_boundary[1]
        corner = 1 + (base_scaling**scale)*epsilon*np.random.randn()
        by = 1 + (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]).round(eps)
        bx = 1 + (base_scaling**scale) * epsilon *tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]).round(eps)
    
    elif f_boundary == 'zero':
        corner = 1
        by = tntt.ones([2]*ni)
        bx = tntt.ones([2]*ni)

    else:
        unif = X_qtt(ni)
        fr = f_boundary[0]  
        by = tntt.TT( tn.tensor(fr([ unif.full().flatten(), tn.ones(2**(ni))]),dtype=tn.float64), [2]*(ni),eps=eps).to(dtype=tn.float64)
        bx = tntt.TT( tn.tensor(fr([ tn.ones(2**(ni)), unif.full().flatten()]),dtype=tn.float64), [2]*(ni),eps=eps).to(dtype=tn.float64)
        corner = fr(1,1)*zkron(delta,delta)

    #Geneare f matrix
    I4 = tn.eye(4,dtype=tn.float64)
    t4 = [ tntt.TT(I4[i],[4]) for i in range(4)]

    ftt = tntt.kron( mps, zkron(t4[0],t4[0]) ) + tntt.kron( smy@mps + zkron(by,delta),  zkron(t4[0], t4[1] )) + tntt.kron(smx@mps + zkron(delta,bx), zkron(t4[1],t4[0]) ) + tntt.kron(smxy@mps + smx@zkron(by,delta)+smy@zkron(delta,bx) + corner*zkron(delta,delta), zkron(t4[1],t4[1]) )

    dxmask = (Dx @ mps).round(1e-15)
    dxftt = tntt.kron(dxmask, zkron(t4[2],t4[0]) ) + tntt.kron(smy@dxmask, zkron(t4[2],t4[1]) ) + tntt.kron(smx@dxmask, zkron(t4[3],t4[0])) + tntt.kron(smxy@dxmask, zkron(t4[3],t4[1]) )
    dxftt = dxftt.round(1e-15)

    dymask = (Dy @ mps).round(1e-15)
    dyftt = tntt.kron(dymask, zkron(t4[0],t4[2])) + tntt.kron(smy@dymask, zkron(t4[0],t4[3])) + tntt.kron(smx@dymask, zkron(t4[1],t4[2]) ) + tntt.kron(smxy@dymask,zkron(t4[1],t4[3]) )
    dyftt = dyftt.round(1e-15)

    dxymask = (Dxy @ mps).round(1e-15)
    dxyftt = tntt.kron(dxymask, zkron(t4[2],t4[2])) + tntt.kron(smy@dxymask, zkron(t4[2],t4[3])) + tntt.kron(smx@dxymask, zkron(t4[3],t4[2])) + tntt.kron(smxy@dxymask, zkron(t4[3],t4[3]) )
    dxyftt = dxyftt.round(1e-15)
    #Geneare f matrix

    fmtt = ftt + dxftt + dyftt + dxyftt
    fmtt = fmtt.round(1e-15)
    #fmtt = apply_local_op(fmtt,-1,Mi)
    #fmtt = apply_local_op(fmtt,0,Mi)
    pols = [tntt.kron( t4[i], tntt.TT(qtt_polynomial_cores(Mi[i], nc)) ) for i in range(4)]
    Ps = sum(pols)
    Pz = zkron(Ps,Ps).round(1e-15)

    mpscks = connect(ftt,Pz,pd=16).round(eps)

    return mpscks


def ncubicspline_i(mps, fscale, f_boundary, eps = 1e-15,epsilon = 0.1):
    # 2D exact cspline mps implementation

    if not isinstance(f_boundary, tuple):
        f_boundary = [f_boundary, None]

    ni = len(mps.N)
    nc = fscale - ni 

    #Exact derivatives
    hh = 1
    H = 2**ni
    #operators
    Sp = R_qtt(2**(ni)-1, ni)
    Sm = L_qtt(1, ni)
    Opb = 4*tntt.eye([2]*ni) + Sp + Sm - dmpo(0,1,ni) 
    Opb = (Opb).round(1e-15)
    rhsOp = Sp - Sm - dmpo(0,1,ni) 
    rhsOp = (3*rhsOp/hh).round(1e-15)
    id = I_qtt(ni)
    # bd mpss
    de0 = dmps(0,ni)
    de1 = dmps(2**ni-1,ni)

    if f_boundary[0] == 'gaussian':
        scale = ni
        base_scaling = 2.0 ** (-4.0 / 3.0)
        r = f_boundary[1]
        corner = 1 + (base_scaling**scale)*epsilon*np.random.randn()
        # Boundary conditions        
        P0 = (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]).round(eps)/H
        P1 = (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]).round(eps)/H
        P1y = (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]).round(eps)/H
        Q0 = (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]).round(eps)/H
        Q1 = (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]).round(eps)/H
        Q1x = (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]).round(eps)/H
        U1x = 1 + (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]).round(eps)
        U1y = 1 + (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]).round(eps)

        #bd S, P 
        s00 = (base_scaling**scale)*epsilon*np.random.randn()
        s01 = (base_scaling**scale)*epsilon*np.random.randn()
        s10 = (base_scaling**scale)*epsilon*np.random.randn()
        s11 = (base_scaling**scale)*epsilon*np.random.randn()
        p01 = (base_scaling**scale)*epsilon*np.random.randn()
        p11 = (base_scaling**scale)*epsilon*np.random.randn()
        q10 = (base_scaling**scale)*epsilon*np.random.randn()
        q11 = (base_scaling**scale)*epsilon*np.random.randn()
        u11 = 1 + (base_scaling**scale)*epsilon*np.random.randn()

    elif f_boundary[0] == 'zero':
        # Boundary conditions        
        P0 = tntt.zeros([2]*ni, dtype=tn.float64)/H
        P1 = tntt.zeros([2]*ni, dtype=tn.float64)/H
        P1y = tntt.zeros([2]*ni, dtype=tn.float64)/H
        Q0 = tntt.zeros([2]*ni, dtype=tn.float64)/H
        Q1x = tntt.zeros([2]*ni, dtype=tn.float64)/H
        Q1 = tntt.zeros([2]*ni, dtype=tn.float64)/H
        U1x = tntt.ones( [2]*ni, dtype=tn.float64)
        U1y = tntt.ones( [2]*ni, dtype=tn.float64)

        #bd S, P 
        s00 = 0
        s01 = 0
        s10 = 0
        s11 = 0
        p01 = 0
        p11 = 0
        q10 = 0
        q11 = 0
        u11 = 1


    else:

        # Boundary conditions
        unif = X_qtt(ni)
        f = f_boundary[0]
        pdx = f_boundary[1][0]
        pdy = f_boundary[1][1]
        pdxy = f_boundary[1][2]
        
        P0 = tntt.TT( tn.tensor(pdx(unif.full(),0),dtype=tn.float64), [2]*ni,eps=1e-15).round(1e-15)/H
        P1 = tntt.TT( tn.tensor(pdx(unif.full(),1),dtype=tn.float64), [2]*ni,eps=1e-15).round(1e-15)/H
        P1y = tntt.TT( tn.tensor(pdxy(unif.full(),1),dtype=tn.float64), [2]*ni,eps=1e-15).round(1e-15)/H
        Q0 = tntt.TT( tn.tensor(pdy(unif.full(),0),dtype=tn.float64), [2]*ni,eps=1e-15).round(1e-15)/H
        Q1 = tntt.TT( tn.tensor(pdy(unif.full(),1),dtype=tn.float64), [2]*ni,eps=1e-15).round(1e-15)/H
        U1x = tntt.TT( tn.tensor( f(1,unif.full() )  ,dtype=tn.float64), [2]*ni,eps=1e-15).round(1e-15)
        U1y = tntt.TT( tn.tensor(  f(unif.full(),1)  ,dtype=tn.float64), [2]*ni,eps=1e-15).round(1e-15)
        Q1x = tntt.TT( tn.tensor( pdxy(unif.full(),1),dtype=tn.float64), [2]*ni,eps=1e-15).round(1e-15)/H

        #bd S, P 
        s00 = pdxy(0,0)/H**2
        s01 = pdxy(0,1)/H**2
        s10 = pdxy(1,0)/H**2
        s11 = pdxy(1,1)/H**2
        p01 = pdx(0,1)/H
        p11 = pdx(1,1)/H
        q10 = pdy(1,0)/H
        q11 = pdy(1,1)/H
        u11 = f(1,1)
    
    corner = u11
    #p operators
    Pop = zkron( Opb , id )
    rhsopx = zkron( rhsOp , id )

    #q operators
    Qop = zkron( id , Opb )
    rhsopy = zkron( id , rhsOp )

    #RHS
    #p
    rhsp = rhsopx @ mps + 4*zkron( de0, P0) + (3/hh)*zkron( de1 , U1x ) - zkron( de1, P1)
    #rhsp = rhsp.round(1e-15)

    #q
    rhsq = rhsopy @ mps + 4*zkron( Q0, de0) + (3/hh)*zkron(U1y, de1 ) - zkron( Q1, de1)
    rhsq = rhsq.round(1e-15)


    #solve the systems
    Pmps = tntt.solvers.amen_solve(Pop, rhsp, eps=1e-15,kickrank=10,nswp=30).round(1e-15)
    Qmps = tntt.solvers.amen_solve(Qop, rhsq, eps=1e-15,kickrank=10,nswp=30).round(1e-15)

    # solve boundary S

    rhss0 = rhsOp @ P0 + 4*de0*s00 + (3/hh)*de1* p01  - de1*s01
    S0x = tntt.solvers.amen_solve(Opb, rhss0, eps=1e-15,kickrank=2,nswp=30).round(1e-15)

    rhss1 = rhsOp @ P1 + 4*de0*s10 +  (3/hh)*de1* p11  - de1*s11
    S1x = tntt.solvers.amen_solve(Opb, rhss1, eps=1e-15,kickrank=2,nswp=30).round(1e-15)

    ops = Opb
    rhss1y = rhsOp @ Q1 + 4*de0*s01 +  (3/hh)*de1* q11  - de1*s11
    S1y = tntt.solvers.amen_solve(ops, rhss1y, eps=1e-15,kickrank=2,nswp=30).round(1e-15)

    #Q10
    #rhsq10 = rhsOp @ U1x + 4*q10*de0 + (3/hh)*u11*de1 - de1*q11
    #rhsq10 = rhsq10.round(1e-15)
    #Q1y = tntt.solvers.amen_solve(Opb, rhsq10, eps=1e-15,kickrank=2,nswp=30).round(1e-15)

    #build S
    rhss = rhsopx @ Qmps + 4*zkron( de0, S0x) + (3/hh)*zkron( de1 , Q1x ) - zkron(de1, S1x)
    rhss = rhss.round(1e-15)

    # solve S
    Smps = tntt.solvers.amen_solve(Pop, rhss, eps=1e-15,kickrank=5,nswp=30).round(1e-15)

    #Generate approximation
    M = tn.tensor([
        [1, 1, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 2],
        [0, 1, 0, 3]
    ], dtype=tn.float64)

    Mi = M.inverse()

    id = I_qtt(ni)
    delta = dmps(2**ni -1,ni)
    sm = R_qtt(2**(ni)-1, ni)
    smx = zkron(sm,id)
    smy = zkron( id , sm)
    smxy = zkron(sm,sm)

    #boundary conditions
    by = U1y
    bx = U1x

    #Geneare f matrix
    I4 = tn.eye(4,dtype=tn.float64)
    t4 = [ tntt.TT(I4[i]) for i in range(4)]
    ftt = tntt.kron(mps,zkron(t4[0],t4[0])) + tntt.kron(smy@mps + zkron(by,delta), zkron(t4[0],t4[1])) + tntt.kron(smx@mps + zkron(delta,bx), zkron(t4[1],t4[0])) + tntt.kron(smxy@mps + smx@zkron(by,delta)+smy@zkron(delta,bx) + corner*zkron(delta,delta), zkron(t4[1],t4[1]) )
    ftt = ftt.round(1e-15)
    dxmask = Pmps
    dxftt = tntt.kron(dxmask, zkron(t4[2],t4[0])) + tntt.kron(smy@dxmask + zkron(P1y,delta), zkron(t4[2],t4[1])) + tntt.kron(smx@dxmask + zkron(delta,P1), zkron(t4[3],t4[0])) + tntt.kron(smy@zkron(delta,P1)+smxy@dxmask + smx@zkron(P1y,delta), zkron(t4[3],t4[1]) )
    dymask = Qmps
    dyftt = tntt.kron(dymask, zkron(t4[0],t4[2])) + tntt.kron(smy@dymask + zkron(Q1,delta), zkron(t4[0],t4[3])) + tntt.kron(smx@dymask+zkron(delta,Q1x), zkron(t4[1],t4[2])) + tntt.kron(smxy@dymask + smx@zkron(Q1,delta) + smy@zkron(delta,Q1x), zkron(t4[1],t4[3]) )
    dxymask = Smps
    dxyftt = tntt.kron(dxymask, zkron(t4[2],t4[2])) + tntt.kron(smy@dxymask + zkron(S1y,delta), zkron(t4[2],t4[3])) + tntt.kron(smx@dxymask + zkron(delta,S1x), zkron(t4[3],t4[2])) + tntt.kron(smxy@dxymask + smy@zkron(delta,S1x) +smx@zkron(S1y,delta), zkron(t4[3],t4[3]) )

    #Geneare f matrix

    fmtt = ftt + dxftt + dyftt + dxyftt
    fmtt = fmtt.round(1e-15)
    pols = [tntt.kron( t4[i], tntt.TT(qtt_polynomial_cores(Mi[i], nc)) ) for i in range(4)]
    Ps = sum(pols)
    Pz = zkron(Ps,Ps).round(1e-15)

    mpscks = connect(ftt,Pz,pd=16).round(eps)

    return mpscks

def skcubic_i(mps, fscale, f_boundary, eps = 1e-15,epsilon = 0.1 , order = 1, boundary = 'lineare'):

    if not isinstance(f_boundary, tuple):
        f_boundary = [f_boundary, None]

    ni = len(mps.N)
    nc = fscale - ni 

    id0 = I_qtt(ni)
    sm10 = R_qtt( 2**ni -1, ni)
    sm20 = R_qtt( 2**ni -2, ni)
    smm10 = L_qtt( 1, ni)
    ones = tntt.ones([1]*ni)
    onesf = tntt.ones([2]*nc)
    delta = dmps(2**ni -1,ni)
    deltae = tntt.kron(delta,onesf)

    if order == 1:
        # Build Kernel interpolant
        Mkc = tn.tensor([
            [0, 2, 0, 0],
            [-1, 0, 1, 0],
            [2, -5, 4, -1],
            [-1, 3, -3, 1]
        ], dtype=tn.float64) /2

        Mkct = Mkc.t()

        Om10 = smm10 + 3*dmpo(0,0,ni) - 3*dmpo(0,1,ni) + dmpo(0,2,ni) 
        O20 = sm20 - 3*dmpo(2**ni-1,2**ni-1,ni) + dmpo(2**ni-1,2**ni-2,ni)
        delta2 = dmps(2**ni -2,ni) + 3*dmps(2**ni -1,ni)
        delta2e = tntt.kron(delta2,onesf)
    elif order == 2:

        Mkct = tn.tensor([
            [1, -3, 3, -1],
            [4, 0, -6, 3],
            [1, 3, 3, -3],
            [0,0, 0, 1]
        ], dtype=tn.float64) /6

        Om10 = smm10 + 3*dmpo(0,0,ni) - 3*dmpo(0,1,ni) + dmpo(0,2,ni) 
        O20 = sm20 - 3*dmpo(2**ni-1,2**ni-1,ni) + dmpo(2**ni-1,2**ni-2,ni)
        delta2 = dmps(2**ni -2,ni) + 3*dmps(2**ni -1,ni)
        delta2e = tntt.kron(delta2,onesf)
    
        if boundary == 'lineare':
            Om10 = smm10 + 2*dmpo(0,0,ni) - dmpo(0,1,ni) 
            O20 = sm20 - dmpo(2**ni-1,2**ni-1,ni) 
            delta2 = dmps(2**ni -2,ni) + 2*dmps(2**ni -1,ni)
        elif boundary == 'mirror':
            Om10 = smm10 +  dmpo(0,1,ni) 
            O20 = sm20 + dmpo(2**ni-1,2**ni-1,ni) 
            delta2 = dmps(2**ni -2,ni) 
        elif boundary == 'clamped':
            Om10 = smm10 +  dmpo(0,0,ni) 
            O20 = sm20 
            delta2 = dmps(2**ni -2,ni) + dmps(2**ni -1,ni)


    if f_boundary[0] == 'gaussian':
        scale = ni
        base_scaling = 2.0 ** (-4.0 / 3.0)
        r = 10
        f11 = 1 + (base_scaling**scale) * epsilon * np.random.randn()
        bx = (1 + (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]) ) * (1-delta) + f11*delta
        by = (1 + (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1])) * (1-delta) + f11*delta
        corner = f11*delta
        corner2 = f11*delta2
    
    elif f_boundary[0] == 'zero':
        f11 = 1
        bx = tntt.ones([2]*ni)
        by = tntt.ones([2]*ni)
        corner = f11*delta
        corner2 = f11*delta2


    else:
        unif = X_qtt(ni)
        uniff = X_qtt(fscale)
        fr = f_boundary[0]
        by = tntt.TT( tn.tensor(fr([ uniff.full().flatten(), tn.ones(2**(ni))]),dtype=tn.float64), [2]*(ni),eps=eps).to(dtype=tn.float64)
        bx = tntt.TT( tn.tensor(fr([ tn.ones(2**(ni)), unif.full().flatten()]),dtype=tn.float64), [2]*(ni),eps=eps).to(dtype=tn.float64)


    mbx = zkron( delta , bx ).round(eps)
    mbx2 = zkron( delta2 , bx ).round(eps)

    pols = [tntt.TT(qtt_polynomial_cores(Mkct[i], nc)) for i in range(4)]
    polsx = [tntt.kron( ones, zkron(tntt.TT(qtt_polynomial_cores(Mkct[i], nc)) , onesf) ).round(eps) for i in range(4)]
    polsy = [tntt.kron( ones, zkron( onesf , tntt.TT( qtt_polynomial_cores(Mkct[i], nc)) ) ).round(eps) for i in range(4)]
    
    zones = zkron(onesf, onesf)
    mps = tntt.kron(mps, zones )
    mbx = tntt.kron(mbx, zones )
    mbx2 = tntt.kron(mbx2,zones )

    oneso = tntt.eye([2]*nc)
    id = tntt.kron(id0,oneso)
    sm1 = tntt.kron(sm10,oneso)
    Om1 = tntt.kron(Om10,oneso)
    O2 = tntt.kron(O20,oneso)

    f_kx =  (zkron(Om1,id) @ mps) *( polsx[0] ) + ( mps *  polsx[1] ) + ( zkron(sm1,id) @ mps + mbx) * (polsx[2]) +  ( zkron(O2,id) @ mps + mbx2) * ( polsx[3])
    f_kx = f_kx.round(eps)

    By = tntt.kron( ( Om10 @ by) , pols[0] ) + tntt.kron( by , pols[1] ) +  tntt.kron( sm10 @ by + corner, pols[2]) +  tntt.kron( O20 @ by + corner2 ,  pols[3])
    By = By.round(eps)
    mby = zkron( By , deltae )
    mby2 = zkron( By , delta2e )
    
    f_kcs = (zkron(id,Om1) @ f_kx) * ( polsy[0] ) + ( f_kx * polsy[1] ) + ( zkron(id,sm1) @ f_kx + mby) * (polsy[2]) + ( zkron(id,O2) @ f_kx + mby2) * (polsy[3]) 
    f_kcs = f_kcs.round(eps)

    return f_kcs




def gen_TT_cascade(Nscales=10, nrank=10, levels=None, seed=None,epsilon=0.1 , method = 'linear', fnoise = None, eps=1e-10, order = 1):

    if levels is None:
        levels = Nscales
    if seed is not None:
        tn.manual_seed(seed)
    if fnoise == None:
        fnoise = 'gaussian'

    # Initialize the stream function uniformly.
    #psi = tntt.ones([4]*Nscales, dtype=tn.float64)
    base_scaling = 2.0 ** (-4.0 / 3.0)
    psi = 1 + (base_scaling**Nscales) * epsilon * ( tntt.randn([4]*Nscales,[1]+[nrank]*(Nscales-1)+[1]) )


    for scale in range(2,Nscales):
        coarse_mps = 1 + (base_scaling**scale) * epsilon * ( tntt.randn([4]*scale,[1]+[nrank]*(scale-1)+[1]) )
        coarse_mps = coarse_mps.round(eps)

        # Interpolate to full resolution using linear interpolation.
        if method =='linear':
            smooth_mps =  linear_i(coarse_mps,Nscales, fnoise, eps=eps, epsilon=epsilon)

        # Interpolate to full resolution using cuadratic interpolation.
        elif method == 'cuadratic':
            if scale == 2:
                smooth_mps = 0
            else:
                smooth_mps =  cuadratic_i(coarse_mps,Nscales, fnoise, eps=eps, epsilon=epsilon)

        # Interpolate to full resolution using splines interpolation.
        elif method == 'acs':
            smooth_mps = acubicspline_i(coarse_mps,Nscales, fnoise, eps=eps, epsilon=epsilon)
        
        elif method == 'kcs':
            smooth_mps = kcubicspline_i(coarse_mps,Nscales, fnoise, eps=eps, epsilon=epsilon, order=order)

        elif method == 'ncs':
            smooth_mps = ncubicspline_i(coarse_mps,Nscales, fnoise, eps=eps, epsilon=epsilon)

        elif method == 'skc':
            smooth_mps = skcubic_i(coarse_mps,Nscales, fnoise, eps=eps, epsilon=epsilon)
        

        psi += smooth_mps
    psi = psi.round(eps)

    return psi


def skcubic3d_i(mps, fscale, f_boundary, eps = 1e-15,epsilon = 0.1, order = 1, boundary = 'periodic'):
    if not isinstance(f_boundary, tuple):
        f_boundary = (f_boundary, None)
    
    if boundary == 'periodic':
        f_boundary = ('periodic', None)

    ni = len(mps.N)
    nc = fscale - ni 
    id0 = I_qtt(ni)
    ones = tntt.ones([1]*ni)
    onesf = tntt.ones([2]*nc)
    delta = dmps(2**ni -1,ni)
    deltae = tntt.kron(delta,onesf)

    if order == 1:
        # Build Kernel interpolant
        Mkc = tn.tensor([
            [0, 2, 0, 0],
            [-1, 0, 1, 0],
            [2, -5, 4, -1],
            [-1, 3, -3, 1]
        ], dtype=tn.float64) /2
        Mkct = Mkc.t()

    elif order == 2:
        Mkct = tn.tensor([
            [1, -3, 3, -1],
            [4, 0, -6, 3],
            [1, 3, 3, -3],
            [0,0, 0, 1]
        ], dtype=tn.float64) /6

    if boundary == "periodic":
        sm10 = P_qtt( 2**ni -1, ni)
        O20 = P_qtt( 2**ni -2, ni)
        Om10 = P_qtt( 1, ni)

    else:    
        sm10 = R_qtt( 2**ni -1, ni)
        sm20 = R_qtt( 2**ni -2, ni)
        smm10 = L_qtt( 1, ni)
        if boundary == 'lineare':
            Om10 = smm10 + 2*dmpo(0,0,ni) - dmpo(0,1,ni) 
            O20 = sm20 - dmpo(2**ni-1,2**ni-1,ni) 
            delta2 = dmps(2**ni -2,ni) + 2*dmps(2**ni -1,ni)
        elif boundary == 'mirror':
            Om10 = smm10 +  dmpo(0,1,ni) 
            O20 = sm20 + dmpo(2**ni-1,2**ni-1,ni) 
            delta2 = dmps(2**ni -2,ni) 
        elif boundary == 'clamped':
            Om10 = smm10 +  dmpo(0,0,ni) 
            O20 = sm20 
            delta2 = dmps(2**ni -2,ni) + dmps(2**ni -1,ni)
        else:
            Om10 = smm10 + 3*dmpo(0,0,ni) - 3*dmpo(0,1,ni) + dmpo(0,2,ni) 
            O20 = sm20 - 3*dmpo(2**ni-1,2**ni-1,ni) + dmpo(2**ni-1,2**ni-2,ni)
            delta2 = dmps(2**ni -2,ni) + 3*dmps(2**ni -1,ni)
            delta2e = tntt.kron(delta2,onesf)

        delta2e = tntt.kron(delta2,onesf)


    if f_boundary[0] == 'gaussian':
        ob = tntt.ones([2]*ni)
        db1 = zkron(ob,delta)
        db2 = zkron(delta,ob)
        scale = ni
        base_scaling = 2.0 ** (-4.0 / 3.0)
        r = f_boundary[1]
        f11 = 1 + (base_scaling**scale)*epsilon*np.random.randn()
        lx = (1 + (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]) )*(1-delta) + f11*delta
        lx = lx.round(eps)
        ly = (1 + (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]) )*(1-delta) + f11*delta
        ly = ly.round(eps)
        lz = (1 + (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]) )*(1-delta) + f11*delta
        lz = lz.round(eps)


        bxy = (1 + (base_scaling**scale) * epsilon * tntt.randn([4]*ni, [1]+[r]*(ni-1)+[1]) )
        bxy = ( bxy*(1-db1-db2) + zkron(delta,ly ) + zkron(lx,delta) ).round(eps)
        bxz = (1 + (base_scaling**scale) * epsilon * tntt.randn([4]*ni, [1]+[r]*(ni-1)+[1]) )
        bxz = (bxz*(1-db1-db2) + zkron(delta,lz) + zkron(lx,delta) ).round(eps)
        byz = (1 + (base_scaling**scale) * epsilon * tntt.randn([4]*ni, [1]+[r]*(ni-1)+[1]))
        byz = (byz*(1-db1-db2) + zkron(delta,lz) + zkron(ly,delta) ).round(eps)

        

        corner = f11*delta
        corner2 = f11*delta2
    
    elif f_boundary[0] == 'zero':
        f11 = 1
        ly = tntt.ones([2]*ni)
        lz = tntt.ones([2]*ni)
        lx = tntt.ones([2]*ni)
        bxy = tntt.ones([4]*ni)
        bxz = tntt.ones([4]*ni)
        byz = tntt.ones([4]*ni)
        corner = f11*delta
        corner2 = f11*delta2



    elif type(f_boundary[0]) == 'function':
        unif = X_qtt(ni)
        by = tntt.TT( tn.tensor(f_boundary([ unif.full().flatten(), tn.ones(2**(ni))]),dtype=tn.float64), [2]*(ni),eps=eps).to(dtype=tn.float64)
        bx = tntt.TT( tn.tensor(f_boundary([ tn.ones(2**(ni)), unif.full().flatten()]),dtype=tn.float64), [2]*(ni),eps=eps).to(dtype=tn.float64)
        f11 = f_boundary([tn.tensor(1),tn.tensor(1)])
        corner = f11*delta
        corner2 = f11* delta2
    
    pols = [tntt.TT(qtt_polynomial_cores(Mkct[i], nc)) for i in range(4)]
    polsi = [tntt.kron( ones, zkron(tntt.TT(qtt_polynomial_cores(Mkct[i], nc)) , onesf) ).round(eps) for i in range(4)]
    polsj = [tntt.kron( ones, zkron( onesf , tntt.TT( qtt_polynomial_cores(Mkct[i], nc)) )).round(eps) for i in range(4)]
    polsx = [tntt.kron( ones, zkron3(tntt.TT(qtt_polynomial_cores(Mkct[i], nc)) , onesf,onesf) ).round(eps) for i in range(4)]
    polsy = [tntt.kron( ones, zkron3( onesf , tntt.TT( qtt_polynomial_cores(Mkct[i], nc)), onesf )).round(eps) for i in range(4)]
    polsz = [tntt.kron( ones, zkron3( onesf , onesf , tntt.TT( qtt_polynomial_cores(Mkct[i], nc)) )).round(eps) for i in range(4)]
    
    zones3 = zkron3(onesf, onesf, onesf)
    zones2 = zkron(onesf, onesf)

    oneso = tntt.eye([2]*nc)
    id = tntt.kron(id0,oneso)
    sm1 = tntt.kron(sm10,oneso)
    Om1 = tntt.kron(Om10,oneso)
    O2 = tntt.kron(O20,oneso)
    mps = tntt.kron(mps, zones3 )

    if boundary == 'periodic':

        #interpolate y
        f_ky = (zkron3(id,Om1,id) @ mps) * ( polsy[0] ) + ( mps * polsy[1] ) + ( zkron3(id,sm1,id) @ mps ) * (polsy[2]) + ( zkron3(id,O2,id) @ mps ) * (polsy[3]) 
        f_ky = f_ky.round(eps)

        #interpolate x
        f_kxy =  (zkron3(Om1,id,id) @ f_ky) *( polsx[0] ) + ( f_ky *  polsx[1] ) + ( zkron3(sm1,id,id) @ f_ky ) * (polsx[2]) +  ( zkron3(O2,id,id) @ f_ky ) * ( polsx[3])
        f_kxy = f_kxy.round(eps)

        #interpolate z
        f_kcs = (zkron3(id,id,Om1) @ f_kxy) * ( polsz[0] ) + ( f_kxy * polsz[1] ) + ( zkron3(id,id,sm1) @ f_kxy ) * (polsz[2]) + ( zkron3(id,id,O2) @ f_kxy ) * (polsz[3]) 
        f_kcs = f_kcs.round(eps)

    else:
        mby0 = zkron( delta , bxz ).round(1e-15)    
        mby20 = zkron( delta2 , bxz ).round(1e-15)
        # index y boundary correctly
        nindices = sum( [ [3*i + 1, 3*i, 3*i+2] for i in range(ni) ], [])
        mby = tntt.reshape( tntt.permute(mby0.to_qtt(), nindices ,eps) , [8]*ni)
        mby2 = tntt.reshape( tntt.permute(mby20.to_qtt(), nindices ,eps) , [8]*ni)
        mps = tntt.kron(mps, zones3 )
        mby = tntt.kron(mby, zones3 )
        mby2 = tntt.kron(mby2,zones3 )
        #interpolate y
        f_ky = (zkron3(id,Om1,id) @ mps) * ( polsy[0] ) + ( mps * polsy[1] ) + ( zkron3(id,sm1,id) @ mps + mby) * (polsy[2]) + ( zkron3(id,O2,id) @ mps + mby2) * (polsy[3]) 
        f_ky = f_ky.round(eps)

        #interpolate boundary yz
        mlz = zkron(delta,lz)
        mlz2 = zkron(delta2,lz)
        mlz = tntt.kron(mlz,zones2)
        mlz2 = tntt.kron(mlz2,zones2) 
        byz = tntt.kron(byz,zones2)
        Byz = (zkron(Om1,id) @ byz) *( polsi[0] ) + ( byz *  polsi[1] ) + ( zkron(sm1,id) @ byz + mlz ) * (polsi[2]) +  ( zkron(O2,id) @ byz + mlz2 ) * ( polsi[3])
        Byz = Byz.round(eps)
        #boundary x
        mbx = zkron( deltae, Byz )
        mbx2 = zkron( delta2e, Byz )

        #interpolate x
        f_kxy =  (zkron3(Om1,id,id) @ f_ky) *( polsx[0] ) + ( f_ky *  polsx[1] ) + ( zkron3(sm1,id,id) @ f_ky + mbx) * (polsx[2]) +  ( zkron3(O2,id,id) @ f_ky + mbx2) * ( polsx[3])
        f_kxy = f_kxy.round(eps)

        # interpolate boundary xy
        mly = zkron(delta,ly)
        mly2 = zkron(delta2,ly)
        mly = tntt.kron(mly,zones2)
        mly2 = tntt.kron(mly2,zones2) 

        # y interpolation
        bxy= tntt.kron(bxy,zones2)
        Bxy = (zkron(Om1,id) @ bxy) *( polsi[0] ) + ( bxy *  polsi[1] ) + ( zkron(sm1,id) @ bxy + mly) * (polsi[2]) +  ( zkron(O2,id) @ bxy + mly2) * ( polsi[3])
        Bxy = Bxy.round(eps)

        #1d boundary lx
        Blx = tntt.kron( ( Om10 @ lx) , pols[0] ) + tntt.kron( lx , pols[1] ) +  tntt.kron( sm10 @ lx + corner, pols[2]) +  tntt.kron( O20 @ lx + corner2 ,  pols[3])
        Blx = Blx.round(eps)
        mlx = zkron( Blx, deltae )
        mlx2 = zkron( Blx, delta2e )

        # x interpolation
        BBxy = (zkron(id,Om1) @ Bxy) * ( polsj[0] ) + ( Bxy * polsj[1] ) + ( zkron(id,sm1) @ Bxy + mlx) * (polsj[2]) + ( zkron(id,O2) @ Bxy + mlx2) * (polsj[3]) 
        BBxy = BBxy.round(eps)

        #final boundary xy
        mbz = zkron( BBxy, deltae )
        mbz2 = zkron( BBxy, delta2e )

        #interpolate z
        f_kcs = (zkron3(id,id,Om1) @ f_kxy) * ( polsz[0] ) + ( f_kxy * polsz[1] ) + ( zkron3(id,id,sm1) @ f_kxy + mbz) * (polsz[2]) + ( zkron3(id,id,O2) @ f_kxy + mbz2) * (polsz[3]) 
        f_kcs = f_kcs.round(eps)

    """ 
    #yx interpolation
    mlx = zkron(lx,delta)
    mlx2 = zkron(lx,delta2)
    mlx = tntt.kron(mlx,zones2)
    mlx2 = tntt.kron(mlx2,zones2) 

    # y interpolation
    bxy= tntt.kron(bxy,zones2)
    Bxy = (zkron(id,Om1) @ bxy) * ( polsj[0] ) + ( bxy * polsj[1] ) + ( zkron(id,sm1) @ bxy + mlx) * (polsj[2]) + ( zkron(id,O2) @ bxy + mlx2) * (polsj[3]) 
    Bxy = Bxy.round(eps)
    #1d boundary lx
    Bly = tntt.kron( ( Om10 @ ly) , pols[0] ) + tntt.kron( ly , pols[1] ) +  tntt.kron( sm10 @ ly + corner, pols[2]) +  tntt.kron( O20 @ ly + corner2 ,  pols[3])
    Bly = Bly.round(eps)
    mly = zkron( deltae,Bly)
    mly2 = zkron( delta2e, Bly )

    BBxy = (zkron(Om1,id) @ Bxy) *( polsi[0] ) + ( Bxy *  polsi[1] ) + ( zkron(sm1,id) @ Bxy + mly) * (polsi[2]) +  ( zkron(O2,id) @ Bxy + mly2) * ( polsi[3])
    BBxy = Bxy.round(eps)
    """

    return f_kcs

def skquad3d_i(mps, fscale, f_boundary, eps = 1e-15,epsilon = 0.1,boundary='lineare'):

    if not isinstance(f_boundary, tuple):
        f_boundary = (f_boundary, None)
        
    if boundary == 'periodic':
        f_boundary = ('periodic', None)

    ni = len(mps.N)
    nc = fscale - ni 
    # Final corrected B-spline matrices (4 samples × 3 monomial terms)
    mk1 = tn.tensor([
        [1/8, -1/2, 1/2],
        [3/4,  0.0, -1.0],
        [1/8,  1/2, 1/2],
        [0.0,  0.0, 0.0]
    ], dtype=tn.float64)
    mk2 = tn.tensor([
        [0.0,  0.0, 0.0],
        [9/8,  -3/2, 1/2],
        [-1/4,  2.0, -1],
        [1/8,  -1/2, 1/2]], dtype=tn.float64)

    id0 = I_qtt(ni)
    sm10 = R_qtt( 2**ni -1, ni)
    sm20 = R_qtt( 2**ni -2, ni)
    smm10 = L_qtt( 1, ni)
    delta = dmps(2**ni -1,ni)
    ones = tntt.ones([1]*ni)
    onesf = tntt.ones([2]*nc)
    deltae = tntt.kron(delta,onesf)

    if boundary == "periodic":
        sm10 = P_qtt( 2**ni -1, ni)
        O20 = P_qtt( 2**ni -2, ni)
        Om10 = P_qtt( 1, ni)

    else:    
        sm10 = R_qtt( 2**ni -1, ni)
        sm20 = R_qtt( 2**ni -2, ni)
        smm10 = L_qtt( 1, ni)
        if boundary == 'lineare':
            Om10 = smm10 + 2*dmpo(0,0,ni) - dmpo(0,1,ni) 
            O20 = sm20 - dmpo(2**ni-1,2**ni-1,ni) 
            delta2 = dmps(2**ni -2,ni) + 2*dmps(2**ni -1,ni)
        elif boundary == 'mirror':
            Om10 = smm10 +  dmpo(0,1,ni) 
            O20 = sm20 + dmpo(2**ni-1,2**ni-1,ni) 
            delta2 = dmps(2**ni -2,ni) 
        elif boundary == 'clamped':
            Om10 = smm10 +  dmpo(0,0,ni) 
            O20 = sm20 
            delta2 = dmps(2**ni -2,ni) + dmps(2**ni -1,ni)
        else:
            Om10 = smm10 + 3*dmpo(0,0,ni) - 3*dmpo(0,1,ni) + dmpo(0,2,ni) 
            O20 = sm20 - 3*dmpo(2**ni-1,2**ni-1,ni) + dmpo(2**ni-1,2**ni-2,ni)
            delta2 = dmps(2**ni -2,ni) + 3*dmps(2**ni -1,ni)
            delta2e = tntt.kron(delta2,onesf)

        delta2e = tntt.kron(delta2,onesf)


    if f_boundary[0] == 'gaussian':
        ob = tntt.ones([2]*ni)
        db1 = zkron(ob,delta)
        db2 = zkron(delta,ob)
        scale = ni
        base_scaling = 2.0 ** (-4.0 / 3.0)
        r = f_boundary[1]
        f11 = 1 + (base_scaling**scale)*epsilon*np.random.randn()
        lx = (1 + (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]) )*(1-delta) + f11*delta
        lx = lx.round(eps)
        ly = (1 + (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]) )*(1-delta) + f11*delta
        ly = ly.round(eps)
        lz = (1 + (base_scaling**scale) * epsilon * tntt.randn([2]*ni, [1]+[r]*(ni-1)+[1]) )*(1-delta) + f11*delta
        lz = lz.round(eps)


        bxy = (1 + (base_scaling**scale) * epsilon * tntt.randn([4]*ni, [1]+[r]*(ni-1)+[1]) )
        bxy = ( bxy*(1-db1-db2) + zkron(delta,ly ) + zkron(lx,delta) ).round(eps)
        bxz = (1 + (base_scaling**scale) * epsilon * tntt.randn([4]*ni, [1]+[r]*(ni-1)+[1]) )
        bxz = (bxz*(1-db1-db2) + zkron(delta,lz) + zkron(lx,delta) ).round(eps)
        byz = (1 + (base_scaling**scale) * epsilon * tntt.randn([4]*ni, [1]+[r]*(ni-1)+[1]))
        byz = (byz*(1-db1-db2) + zkron(delta,lz) + zkron(ly,delta) ).round(eps)

        corner = f11*delta
        corner2 = f11*delta2
    
    elif f_boundary[0] == 'zero':
        f11 = 1
        ly = tntt.ones([2]*ni)
        lz = tntt.ones([2]*ni)
        lx = tntt.ones([2]*ni)
        bxy = tntt.ones([4]*ni)
        bxz = tntt.ones([4]*ni)
        byz = tntt.ones([4]*ni)
        corner = f11*delta
        corner2 = f11*delta2


    elif type(f_boundary[0]) == 'function':
        unif = X_qtt(ni)
        by = tntt.TT( tn.tensor(f_boundary([ unif.full().flatten(), tn.ones(2**(ni))]),dtype=tn.float64), [2]*(ni),eps=eps).to(dtype=tn.float64)
        bx = tntt.TT( tn.tensor(f_boundary([ tn.ones(2**(ni)), unif.full().flatten()]),dtype=tn.float64), [2]*(ni),eps=eps).to(dtype=tn.float64)
        f11 = f_boundary([tn.tensor(1),tn.tensor(1)])
        corner = f11*delta
        corner2 = f11* delta2

    def qtt2pol(i):
        phi0 = hs(nc,'1')*tntt.TT(qtt_polynomial_cores(mk1[i], nc)) 
        phi1 = hs(nc,'0')*tntt.TT(qtt_polynomial_cores(mk2[i], nc)) 
        return (phi0+phi1).round(eps)
    

    pols = [qtt2pol(i) for i in range(4)]
    polsi = [tntt.kron( ones, zkron(qtt2pol(i), onesf) ).round(eps) for i in range(4)]
    polsj = [tntt.kron( ones, zkron( onesf , qtt2pol(i) )).round(eps) for i in range(4)]
    polsx = [tntt.kron( ones, zkron3(qtt2pol(i), onesf,onesf) ).round(eps) for i in range(4)]
    polsy = [tntt.kron( ones, zkron3( onesf , qtt2pol(i), onesf )).round(eps) for i in range(4)]
    polsz = [tntt.kron( ones, zkron3( onesf , onesf , qtt2pol(i) )).round(eps) for i in range(4)]
    
    zones = zkron3(onesf, onesf, onesf)
    zones2 = zkron(onesf, onesf)

    oneso = tntt.eye([2]*nc)
    id = tntt.kron(id0,oneso)
    sm1 = tntt.kron(sm10,oneso)
    Om1 = tntt.kron(Om10,oneso)
    O2 = tntt.kron(O20,oneso)

    oneso = tntt.eye([2]*nc)
    id = tntt.kron(id0,oneso)
    sm1 = tntt.kron(sm10,oneso)
    Om1 = tntt.kron(Om10,oneso)
    O2 = tntt.kron(O20,oneso)
    mps = tntt.kron(mps, zones )

    if boundary == 'periodic':

        #interpolate y
        f_ky = (zkron3(id,Om1,id) @ mps) * ( polsy[0] ) + ( mps * polsy[1] ) + ( zkron3(id,sm1,id) @ mps ) * (polsy[2]) + ( zkron3(id,O2,id) @ mps ) * (polsy[3]) 
        f_ky = f_ky.round(eps)

        #interpolate x
        f_kxy =  (zkron3(Om1,id,id) @ f_ky) *( polsx[0] ) + ( f_ky *  polsx[1] ) + ( zkron3(sm1,id,id) @ f_ky ) * (polsx[2]) +  ( zkron3(O2,id,id) @ f_ky ) * ( polsx[3])
        f_kxy = f_kxy.round(eps)

        #interpolate z
        f_kcs = (zkron3(id,id,Om1) @ f_kxy) * ( polsz[0] ) + ( f_kxy * polsz[1] ) + ( zkron3(id,id,sm1) @ f_kxy ) * (polsz[2]) + ( zkron3(id,id,O2) @ f_kxy ) * (polsz[3]) 
        f_kcs = f_kcs.round(eps)

    else:
        mby0 = zkron( delta , bxz ).round(1e-15)    
        mby20 = zkron( delta2 , bxz ).round(1e-15)
        # index y boundary correctly
        nindices = sum( [ [3*i + 1, 3*i, 3*i+2] for i in range(ni) ], [])
        mby = tntt.reshape( tntt.permute(mby0.to_qtt(), nindices ,eps) , [8]*ni)
        mby2 = tntt.reshape( tntt.permute(mby20.to_qtt(), nindices ,eps) , [8]*ni)
        mby = tntt.kron(mby, zones )
        mby2 = tntt.kron(mby2,zones )
        #interpolate y
        f_ky = (zkron3(id,Om1,id) @ mps) * ( polsy[0] ) + ( mps * polsy[1] ) + ( zkron3(id,sm1,id) @ mps + mby) * (polsy[2]) + ( zkron3(id,O2,id) @ mps + mby2) * (polsy[3]) 
        f_ky = f_ky.round(eps)

        #interpolate boundary yz
        mlz = zkron(delta,lz)
        mlz2 = zkron(delta2,lz)
        mlz = tntt.kron(mlz,zones2)
        mlz2 = tntt.kron(mlz2,zones2) 
        byz = tntt.kron(byz,zones2)
        Byz = (zkron(Om1,id) @ byz) *( polsi[0] ) + ( byz *  polsi[1] ) + ( zkron(sm1,id) @ byz + mlz ) * (polsi[2]) +  ( zkron(O2,id) @ byz + mlz2 ) * ( polsi[3])
        Byz = Byz.round(eps)
        #boundary x
        mbx = zkron( deltae, Byz )
        mbx2 = zkron( delta2e, Byz )

        #interpolate x
        f_kxy =  (zkron3(Om1,id,id) @ f_ky) *( polsx[0] ) + ( f_ky *  polsx[1] ) + ( zkron3(sm1,id,id) @ f_ky + mbx) * (polsx[2]) +  ( zkron3(O2,id,id) @ f_ky + mbx2) * ( polsx[3])
        f_kxy = f_kxy.round(eps)

        # interpolate boundary xy
        mly = zkron(delta,ly)
        mly2 = zkron(delta2,ly)
        mly = tntt.kron(mly,zones2)
        mly2 = tntt.kron(mly2,zones2) 

        # y interpolation
        bxy= tntt.kron(bxy,zones2)
        Bxy = (zkron(Om1,id) @ bxy) *( polsi[0] ) + ( bxy *  polsi[1] ) + ( zkron(sm1,id) @ bxy + mly) * (polsi[2]) +  ( zkron(O2,id) @ bxy + mly2) * ( polsi[3])
        Bxy = Bxy.round(eps)

        #1d boundary lx
        Blx = tntt.kron( ( Om10 @ lx) , pols[0] ) + tntt.kron( lx , pols[1] ) +  tntt.kron( sm10 @ lx + corner, pols[2]) +  tntt.kron( O20 @ lx + corner2 ,  pols[3])
        Blx = Blx.round(eps)
        mlx = zkron( Blx, deltae )
        mlx2 = zkron( Blx, delta2e )

        # x interpolation
        BBxy = (zkron(id,Om1) @ Bxy) * ( polsj[0] ) + ( Bxy * polsj[1] ) + ( zkron(id,sm1) @ Bxy + mlx) * (polsj[2]) + ( zkron(id,O2) @ Bxy + mlx2) * (polsj[3]) 
        BBxy = BBxy.round(eps)

        #final boundary xy
        mbz = zkron( BBxy, deltae )
        mbz2 = zkron( BBxy, delta2e )

        #interpolate z
        f_kcs = (zkron3(id,id,Om1) @ f_kxy) * ( polsz[0] ) + ( f_kxy * polsz[1] ) + ( zkron3(id,id,sm1) @ f_kxy + mbz) * (polsz[2]) + ( zkron3(id,id,O2) @ f_kxy + mbz2) * (polsz[3]) 
        f_kcs = f_kcs.round(eps)


    return f_kcs


def gen_TT_cascade_3d(Nscales=10, nrank=10, levels=None, seed=None,epsilon=0.1 , method = 'skc', fnoise = None, eps=1e-10,var=1,boundary='lineare',order=1):

    if levels is None:
        levels = Nscales
    if seed is not None:
        tn.manual_seed(seed)
    if fnoise == None:
        fnoise = 'gaussian'

    # Initialize the stream function uniformly.
    psix = tntt.ones([8]*Nscales, dtype=tn.float64)
    psiy = tntt.ones([8]*Nscales, dtype=tn.float64)
    psiz = tntt.ones([8]*Nscales, dtype=tn.float64)
    base_scaling = 2.0 ** (-4.0 / 3.0)
    #psix = 1 + (base_scaling**Nscales) * epsilon * ( tntt.randn([8]*Nscales,[1]+[nrank]*(Nscales-1)+[1]) )
    #psiy = 1 + (base_scaling**Nscales) * epsilon * ( tntt.randn([8]*Nscales,[1]+[nrank]*(Nscales-1)+[1]) )
    #psiz = 1 + (base_scaling**Nscales) * epsilon * ( tntt.randn([8]*Nscales,[1]+[nrank]*(Nscales-1)+[1]) )

    for scale in range(2,Nscales):
        coarse_mpsx = (base_scaling**scale) * epsilon * ( tntt.randn([8]*scale,[1]+[nrank]*(scale-1)+[1],var=var) )
        coarse_mpsy = (base_scaling**scale) * epsilon * ( tntt.randn([8]*scale,[1]+[nrank]*(scale-1)+[1],var=var) )
        coarse_mpsz =  (base_scaling**scale) * epsilon * ( tntt.randn([8]*scale,[1]+[nrank]*(scale-1)+[1],var=var) )

        # Interpolate to full resolution using linear interpolation.
        if method =='linear':
            raise Exception('Not implemented')

        # Interpolate to full resolution using cuadratic interpolation.
        elif method == 'cuadratic':
            raise Exception('Not implemented')

        # Interpolate to full resolution using splines interpolation

        elif method == 'skc':
            smooth_mpsx = skcubic3d_i(coarse_mpsx,Nscales, fnoise, eps=eps, epsilon=epsilon, order=order, boundary=boundary)
            smooth_mpsy = skcubic3d_i(coarse_mpsy,Nscales, fnoise, eps=eps, epsilon=epsilon, order=order, boundary=boundary)
            smooth_mpsz = skcubic3d_i(coarse_mpsz,Nscales, fnoise, eps=eps, epsilon=epsilon, order=order, boundary=boundary)
        
        elif method == 'skq':
            smooth_mpsx = skquad3d_i(coarse_mpsx,Nscales, fnoise, eps=eps, epsilon=epsilon,boundary=boundary)
            smooth_mpsy = skquad3d_i(coarse_mpsy,Nscales, fnoise, eps=eps, epsilon=epsilon,boundary=boundary)
            smooth_mpsz = skquad3d_i(coarse_mpsz,Nscales, fnoise, eps=eps, epsilon=epsilon,boundary=boundary)

        psix += smooth_mpsx
        psix = psix.round(eps)

        psiy += smooth_mpsy
        psiy = psiy.round(eps)

        psiz += smooth_mpsz
        psiz = psiz.round(eps)

    return psix, psiy,psiz


# 1) Velocity from vector potential
@nb.njit(parallel=True)
def compute_velocity_from_vector_potential(Ax, Ay, Az, L):
    N = Ax.shape[0]
    inv2dx = N / (2.0 * L)  # = 1/(2*dx)
    u = np.empty_like(Ax)
    v = np.empty_like(Ax)
    w = np.empty_like(Ax)
    for i in nb.prange(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):
                u[i,j,k] = ((Az[i, j+1, k] - Az[i, j-1, k])
                           - (Ay[i, j, k+1] - Ay[i, j, k-1])) * inv2dx
                v[i,j,k] = ((Ax[i, j, k+1] - Ax[i, j, k-1])
                           - (Az[i+1, j, k] - Az[i-1, j, k])) * inv2dx
                w[i,j,k] = ((Ay[i+1, j, k] - Ay[i-1, j, k])
                           - (Ax[i, j+1, k] - Ax[i, j-1, k])) * inv2dx
    return u, v, w


# 2) Flatness computation
@nb.njit(parallel=True)
def compute_flatness(u, max_sep, L):
    N = u.shape[0]
    dx = L / N
    seps = np.empty(max_sep, dtype=np.float64)
    flats = np.empty(max_sep, dtype=np.float64)
    for dr in nb.prange(1, max_sep+1):
        seps[dr-1] = dr * dx
        diff = u[dr:, :, :] - u[:N-dr, :, :]
        # reduction in Numba:
        S2 = 0.0
        S4 = 0.0
        cnt = diff.size
        for idx in range(cnt):
            d = diff.flat[idx]
            S2 += d*d
            S4 += d*d*d*d
        S2 /= cnt
        S4 /= cnt
        flats[dr-1] = S4 / (S2*S2)
    return seps, flats


# 3) Kolmogorov‐field generator (hybrid Numba/NumPy)
def generate_divergence_free_kolmogorov_3d(N=128, L=2*np.pi, exponent=-5/3, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # wave‑number grids (NumPy)
    k1 = np.fft.fftfreq(N, d=L/N) * 2*np.pi
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    k2[0,0,0] = 1.0

    # amplitude & random phases (NumPy)
    amp = k2**((exponent-2)/2.0)
    amp[0,0,0] = 0.0
    φx = np.exp(1j*2*np.pi*np.random.rand(N,N,N))
    φy = np.exp(1j*2*np.pi*np.random.rand(N,N,N))
    φz = np.exp(1j*2*np.pi*np.random.rand(N,N,N))

    wx = amp * φx
    wy = amp * φy
    wz = amp * φz

    # project to divergence‑free in k‑space (Numba helper)
    ux, uy, uz = _project_divfree_nb(KX, KY, KZ, wx.real, wx.imag,
                                                  wy.real, wy.imag,
                                                  wz.real, wz.imag)

    # back to complex
    u_hat = ux[0] + 1j*ux[1]
    v_hat = uy[0] + 1j*uy[1]
    w_hat = uz[0] + 1j*uz[1]

    # inverse FFT to real space
    u = np.fft.ifftn(u_hat).real
    v = np.fft.ifftn(v_hat).real
    w = np.fft.ifftn(w_hat).real

    return u, v, w


@nb.njit(parallel=True)
def _project_divfree_nb(KX, KY, KZ,
                        wx_r, wx_i,
                        wy_r, wy_i,
                        wz_r, wz_i):
    N = KX.shape[0]
    # we’ll store real & imag parts separately
    ux_r = np.empty((2, N, N, N))
    uy_r = np.empty((2, N, N, N))
    uz_r = np.empty((2, N, N, N))
    for i in nb.prange(N):
        for j in range(N):
            for k in range(N):
                k2 = KX[i,j,k]**2 + KY[i,j,k]**2 + KZ[i,j,k]**2
                if k2 == 0.0:
                    ux_r[0,i,j,k] = ux_r[1,i,j,k] = 0.0
                    uy_r[0,i,j,k] = uy_r[1,i,j,k] = 0.0
                    uz_r[0,i,j,k] = uz_r[1,i,j,k] = 0.0
                else:
                    # dot = K·w  (real and imag)
                    dr = KX[i,j,k]*wx_r[i,j,k] - KX[i,j,k]*0 \
                       + KY[i,j,k]*wy_r[i,j,k] - KY[i,j,k]*0 \
                       + KZ[i,j,k]*wz_r[i,j,k] - KZ[i,j,k]*0
                    di = KX[i,j,k]*wx_i[i,j,k] + KY[i,j,k]*wy_i[i,j,k] + KZ[i,j,k]*wz_i[i,j,k]

                    # û = w_x − KX*(dot)/k2
                    ux_r[0,i,j,k] = wx_r[i,j,k] - KX[i,j,k]*dr/k2
                    ux_r[1,i,j,k] = wx_i[i,j,k] - KX[i,j,k]*di/k2
                    uy_r[0,i,j,k] = wy_r[i,j,k] - KY[i,j,k]*dr/k2
                    uy_r[1,i,j,k] = wy_i[i,j,k] - KY[i,j,k]*di/k2
                    uz_r[0,i,j,k] = wz_r[i,j,k] - KZ[i,j,k]*dr/k2
                    uz_r[1,i,j,k] = wz_i[i,j,k] - KZ[i,j,k]*di/k2
    return ux_r, uy_r, uz_r
