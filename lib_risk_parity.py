# Copyright 2020, Anish R. Shah
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

import numpy as np
import scipy.optimize as optimize

# to solve standard risk parity
def std_risk_parity(target, C, x0=None, normalize=True, gradient=True, hessian=False):
    """
    Function does risk parity - finds weights where security contributions meet the target variance budgets
    target = (length N) target variance contributions. Scale doesn't matter if normalizing weights
    C = (N x N) covariance
    x0 = (None or length N) initial guess of portfolio weights
    normalize = if True, treats targets as relative to total and scales weights to sum to 1
    gradient = (True or False) if True use gradient in solver
    hessian = (True or False)  if True use gradient and Hessian in solver.
              Note: non-negative bounds, because of algorithms available, are imposed only when gradient=True, hessian=False
    returns (length N) risk parity weights
    """
    # ARS: could do with fewer calculations by returning value, gradient, and hessian in one function
    def _rp_std_loss(x, target, C):
        """
        Calculates value of function minimized in risk parity: 1/2 variance + regularization penalty on small weights
        
        x = (length N) portfolio weights
        target = (length N) target variance contributions
        C = (N x N) covariance
        returns (scalar) value of loss function
        """
        w = np.reshape(x, (-1,1))
        return 0.5* w.T.dot(C).dot(w) - target.dot(np.log(abs(w)))

    def _rp_std_loss_gradient(x, target, C):
        """
        Calculates d/dx of rp_std_loss(x)
        returns (length N) gradient
        """
        w = np.reshape(x, (-1,1))
        return C.dot(w)[:,0] - target / x  # [:,0] converts (N x 1) to length N

    def rp_std_loss_hessian(x, target, C):
        """
        Calculates 2nd derivative of rp_std_loss(x)
        returns (N x N) Hessian
        """
        return C + np.diag(target / x**2)
    
    N = C.shape[0]
    if x0 is None:
        x0 = np.ones(N) / N
    if len(x0.shape) > 1:
        x0 = x0[:,0]  # convert to vector if N x 1
    f = _rp_std_loss
    method = 'nelder-mead'
    gradf = None
    hessianf = None
    bounds = None
    if gradient or hessian:
        gradf = _rp_std_loss_gradient
        method = 'TNC'
        eps = 0.00000001 # small number close to 0
        bounds = optimize.Bounds(lb=np.full(N,eps), ub=np.full(N,np.inf), keep_feasible=True)
    if hessian:
        hessianf = rp_std_loss_hessian
        method='Newton-CG' # ARS: note - the solution sometimes jumps out of the positive orthant with this since no bounds
    o = optimize.minimize(f, method=method, x0=x0, args=(target,C), jac=gradf, hess=hessianf, bounds=bounds)
    x = o.x
    if normalize:
        x /= sum(x)
    return x


# uncertain risk parity with target fraction, nonnegative weights summing to 1
def uncertain_risk_parity(target, C, covC, x0=None):
    """
    Function solves uncertain risk parity minimizing mean squared error from variance fractions with weights>0 and summing to 1
    
    target = (length N) target variance contributions. Scale doesn't matter, in preprocessing these are normalized to sum to 1
    C = (N x N) covariance
    covC = (N x N x N x N) covariance of elements of covariance matrix
    x0 = (None or length N) initial guess of portfolio weights
    
    returns (length N) non-negative risk parity weights that sum to 1
    """
    
    def _rp_utf_loss(x, target, C, covC):
        # uncertain target fraction risk parity loss
        # x = (length N) weights
        # target = (length N) target variance fractions
        #
        # returns (scalar) loss
        #
        # loss = sum_i E[(varcontr_i - target_i*var_portfolio)^2]
        # = sumi E[vi - ti*vp]^2 + var[vi - ti*vp]
        # = sumi E[vi - ti*vp]^2 + var[vi] + ti^2 var[vp] - 2 ti cov[vi, vp]
        # where vi = var contribution of security i
        #       vp = variance of portfolio = sum(vi)
        #       ti = target variance fraction of security i
        N = len(x)
        v = x * C.dot(x)  # variance contribution of each investment
        vp = v.sum()      # variance of portfolio
        varv = np.zeros(N)   # holds variance of elements of v
        covvp = np.zeros(N)  # holds cov(v, vp)
        sumx = x.sum() # used in loop below because np.average() normalizes weights to sum to 1
        _tmp = np.average(covC,axis=2,weights=x) # used for covvp - integrate out 3rd dimension weighting by x / sum(x)
        for i in range(N):
            varv[i] = x[i]**2 * x.T.dot(covC[i,:,i,:]).dot(x)  # variance of each contribution
            covvp[i] = x[i] * x.T.dot(_tmp[i,:,:]).dot(x) * sumx # covariance of each contribution with portfolio variance
            # port variance = covvp.sum()
        varp = covvp.sum() # variance of portfolio variance
        loss = ((v - target*vp) ** 2).sum() + varv.sum() + (target ** 2).sum() * varp - 2. *target.T.dot(covvp) 
        return loss
        
    N = C.shape[0]
    if x0 is None:
        x0 = np.ones(N) / N
    if len(x0.shape) > 1:
        x0 = x0[:,0]  # convert to vector if N x 1
    target = target / sum(target) # normalize target fractions

    f = _rp_utf_loss
    eps = 0.00000001 # small number close to 0
    bounds = optimize.Bounds(lb=np.full(N,eps), ub=np.full(N,np.inf), keep_feasible=True)
    constraint = optimize.LinearConstraint(A=np.ones((1,N)), lb=1., ub=1.) # budget constraint
    o = optimize.minimize(f, method='trust-constr', x0=x0, args=(target,C,covC), bounds=bounds, constraints=constraint)
    return o.x