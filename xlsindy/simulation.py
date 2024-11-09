import numpy as np
from .dynamics_modeling import *
from .catalog_gen import *
from .euler_lagrange import *
from .optimization import *


def Execute_Regression(t_values,thetas_values,t,Symb,Catalog,F_ext_func,Noise=0,troncature=5,Subsample=1,Hard_tr=10**-3,q_d_v=[],q_dd_v=[],reg=True,norm=True):

    if Subsample == 0:
        Subsample =1

    Nb_t = len(t_values)

    Coord_number = thetas_values.shape[1]

    Exp_matrix,t_values_s = Catalog_to_experience_matrix(Nb_t,Coord_number,Catalog,Symb,t,thetas_values,t_values,subsample=Subsample,Frottement=True,troncature=troncature,noise=Noise,q_d_v=q_d_v,q_dd_v=q_dd_v)

    Forces_vec = Forces_vector(F_ext_func, t_values_s)

    Cov_all = None
    Solution = None

    if reg:

        Exp_norm,reduction,Variance = Normalize_exp(Exp_matrix,null_effect=norm)

        coeff = Lasso_reg(Forces_vec,Exp_norm)

        Solution = Un_normalize_exp(coeff,Variance,reduction,Exp_matrix)

        Solution[np.abs(Solution)< np.max(np.abs(Solution))*Hard_tr] = 0

        #Covariance estimation through the OLS theory
        Solution_f = Solution.flatten()

        Nz_ind = np.nonzero(np.abs(Solution_f)>0)[0]

        Exp_matrix_red = Exp_matrix[:,Nz_ind]

        Covariance = np.cov(Exp_matrix_red.T)

        Cov_all=np.zeros((Solution.shape[0],Solution.shape[0]))

        Cov_all[Nz_ind[:,np.newaxis],Nz_ind]=Covariance

        residuals = Forces_vec - Exp_matrix@Solution

        sigma2 =  1/(Exp_matrix.shape[0]-Exp_matrix.shape[1])*residuals.T@residuals

        Cov_all = sigma2*Cov_all

        # ----------------



    return Solution,Exp_matrix,t_values_s ,Cov_all