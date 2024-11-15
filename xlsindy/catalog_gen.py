import numpy as np
import sympy as sp


def Catalog_gen_p(f_cat,qk,prof,im,jm,puissance,puissance_init): # Not everything lol

    if prof==0 :
        return [1] #,f_cat[f].format(v_cat[v])
    else:
        ret = []
        for i in range(im,len(f_cat)):
            for j in range(jm,qk):

                if i == im and j == jm:
                    if puissance > 0:
                        res = Catalog_gen_p(f_cat,qk,prof-1,i,j,puissance-1,puissance_init)
                    else:
                        res = []

                    fun_p = f_cat[i]
                    res_add = [res[l]*fun_p(j) for l in range(len(res))]

                else:

                    res = Catalog_gen_p(f_cat,qk,prof-1,i,j,puissance_init-1,puissance_init)

                    fun_p = f_cat[i]
                    res_add = [res[l]*fun_p(j) for l in range(len(res))]

                ret += res_add


        return ret

def Concat_Func_var(f_cat,qk):

    ret = []

    for i in range(len(f_cat)):

        for j in range(qk):

            fun_p = f_cat[i]
            ret += [fun_p(j)]

    return ret

def Catalog_gen_c(cat,prof,im,puissance,puissance_init):

    if prof==0 :
        return [1] #,f_cat[f].format(v_cat[v])
    else:
        ret = []
        for i in range(im+1, len(cat)):

            res = Catalog_gen_c(cat,prof - 1, i, puissance_init - 1, puissance_init)

            ret += [res[l] * cat[i] for l in range(len(res))]

        if puissance > 0:

            res = Catalog_gen_c(cat, prof - 1, im, puissance - 1, puissance_init)
            ret += [res[l] * cat[im] for l in range(len(res))]

        return ret

def Catalog_gen(f_cat,qk,degre,puissance=None):
    """Create a catalog of linear combinaison of an array of function
    
    """
    catalog = []

    if puissance==None:

        puissance = degre

    sub_cat = Concat_Func_var(f_cat,qk)

    for i in range(degre):
        catalog += Catalog_gen_c(sub_cat, i + 1, 0,puissance,puissance)

    return catalog


def Symbol_Matrix_g(Coord_number,t):
    """

    """

    ret = np.zeros((4,Coord_number),dtype='object')
    ret[0,:]= [sp.Function("Fext{}".format(i))(t) for i in range(Coord_number)]
    ret[1, :] = [sp.Function("q{}".format(i))(t) for i in range(Coord_number)]
    ret[2, :] = [sp.Function("q{}_d".format(i))(t) for i in range(Coord_number)]
    ret[3, :] = [sp.Function("q{}_dd".format(i))(t) for i in range(Coord_number)]
    return ret

def Forces_vector(F_fun,t_v):

    F_vec = F_fun(t_v)

    F_vec[0:-1,:] -= F_vec[1:,:]

    return np.transpose(np.reshape(F_vec,(1,-1)))

def Make_Solution_vec(exp,Catalog,Frottement=[]):

    exp_arg = sp.expand(sp.expand_trig(exp)).args
    #print("Expression ",exp_arg)
    #print("Reduction ",sp.expand(sp.expand_trig(exp)))
    Solution = np.zeros((len(Catalog)+len(Frottement),1))

    for i in range(len(exp_arg)):

        for v in range(len(Catalog)):

            test = exp_arg[i]/Catalog[v]

            if(len(test.args)==0):

                Solution[v,0] = test

    for i in range(len(Frottement)):
        Solution[len(Catalog)+i] = Frottement[i]

    return Solution

def Make_Solution_exp(Solution,Catalog,Frottement=0):

    Modele = 0

    for i in range(len(Solution)-Frottement):

        Modele += Solution[i]*Catalog[i]

    return Modele





