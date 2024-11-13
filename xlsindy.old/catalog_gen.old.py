import numpy as np
import sympy as sp



# def catalog_gen_p(f_cat,qk,prof,im,jm,puissance,puissance_init): # Not everything lol

#     if prof==0 :
#         return [1] #,f_cat[f].format(v_cat[v])
#     else:
#         ret = []
#         for i in range(im,len(f_cat)):
#             for j in range(jm,qk):

#                 if i == im and j == jm:
#                     if puissance > 0:
#                         res = catalog_gen_p(f_cat,qk,prof-1,i,j,puissance-1,puissance_init)
#                     else:
#                         res = []

#                     fun_p = f_cat[i]
#                     res_add = [res[l]*fun_p(j) for l in range(len(res))]

#                 else:

#                     res = catalog_gen_p(f_cat,qk,prof-1,i,j,puissance_init-1,puissance_init)

#                     fun_p = f_cat[i]
#                     res_add = [res[l]*fun_p(j) for l in range(len(res))]

#                 ret += res_add


#         return ret

def concat_Func_var(f_cat,qk):

    ret = []

    for i in range(len(f_cat)):

        for j in range(qk):

            fun_p = f_cat[i]
            ret += [fun_p(j)]

    return ret

def catalog_gen_c(cat,prof,im,puissance,puissance_init):

    if prof==0 :
        return [1] #,f_cat[f].format(v_cat[v])
    else:
        ret = []
        for i in range(im+1, len(cat)):

            res = catalog_gen_c(cat,prof - 1, i, puissance_init - 1, puissance_init)

            ret += [res[l] * cat[i] for l in range(len(res))]

        if puissance > 0:

            res = catalog_gen_c(cat, prof - 1, im, puissance - 1, puissance_init)
            ret += [res[l] * cat[im] for l in range(len(res))]

        return ret

def catalog_gen(f_cat,qk,degre,puissance=None):
    """Create a catalog of linear combinaison of an array of function
    
    """
    catalog = []

    if puissance==None:

        puissance = degre

    sub_cat = concat_Func_var(f_cat,qk)

    for i in range(degre):
        catalog += catalog_gen_c(sub_cat, i + 1, 0,puissance,puissance)

    return catalog




def forces_vector(f_fun,t_v):

    f_vec = f_fun(t_v)

    f_vec[0:-1,:] -= f_vec[1:,:]

    return np.transpose(np.reshape(f_vec,(1,-1)))

def make_solution_vec(exp,catalog,frottement=[]):

    exp_arg = sp.expand(sp.expand_trig(exp)).args
    #print("Expression ",exp_arg)
    #print("Reduction ",sp.expand(sp.expand_trig(exp)))
    solution = np.zeros((len(catalog)+len(frottement),1))

    for i in range(len(exp_arg)):

        for v in range(len(catalog)):

            test = exp_arg[i]/catalog[v]

            if(len(test.args)==0):

                solution[v,0] = test

    for i in range(len(frottement)):
        solution[len(catalog)+i] = frottement[i]

    return solution

def make_solution_exp(solution,catalog,frottement=0):

    modele = 0

    for i in range(len(solution)-frottement):

        modele += solution[i]*catalog[i]

    return modele