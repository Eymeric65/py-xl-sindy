import numpy as np
import sympy as sp
import time


def Euler_lagranged(expr, Smatrix, t, qi):  #Euler Lagrange en symbolique take an expression of the lagrangian, an Symbol matrix, the symbolic for t, and the indice for the transformation
    dL_dq = sp.diff(expr, Smatrix[1, qi])
    dL_dq_d = sp.diff(expr, Smatrix[2, qi])
    d_dt = (sp.diff(dL_dq_d, t))

    for j in range(Smatrix.shape[1]):  # Time derivative replacement d/dt q -> q_d

        d_dt = d_dt.replace(sp.Derivative(Smatrix[1, j], t), Smatrix[2, j])
        d_dt = d_dt.replace(sp.Derivative(Smatrix[2, j], t), Smatrix[3, j])

    return dL_dq - d_dt

def Lagrangian_to_Acc_func(L, Symbol_matrix, t, Substitution,fluid_f = [],Verbose = False,Clever_Solve=True): # Turn the Lagrangian into the complete Array function
    Qk = Symbol_matrix.shape[1]

    if len(fluid_f) ==0 :
        fluid_f = [0 for i in range(Qk)]

    Acc = np.zeros((Qk, 1), dtype="object")
    Acc_Lambda = None

    dyn = np.zeros((Qk,), dtype="object")

    Valid = True

    for i in range(Qk):  # Derive the k expression for dynamics #LOL les equations sont couplées en gros mdrrrrrrr

        if (Verbose == True):
            Time_start = time.time()

        Dyn = Euler_lagranged(L, Symbol_matrix, t, i)  # + fluid_f[i]*Symbol_matrix[2, i]  # Add the Fext term

        if (Verbose == True):
            print("Time do derive {} : {}".format(i,time.time()-Time_start))

        Dyn -= Symbol_matrix[0, i] # Force action

        # Hardcodage matrice de dissipation pour un systeme en chaine (interagit deux a deux)

        Dyn += fluid_f[i]*Symbol_matrix[2, i]

        if(i<(Qk-1)):
            Dyn += fluid_f[i+1]*Symbol_matrix[2, i]
            Dyn += - fluid_f[i+1]*Symbol_matrix[2, i+1]

            Dyn += Symbol_matrix[0, i + 1]# Reaction Fext

        if(i>0):

            Dyn+= - fluid_f[i]*Symbol_matrix[2, i-1]

        # ------------------------------------

        #if( Symbol_matrix[3,i] in Dyn.atoms(sp.Function) ):
        if(str(Symbol_matrix[3,i]) in str(Dyn)):

            dyn[i] = Dyn.subs(Substitution)

        else:
            Valid = False
            break

    if(Valid):

        if (Verbose == True):
            print("Dynamics {} : ".format(len(dyn)),dyn)
            Time_start = time.time()

        if Clever_Solve:

            # Isolate the system, lambdify and solve matrix after ward

            System_term = np.empty((Qk,Qk),dtype=object)

            F_term = np.empty((Qk,1),dtype=object)

            for i in range(Qk):

                eq_i = dyn[i]

                for j in range(Qk):

                    eq_i = eq_i.collect(Symbol_matrix[3,j])

                    term = eq_i.coeff(Symbol_matrix[3,j])

                    System_term[i,j] = - term
                    eq_i = eq_i - term* Symbol_matrix[3,j]

                F_term[i,0] = eq_i

            #print(System_term,F_term,dyn) # seems fine !

            F_System_t = sp.lambdify([Symbol_matrix],System_term)
            F_F_t = sp.lambdify([Symbol_matrix],F_term)

            def Bypass_Lambda(Input):

                System_t_ev = F_System_t(Input)
                F_t_ev = F_F_t(Input)

                sol = np.linalg.solve(System_t_ev,F_t_ev)

                return sol


            # def Bypass_Lambda(Input): # Clever bypass but the cost is in the best case *4 times and worst *10
            #
            #     mariage = []
            #
            #     Symb_fl = Symbol_matrix[:3,:].flatten()
            #     Inp_fl = Input[:3,:].flatten()
            #
            #     for i in range(len(Symb_fl)):
            #         mariage += [(Symb_fl[i],Inp_fl[i])]
            #
            #     semi_rep = list(map(lambda x : x.subs(mariage).evalf(),dyn))
            #
            #     Solution = sp.solve(semi_rep,Symbol_matrix[3, :])
            #
            #     ret = list(Solution.values())
            #
            #     return np.reshape(ret,(-1,1)).astype(float)

            Acc_Lambda = Bypass_Lambda

        else:

            Solution_S = sp.solve(dyn,Symbol_matrix[3, :])

            #print("Analytique Sol",Solution_S)

            if isinstance(Solution_S,dict) :

                Sol = list(Solution_S.values())
            else:
                Sol = list(Solution_S[0])

            Acc[:,0]= Sol #LA SOLUTION

            if (Verbose == True):
                print("Time do Solve : {}".format(time.time()-Time_start))
                Time_start = time.time()

            Acc_Lambda = sp.lambdify([Symbol_matrix], Acc)  # Lambdify under the input of Symbol_matrix

        if (Verbose == True):
            print("Time do Lambdify : {}".format(time.time()-Time_start))

    return Acc_Lambda,Valid

def Clever_Linear_Solve(dyn,q_dd_symb):

    A = np.zeros((len(q_dd_symb),len(q_dd_symb)),dtype="object")

    b = np.zeros((1,len(q_dd_symb)),dtype="object")

    for i in range(len(q_dd_symb)):

        exp = dyn[i]

        for j in range(len(q_dd_symb)):

            exp = sp.collect(exp,q_dd_symb[j])

            #print("exp collected {} , {}".format(j,exp))

            A[i,j] = exp.coeff(q_dd_symb[j],1)

            #print("Coeff A[{},{}] : {} ".format(i,j, A[i,j] ))

            exp = exp - A[i,j]*q_dd_symb[j]

        b[0,i] = exp
        #print("Coeff b[{}] : {} ".format(i, b[0,i]))

    return sp.linsolve((sp.Matrix(A),sp.Matrix(b)),*q_dd_symb)
    #return np.linalg.solve(A,b)

def Catalog_to_experience_matrix(Nt,Qt,Catalog,Sm,t,q_v,q_t,subsample=1,noise=0,Frottement=False,troncature=0,q_d_v=[],q_dd_v=[]):
    #print(Nt)
    #print(Nt//subsample)
    #print(Nt/2 != Nt//2)

    Nt_s = len(q_v[troncature::subsample])

    Exp_Mat = np.zeros(((Nt_s) * Qt, len(Catalog)+int(Frottement)*Qt))

    if len(q_d_v) == 0 :
        print("Usage of approximation of speed")
        q_d_v = np.gradient(q_v,q_t,axis=0,edge_order=2)
    if len(q_dd_v) == 0:
        print("Usage of approximation of acceleration")
        print("Shape of array :",q_d_v.shape,q_t.shape)
        q_dd_v= np.gradient(q_d_v,q_t,axis=0,edge_order=2)

    q_matrix = np.zeros((Sm.shape[0],Sm.shape[1],Nt_s))

    q_matrix[1, :, :] = np.transpose(q_v[troncature::subsample])
    q_matrix[2, :, :] = np.transpose(q_d_v[troncature::subsample])
    q_matrix[3, :, :] = np.transpose(q_dd_v[troncature::subsample])

    #q_matrix = q_matrix + np.random.normal(0,noise,q_matrix.shape)

    for i in range(Qt):

        #print("Valeur Qt : ",i)
        Catalog_lagranged = list(map(lambda x: Euler_lagranged(x, Sm,t,i), Catalog))

        #print(Catalog_lagranged)

        #print("--------------")

        Catalog_lambded = list(map(lambda x: sp.lambdify([Sm], x, modules="numpy"), Catalog_lagranged))

        # print(Catalog_lagranged)
        # print([Symb_l,Symb_d_l,Symb_dd_l])
        # print(Catalog_lambded)
        #
        # fun_l = Catalog_lambded[0]
        # print(fun_l([np.array([10,10])],[np.array([10,10])],[np.array([10,10])]))

        for j in range(len(Catalog_lambded)):
            # print(str(signature(Catalog_lambded[j])))
            Func_pick = Catalog_lambded[j]
            Exp_Mat[i * Nt_s:(i + 1) * (Nt_s), j] = Func_pick(q_matrix)

        if(Frottement): # Ajout des forces de frottement fluides 2 a 2
            Exp_Mat[i * Nt_s:(i + 1) * (Nt_s), len(Catalog_lambded)+i] += q_d_v[troncature::subsample,i]

            if i < (Qt-1): # Ajout de la force de reaction si on est pas en bout de chaine

                Exp_Mat[i * Nt_s:(i + 1) * (Nt_s), len(Catalog_lambded) + i+1] +=  q_d_v[troncature::subsample,i] - q_d_v[troncature::subsample, i+1]

            if i>0: # Ajout de la difference des vitesses si on est pas lie a la base

                Exp_Mat[i * Nt_s:(i + 1) * (Nt_s), len(Catalog_lambded) + i] -= q_d_v[troncature::subsample,i-1]

    return Exp_Mat,q_t[troncature::subsample]

def Covariance_exp_Matrix(Exp_matrix,Solution,Qt): # Grossse chiasse

    print( Exp_matrix.shape)

    covSol = np.linalg.inv( Exp_matrix.T@Exp_matrix )

    Nt = int(Exp_matrix.shape[0]/Qt)

    print(Nt,covSol.shape)

    Var = np.zeros((Nt,))

    for i in range(Qt):

        App = Exp_matrix[i * Nt:(i + 1) * (Nt),:]

        print("app shape : ",App.shape," Cov sol shape : ",covSol.shape)

        Var = Var + (covSol*App@App.T).diagonal()

    return Var

def Covariance_vec(Exp_matrix,covSol,Qt): # Grossse chiasse

    print( Exp_matrix.shape)

    Nt = int(Exp_matrix.shape[0]/Qt)

    print(Nt,covSol.shape)

    Var = np.zeros((Nt,))

    for i in range(Qt):

        App = Exp_matrix[i * Nt:(i + 1) * (Nt),:]

        print("app shape : ",App.shape," Cov sol shape : ",covSol.shape)

        Var = Var + (App@covSol@App.T).diagonal()

    return Var

