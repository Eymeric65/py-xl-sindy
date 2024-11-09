import numpy as np

from function.Simulation import *
import matplotlib.pyplot as plt
# Setup problem

regression = True

#Exp1
# L1t = .2
# L2t = .2
# m_1 = .1
# m_2 = .1
# Y0 = np.array([[2, 0], [0, 0]])  # De la forme (k,2)
# Frotement = [-.4,-.2]
# M_span = [0.8,0.5] # Max span
# periode = 1 #
# periode_shift = 0.2
# Surfacteur=30 # La base
# N_periode = 5# In one periode they will be Surfacteur*N_Periode/Cat_len time tick

L1t = 1.
L2t = 1.
m_1 = .8
m_2 = .8
Y0 = np.array([[2, 0], [0, 0]])  # De la forme (k,2)
Frotement = [-1.4,-1.2]
M_span = [15.8,4.5] # Max span
#M_span = [3.8,3.5] # Max span less determination
periode = 1 #
periode_shift = 0.2
Surfacteur=10 # La base
N_periode = 5# In one periode they will be Surfacteur*N_Periode/Cat_len time tick

t = sp.symbols("t")

Coord_number = 2
Symb = Symbol_Matrix_g(Coord_number,t)

# Ideal model creation

theta1 = Symb[1,0]
theta1_d = Symb[2,0]
theta1_dd = Symb[3,0]

theta2 = Symb[1,1]
theta2_d = Symb[2,1]
theta2_dd = Symb[3,1]

m1, l1, m2, l2, g = sp.symbols("m1 l1 m2 l2 g")

Lt = L1t + L2t
Substitution = {"g": 9.81, "l1": L1t, "m1": m_1, "l2": L2t, "m2": m_2}

L = (1 / 2 * (m1 + m2) * l1 ** 2 * theta1_d ** 2 + 1 / 2 * m2 * l2 ** 2 * theta2_d ** 2 + m2 * l1 * l2 * theta1_d
     * theta2_d * sp.cos(theta1 - theta2) + (m1 + m2) * g * l1 * sp.cos(theta1) + m2 * g * l2 * sp.cos(
    theta2))

# -------------------------------

# Creation catalogue

Degree_function = 4

Puissance_model= 2

#Suited catalog creation

function_catalog_1 = [
    lambda x: Symb[2,x]
]

function_catalog_2 = [
     lambda x : sp.sin(Symb[1,x]),
     lambda x : sp.cos(Symb[1,x])
]

Catalog_sub_1 = np.array(Catalog_gen(function_catalog_1,Coord_number,2))

Catalog_sub_2 = np.array(Catalog_gen(function_catalog_2,Coord_number,2))

Catalog_crossed = np.outer(Catalog_sub_2,Catalog_sub_1)

Catalog = np.concatenate((Catalog_crossed.flatten(),Catalog_sub_1,Catalog_sub_2))

Solution_ideal = Make_Solution_vec(sp.expand_trig(L.subs(Substitution)),Catalog,Frottement=Frotement)#,Frottement=Frotement)

Cat_len = len(Catalog)

#--------------------------

# Force generation

# Parametre

Time_end = periode * Cat_len/N_periode

print("Temps de l'experience {} et longueur du Catalogue {} ".format(Time_end,Cat_len))

#----------------External Forces--------------------

F_ext_func = F_gen_opt(Coord_number,M_span,Time_end,periode,periode_shift,aug=15)
#F_ext_func = F_gen_c(M_span,periode_shift,Time_end,periode,Coord_number,aug=14)

# ---------------------------
troncature = 5
# Creation des schema de simulation

Acc_func,_ = Lagrangian_to_Acc_func(L, Symb, t, Substitution, fluid_f=Frotement)

Dynamics_system = Dynamics_f(Acc_func,F_ext_func)

t_values_w, phase = Run_RK45(Dynamics_system, Y0, Time_end,max_step=0.01)


thetas_values_w = phase[:,::2]

q_d_v = phase[:,1::2]

#q_d_v_g = np.gradient(phase[:,::2], t_values_w,axis=0,edge_order=2)
#print("Deviation gradient q0",np.linalg.norm(q_d_v[troncature:] - q_d_v_g[troncature:])/np.linalg.norm(q_d_v[troncature:])) #tres important



phase_acc =  np.gradient(phase[:,1::2], t_values_w,axis=0,edge_order=2)

#Ajout du bruit

Noise_sigma= 0 #  10**-3*0

# --------------------------------------- Regression 1 -----------------------------------


#distance_subsampling = 1.5
distance_subsampling = 0.3
Indices_sub = Optimal_sampling(phase,distance_subsampling)

#Indices_sub = Optimal_sampling(phase_acc,distance_subsampling)

print("Reduction des indices : ",len(Indices_sub))
print("Formal way len : ", (Surfacteur*Cat_len))

Solution_2,Exp_matrix_2,t_values_s_2,Covariance_param_2 = Execute_Regression(t_values_w[Indices_sub],thetas_values_w[Indices_sub,:],t,Symb,Catalog,F_ext_func,q_d_v=q_d_v[Indices_sub,:],q_dd_v=phase_acc[Indices_sub,:],reg=regression,Hard_tr=3*10**-3)

dt_l = np.gradient(t_values_s_2,edge_order=2)

print("Variation du gradient dt :",np.var(dt_l)/np.mean(dt_l))


if regression:

    Erreur_2 = np.linalg.norm( Solution_2/np.max(Solution_2)-Solution_ideal/np.max(Solution_ideal))/np.linalg.norm(Solution_ideal/np.max(Solution_ideal))
    print("Erreur de resolution coeff :",Erreur_2)
    print("sparsity : ",np.sum(np.where(np.abs(Solution_2) > 0,1,0)))


Modele_fit_2 = Make_Solution_exp(Solution_2[:,0],Catalog,Frottement=len(Frotement))


Acc_func3 , Model_Valid_2 = Lagrangian_to_Acc_func(Modele_fit_2, Symb, t, Substitution,
                                                 fluid_f=Solution_2[-len(Frotement):, 0])

#------------------------ Regression 2 -------------------



Nb_t = len(t_values_w)

#Subsample = Nb_t//(Surfacteur*Cat_len)
Subsample = Nb_t//len(Indices_sub)
#Subsample = 1

Modele_ideal = Make_Solution_exp(Solution_ideal[:,0],Catalog,Frottement=len(Frotement))
print("Modele ideal",Modele_ideal)

Solution,Exp_matrix,t_values_s,Covariance_param = Execute_Regression(t_values_w,thetas_values_w,t,Symb,Catalog,F_ext_func,Subsample=Subsample,q_d_v=q_d_v,q_dd_v=phase_acc,reg=regression,Hard_tr=3*10**-3)

if regression:

    Erreur = np.linalg.norm( Solution/np.max(Solution)-Solution_ideal/np.max(Solution_ideal))/np.linalg.norm(Solution_ideal/np.max(Solution_ideal))
    print("Erreur de resolution coeff :",Erreur)
    print("sparsity : ",np.sum(np.where(np.abs(Solution) > 0,1,0)))


Modele_fit = Make_Solution_exp(Solution[:,0],Catalog,Frottement=len(Frotement))


Acc_func2 , Model_Valid = Lagrangian_to_Acc_func(Modele_fit, Symb, t, Substitution,
                                                 fluid_f=Solution[-len(Frotement):, 0])




fig, axs = plt.subplots(3, 4)

fig.suptitle("Resultat Experience Double pendule"+str(Noise_sigma))

#Simulation temporelle

axs[0,0].set_title("q0")
axs[1,0].set_title("q1")

axs[0,0].plot(t_values_w,thetas_values_w[:,0])
axs[1,0].plot(t_values_w,thetas_values_w[:,1])

if (Model_Valid):
    Dynamics_system_2 = Dynamics_f(Acc_func2, F_ext_func)
    t_values_v, thetas_values_v = Run_RK45(Dynamics_system_2, Y0, Time_end, max_step=0.05)

    axs[0, 0].plot(t_values_v, thetas_values_v[:, 0], "--", label="found model Reg classic")

    axs[1, 0].plot(t_values_v, thetas_values_v[:, 2], "--", label="found model Reg classic")

if (Model_Valid_2):
    Dynamics_system_3 = Dynamics_f(Acc_func3, F_ext_func)
    t_values_v2, thetas_values_v2 = Run_RK45(Dynamics_system_3, Y0, Time_end, max_step=0.05)

    axs[0, 0].plot(t_values_v2, thetas_values_v2[:, 0], "--", label="found model Reg modif")

    axs[1, 0].plot(t_values_v2, thetas_values_v2[:, 2], "--", label="found model Reg modif")

axs[0, 0].legend()
axs[1, 0].legend()

axs[1,2].set_title("temporal error")

# if(Model_Valid):
#
#     interp_other_sim = np.interp(t_values_w,t_values_v,thetas_values_v[:,0])
#     amp0 = thetas_values_w[:,0].max() -thetas_values_w[:,0].min()
#     axs[1,2].plot(t_values_w,(thetas_values_w[:,0]-interp_other_sim )/amp0, label="Q0")
#
#     interp_other_sim = np.interp(t_values_w,t_values_v,thetas_values_v[:,2])
#     amp1 = thetas_values_w[:, 1].max() - thetas_values_w[:, 1].min()
#     axs[1,2].plot(t_values_w,(thetas_values_w[:,1]-interp_other_sim)/amp1,label="Q1")



axs[2,2].set_title("Forces")
axs[2,2].plot(t_values_w,F_ext_func(t_values_w).T,label=["F_1", "F_2"])

axs[0,1].set_title("q0_d")
axs[1,1].set_title("q1_d")
axs[0,1].plot(t_values_w,q_d_v[:,0])
axs[1,1].plot(t_values_w,q_d_v[:,1])

axs[0,2].set_title("Phase q0")
axs[1,2].set_title("Phase q1")
axs[0,2].plot(thetas_values_w[:,0],q_d_v[:,0])
axs[1,2].plot(thetas_values_w[:,1],q_d_v[:,1])

axs[0,2].scatter(thetas_values_w[::Subsample,0],q_d_v[::Subsample,0],label="Reg classic")
axs[1,2].scatter(thetas_values_w[::Subsample,1],q_d_v[::Subsample,1],label="Reg classic")

axs[0,2].scatter(thetas_values_w[Indices_sub,0],q_d_v[Indices_sub,0],label="Reg m")
axs[1,2].scatter(thetas_values_w[Indices_sub,1],q_d_v[Indices_sub,1],label="Reg m")

axs[1,2].legend()
axs[0,2].legend()
# Regression error

F_vec = Forces_vector(F_ext_func,t_values_s)

axs[2, 0].set_title("Regression error")
axs[2, 0].plot(np.repeat(t_values_s,Coord_number)*2,(Exp_matrix@Solution_ideal-F_vec),label="ideal solution")

if regression:
    axs[2, 0].plot(np.repeat(t_values_s, Coord_number) * 2, (Exp_matrix @ Solution - F_vec),label="solution")

axs[2, 0].legend()

axs[2, 1].set_title("Model retrieved")
Bar_height_ideal = np.abs(Solution_ideal) / np.max(np.abs(Solution_ideal))
axs[2, 1].bar(np.arange(len(Solution_ideal)), Bar_height_ideal[:, 0], width=1, label="True model")
if regression :
    Bar_height_found = np.abs(Solution) / np.max(np.abs(Solution))
    axs[2, 1].bar(np.arange(len(Solution_ideal)), Bar_height_found[:, 0], width=0.5, label="Model Found")

    Bar_height_found = np.abs(Solution_2) / np.max(np.abs(Solution_2))
    axs[2, 1].bar(np.arange(len(Solution_ideal)), Bar_height_found[:, 0], width=0.25, label="Model Found2")

axs[2, 1].legend()

axs[0,3].set_title("Variance sol")

Var_modif = Covariance_vec(Exp_matrix_2,Covariance_param_2,Coord_number)
axs[0,3].plot(t_values_s_2,Var_modif,label="Variance modif")

Var_class = Covariance_vec(Exp_matrix,Covariance_param,Coord_number)
axs[0,3].plot(t_values_s,Var_class,label="Variance Class")

axs[0,3].legend()


plt.show()