import numpy as np
from .render import printProgress
from scipy import interpolate
from scipy.integrate import RK45


def Dynamics_f(Acc, Fext): # Transform a function Array like this [[q0_dd(q0,q0_d,f0)],...,[qk_dd(qk,qk_d,fk)]] into a function Dynamics(t,[q0,q0_d,...,qk,qk_d])
    def func(t, State):
        #State is a list [q0,q0_d,...,qk,qk_d]
        State = np.reshape(State, (-1, 2))
        State_f = np.transpose(State)

        # Input is the same as Symbol matrix
        Input = np.zeros((State_f.shape[0] + 2, State_f.shape[1]))
        Input[1:3, :] = State_f
        Input[0, :] = Fext(t)

        ret = np.zeros(State.shape)

        ret[:, 0] = State[:, 1]
        ret[:, 1] = Acc(Input)[:, 0]
        return np.reshape(ret, (-1,))

    return func

def Dynamics_f_extf(Acc):

    def func(t,State,F_ext):
        # State is a list [q0,q0_d,...,qk,qk_d]
        State = np.reshape(State, (-1, 2))
        State_f = np.transpose(State)

        # Input is the same as Symbol matrix
        Input = np.zeros((State_f.shape[0] + 2, State_f.shape[1]))
        Input[1:3, :] = State_f
        Input[0, :] = F_ext

        ret = np.zeros(State.shape)

        ret[:, 0] = State[:, 1]
        ret[:, 1] = Acc(Input)[:, 0]
        return np.reshape(ret, (-1,))

    return func

def Run_RK45(dynamics, Y0, Time_end,max_step=0.05):  #Run a RK45 integration on our model Y0 is still under the form (k,2)

    Y0_f = np.reshape(Y0, (-1,))  # de la forme(2*k,)
    Model = RK45(dynamics, 0, Y0_f, Time_end, max_step, 0.001, np.e ** -6)
    #Model = RK45(dynamics, 0, Y0_f, Time_end, max_step, 10**-9 , 10 ** -16)
    # collect data
    t_v = []
    q_v = []

    try:

        # for i in range(40000):
        #     # get solution step state
        #     Model.step()
        #     t_v.append(Model.t)
        #     q_v.append(Model.y)
        #     # break loop after modeling is finished
        #     if Model.status == 'finished':
        #
        #         #print("End step : ", i)
        #
        #         break
        while Model.status != "finished":

            for _ in range(200):
                if Model.status != "finished":

                    Model.step()
                    t_v.append(Model.t)
                    q_v.append(Model.y)



            printProgress(Model.t,Time_end)

    except RuntimeError:

        print("RuntimeError of RK45 Experiment")

    t_v = np.array(t_v)
    q_v = np.array(q_v)

    return t_v, q_v

# Forces creation

def F_gen_v(M_var,periode_shift,Time_end,periode,Coord_number):

    F_ext_time = np.arange(0,Time_end+periode,periode)

    f_nbt = len(F_ext_time)

    F_ext_time = F_ext_time + (np.random.random((f_nbt,))-0.5)*2*periode_shift

    F_ext_Value = (np.random.random_sample((Coord_number,f_nbt))*2-1)

    F_ext_Value = F_ext_Value/np.std(F_ext_Value)*np.sqrt(M_var)/0.87



    return interpolate.CubicSpline(F_ext_time, F_ext_Value, axis=1)

def F_gen_a(M_span,periode_shift,Time_end,periode,Coord_number):

    F_ext_time = np.arange(0,Time_end+periode,periode)

    f_nbt = len(F_ext_time)

    F_ext_time = F_ext_time + (np.random.random((f_nbt,))-0.5)*2*periode_shift



    F_ext_Value = (np.random.normal((Coord_number,f_nbt))*2-1)*M_span

    amp = (np.max(F_ext_Value)-np.min(F_ext_Value))/2

    F_ext_Value = F_ext_Value/amp*M_span

    print("max",M_span)

    return interpolate.CubicSpline(F_ext_time, F_ext_Value, axis=1)

def concat_f(arr):

    def ret(t):

        out = arr[0](t)

        for f in arr[1:]:

            out += f(t)

        return out

    return ret

def F_gen_c(M_span,periode_shift,Time_end,periode,Coord_number,aug=50,method="a"):

    f_arr = []

    p=0

    if method == "v":
        F_gen = F_gen_v
    else :
        F_gen = F_gen_a

    for i in range(1,aug):

        p+= 1/(i)

    for i in range(1, aug):
        f_arr += [F_gen(M_span / p / i, periode_shift / (i*2), Time_end, periode / (i*2),Coord_number)]

    return concat_f(f_arr)

def F_gen_r(Time_end,aug,aug_i,periode_i,periode_shift_i,c_number):

    if aug == aug_i:

        return lambda x : x*np.zeros((c_number,1))

    else:


        F_b = F_gen_r(Time_end,aug+1,aug_i,periode_i,periode_shift_i,c_number)

        m = (aug_i-aug)

        periode = periode_i/m
        periode_shift = periode_shift_i/m
        variance = 1/m

        time_list = np.arange(0, Time_end + periode, periode)
        tl = len(time_list)
        time_list = time_list + (np.random.random((tl,))-0.5)*2*periode_shift

        F_ext_Value = (np.random.random_sample((c_number, tl)) * 2 - 1) * variance + F_b(time_list)

        F_ext_Value = F_ext_Value/np.std(F_ext_Value)

        return interpolate.CubicSpline(time_list, F_ext_Value, axis=1)

def F_gen_opt(c_n,V,Time_end,periode,periode_shift,aug=50):


    V = np.reshape(V,(c_n,1))# Top disgusting Python line

    F = F_gen_r(Time_end,0,aug,periode,periode_shift,c_n)

    def norm(t):

        if(len(F(t).shape)==1):
            return F(t) * V.flatten()
        else:
            return F(t)*V

    return norm







