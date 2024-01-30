import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#QUESTION 1 

#function that simulates the neural network
def sim_bg(I_in, w_in):

    # simulation parameters
    n_simulations = 5
    n_acquisition_trials = 25
    n_extinction_trials = 25
    n_reacquisition_trials = 25
    n_trials = n_acquisition_trials + n_extinction_trials + n_reacquisition_trials

    # ctx-d1 and ctx-d2 learning rate parameters
    alpha_ltp_d1 = 2.5e-15
    alpha_ltd_d1 = 3e-14
    alpha_ltp_d2 = 5e-14
    alpha_ltd_d2 = 3e-14

    # reward prediction learning rate -> determines strength of inhibitory connections
    gamma = 0.01 #maybe change to 1.0

    # response threshold - determines whether response is significant or not 
    resp_thresh = 400

    # response of each spike on post synaptic membrane v
    psp_amp = 1e5
    psp_decay = 100 #how fast PSP decays over time

    # arrays to store behaviour, reward, etc
    resp = np.zeros((n_trials, n_simulations))
    r_obtained = np.zeros((n_trials, n_simulations))
    r_predicted = np.zeros((n_trials, n_simulations))
    delta = np.zeros((n_trials, n_simulations))

    # arrays for each neuron
    v = np.zeros((n_cells, n_steps))
    u = np.zeros((n_cells, n_steps))
    g = np.zeros((n_cells, n_steps))
    spike = np.zeros((n_cells, n_steps))
    v[:, 0] = iz_params[:, 1] + np.random.rand(n_cells) * 100

    # connection weight matrix
    w = np.zeros((n_cells, n_cells))

    # record keeping for ctx-d1 and ctx-d2 weights
    w_record = np.zeros((2, n_trials, n_simulations))

    # direct pathway
    w[0, 1] = 0.0 #ctx-d1
    w[1, 3] = -2.5 #d1-gpi

    # indirect pathway
    w[0, 2] = 0.1 #ctx-d2 
    w[2, 4] = -1.5 #d2-gpe
    w[4, 3] = -0.1 #gpe-gpi

    # lateral inhibition d1-d2
    w[1, 2] = -0.0 #d1-d2
    w[2, 1] = -2.0 #d2-d1

    # hyperdirect pathway
    w[0, 6] = 0 #ctx-stn
    w[6, 3] = 0 #stn-gpi

    # stn-gpe feedback
    w[6, 4] = 0 #stn-gpe
    w[4, 6] = 0 #gpe-stn

    # output
    w[3, 5] = -2.5 #gpi-thal
    w[5, 0] = 0 #thal-ctx

    # input into cells from other cells
    I_net = np.zeros((n_cells, n_steps))

    for sim in range(n_simulations):
        for trl in range(n_trials - 1):
            print(sim, trl)

            # reset between each trial
            v = np.zeros((n_cells, n_steps))
            u = np.zeros((n_cells, n_steps))
            g = np.zeros((n_cells, n_steps))
            spike = np.zeros((n_cells, n_steps))
            v[:, 0] = iz_params[:, 1] + np.random.rand(n_cells) * 100

            for i in range(1, n_steps):

                dt = t[i] - t[i - 1]

                I_net = np.zeros((n_cells, n_steps))
                for jj in range(n_cells):
                    for kk in range(n_cells):
                        if jj != kk:
                            I_net[jj, i - 1] += w[kk, jj] * g[kk, i - 1]

                    I_net[jj, i - 1] += w_in[jj] * I_in[i - 1]

                    C = iz_params[jj, 0]
                    vr = iz_params[jj, 1]
                    vt = iz_params[jj, 2]
                    vpeak = iz_params[jj, 3]
                    a = iz_params[jj, 4]
                    b = iz_params[jj, 5]
                    c = iz_params[jj, 6]
                    d = iz_params[jj, 7]
                    k = iz_params[jj, 8]

                    dvdt = (k * (v[jj, i - 1] - vr) * (v[jj, i - 1] - vt) -
                            u[jj, i - 1] + I_net[jj, i - 1] +
                            np.random.normal(E_mu[jj], E_sig[jj])) / C
                    dudt = a * (b * (v[jj, i - 1] - vr) -
                                u[jj, i - 1]) + u_input[jj, i - 1]
                    dgdt = (-g[jj, i - 1] +
                            psp_amp * spike[jj, i - 1]) / psp_decay

                    v[jj, i] = v[jj, i - 1] + dvdt * dt
                    u[jj, i] = u[jj, i - 1] + dudt * dt
                    g[jj, i] = g[jj, i - 1] + dgdt * dt

                    if v[jj, i] >= vpeak:
                        v[jj, i - 1] = vpeak
                        v[jj, i] = c
                        u[jj, i] = u[jj, i] + d
                        spike[jj, i] = 1

                # press the lever if thalamus crosses a threshold
                # NOTE: the n//3 bit is a hack to omit initialisation spikes
                if i > n_steps // 3:
                    if g[5, i] > resp_thresh:
                        resp[trl, sim] = 1
                        break

            # press lever on a random % of all trials
            if np.random.uniform(0, 1) > 0.5:
                resp[trl, sim] = 1

            if trl < n_acquisition_trials:
                # give reward if lever is pressed
                if resp[trl, sim] == 1.0:
                    r_obtained[trl, sim] = 1.0

            if trl > n_acquisition_trials and trl > n_acquisition_trials + n_extinction_trials:
                # reward is fixed at zero during extinction
                r_obtained[trl, sim] = 0.0

            if trl > n_acquisition_trials + n_extinction_trials:
                # give reward if lever is pressed
                if resp[trl, sim] == 1.0:
                    r_obtained[trl, sim] = 1.0

            # update reward prediction
            delta[trl, sim] = r_obtained[trl, sim] - r_predicted[trl, sim]
            r_predicted[trl + 1] = r_predicted[trl, sim] + gamma * delta[trl, sim]

            # update synaptic weights (ctx-d1)
            pre = g[0, :].sum()
            post = g[1, :].sum()

            if delta[trl, sim] <= 0:
                delta_w = alpha_ltd_d1 * pre * post * delta[trl, sim] * (w[0,1])

            if delta[trl, sim] > 0:
                delta_w = alpha_ltp_d1 * pre * post * delta[trl, sim] * (1. - w[0, 1])

            w[0, 1] += delta_w
            w[0, 1] = np.clip(w[0, 1], 0.1, 1)

            # update synaptic weights (ctx-d2)
            pre = g[0, :].sum()
            post = g[2, :].sum()

            if delta[trl, sim] <= 0:
                delta_w = -alpha_ltp_d2 * pre * post * delta[trl, sim] * (1. - w[0, 2])

            if delta[trl, sim] > 0:
                delta_w = -alpha_ltd_d2 * pre * post * delta[trl, sim] * (w[0,2])

            w[0, 2] += delta_w
            w[0, 2] = np.clip(w[0, 2], 0.1, 1)

            # record trial
            w_record[0, trl + 1, sim] = w[0, 1]
            w_record[1, trl + 1, sim] = w[0, 2]

    fig, ax = plt.subplots(3, 1, squeeze=False)
    x = np.arange(0, n_trials - 1, 1)
    ax[0, 0].plot(x, resp[:-1, :].mean(1), 'o', label='response')
    ax[0, 0].plot([n_acquisition_trials, n_acquisition_trials], [0, 1], '--k')
    ax[0, 0].plot([n_acquisition_trials + n_extinction_trials, n_acquisition_trials + n_extinction_trials], [0, 1], '--k')
    ax[0, 0].plot([25, 25], [0, 1], '--k')
    ax[0, 0].set_title("Average Response Rate") 

    #ax[1, 0].plot([0, n_trials - 2], [resp_thresh, resp_thresh], '--k')
    ax[1, 0].plot(x, r_obtained[:-1, :].mean(1), label='obtained reward')
    ax[1, 0].plot(x, r_predicted[:-1, :].mean(1), label='predicted reward')
    ax[1, 0].plot(x, delta[:-1, :].mean(1), label='delta')
    ax[1,0].set_xlabel('Trial')
    ax[1,0].set_ylabel('')
    ax[1, 0].set_title("Comparison of Obtained and Predicted Rewards")

    ax[2, 0].plot(x, w_record[0, :-1, :].mean(1), label='w ctx-d1')
    #ax[2, 0].plot(x, w_record[1, :-1, :].mean(1), label='w ctx-d2')
    ax[2,0].set_ylabel('Synaptic Weight (w)')
    ax[2, 0].set_title("Average Synaptic Weight")

    [x.legend() for x in ax.flatten()]
    [x.set_xticks(np.arange(0, n_trials - 1, 5)) for x in ax.flatten()]
    plt.tight_layout()
    plt.show()


tau = 0.1
T = 3000
t = np.arange(0, T, tau)
n_steps = t.shape[0]

# Cells: CTX, D1, D2, GPi, GPe, Thal, STN
# Direct: CTX -> D1 -> GPi
# Indirect: CTX -> D2 -> GPe -> GPi
# Hyperdirect: CTX -> STN -> GPi
# Output: GPi -> Thal -> CTX
# Gain limiter: STN <-> GPe
iz_params = np.array([
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # ctx (rs) 0
    [50, -80, -25, 40, 0.01, -20, -55, 150, 1],  # d1 (spn) 1
    [50, -80, -25, 40, 0.01, -20, -55, 150, 1],  # d2 (spn) 2
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # gpi (rs) 3
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # gpe (rs) 4
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # thal (rs) 5
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # stn (rs) 6
    [100, -75, -45, 60, 0.01, 5, -56, 130, 1.2]  # tan (special case) 7
])

# baseline firing (high for GPi, GPe, thal, and tan)
E_mu = np.array([0, 0, 0, 300, 300, 300, 0, 300])
E_sig = np.array([0, 0, 0, 0, 0, 0, 0, 300])

n_cells = iz_params.shape[0]

# define input into u equations
# this only applies to TAN (index -1) but we program it more generally
u_input = np.zeros((n_cells, n_steps))
u_input[-1, n_steps // 3:2 * n_steps // 3] = 3.0
for i in range(2 * n_steps // 3, n_steps):
    u_input[-1, i] = u_input[-1, i - 1] * 0.05

# define input signal
I_in = np.zeros(n_steps)
I_in[n_steps // 3:2 * n_steps // 3] = 5e1

w_in = np.zeros(n_cells)

# add input only to ctx
w_in[0] = 75

sim_bg(I_in, w_in)

##QUESTION 2

def sim_bg_2(I_in, w_in):

    # simulation parameters
    n_simulations = 5
    n_acquisition_trials = 25
    n_extinction_trials = 25
    n_reacquisition_trials = 25
    n_trials = n_acquisition_trials + n_extinction_trials + n_reacquisition_trials

    # ctx-d1 and ctx-d2 learning rate parameters
    alpha_ltp_d1 = 2.5e-15
    alpha_ltd_d1 = 3e-14
    alpha_ltp_d2 = 1e-13 #increase 
    alpha_ltd_d2 = 3e-14

    # reward prediction learning rate -> determines strength of inhibitory connections
    gamma = 0.10 #increase

    # response threshold - determines whether response is significant or not 
    resp_thresh = 400

    # response of each spike on post synaptic membrane v
    psp_amp = 1e5
    psp_decay = 100 #how fast PSP decays over time

    # arrays to store behaviour, reward, etc
    resp = np.zeros((n_trials, n_simulations))
    r_obtained = np.zeros((n_trials, n_simulations))
    r_predicted = np.zeros((n_trials, n_simulations))
    delta = np.zeros((n_trials, n_simulations))

    # arrays for each neuron
    v = np.zeros((n_cells, n_steps))
    u = np.zeros((n_cells, n_steps))
    g = np.zeros((n_cells, n_steps))
    spike = np.zeros((n_cells, n_steps))
    v[:, 0] = iz_params[:, 1] + np.random.rand(n_cells) * 100

    # connection weight matrix
    w = np.zeros((n_cells, n_cells))

    # record keeping for ctx-d1 and ctx-d2 weights
    w_record = np.zeros((2, n_trials, n_simulations))

    # direct pathway
    w[0, 1] = 0.0 #ctx-d1
    w[1, 3] = -2.5 #d1-gpi

    # indirect pathway
    w[0, 2] = 0.1 #ctx-d2 
    w[2, 4] = -1.5 #d2-gpe
    w[4, 3] = -0.1 #gpe-gpi

    # lateral inhibition d1-d2
    w[1, 2] = -2.0 #d1-d2
    w[2, 1] = 0.0 #d2-d1 #change this to 0

    # hyperdirect pathway
    w[0, 6] = 0 #ctx-stn
    w[6, 3] = 0 #stn-gpi

    # stn-gpe feedback
    w[6, 4] = 0 #stn-gpe
    w[4, 6] = 0 #gpe-stn

    # output
    w[3, 5] = -2.5 #gpi-thal
    w[5, 0] = 0 #thal-ctx

    # input into cells from other cells
    I_net = np.zeros((n_cells, n_steps))

    for sim in range(n_simulations):
        for trl in range(n_trials - 1):
            print(sim, trl)

            # reset between each trial
            v = np.zeros((n_cells, n_steps))
            u = np.zeros((n_cells, n_steps))
            g = np.zeros((n_cells, n_steps))
            spike = np.zeros((n_cells, n_steps))
            v[:, 0] = iz_params[:, 1] + np.random.rand(n_cells) * 100

            for i in range(1, n_steps):

                dt = t[i] - t[i - 1]

                I_net = np.zeros((n_cells, n_steps))
                for jj in range(n_cells):
                    for kk in range(n_cells):
                        if jj != kk:
                            I_net[jj, i - 1] += w[kk, jj] * g[kk, i - 1]

                    I_net[jj, i - 1] += w_in[jj] * I_in[i - 1]

                    C = iz_params[jj, 0]
                    vr = iz_params[jj, 1]
                    vt = iz_params[jj, 2]
                    vpeak = iz_params[jj, 3]
                    a = iz_params[jj, 4]
                    b = iz_params[jj, 5]
                    c = iz_params[jj, 6]
                    d = iz_params[jj, 7]
                    k = iz_params[jj, 8]

                    dvdt = (k * (v[jj, i - 1] - vr) * (v[jj, i - 1] - vt) -
                            u[jj, i - 1] + I_net[jj, i - 1] +
                            np.random.normal(E_mu[jj], E_sig[jj])) / C
                    dudt = a * (b * (v[jj, i - 1] - vr) -
                                u[jj, i - 1]) + u_input[jj, i - 1]
                    dgdt = (-g[jj, i - 1] +
                            psp_amp * spike[jj, i - 1]) / psp_decay

                    v[jj, i] = v[jj, i - 1] + dvdt * dt
                    u[jj, i] = u[jj, i - 1] + dudt * dt
                    g[jj, i] = g[jj, i - 1] + dgdt * dt

                    if v[jj, i] >= vpeak:
                        v[jj, i - 1] = vpeak
                        v[jj, i] = c
                        u[jj, i] = u[jj, i] + d
                        spike[jj, i] = 1

                # press the lever if thalamus crosses a threshold
                # NOTE: the n//3 bit is a hack to omit initialisation spikes
                if i > n_steps // 3:
                    if g[5, i] > resp_thresh:
                        resp[trl, sim] = 1
                        break

            # press lever on a random % of all trials
            if np.random.uniform(0, 1) > 0.5:
                resp[trl, sim] = 1

            if trl < n_acquisition_trials:
                # give reward if lever is pressed
                if resp[trl, sim] == 1.0:
                    r_obtained[trl, sim] = 1.0

            if trl > n_acquisition_trials and trl > n_acquisition_trials + n_extinction_trials:
                # reward is fixed at zero during extinction
                r_obtained[trl, sim] = 0.0

            if trl > n_acquisition_trials + n_extinction_trials:
                # give reward if lever is pressed
                if resp[trl, sim] == 1.0:
                    r_obtained[trl, sim] = 1.0

            # update reward prediction
            delta[trl, sim] = r_obtained[trl, sim] - r_predicted[trl, sim]
            r_predicted[trl + 1] = r_predicted[trl, sim] + gamma * delta[trl, sim]

            # update synaptic weights (ctx-d1)
            pre = g[0, :].sum()
            post = g[1, :].sum()

            if delta[trl, sim] <= 0:
                delta_w = alpha_ltd_d1 * pre * post * delta[trl, sim] * (w[0,1])

            if delta[trl, sim] > 0:
                delta_w = alpha_ltp_d1 * pre * post * delta[trl, sim] * (1. - w[0, 1])

            w[0, 1] += delta_w
            w[0, 1] = np.clip(w[0, 1], 0.1, 1)

            # update synaptic weights (ctx-d2)
            pre = g[0, :].sum()
            post = g[2, :].sum()

            if delta[trl, sim] <= 0:
                delta_w = -alpha_ltp_d2 * pre * post * delta[trl, sim] * (1. - w[0, 2])

            if delta[trl, sim] > 0:
                delta_w = -alpha_ltd_d2 * pre * post * delta[trl, sim] * (w[0,2])

            w[0, 2] += delta_w
            w[0, 2] = np.clip(w[0, 2], 0.1, 1)

            # record trial
            w_record[0, trl + 1, sim] = w[0, 1]
            w_record[1, trl + 1, sim] = w[0, 2]

    fig, ax = plt.subplots(3, 1, squeeze=False)
    x = np.arange(0, n_trials - 1, 1)
    ax[0, 0].plot(x, resp[:-1, :].mean(1), 'o', label='response')
    ax[0, 0].plot([n_acquisition_trials, n_acquisition_trials], [0, 1], '--k')
    ax[0, 0].plot([n_acquisition_trials + n_extinction_trials, n_acquisition_trials + n_extinction_trials], [0, 1], '--k')
    ax[0, 0].plot([25, 25], [0, 1], '--k')
    ax[0, 0].set_title("Average Response Rate") 


    #ax[1, 0].plot([0, n_trials - 2], [resp_thresh, resp_thresh], '--k')
    ax[1, 0].plot(x, r_obtained[:-1, :].mean(1), label='obtained reward')
    ax[1, 0].plot(x, r_predicted[:-1, :].mean(1), label='predicted reward')
    ax[1, 0].plot(x, delta[:-1, :].mean(1), label='delta')
    ax[1, 0].set_title("Comparison of Obtained and Predicted Rewards")

    ax[2, 0].plot(x, w_record[0, :-1, :].mean(1), label='w ctx-d1')
    ax[2, 0].plot(x, w_record[1, :-1, :].mean(1), label='w ctx-d2')
    ax[2,0].set_ylabel('Synaptic Weight (w)')
    ax[2, 0].set_title("Average Synaptic Weight")
    [x.legend() for x in ax.flatten()]
    [x.set_xticks(np.arange(0, n_trials - 1, 5)) for x in ax.flatten()]
    plt.tight_layout()
    plt.show()


tau = 0.1
T = 3000
t = np.arange(0, T, tau)
n_steps = t.shape[0]

# Cells: CTX, D1, D2, GPi, GPe, Thal, STN
# Direct: CTX -> D1 -> GPi
# Indirect: CTX -> D2 -> GPe -> GPi
# Hyperdirect: CTX -> STN -> GPi
# Output: GPi -> Thal -> CTX
# Gain limiter: STN <-> GPe
iz_params = np.array([
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # ctx (rs) 0
    [50, -80, -25, 40, 0.01, -20, -55, 150, 1],  # d1 (spn) 1
    [50, -80, -25, 40, 0.01, -20, -55, 150, 1],  # d2 (spn) 2
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # gpi (rs) 3
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # gpe (rs) 4
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # thal (rs) 5
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # stn (rs) 6
    [100, -75, -45, 60, 0.01, 5, -56, 130, 1.2]  # tan (special case) 7
])

# baseline firing (high for GPi, GPe, thal, and tan)
E_mu = np.array([0, 0, 0, 300, 300, 300, 0, 300])
E_sig = np.array([0, 0, 0, 0, 0, 0, 0, 300])

n_cells = iz_params.shape[0]

# define input into u equations
# this only applies to TAN (index -1) but we program it more generally
u_input = np.zeros((n_cells, n_steps))
u_input[-1, n_steps // 3:2 * n_steps // 3] = 3.0
for i in range(2 * n_steps // 3, n_steps):
    u_input[-1, i] = u_input[-1, i - 1] * 0.05

# define input signal
I_in = np.zeros(n_steps)
I_in[n_steps // 3:2 * n_steps // 3] = 5e1

w_in = np.zeros(n_cells)

# add input only to ctx
w_in[0] = 75

sim_bg_2(I_in, w_in)