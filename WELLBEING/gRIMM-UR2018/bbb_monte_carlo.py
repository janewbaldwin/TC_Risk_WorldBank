v_init = 1.0
f_a = 0.2
delta_v = 0.1

v_t_uncorr = [v_init]
v_t_corr   = [v_init]

disaster_years = 10

for i in range(disaster_years):

    v_t_uncorr.append(v_t_uncorr[-1]*(f_a*(1-delta_v)+(1-f_a)))
    v_t_corr.append(v_t_corr[-1]*(1-delta_v))

print(v_t_uncorr[10])
print(v_t_corr[10])
#print(v_t_corr)
