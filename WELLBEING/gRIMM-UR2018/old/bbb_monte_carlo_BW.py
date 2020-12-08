v_init = 1.0
f_a = 0.2
delta_v = 0.1

v_t_uncorr = [v_init]
v_t_corr   = [v_init]

disaster_years = 10

for i in range(disaster_years):

    v_t_uncorr.append(v_t_uncorr[-1]*(f_a*(1-delta_v)+(1-f_a)))

    if i ==1:
        v_t_corr.append((v_t_corr[-1]-(1-f_a)*v_init)*(1-delta_v)+(1-f_a)*v_init)

#print(v_t_uncorr[-1])
#print(v_t_corr[-1])


_f_a = f_a
print(f_a*(1-f_a)**disaster_years)
for i in range(disaster_years):
    _f_a *= (1-_f_a)
    
print(_f_a)
    
