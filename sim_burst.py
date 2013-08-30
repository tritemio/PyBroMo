#cd ../burst/src/
ph_times_d = numpy.load("sim/ph_times_EM2e5_BG6e3_90+80s_05us_40p_86pM.dat")
ph_times_a = numpy.load("sim/ph_times_EM1e4_BG4e3_90+80s_05us_40p_86pM.dat")

clk_p = t_step/32. # with t_step=0.5us -> 156.25 ns
ph_times, a_em = forge_ph_times(ph_times_d, ph_times_a)
ph_times_int = (ph_times/clk_p).astype('int64')
name = 'BroSim_DO_EM2e5_BT05'
d = Data(fname=name, nch=1, gamma=1, BT=0, 
        ph_times_m=[ph_times_int], A_em=[a_em], clk_p=clk_p, ALEX=0)
d.calc_bg(bg_calc_exp, time_s=5, tail_min_p=0.1)

#d.burst_search_t(L=10,m=10,P=None,F=6, ph_sel='DA', nofret=1)
#d.cal_ph_num()
#dplot(d, bt_fit_comp_range)

d.burst_search_t(L=10,m=10,P=None,F=6, ph_sel='DA')
d.update_bt(BT=0.05)
dplot(d, hist_size); xlim(0,200)
#savefig(name+"_hist_size.png", dpi=250)
dplot(d, hist_width); xlim(0,2)
#savefig(name+"_hist_width.png", dpi=250)
dplot(d, scatter_width_size); xlim(0,2); ylim(0,200)
#savefig(name+"_scatter_width_size.png", dpi=250)
dplot(d, timetrace_bg)
#savefig(name+"_timetrace_bg.png", dpi=250)
