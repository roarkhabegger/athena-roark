import numpy as np 
import astropy.units as u
import astropy.constants as c

# input parameters 

nx1 = 256
x1min = -100
x1max = 100

nx2 = 256
x2min = -100 
x2max = 100 

nx3 = 256
x3min = -100
x3max = 100

t_end = 500*u.Myr

injH = 100
injL = 4
SF_rate = 10**(-2.5) * u.Msun / u.yr / u.kpc**2 
Msun_per_SN = 100  # 1/Number of supernovae per solar mass of star formation

##### After here is all defined by above.

SN_rate = SF_rate* (x1max - x1min) * (x2max-x2min) *u.pc**2  / (Msun_per_SN * u.M_sun) 
print("Rate = {:0.2f} SN per Myr".format(SN_rate.to("Myr-1").value))
print("Expected number of injections for {:0.1e} Myr: {:0.2e}".format(t_end.to("Myr").value, (SN_rate * t_end).to("").value))
response = input(" Generate injection table? (y/n): ",)

if response != "y":
    print("Exiting without generating injection table. ")
    print("If table was desired, respond with y next time")
    exit()
# times = np.

print("Generating injection table...")


N_injs = np.random.poisson((SN_rate * t_end).to("").value)
dx1 = (x1max - x1min)/nx1
dx2 = (x2max - x2min)/nx2
dx3 = (x3max - x3min)/nx3
x1_lims = (x1min + injL*dx1, x1max - (injL+1)*dx1)
x2_lims = (x2min + injL*dx2, x2max - (injL+1)*dx2)
x3_lims = (-1*injH+ injL*dx3, injH- (injL+1)*dx3)

injTime = sorted(np.random.uniform(0, t_end.to("Myr").value, N_injs))
x1pos = (np.floor((np.random.uniform(*x1_lims, N_injs) - x1min) / dx1 + 0.5) + 0.5) * dx1 + x1min
x2pos = (np.floor((np.random.uniform(*x2_lims, N_injs) - x2min) / dx2 + 0.5) + 0.5) * dx2 + x2min
x3pos = (np.floor((np.random.uniform(*x3_lims, N_injs) - x3min) / dx3 + 0.5) + 0.5) * dx3 + x3min
injection_table = np.vstack((injTime, x1pos, x2pos, x3pos)).T

np.savetxt("injection_smallbox.txt", injection_table, header="Time[Myr] x1[pc] x2[pc] x3[pc]", fmt="%0.6f")