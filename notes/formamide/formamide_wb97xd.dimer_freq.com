%nprocshared=2
%mem=2GB
%chk=formamide_wb97xd.dimer_freq.chk
# wb97xd/6-31+g* Pop=NaturalOrbital freq

Formamide dimer frequency calculation

0 1
C   0.000   0.421   0.000
O   1.199   0.227   0.000
N  -0.940  -0.558   0.000
H  -0.437   1.436   0.000
H  -1.925  -0.346   0.000
H  -0.648  -1.525   0.000
C   0.298   0.298   5.000
O   1.008  -0.687   5.000
N  -1.059   0.270   5.000
H   0.706   1.324   5.000
H  -1.606   1.117   5.000
H  -1.537  -0.620   5.000

