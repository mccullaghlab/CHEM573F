%nprocshared=2
%mem=2GB
%chk=formamide_wb97xd.dimer_freq.chk
# wb97xd/6-31+g* Pop=NaturalOrbital freq

Formamide dimer frequency calculation

0 1
C     0.458   0.479   0.000
O     1.657   0.285   0.000
N    -0.481  -0.501   0.000
H     0.021   1.494   0.000
H    -1.467  -0.288   0.000
H    -0.190  -1.467   0.000
C     0.663   0.015   5.000
O     1.373  -0.970   5.000
N    -0.694  -0.014   5.000
H     1.071   1.042   5.000
H    -1.241   0.834   5.000
H    -1.172  -0.903   5.000

