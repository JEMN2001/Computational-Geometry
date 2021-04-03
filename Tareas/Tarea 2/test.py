import tarea2 as t
import random as rd
import numpy as np

a = t.Airports("airports_CO.dat", "borders_CO.dat")
a.plot()
m, M = a.get_minmax_airports()
print('Airport with minimal area: ', m)
print('Airport with maximal area: ', M)
l, m = a.get_mostless_airports()
print("Aitport with less neighbors: ", l)
print("Aitport with most neighbors: ", m)
airport1 = rd.choice(a.names)
airport2 = rd.choice(a.names)
a.get_Airport_based_path(airport1, airport2)
a.get_Threat_based_path(airport1, airport2)
