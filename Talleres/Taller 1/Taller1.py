import numpy as np

# This function determines if there exist 3 colinear points in a set of points in O(N^3)
# P: set of points (numpy arrays)
# Return: Boolean (True if there are 3 colinear points, False if not), tuple (the 3 colinear points if there are, None otherwise)
def colinear1(P):
	for p in P:
		for q in P:
			for r in P:
				if np.array_equal(p,q) or np.array_equal(p,r) or np.array_equal(q,r):
					continue
				colin = np.cross(q-p,r-p)
				if colin == 0:
					return True, [tuple(p),tuple(q),tuple(r)]
	return False, None

# This function determines if there exist 3 colinear points in a set of points in O(N^2*log(N))
# P: set of points(tuples)
# Return: Boolean (True if there are 3 colinear points, False if not), tuple (the 3 colinear points if there are, None otherwise)
def colinear2(P):
	slopes = {}
	for p in P:
		for q in P:
			if p == q:
				continue
			if p[0] == q[0]:
				m = 'inf'
			else:
				m = (p[1]-q[1])/(p[0]-q[0])
			if m not in slopes:
				slopes[m] = [(p,q)]
			else:
				if (p,q) in slopes[m] or (q,p) in slopes[m]:
					continue
				slopes[m].append((p,q))

	#Print purposes
	#for slope in slopes:
		#print(slope,": ",slopes[slope])
	#End print purposes

	for slope in slopes:
		lines = slopes[slope]
		if len(lines)<2:
			continue
		for line in lines:
			p = line[0]
			q = line[1]
			for line2 in lines:
				if line2 == line:
					continue
				if p in line2 or q in line2:
					pts = [p,q,line2[0],line2[1]]
					pts = list(dict.fromkeys(pts))
					return True, pts
	return False, None

# This function determines if the line segments p0p1 and p2p3 intersects
# p0, p1, p2, p3: points (numpy.arrays)
# Return: True or False
def intersect(p0,p1,p2,p3):
	s1 = np.cross(p1-p0,p3-p1)
	s2 = np.cross(p1-p0,p2-p1)
	if s1*s2 < 0:
		return True
	if s1*s2 > 0:
		return False
	x_min = min(p0[0],p1[0])
	x_max = max(p0[0],p1[0])
	y_min = min(p0[1],p1[1])
	y_max = max(p0[1],p1[1])
	if s1 == 0:
		x = p3[0]
		y = p3[1]
		if x_min <= x and x <= x_max and y_min <= y and y <= y_max:
			return True
	if s2 == 0:
		x = p2[0]
		y = p2[1]
		if x_min <= x and x <= x_max and y_min <= y and y <= y_max:
			return True
	return False


# This function determines if the line segment p2p3 intersects with the left horizontal ray of p0
# p0, p1, p2, p3: points (numpy.arrays)
# horizontal: True if it's a ray paralel to the x axis, False if not
# positive, True if the ray is directed positively to it's axis, Flase if not
# Return: True or False
def ray_intersects(p0,p1,p2,horizontal,possitive):
	if horizontal:
		if possitive:
			x = max(p1[0],p2[0])
			if x < p0[0]:
				return False
		else:
			x = min(p1[0],p2[0])
			if x > p0[0]:
				return False
		p = np.array((x,p0[1]))
	else:
		if possitive:
			y = max(p1[1],p2[1])
			if y < p0[1]:
				return False
		else:
			y = min(p1[1],p2[1])
			if y > p0[1]:
				return False
		p = np.array((p0[0],y))
	return intersect(p,p0,p1,p2)

def Inside_convex_poly(P,p):
	n = len(P)
	ray = 0
	for horizontal in [True, False]:
		for possitive in [True, False]:
			r_valid = False
			for i in range(n):
				d = np.linalg.norm(P[i%n]-P[(i+1)%n])
				d1 = np.linalg.norm(p-P[(i+1)%n])
				d2 = np.linalg.norm(P[i%n]-p)
				if d == d1+d2:
					return False
				if ray_intersects(p,P[i%n],P[(i+1)%n],horizontal,possitive):
					r_valid = True
					break
			if not r_valid:
				return False
	return True

if __name__ == '__main__':
	P = [(3, -4), (-7, 10), (7, -6), (-2, 8), (3, -10), (1, 10), (1, -8), (-2, 8), (-3, 1), (-9, -9), (8, 2), (0, -5), (-2, 7), (8, -2), (-9, -9), (-4, -3), (7, -6), (-9, -3), (6, 1), (5, 7), (8, 5), (4, 6), (3, -10), (2, -7), (0, 1), (-1, 8), (-6, -10), (-2, -3), (10, 8), (3, 8), (1, -8), (-1, 2), (2, -4), (3, 6), (-3, -1), (-6, 10), (7, -5), (-9, 2), (-8, -7), (5, -4), (4, -9), (-1, -4), (-5, -4), (7, 10), (-8, 9), (-4, -6), (-5, -7), (10, -7), (-6, -5), (8, 1)]
	
	P_np = [np.array(x) for x in P]
	print(colinear1(P_np))

	print(colinear2(P))

	P = [(0,3),(2,1),(2,-2),(-2,-4),(-6,0)]
	P_np = [np.array(x) for x in P]
	p = (-2,0)
	p_np = np.array(p)
	print(Inside_convex_poly(P_np,p_np))