from pandas import read_csv
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from scipy.spatial.qhull import QhullError
import numpy as np
from matplotlib import pyplot as plt

'''
Class to represent store airports and its corresponding Voronoi diagram

Attributes:
    _division: scipy.spatial.Voronoi; the Voronoi subdivision of
                                      the plane for the airports
    altitudes: list; where the ith posotion is the altitude of the ith airport
    cities: list; where the ith position is the city of the ith airport
    departaments: list; where the ith position is the
                        departament of the ith airport
    names: list; where the ith psoition is the name of the ith airport
    _map_x: np.array; representing the x coordinates of the country borders
    _map_y: np.array; representing the y coordinates of the country borders

Raises:
    Assert error
    FileNotFoundError
    QhullError
'''


class Airports:

    '''
    Constructor

    Arguments:
        points_filename: str; name of the file storing the airports data
        map_filename: str; name of the file storing the borders
                           of the country map (optional)

    Raises:
        Assert error
        FileNotFoundError
        QhullError
    '''
    def __init__(self,
                 points_filename,
                 map_filename=None):

        assert type(points_filename) == str, \
               "points_filename has to be a string"

        try:
            df = read_csv(points_filename, sep=r'\s+',
                          names=['latitude', 'longitude', 'altitude',
                                 'city', 'departament', 'name'])
            points = np.array([
                              [df['longitude'][i], df['latitude'][i]]
                              for i in range(len(df))])
            self._division = Voronoi(points)
            self.altitudes = np.array(df['altitude'])
            self.cities = np.array(df['city'])
            self.departaments = np.array(df['departament'])
            self.names = np.array(df['name'])

        except FileNotFoundError:
            raise Exception("The file "+points_filename+" doesn't exist")

        except QhullError:
            raise Exception("Couldn't create the voronoi subdivision")

        if (map_filename is not None):
            try:
                df = read_csv(map_filename,
                              sep=r'\s+',
                              names=['latitude', 'longitude'])
                self._map_x = np.array(
                                       [df['longitude'][i]
                                        for i in range(len(df))])
                self._map_y = np.array(
                                       [df['latitude'][i]
                                        for i in range(len(df))])

            except FileNotFoundError:
                raise Exception("The file "+map_filename+" doesn't exist")

        else:
            self._map_x = np.array([])
            self._map_y = np.array([])

    '''
    Method to plot the voronoi subdivision and the country borders if there are
    '''
    def plot(self):
        fig = voronoi_plot_2d(self._division,
                              show_vertices=False,
                              line_colors='orange')
        fig.set_size_inches(8, 7)
        plt.plot(self._map_x, self._map_y, 'k')
        plt.ylabel('Latitude')
        plt.xlabel('Longitude')
        plt.title("Voronoi Map")
        plt.show()

    '''
    Function that reports the area of a Voronoi region

    Attributes:
        region: list; a valid region for the Voronoi subdivision

    Returns:
        float; area of the region

    Complexity:
        O(N) with respect of the number of vertices of the region,
        O(1) with respect of the number of airports
    '''
    def _area(self, region):
        vertices = [self._division.vertices[i] for i in region]
        n = len(vertices)
        out = 0
        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i+1) % n]
            out += p1[0]*p2[1]-p1[1]*p2[0]
        out /= 2
        return abs(out)

    '''
    Function that reports the airport with max area and min
    area in the subdivision, only compares finite rigions

    Returns:
        str, str; names of the airport with bigest coverage area
                  and min coverage area

    Complexity:
        O(E*N) where E is the mean value of vertices per region,
               and N the amount of airports
    '''
    def get_minmax_airports(self):
        index_maxarea = -1
        index_minarea = -1
        minarea = np.inf
        maxarea = -1
        p_r = self._division.point_region
        regions = self._division.regions
        n = len(p_r)
        for i in range(n):
            region = regions[p_r[i]]
            if -1 not in region and region != []:
                area = self._area(region)
                if area <= minarea:
                    minarea = area
                    index_minarea = i
                if area >= maxarea:
                    maxarea = area
                    index_maxarea = i
        mn = self.names[index_minarea]
        mx = self.names[index_maxarea]
        return mn, mx

    '''
    Function taht reports the airports with most neighbors and least neighbors

    Returns:
        str, str; names of the airports with most and less neighbors

    Complexity:
        O(N) with respect to the number of airports
    '''
    def get_mostless_airports(self):
        n = len(self._division.points)
        edges = self._division.ridge_points
        neighbors = np.zeros(n)
        for e in edges:
            neighbors[e[0]] += 1
            neighbors[e[1]] += 1
        min_index = np.argmin(neighbors)
        max_index = np.argmax(neighbors)
        ls = self.names[min_index]
        mt = self.names[max_index]
        return ls, mt

    '''
    Function that reports all the neighbors of a point

    Attributes:
        point: np.array; an array with the x and y coordinates of the point
        tp: str; if it is 'point' it handle the point as such and
                 compute the neighbor points if it is 'vertex' it
                 handle the point as a voronoi vertex and returns the
                 neighbor vertices

    Returns:
        np.array; an array with the neighbor points or vertices

    Raises:
        Assertion error
        IndexError

    Complexity:
        O(N) with respect to the number of airports
    '''
    def _get_neighbors(self, point, tp='point'):
        assert tp == 'point' or tp == 'vertex', "Invalid type"
        if tp == 'point':
            points = self._division.points
            all_neighbors = self._division.ridge_points
        else:
            points = self._division.vertices
            all_neighbors = self._division.ridge_vertices
        out = []
        try:
            index = np.where(np.all(points == point, axis=1))[0][0]
            incident_edges = [x for x in all_neighbors if index in x]
            for inc in incident_edges:
                if inc[0] == index and inc[1] != -1:
                    out.append(inc[1])
                elif inc[1] == index and inc[0] != -1:
                    out.append(inc[0])
            return np.array([points[i] for i in out])
        except IndexError:
            raise Exception("The point {} isn't a valid point".format(point))

    '''
    Method that plots the Airport based path between two airports
    This path moves from one airport to another until reach the destination

    Attributes:
        origin_n: str; name of the origin airport
        destination_n: str; name of the destination airport

    Raises:
        Assertion error

    Complexity:
        O(P*N) where P is the amount of airports visited in the path,
               and N the total amount of Airports
        This complexity was calculated only for the path generation,
        and not for the path plotting.
    '''
    def get_Airport_based_path(self, origin_n, destination_n):
        assert origin_n in self.names, \
               "The origin is not an airport"
        assert destination_n in self.names, \
               "The destination is not an airport"
        idx1 = np.where(
                        self.names == origin_n)[0][0]
        idx2 = np.where(
                        self.names == destination_n)[0][0]
        origin = self._division.points[idx1]
        destination = self._division.points[idx2]
        current = origin
        path = [list(origin)]
        while (not np.all(current == destination)):
            ngh = self._get_neighbors(current)
            distances = cdist(ngh, destination.reshape(1, 2))
            idx = np.argmin(distances)
            current = ngh[idx]
            path.append(list(current))
        path.append(list(destination))
        fig = voronoi_plot_2d(self._division,
                              show_vertices=False,
                              line_colors='k')
        fig.set_size_inches(8, 7)
        plt.plot(self._map_x, self._map_y, 'gray')
        plt.ylabel('Latitude')
        plt.xlabel('Longitude')
        plt.plot([x[0] for x in path], [x[1] for x in path], 'r', linewidth=2)
        plt.title(origin_n+" to "+destination_n+" airport based path")
        plt.show()

    '''
    Function that reports the closest point laying on a
    voronoi edge to a given point and the index of the edge
    where it lays

    Attributes:
        origin: np.array; array with the x and y coordinates of the point

    Returns:
        list, int; list with the x and y coordinates of the output point
                   and the index of the edge where the point laysS
    Complexity:
        O(N) with respect to the number of airports
    '''
    def _get_closest_edge_point(self, origin):
        ngh = self._get_neighbors(origin)
        distances = cdist(ngh, origin.reshape(1, 2))
        idx = np.argmin(distances)
        point2 = ngh[idx]
        r_p = self._division.ridge_points
        p = self._division.points
        idx1 = np.where(p == origin)[0][0]
        idx2 = np.where(p == point2)[0][0]
        try:
            idx = np.where(np.all(r_p == np.array([idx1, idx2]), axis=1))[0][0]
        except IndexError:
            idx = np.where(np.all(r_p == np.array([idx2, idx1]), axis=1))[0][0]
        return 0.5*(origin+point2), idx

    '''
    Method that plots the Threat based path between two airports
    this path moves first to the closest Voronoi edge, to then start
    moving through edges until reaching the closest points of an edge
    to the destination, to then move to the destination

    Attributes:
        origin_n: str; name of the origin airport
        destination_n: name of the destination airport

    Raises:
        Assert error

    Complexity:
        O(P*N) where P is the amount of voronoi vertices visited in the path,
               and N is the total amount of airports
        This complexity was only caculated for the path computation,
        and not for the path plotting
    '''
    def get_Threat_based_path(self, origin_n, destination_n):
        assert origin_n in self.names, \
               "The origin is not an airport"
        assert destination_n in self.names, \
               "The destination is not an airport"
        idx1 = np.where(
                        self.names == origin_n)[0][0]
        idx2 = np.where(
                        self.names == destination_n)[0][0]
        origin = self._division.points[idx1]
        if (origin_n == destination_n):
            return [origin]
        destination = self._division.points[idx2]
        closest_origin, idx1 = self._get_closest_edge_point(origin)
        closest_destination, idx2 = self._get_closest_edge_point(destination)
        e1 = self._division.ridge_vertices[idx1]
        e2 = self._division.ridge_vertices[idx2]
        if -1 in e1:
            vertex_o = e1.index(-1)
            idx = e1[1-vertex_o]
            vertex_o = self._division.vertices[idx]
        else:
            dist = cdist(self._division.vertices[e1],
                         destination.reshape(1, 2))
            vertex_o = self._division.vertices[e1[np.argmin(dist)]]
            idx = e1[np.argmin(dist)]
        path = [origin, closest_origin, vertex_o]
        current = vertex_o
        while (True):
            if idx in e2:
                path.append(list(current))
                break
            ngh = self._get_neighbors(current, tp='vertex')
            distances = cdist(ngh, closest_destination.reshape(1, 2))
            idx = np.argmin(distances)
            current = ngh[idx]
            if (True not in [np.all(current == x) for x in path]):
                path.append(list(current))
            else:
                ngh = np.delete(ngh, idx, axis=0)
                distances = cdist(ngh, closest_destination.reshape(1, 2))
                idx = np.argmin(distances)
                current = ngh[idx]
                path.append(current)
            idx = np.where(
                           np.all(
                                  self._division.vertices == np.array(current),
                                  axis=1))[0][0]
        path.append(closest_destination)
        path.append(destination)
        fig = voronoi_plot_2d(self._division,
                              show_vertices=False,
                              line_colors='k')
        fig.set_size_inches(8, 7)
        plt.plot(self._map_x, self._map_y, 'gray')
        plt.ylabel('Latitude')
        plt.xlabel('Longitude')
        plt.plot([x[0] for x in path], [x[1] for x in path], 'b', linewidth=2)
        plt.title(origin_n+" to "+destination_n+" threat based path")
        plt.show()
