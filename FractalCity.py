#!/usr/bin/env python
# coding: utf-8

# In[370]:


import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import shapely.ops as so
import shapely.wkt
import matplotlib.pyplot as plt
import powerlaw

from shapely import ops
from scipy import stats
from scipy.stats import t
from shapely.geometry import Point, box, GeometryCollection
from shapely.strtree import STRtree


# In[369]:


class ReadData:
    """A class to read information from nodes (osmid, x, y, lon, lat, geometry) and links
       (u, v, length, geometry) from the csv's files to generate the street network.
       
       Parameters
       ----------
       str : path
            Path where the Brazil folder is located with the data in the nodes
            and the street network links.
            
       Methods
       -------
       read_nodes_and_links(name, uf, region)
          Read the csv with nodes and links information.
          
       read_polygons_populations(name, uf, region)
          Read the csv with geometries and populations.
            
    """
    def __init__(self, path):
        self.path = path

    
    def read_nodes_and_links(self, name, uf, region):
        """Read the dataframes (pandas.core.frame.DataFrame) of nodes and links of the street network.
        
           Parameters
           ----------
           str : name
               City name (e.g., 'São Gonçalo do Sapucaí', "Lavras", etc).
           str : uf
               State abbreviation (e.g., 'MG', "SP", etc).
           str : region
               Rerion name (e.g., 'Sudeste', "Norte", etc).
            
            
           Returns
           -------
           tuple:
               The first component of the tuple is a 'pandas.core.frame.DataFrame' with the node information
               (osmid, x, y, lon, lat, geometry).
            
               Details
               
               ---------------------------------------------------------------
               variable | interpretation
               ---------------------------------------------------------------
               ---------------------------------------------------------------
               osmid    | node id (an integer, e.g., 94938203, 63278324, etc)
               ---------------------------------------------------------------
                  x     | node abyss
               ---------------------------------------------------------------
                  y     | node ordinate
               ---------------------------------------------------------------
                 lon    | node longitude (format: epsg 4326)
               ---------------------------------------------------------------
                 lat    | node latitude (format: epsg 4326)
               ---------------------------------------------------------------
               geometry | node geometry (shapely.geometry.point.Point)
               ---------------------------------------------------------------
            
            
               The second component of the tuple is a 'pandas.core.frame.DataFrame' with the link information
               (u, v, length, geometry).
            
               Details
               
               ------------------------------------------------------------------
               variable |  interpretation
               ------------------------------------------------------------------
               ------------------------------------------------------------------
                  u     | identification of a link node (an integer,
                        | e.g., 94938203, 63278324, etc.)
               ------------------------------------------------------------------
                  v     | identification of another node on the link (an integer,
                        | e.g., 94938203, 63278324, etc.)
               -------------------------------------------------- ---------------
               length   | link length in meters
               -------------------------------------------------- ---------------
               geometry | link geometry (shapely.geometry.linestring.LineString)
               -------------------------------------------------- ---------------
            
               Useful links:
            
               1. https://shapely.readthedocs.io/en/stable/manual.html
            
               2. https://pandas.pydata.org/docs/user_guide/index.html#user-guide
            
               3. https://geopandas.org/en/stable/docs/user_guide.html
        """
        
        path = self.path +'Brasil/Região/'+ region + '/' + uf + '/'+ 'Municípios/'
        df_nodes = pd.read_csv(path +'nos_proj_'+ name + '.csv', usecols=['osmid', 'x', 'y', 'lon', 'lat', 'geometry'])
        df_links = pd.read_csv(path +'links_proj_'+ name + '.csv', usecols = ['u', 'v', 'length', 'geometry'])
        
        return df_nodes, df_links
    
    def read_polygons_populations(self, name, uf, region):
        """Method for get the polygons and population live in each polygon.
        
           Parameters
           ----------
           name : str
               City name (e.g., São Gonçalo do Sapucaí, Lavras, Belo Horizonte, etc.).
        
           uf : str
               City state abbreviation (e.g., MG, SP, AM, etc.).
        
           Returns
           -------
        
           polygon, population : list, list
              A tuple of lists: the first component of the tuple is a list of polygons (of the census track geted 
              from IBGE) of the city and the second component of the tuple is a list of population (of the census
              track geted from IBGE). OBS: don't sort the population list!
           
        """
        path = self.path+"Geoms/Região/"+region+"/"+uf+"_geoms_populs.csv"
        df = pd.read_csv(path)
        city = df.loc[df.nome==name]
    
        geo = [shapely.wkt.loads(eval(list(city.geometry)[0])[i]).buffer(0) for i in range(len(eval(list(city.geometry)[0])))]
        polygon = GeometryCollection(geo)
    
        s = list(city.popul_setor)[0]
        p = list(np.fromstring(s[1:-1], sep=', '))
        population = list(np.nan_to_num(p))
    
        return polygon, population

class StreetNetwork:
    """A class to generate the street network from node information
       (osmid, x, y, lon, lat, geometry) and links (u, v, length, geometry)
       of previously read csv files.
       
       Methods
       -------
       generate_network(df_nodes, df_links)
           Method to generate the street network from the information of the nodes and the
           links.
            
       -----------------------------------------
            
       cyclomatic_number(G)
           It describes the number of edges that must be removed from a network to ensure
           that no network cycles remain.
       
       alpha_number(G)
           It describes the ratio between the number of circuits (loops) and the maximum
           number of circuits in the network with the same number of nodes.
       
       beta_number(G)
           It measures the frequency of connections based on the relationship between the 
           number of links and the number of nodes in the network.
       
       gamma_number(G)
           It measures the frequency of links and is defined as the ratio between the number
           of links and the maximum possible number of links.
           
       Ref. 
        
       Sharifi, A. Resilient Urban Forms: A Review of Literacture on Street and Street Network. 2018.

       -----------------------------------------
       
       network_area(G)
          Convex hull area formed by the network nodes.
       
       network_perimeter(G)
          Convex hull perimeter formed by the network nodes.
          
       -----------------------------------------
       
       network_efficiency(G)
          Calculate the street network efficiency.
          
          Ref. 
          
          "Paolo Crucitti, Vito Latora, and Sergio Porta. Centrality measures in
           spatial networks of urban streets. Phys. Rev. E 73, 036125 – Published 24 March 2006"
       
       ----------------------------------------
       
       generate_routes(G)
          Generate 1000 routes on the street network as default.
          
       plot_network(G)
          Method to plot the street network.
    """
    
    def generate_network(self, df_nodes, df_links):
        """Method to generate the street network from the information of the nodes and the
           links. The node positions are longitude and latitude.
            
           Parameters
           ----------
           pandas.core.frame.DataFrame : df_nodes
              Dataframe with node information.
        
           pandas.core.frame.DataFrame : df_links
              Dataframe with the link information.
           
           returns
           -------
           G : networkx.classes.graph.Graph
              Street network.
        """ 
        nodes = list(df_nodes.osmid)
        coords = list(zip(list(df_nodes.lon), list(df_nodes.lat)))
        links = list(zip(list(df_links.u), list(df_links.v)))
        dict_nodes_pos = dict(zip(nodes, coords))
            
        w_links = []
        for link in links:
            node0, node1 = link
            x0, y0 = dict_nodes_pos[node0]
            x1, y1 = dict_nodes_pos[node1]
            d = ((x1-x0)**2+(y1-y0)**2)**0.5
            w_links.append(d)
                
        G = nx.Graph()
        G.add_edges_from(links)
        nx.set_node_attributes(G, dict_nodes_pos, 'pos')
        weight = [{'weight': w} for w in w_links]
        links_weighted = dict(zip(links, weight))
        nx.set_edge_attributes(G, links_weighted)
        
        return G
            
    def cyclomatic_number(self, G):
        """Method for calculating the cyclomatic number, which describes the number of edges 
           that must be removed from a network to ensure that no network cycles remain.
           
           Parameters
           ----------
           networkx.classes.graph.Graph : G
              Street network.
           
           Returns
           -------
           float
              Cyclomatic number.
        """
        cyclomatic_number = G.number_of_edges()-G.number_of_nodes()+1
        return cyclomatic_number
    
    def alpha_number(self, G):
        """Method for calculating the alpha number, which describes the ratio between the number of circuits 
           (loops) and the maximum number of circuits in the network with the same number of nodes.
          
           Parameters
           ----------
           networkx.classes.graph.Graph : G
             Street network.
           
           Returns
           -------
           float
              Alpha number.
        
        """
        alpha_index = (G.number_of_edges()-G.number_of_nodes()+1)/(2*G.number_of_nodes()-5)
        return alpha_index
    
    def beta_number(self, G):
        """Method for calculating the beta number, which measures the frequency of connections based on the 
           relationship between the number of links and the number of nodes in the network.
           
           Parameters
           ----------
           networkx.classes.graph.Graph : G
             Street network.
           
           Returns
           -------
           float
              Beta number.
        """
        beta_index = G.number_of_edges()/G.number_of_nodes()
        return beta_index
    
    def gamma_number(self, G):
        """Method for calculating the gamma number, measures the frequency of links and is defined as the 
           ratio between the number of links and the maximum possible number of links.
           
           Parameters
           ----------
           networkx.classes.graph.Graph : G
             Street network.
           
           Returns
           -------
           float
              Gamma number.
        """
        gamma_index = G.number_of_edges()/(3*(G.number_of_nodes()-2))
        return gamma_index
    
    def network_area(self, G):
        """Method for calculating the network area based on the convex hull.
           
           Parameters
           ----------
           networkx.classes.graph.Graph : G
             Street network.
           
           Returns
           -------
           float
              Convex hull area formed by the street network nodes.
        """

        lonlat = [G.nodes(data=True)[node]['pos'] for node in list(G.nodes())]

        points = [Point(pos[0],pos[1]) for pos in lonlat]
        
        union_points = ops.unary_union(points)
        
        hull = union_points.convex_hull
        
        gdf = gpd.GeoDataFrame({'geometry': [hull]}, crs = 'epsg:4326')
        
        gdf_convert = gdf.to_crs(5837)

        return gdf_convert.area.values[0]
    
    def network_perimeter(self, G):
        """Method for calculating the network perimeter based on the convex hull.
        
           Parameters
           ----------
           networkx.classes.graph.Graph : G
             Street network.
           
           Returns
           -------
           float
              Convex hull perimeter formed by the street network nodes.
        
        """

        lonlat = [G.nodes(data=True)[node]['pos'] for node in list(G.nodes())]

        points = [Point(pos[0],pos[1]) for pos in lonlat]
        
        union_points = ops.unary_union(points)
        
        hull = union_points.convex_hull
        
        gdf = gpd.GeoDataFrame({'geometry': [hull]}, crs = 'epsg:4326')
        
        gdf_convert = gdf.to_crs(5837)

        return gdf_convert.exterior.length.values[0]
    
    def network_efficiency(self, G):
        """Method to calculate the street network efficiency.
        
           Parameters
           ----------
           networkx.classes.graph.Graph : G
               Street network.
             
           Returns
           -------
           float
              Network efficiency.
        
              Ref. 
           
              "Paolo Crucitti, Vito Latora, and Sergio Porta. Centrality measures in
               spatial networks of urban streets. Phys. Rev. E 73, 036125 – Published 24 March 2006"
        """
    
        def degree_meter(A):
            return (2*np.pi*A*6378137)/360

        nodes = list(G.nodes())
        dict_nodes_pos = dict(zip(nodes, [G.nodes(data=True)[node]['pos'] for node in nodes]))
        M = [[(nodes[i], node) for node in nodes] for i in range(len(nodes))]    

        summ = 0
        for i in range(len(M)):
            for j in range(len(M)):
                if i != j:
                    route = nx.shortest_path(G, M[i][j][0], M[i][j][1])

                    source, target = route[0], route[-1]
                    x0, y0 = dict_nodes_pos[source]
                    x1, y1 = dict_nodes_pos[target]

                    euclid_dist = degree_meter(((x1-x0)**2+(y1-y0)**2)**0.5)

                    shp_dist = 0
                    for k in range(len(route)-1):
                        x0, y0 = dict_nodes_pos[route[k]]
                        x1, y1 = dict_nodes_pos[route[k+1]]
                        shp_dist += degree_meter(((x1-x0)**2+(y1-y0)**2)**0.5)

                    summ += euclid_dist/shp_dist

        efficiency = summ/(len(M)*(len(M)-1))

        return efficiency
    
    def generate_routes(self, G, routes=1000):
        """Method to generate 1000 routes as default on the street nework.
        
           Parameters
           ----------
           networkx.classes.graph.Graph: G
              Street network.
              
           int : routes
               Number of routes that will be generated.
               
           Returns
           -------
           dict 
              Dictionary with euclidian, chemical, average euclidian distances,
              the routes (list of nodes) and position nodes of each route.
              
              
              Details
              
              ----------------------------------------------------------------------
                    variable               |     interpretation
              ----------------------------------------------------------------------
              ----------------------------------------------------------------------
              'euclidian_distance'         |  list of the euclidian distance
              ----------------------------------------------------------------------
              'average_euclidian_distance' |  list of the average euclidian distance
              ----------------------------------------------------------------------
              'chemical_distance'          |  list of the chemical distance
              ----------------------------------------------------------------------
              'chemical_routes'            |  list of the chemical routes
              ----------------------------------------------------------------------
              'chemical_routes_pos'        |  list of the chemical routes position
              ----------------------------------------------------------------------
           
        """
    
        def degree_meters(A):
            return (2*np.pi*A*6378137)/360

        def average_euclidian_distance(euclid_dist, chem_dist):

            average_euclid_dist = []

            for cd in list(chem_dist):
                cont = 0
                sum_ed = 0
                for i in range(len(list(euclid_dist))):
                    ed = list(euclid_dist)[i]
                    if ed <= cd:
                        sum_ed += ed
                        cont += 1
                aed = sum_ed/cont
                average_euclid_dist.append(aed)

            return average_euclid_dist

        dict_nodes_pos = dict(zip(list(G.nodes()),[G.nodes(data=True)[node]['pos'] for node in list(G.nodes())]))

        ed_list = []
        cd_list = []
        cr_list = []
        cr_pos_list = []

        count = 0

        while count <= routes-1:

            nodes =  list(dict_nodes_pos.keys())

            source, target = np.random.choice(nodes, 2, replace = False)

            x1, y1 = dict_nodes_pos[source]
            x2, y2 = dict_nodes_pos[target]
            
            ed = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            ed_list.append(degree_meters(ed))

            cr = nx.shortest_path(G, source, target)
            cr_list.append(cr)
            
            cr_pos = [dict_nodes_pos[node] for node in cr]
            cr_pos_list.append(cr_pos)
            
            sum_d = 0
            for i in range(len(cr)-1):
                inode = cr[i]
                fnode = cr[i+1]
                x1, y1 = dict_nodes_pos[inode]
                x2, y2 = dict_nodes_pos[fnode]
                d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                sum_d += degree_meters(d)
            cd = sum_d
            cd_list.append(cd)

            count +=1

        aed_list = average_euclidian_distance(ed_list, cd_list)

        dict_routes = {'euclidian_distance': ed_list, 'average_euclidian_distance': aed_list,
                      'chemical_distance': cd_list,'chemical_routes': cr_list,
                      'chemical_routes_pos': cr_pos_list}
        
        return dict_routes

            
    def plot_network(self, G, kwargs):
        """Generates the image of the street network of the city.

           Parameters
           ----------
           networkx.classes.graph.Graph : G
              Street network.
        
           dict : kwargs
              Dictionary with parameters for the plot.
               
              Details:
              --------------------------------------
              variable     | type         | exemple
              --------------------------------------
              --------------------------------------
              'figsize'    | tuple of int | (10, 10)
              --------------------------------------
              'node_size'  | int          |    0
              --------------------------------------
              'edge_width' | int          |    1
              --------------------------------------
            
           Returns
           -------
           plt.image
               Image of the street network.

        """
        
        dict_nodes_pos = dict(zip(list(G.nodes()),[G.nodes(data=True)[node]['pos'] for node in list(G.nodes())]))
    
        fig, ax = plt.subplots(figsize=kwargs['figsize'])
        nx.draw(G, pos=dict_nodes_pos, ax=ax, node_size=kwargs['node_size'], width=kwargs['edge_width']) 
        
        return plt.show()
    
class point:
    """A class to generate point objects for calculating the fractal dimension using 
       the "sandbox" method.
    
       References
       -----------
       
       1. Tél, T., Á. Fülöp, and T. Vicsek (1989), Determination of fractal dimensions
          for geometrical multifractals, Physica A, 159, 155 – 166.
       
       2. Vicsek, T. (1990), Mass multifractals, Physica A, 168, 490 – 497.
       
       3. Vicsek, T., F. Family, and P. Meakin (1990), Multifractal geometry of
          diffusion-limited aggregates, Europhys. Lett., 12(3), 217 – 222.
       
       4. S. G. De Bartolo, R. Gaudio, and S. Gabriele, “Multifractal analysis of river networks:
          Sandbox approach,” Water Resources Research, vol. 40, no. 2, 2004.
    """
    pass
    
class SandBox:
    """Class to calculate the generalized fractal dimension of the street network and 
       population using the "sandbox" method.
            
       Methods
       -------
       generate_grid(df, threshold = 50.0, gridsize = 15, epsg = True):
           Method to generate a grid over the street network.
           
       generate_points(x_red, y_red, x_blue, y_blue, x_gray, y_gray): 
           Method to generate points for estimating the object's fractal dimension.
           
       apply_sandbox_on_street_network(num_div_rmin = 100, num_div_rmax = 10,
                                       num_intervals = 100, fraction = 2):
           Method to carry out the steps of the "sandbox" algorithm on street network.
       
       get_sandbox_data_from_street_network:
           Method for obtaining data from calculations on street network.
           
       apply_sandbox_on_population(multipolygon, populs, x_red, y_red, x_gray, y_gray,
                                   num_div_rmin = 100, num_div_rmax = 10, num_intervals = 100, fraction = 2):
           Method for obtaining data from calculations on population geometries.
           
       street_network_generalized_dimension(data, moment_list)
           Method to estimating the generalized fractal dimension of the street network.
       
       street_network_mass_exponent(dq_list, moment_list)
           Method to estimating the mass exponents of the street network.
          
       street_network_multifractal_spectrum(tau_list, moment_list)
           Method to estimating the multifractal spectrum of the street network.
       
       population_generalized_dimension(data, moment_list)
           Method to estimating the generalized fractal dimension of the population.
       
       population_mass_exponent(dq_list, moment_list)
           Method to estimating the mass exponents of the population.
           
       population_multifractal_spectrum(tau_list, moment_list)
           Method to estimating the generalized fractal dimension of the population.
       
       References
       ----------
       
       1. Tél, T., Á. Fülöp, and T. Vicsek (1989), Determination of fractal dimensions
          for geometrical multifractals, Physica A, 159, 155 – 166.
       
       2. Vicsek, T. (1990), Mass multifractals, Physica A, 168, 490 – 497.
       
       3. Vicsek, T., F. Family, and P. Meakin (1990), Multifractal geometry of
          diffusion-limited aggregates, Europhys. Lett., 12(3), 217 – 222.
       
       4. S. G. De Bartolo, R. Gaudio, and S. Gabriele, “Multifractal analysis of river networks:
          Sandbox approach,” Water Resources Research, vol. 40, no. 2, 2004.
            
    """
    
    def __init__(self):
        self.points_found = []
        self.defined_radios = []
        self.diagonal = []
        self.selected_points = []
        self.total_points = []
    
    
    def generate_grid(self, df, threshold = 50.0, gridsize = 15, epsg = True):
        """Method for finding the points of a grid, defined by "gridsize" (default: 15 x 15 grid),
           closest to a certain distance, defined by "threshold" (default: 50 meters), from the po-
           ints of a given object (e.g., street network). The coordinates format it will be epsg 
           4326 (latitude and longitude).
            
           Parameters
           ----------
           float : threshold
              Threshold to find the points (reds). Default: 50 meters.
        
           int : gridsize
              Grid size. Default: 15 x 15 grid. (e.g., 15 x 15 = 225 evenly spaced dots).
            
           Returns
           -------
           tuple of lists: (x_red, y_red, x_blue, y_blue, x_gray, y_gray)
               
              Details:
               
              ----------------------------------------------------------------------------
              variable | type   | description
              ----------------------------------------------------------------------------
              ----------------------------------------------------------------------------
              x_red    | list   | List of x's or lon's coordinates (epsg format: 4326)
                       |        | of the closest found points of the points of the object
                       |        | that you want to estimate the fractal dimension.
              ----------------------------------------------------------------------------
              y_red    | list   | List of y's or lat's coordinates (epsg format: 4326)
                       |        | of the closest found points of the points of the object
                       |        | that you want to estimate the fractal dimension.
              ----------------------------------------------------------------------------
              x_blue   | list   | List of x's or lon's coordinates (epsg format: 4326)
                       |        | of the object's points that are wants to estimate 
                       |        | the fractal dimension.
              ----------------------------------------------------------------------------
              y_blue   | list   | List of y's or lat's coordinates (epsg format: 4326)
                       |        | of the object's points that are wants to estimate
                       |        | the fractal dimension.
              ----------------------------------------------------------------------------
              x_gray   | list   | List of x's or lon's coordinates (epsg format: 4326)
                       |        | of the grid.
              ----------------------------------------------------------------------------
              y_gray   | list   | List of y's or lat's coordinates (epsg format: 4326)
                       |        | of the grid.
              ----------------------------------------------------------------------------      
           
        """
        
        def degree_to_meter(degree):
            return (np.pi*degree*6378137)/180
        
        def meter_to_degree(meter):
            return (meter*180)/(np.pi*6378137)
        
            
        lon_blue, lat_blue = list(df.lon), list(df.lat)
        lonmin, lonmax, latmin, latmax = min(df.lon), max(df.lon), min(df.lat), max(df.lat)

        lon_space = np.linspace(lonmin, lonmax, gridsize)
        lat_space = np.linspace(latmin, latmax, gridsize)
        lonv, latv = np.meshgrid(lon_space, lat_space, indexing='ij')
        coords_grid = np.array([lonv, latv])
            
        lon_gray = []
        lat_gray = []
        for i in range(coords_grid.shape[1]):
            for j in range(coords_grid.shape[1]):
                u, v = coords_grid[:, i, j]
                lon_gray.append(u)
                lat_gray.append(v)
            
        points_gray = [Point(lon, lat) for lon, lat in zip(lon_gray, lat_gray)] 
        collection_points_gray = GeometryCollection(points_gray)
            
        points_blue = [Point(lon, lat).buffer(meter_to_degree(threshold)) for lon, lat in zip(lon_blue, lat_blue)]
        union_points_blue = ops.unary_union(points_blue)
    
        intersection_points = list(union_points_blue.intersection(collection_points_gray).geoms)
        coords_intersection = [list(intersection_points[i].coords)[0] for i in range(len(intersection_points))]
        lon_red, lat_red = [p[0] for p in coords_intersection], [p[1] for p in coords_intersection]

        return lon_red, lat_red, lon_blue, lat_blue, lon_gray, lat_gray
        
    def generate_points(self, x_red, y_red, x_blue, y_blue, x_gray, y_gray):
        """Method to create the points for estimating the object's fractal dimension.
           
           Parameters
           ----------
           list : x_red
              List of x's or lon's coordinates (epsg format: 4326) of the closest found points
              of the points of the object that you want to mediate the fractal dimension.

           list : y_red
              List of y's or lat's coordinates (epsg format: 4326) of the closest found points
              of the points of the object that you want to mediate the fractal dimension.

           list : x_blue
              List of x's or lon's coordinates (epsg format: 4326) of the object's points that are
              wants to mediate the fractal dimension.

           list : y_blue
              List of y's or lat's coordinates (epsg format: 4326) of the object points that
              wants to mediate the fractal dimension.

           list : x_gray
              List of x's or lon's coordinates (epsg format: 4326) of the surrounding daw points
              to the object you want to measure the fractal dimension.

           list : x_gray
              List of y's or lat's coordinates (epsg format: 4326) of the surrounding daw points
              to the object you want to measure the fractal dimension.
            
           Returns
           -------
           str : message
              Message confirming the creation of the points.
           
        """
        global points
        points = []
    
        for i in range(len(x_blue)):
            p = point()
            p.x = x_blue[i]
            p.y = y_blue[i]
            p.type = 'blue'
            points.append(p)
            self.total_points.append(len(x_blue))
        
        for i in range(len(x_red)):
            p = point()
            p.x = x_red[i]
            p.y = y_red[i]
            p.type = 'red'
            points.append(p)
        
        for i in range(len(x_gray)):
            p = point()
            p.x = x_gray[i]
            p.y = y_gray[i]
            p.type = 'gray'
            points.append(p)
            
        return print('Points created!')
            
    def apply_sandbox_on_street_network(self, num_div_rmin = 100, num_div_rmax = 10,
                                        num_intervals = 100, fraction = 2):
        """Method to carry out the steps of the "sandbox" algorithm.
            
           Parameters
           ----------
        
           int : um_div_rmin
              Number of divisions to obtain the minimum radius in relation to the diagonal of the enclosing box.
              (e.g., num_div_rmin = 100 means to choose a minimum radius corresponding to 1% of the size
              the diagonal of the enclosing box).
           
           int : num_div_rmax
              Number of divisions to obtain the minimum radius in relation to the diagonal of the enclosing box.
              (e.g., num_div_rmax = 10 means choose a maximum radius corresponding to 10% of the size
              the diagonal of the enclosing box).
        
           int : num_intervals
              Number of rays that will be generated.
           
           int : fraction
               Fraction of the points obtained that are at a certain distance from the object points (e.g., network)
               in which the rays will be generated and the measurements will be carried out.
            
               Details:
               ----------------------------
               fraction      interpretation
               ----------------------------
               ----------------------------
                 1        |  100% of the
                          |  points
               ----------------------------
                 2        |  50% of the 
                          |  points
               ----------------------------
                 4        |  25% of the 
                          |  points
               ----------------------------
                 6        |  12.5% of the 
                          |  points
               ----------------------------
                 8        |  6.25% of the
                          |  points
               ----------------------------
            
           Returns
           -------
           str : message
              Message confirming the execution of the calculation.
           
        """
        global points
        
        def degree_to_meter(degree):
            return (np.pi*degree*6378137)/180
    
        px_gray = [p.x for p in points if p.type == 'gray']
        py_gray = [p.y for p in points if p.type == 'gray']
    
        xmin, xmax, ymin, ymax = min(px_gray), max(px_gray), min(py_gray), max(py_gray)
    
        diag = ((xmax-xmin)**2+(ymax-ymin)**2)**0.5
        self.diagonal.append(diag)
    
        rmax =  diag/num_div_rmax
        rmin = diag/num_div_rmin
         
        reds = [p for p in points if p.type == 'red']
        
        chosed_reds = list(np.random.choice(reds, int(len(reds)/fraction)))
        p_reds = [(p.x, p.y) for p in chosed_reds]
        self.selected_points.append(p_reds)
    
        radius = list(np.linspace(rmin, rmax, num_intervals))
        
        for red in chosed_reds:
            num_points_in_r = []
            for i in range(len(radius)):
                r = radius[i]
                blues = [p for p in points if p.type == 'blue']
                pe = [p for p in blues if (red.x-p.x)**2+(red.y-p.y)**2 < r**2 and p != red]
                num_points = len(pe)
                num_points_in_r.append(num_points)
            
            self.points_found.append(num_points_in_r)
            self.defined_radios.append(radius)

        return print('Sandbox method successfully applied!')
    
    def get_sandbox_data_from_street_network(self):
        """Method for obtaining data from calculations.
            
           Returns
           -------
           dict : dict_data
              Dictionary with data from calculations.

           
              Details:
              ---------------------------------------------------------------------
                variable              | type     | interpretation
              ---------------------------------------------------------------------
              ---------------------------------------------------------------------
              'r'                     | List     | Radii (meter) of the generated 
                                      | of lists | circles
               --------------------------------------------------------------------
              'N'                     | List     | Number of points
                                      | of lists | found
              ---------------------------------------------------------------------
              'total_points'          | float    | Total number of points
                                      |          | of the object
              ---------------------------------------------------------------------
              'box_size'              | float    | box size length (meter)
              ---------------------------------------------------------------------
              'analized_points'       | List of  | Fraction of points (coordinades)
                                      | tuple    | where the the measurements will
                                      |          | be taken
              ---------------------------------------------------------------------
           
        """
        global points
        
        def degree_to_meter(degree):
            return (2*np.pi*degree*6378137)/360
    
        diagonal = self.diagonal[0]
        L = diagonal/np.sqrt(2)
        radios = [[degree_to_meter(self.defined_radios[j][i]) for i in range(len(self.defined_radios[0]))] for j in range(len(self.defined_radios))]
        
        dict_data = {'r': radios, 'N': self.points_found,
                     'total_points': self.total_points[0], 'box_size': L,
                     'analized_points': self.selected_points[0]}
        return dict_data
    
    def apply_sandbox_on_population(self, multipolygon, populs, x_red, y_red, x_gray, y_gray,
                                    num_div_rmin = 100, num_div_rmax = 10, num_intervals = 100, fraction = 2):
        """Method to calculating the fractal dimension of population.
        
           Parameters
           ----------
            
           shapely.geometry.GeometryCollection : multipolygon
               Collection of geometries of census tracts.
            
           list : populs
               List of population in each census tract.
               
           list : x_red
              List of x's or lon's coordinates (epsg format: 4326) of the closest found points
              of the points of the object that you want to mediate the fractal dimension.

           list : y_red
              List of y's or lat's coordinates (epsg format: 4326) of the closest found points
              of the points of the object that you want to mediate the fractal dimension.

           list : x_gray
              List of x's or lon's coordinates (epsg format: 4326) of the surrounding daw points
              to the object you want to measure the fractal dimension.

           list : x_gray
              List of y's or lat's coordinates (epsg format: 4326) of the surrounding daw points
              to the object you want to measure the fractal dimension.
        
           int : num_div_rmin
              Number of divisions to obtain the minimum radius in relation to the diagonal of the enclosing box.
              (e.g., num_div_rmin = 100 means to choose a minimum radius corresponding to 1% of the size
              the diagonal of the enclosing box). Default num_div_rmin = 100.
           
           int : num_div_rmax
              Number of divisions to obtain the minimum radius in relation to the diagonal of the enclosing box.
              (e.g., num_div_rmax = 10 means choose a maximum radius corresponding to 10% of the size
              the diagonal of the enclosing box). Default num_div_rmax = 10.
        
           int : num_intervals
              Number of rays that will be counted.
              
           int : fraction
               Fraction of the points obtained that are at a certain distance from the object points (e.g., network)
               in which the rays will be generated and the measurements will be carried out. Default fractaion = 2.
            
               Details:
               ----------------------------
               fraction      interpretation
               ----------------------------
               ----------------------------
                 1        |  100% of the
                          |  points
               ----------------------------
                 2        |  50% of the 
                          |  points
               ----------------------------
                 4        |  25% of the 
                          |  points
               ----------------------------
                 6        |  12.5% of the 
                          |  points
               ----------------------------
                 8        |  6.25% of the
                          |  points
               ----------------------------
           
           Returns
           -------
           dict : dict_data_population
              Dictionary with data from calculations.

           
              Details:
              ---------------------------------------------------------------------
                Variable              | Type     | Interpretation
              ---------------------------------------------------------------------
              ---------------------------------------------------------------------
              'r'                     | List     | Radii (meter) of the generated 
                                      | of lists | circles
               --------------------------------------------------------------------
              'N'                     | List     | Number of populations
                                      | of lists | found
              ---------------------------------------------------------------------
              'total_population'      | float    | Total population
              ---------------------------------------------------------------------
              'box_size'              | float    | box size length (meter)
              ---------------------------------------------------------------------
              'analized_points'       | List of  | Fraction of points (coordinades)
                                      | tuple    | where the the measurements will
                                      |          | be taken
              ---------------------------------------------------------------------
        
        """
        
        def get_population(lonlat, multipolygon, populs, diagonal):
            
            def degree_meters(A):
                return (2*np.pi*A*6378137)/360

            str_polygons = []
            for p in multipolygon.geoms:
                str_pol = str(p)
                str_polygons.append(str_pol)

            dict_polygon_population = dict(zip(str_polygons, populs))

            maxradii = diagonal/num_div_rmin
            minradii = diagonal/num_div_rmax

            radius = list(np.linspace(minradii, maxradii, num_intervals))

            population_list = []
            radius_list = []

            for radii in radius:
                circle = Point(lonlat).buffer(radii)
                radius_list.append(degree_meters(radii))

                tree = STRtree(multipolygon.geoms)
                query_geo_list = [o.wkt for o in tree.query(circle)]

                list_wkt_pol = []
                for pol in query_geo_list:
                    wkt_pol = shapely.wkt.loads(pol)
                    list_wkt_pol.append(wkt_pol.buffer(0))


                sum_population = 0
                for i in range(len(list_wkt_pol)):

                    p = list_wkt_pol[i]
                    str_pol = str(p)

                    query_population = dict_polygon_population[str_pol]
                    
                    total_area = p.area
                    
                    query_area = p.difference(p.difference(circle)).area

                    qpop = (query_area/total_area)*query_population
                    sum_population += qpop

                population_list.append(sum_population)

            dict_radius_population = {'r': radius_list, 'N': population_list}

            return dict_radius_population
        
        def degree_meters(A):
            return (2*np.pi*A*6378137)/360
            
        center_points = list(zip(x_red, y_red))      
        dict_points = dict(zip(range(len(center_points)), center_points))
        
        keys = list(dict_points.keys())        
        chosed_keys = list(np.random.choice(keys, int(len(keys)/fraction), replace=False))      
        chosed_points = [dict_points[p] for p in chosed_keys]

        diagonal = ((max(x_gray)-min(x_gray))**2+(max(y_gray)-min(y_gray))**2)**0.5    
        L = degree_meters(diagonal)/np.sqrt(2)

        radius = []
        population_in_radius = []

        for lonlat in chosed_points:
            data = get_population(lonlat, multipolygon, populs, diagonal)
            radius.append(data['r'])
            population_in_radius.append(data['N'])
            
        
        dict_data_population = {'r': radius, 'N': population_in_radius, 
                                'total_population': sum(populs),'box_size': L,
                                'analized_points': chosed_points}
        
        return dict_data_population
    
    def street_network_generalized_dimension(self, data, moment_list):
        """Method to calculate the generalized fractal dimension based on "sandbox" strategy
           of the street network.
        
           Parameters
           ----------
           
           dict : data
               The dictionary with data generated by the "get_sandbox_data_from_street_network".
               
           list : moment_list
               The list of order moments (e.g, a linear space betweenn [-q, q], when q 
               is a real number).
               
           Returns
           -------
           dict : dict_gen_dim
              Dictionary with data generated.
              
              
              Details:
              --------------------------------------------------------------------------------------
              variable                            |    type    |     description
              --------------------------------------------------------------------------------------
              --------------------------------------------------------------------------------------
              'gen_dim_street_network'            |    list    |  generalized fractal dimensions
              --------------------------------------------------------------------------------------
              'gen_dim_street_network_std'        |    list    |  standard error of the generalized
                                                  |            |  fractal dimensions
              --------------------------------------------------------------------------------------
              'gen_dim_street_network_intercept'  |    list    |  intercept of the generalized 
                                                  |            |  fractal dimensions
              --------------------------------------------------------------------------------------
              'gen_dim_street_network_r2'         |    list    |  R2 of the generalized 
                                                  |            |  fractal dimensions
              --------------------------------------------------------------------------------------
        
        """
        
        def get_average(data, q):
    
            if round(q,1) == 1.0:
                average = [np.mean([data['N'][i][j]/data['total_points'] for i in range(len(data['N']))]) for j in range(len(data['N'][0]))]
                return average
            else:
                average = [np.mean([(data['N'][i][j]/data['total_points'])**(q-1) for i in range(len(data['N']))]) for j in range(len(data['N'][0]))]
                return average
            
        def fit_average(x, y, q):

            def power_law(x, a, b):
                return b*x**a

            tinv = lambda p, dff: abs(t.ppf(p/2, dff))

            u, v = x, y

            if round(q,1) == 1.0:

                res = stats.linregress(np.log(u), np.log(v))
                ts = tinv(0.05, len(u)-2)
                a, a_std, b, r2 = res.slope, ts*res.stderr, res.intercept, res.rvalue**2

                return a, a_std, b, r2
            else:
                res = stats.linregress(np.log(u), np.log(v)/(q-1))
                ts = tinv(0.05, len(u)-2)
                a, a_std, b, r2 = res.slope, ts*res.stderr, res.intercept, res.rvalue**2

                return a, a_std, b, r2    
        
        dq_list = []
        dq_std_list = []
        dq_intercep_list = []
        r2_list = []

        for q in moment_list:
            x = list(np.array(data['r'][0])/data['box_size'])
            y = get_average(data, q)
            d, d_std, b, r2 = fit_average(x, y, q)

            dq_list.append(d)
            dq_std_list.append(d_std)
            dq_intercep_list.append(b)
            r2_list.append(r2)
            
        dict_gen_dim = {'gen_dim_street_network' : dq_list,
                        'gen_dim_street_network_std' : dq_std_list,
                        'gen_dim_street_network_intercept' : dq_intercep_list,
                        'gen_dim_street_network_r2' : r2_list} 
            
        return dict_gen_dim
    
    def street_network_mass_exponent(self, dq_list, moment_list):
        """Method for calculating mass exponents. The exponents control
           how the probability moment orders, q, scale with the side size of the covering balls.
        
           Parameters
           ----------
           list : dq_list
               List of the generalized dimensions of the street network.
           
           list : moment_list
               List of orders moment (e.g, a linear space betweenn [-q, q], when q 
               is a real number).
               
           Returns
           -------
           list : taus
              List of mass expontents of the street network.
               
        """
        
        tau_list = [(1-q)*d for q, d in zip(moment_list, dq_list)]
        
        return tau_list
    
    def street_network_multifractal_spectrum(self, tau_list, moment_list):
        """Method to calculating multifractal spectrum. "alpha" are the Lipschitz-Hölder exponents
           and f(alpha)" are the multifractal spectra.
           
           Parameters
           ----------
           list : tau_list
               List of the mass exponents of the street network.
           
           list : moment_list
               List of orders moment (e.g, a linear space betweenn [-q, q], when q 
               is a real number).
              
           Returns
           -------
           tuple (list, list) : alpha_list, falpha_list
                'alpha_list' is the list of Lipschitz-Hölder exponents and 'falpha_list'
                is the list of multifractal spectra of the street network.
           
        """
        
        alpha_list = list(-np.diff(tau_list)/np.diff(moment_list))
        falpha_list = [t+q*a for t, q, a in zip(tau_list, moment_list, alpha_list)]
        
        return alpha_list, falpha_list
    
    def population_generalized_dimension(self, data, moment_list):
        """Method to calculate the generalized fractal dimension based on "sandbox" strategy
           of the population.
        
           Parameters
           ----------
           
           dict : data
               The dictionary with data generated by the "apply_sandbox_on_population".
               
           list : moment_list
               The list of order moments (e.g, a linear space betweenn [-q, q], when q 
               is a real number).
               
           Returns
           -------
           dict : dict_gen_dim
              Dictionary with data generated.
              
              
              Details:
              --------------------------------------------------------------------------------------
              variable                        |    type    |     description
              --------------------------------------------------------------------------------------
              --------------------------------------------------------------------------------------
              'gen_dim_population'            |    list    |  generalized fractal dimensions
              --------------------------------------------------------------------------------------
              'gen_dim_population_std'        |    list    |  standard error of the generalized
                                              |            |  fractal dimensions
              --------------------------------------------------------------------------------------
              'gen_dim_population_intercept'  |    list    |  intercept of the generalized 
                                              |            |  fractal dimensions
              --------------------------------------------------------------------------------------
              'gen_dim_population_r2'         |    list    |  R2 of the generalized 
                                              |            |  fractal dimensions
              --------------------------------------------------------------------------------------
        
        """
        
        def get_average(data, q):
    
            if round(q,1) == 1.0:
                average = [np.mean([data['N'][i][j]/data['total_population'] for i in range(len(data['N']))]) for j in range(len(data['N'][0]))]
                return average
            else:
                average = [np.mean([(data['N'][i][j]/data['total_population'])**(q-1) for i in range(len(data['N']))]) for j in range(len(data['N'][0]))]
                return average
            
        def fit_average(x, y, q):

            def power_law(x, a, b):
                return b*x**a

            tinv = lambda p, dff: abs(t.ppf(p/2, dff))

            u, v = x, y

            if round(q,1) == 1.0:

                res = stats.linregress(np.log(u), np.log(v))
                ts = tinv(0.05, len(u)-2)
                a, a_std, b, r2 = res.slope, ts*res.stderr, res.intercept, res.rvalue**2

                return a, a_std, b, r2
            else:
                res = stats.linregress(np.log(u), np.log(v)/(q-1))
                ts = tinv(0.05, len(u)-2)
                a, a_std, b, r2 = res.slope, ts*res.stderr, res.intercept, res.rvalue**2

                return a, a_std, b, r2    
        
        dq_list = []
        dq_std_list = []
        dq_intercep_list = []
        r2_list = []

        for q in moment_list:
            x = list(np.array(data['r'][0])/data['box_size'])
            y = get_average(data, q)
            d, d_std, b, r2 = fit_average(x, y, q)

            dq_list.append(d)
            dq_std_list.append(d_std)
            dq_intercep_list.append(b)
            r2_list.append(r2)
            
        dict_gen_dim = {'gen_dim_population' : dq_list,
                        'gen_dim_population_std' : dq_std_list,
                        'gen_dim_population_intercept' : dq_intercep_list,
                        'gen_dim_population_r2' : r2_list} 
            
        return dict_gen_dim
    
    def population_mass_exponent(self, dq_list, moment_list):
        """Method for calculating mass exponents of the population. The exponents control
           how the probability moment orders, q, scale with the side size of the covering balls.
        
           Parameters
           ----------
           list : dq_list
               List of the generalized dimensions of the population.
           
           list : moment_list
               List of orders moment (e.g, a linear space betweenn [-q, q], when q 
               is a real number).
               
           Returns
           -------
           list : taus
              List of mass expontents of the population.
               
        """
        
        taus_list = [(1-q)*d for q, d in zip(moment_list, dq_list)]
        
        return taus_list
    
    def population_multifractal_spectrum(self, taus_list, moment_list):
        """Method to calculating multifractal spectrum. "alpha" are the Lipschitz-Hölder exponents
           and f(alpha)" are the multifractal spectra.
           
           Parameters
           ----------
           list : tau_list
               List of the mass exponents of the population.
           
           list : moment_list
               List of orders moment (e.g, a linear space betweenn [-q, q], when q 
               is a real number).
              
           Returns
           -------
           tuple (list, list) : alpha_list, falpha_list
                'alpha_list' is the list of Lipschitz-Hölder exponents and 'falpha_list'
                is the list of multifractal spectra of the population.
           
        """
        
        alphas_list = list(-np.diff(taus_list)/np.diff(moment_list))
        falphas_list = [t+q*a for t, q, a in zip(taus_list, moment_list, alphas_list)]
        
        return alphas_list, falphas_list

class PlotStreetPopulation:
    """Class to plot the street network and population in each census track.
    
       Atrributes
       ----------
       list : geo
           List of the geometries of each census track.
       
       list : popul
           List of the population of each census track.
           
       pandas.core.frame.DataFrame : df_nodes
           Data frame with the nodes information.
       
       pandas.core.frame.DataFrame : df_links
          Data frame with the links information.
          
       Method
       ------
       plot_geo_population_network(kwargs)
          Plot of the street network and population.
       
           
       
    """
    
    def __init__(self, geo, popul, df_nodes, df_links):
        self.geo = geo
        self.popul = popul
        self.df_nodes = df_nodes
        self.df_links = df_links
        
    
    def plot_geo_population_network(self, kwargs):
        """Method to plot the street network and population.
        
           Parameters
           ----------
           dict :  kwargs
           
           Returns
           -------
           plt.image
              Image of the street network and population.
           
              Details:
              -----------------------------------------------------------
              variable       | type   |    exemple
              -----------------------------------------------------------
              -----------------------------------------------------------
              'figsize'      | tuple  |   (10, 10)
              -----------------------------------------------------------
              'cmap'         | str    |   'viridis'
              -----------------------------------------------------------
              'legend'       | bool   |   True
              -----------------------------------------------------------
              'legend_kwds'  | dict   |   {"label": "Population in 2010",
                             |        |    "orientation": "horizontal"}
              -----------------------------------------------------------
              'node_color'   | str    |   'white'
              -----------------------------------------------------------
              'markersize'   | int    |   1
              -----------------------------------------------------------
              'edge_color'   | str    |   'white'
              -----------------------------------------------------------
              'linewidth'    | int    |   1
              -----------------------------------------------------------
              'xlabel'       | str    |   'lon'
              -----------------------------------------------------------
              'ylabel'       | str    |   'lat'
              -----------------------------------------------------------
              'xlim'         | tuple  |   (float, float)
              -----------------------------------------------------------
              'ylim'         | tuple  |   (float, float)
              -----------------------------------------------------------
        """
        
        gpdf_popul = gpd.GeoDataFrame({'population': self.popul, 'geometry': list(self.geo.geoms)}, crs='epsg:4326')
        gpdf_nodes = gpd.GeoDataFrame(self.df_nodes, geometry = [shapely.wkt.loads(g) for g in self.df_nodes.geometry], crs='epsg:4326')
        gpdf_links = gpd.GeoDataFrame(self.df_links, geometry = [shapely.wkt.loads(g) for g in self.df_links.geometry], crs='epsg:4326')

        fig, ax = plt.subplots(figsize=kwargs['figsize'])
        
        gpdf_popul.plot(ax = ax, column='population', cmap = kwargs['cmap'], legend=kwargs['legend'], legend_kwds=kwargs['legend_kwds'])
        gpdf_nodes.plot(ax = ax, color=kwargs['node_color'], markersize=kwargs['markersize'])
        gpdf_links.plot(ax = ax, color=kwargs['edge_color'], linewidth=kwargs['linewidth'])
        
        ax.set_xlabel(kwargs['xlabel'])
        ax.set_ylabel(kwargs['ylabel'])
        ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
        ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])
        
        return plt.show()
        

