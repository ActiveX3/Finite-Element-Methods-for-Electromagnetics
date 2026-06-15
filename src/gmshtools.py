import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import subprocess
import gmsh
import sys
import colorsys


##########################################
#
#
# Hilfsmittel für den Umgang mit Gmsh, by Jürgen Weizenecker
#
# Defintion Klasse und zusätzliche Funktionen
#    - NormalDerivative
#    - ComputeGradient
#
##########################################


class ElementMsh:
    '''
       Zusatzinfos von Objekten:
       -- elemente
       -- gegebnefalls Knoten
       -- gegebenfalls Verknüpfung zwischen entities
       -- Elemente können geplottet werden
          plot(mesh,ax=None, color='k', alpha=0.7, node_label=False):
       -- Aus Elementen können durch eine Auswahl mittels einer Funktion neue Elemente gewählt werden
          def select(Attributname, Auswahlfunktion):
       -- Info wird angezeigt
    '''
    def __init__(self, parent, tt, name, node=False, connect=None):
        # Knotenkoordinaten sind nicht öffentlich, werden aber hier gebraucht zum plotten
        self.elements = tt
        self.info = name
        self.parent = parent
        if node:
            self.nodes = np.unique( self.elements.flatten() )
        # Bei Linienelemente Zuordnung zu den 2D-Elementen ermöglichen
        if connect is not None:
            if tt.shape[1] == 2:
                self.connect = [ connect[tuple(np.sort(t).astype(int))] for t in self.elements ]
        
                
    # methode plotten    
    def plot(self, ax=None, newaxis=False, color='grey', alpha=0.3, node_label=False, direction=False, *args, **kwargs):
        '''
        Plotte die Elemente mit matplotlib zum aktuellen Attribut
        Direction ist im Moment nur für Linien möglich
        '''
        # eingeführt, da es in jupyter dazu kam, dass plt.figures in aktuelle bx figures geplottet wurden.
        # Hatte auch was mit plt.show() zu tun.
        if newaxis == True:
            self.parent.ax = None
        
        if self.parent.ax is None and ax is None:
            fig = plt.figure()
            
            if self.parent.dim == 3:
                self.parent.ax = fig.add_subplot(111, projection="3d")
                self.parent.ax.set_xlabel("x")
                self.parent.ax.set_ylabel("y")                
                self.parent.ax.set_zlabel("z")
                self.parent.ax.set_box_aspect((1, 1, 1))
            else:
                self.parent.ax = fig.add_subplot(111)
                self.parent.ax.set_xlabel("x")
                self.parent.ax.set_ylabel("y")
                
                    
        if ax is not None:
            # Prüfen, ob ax eine 3D-Achse ist
            if self.parent.dim == 3 and not hasattr(ax, "zaxis"):
                raise ValueError("Übergebenes ax ist keine 3D-System.")
            bx = ax
        else:
            if self.parent.ax.figure.number not in plt.get_fignums():
                print("Unerwartetes Verhalten")
            bx = self.parent.ax
        
        
        verts = [self.parent.points[t] for t in self.elements]
        if self.elements.shape[1] ==2:
            linw = 2.2
        else:
            linw =1.0
        
        # Dreiecke, Linien, Tetraeder usw. malen    
        if self.elements.shape[1] >1:    
            if self.parent.dim == 3:    
                poly = Poly3DCollection(verts, alpha=alpha, linewidths=linw, *args, **kwargs)
                poly.set_edgecolor(color)
                bx.add_collection3d(poly)
            else:
                patches_list = []
                for v in verts:
                    xy = v[:, :2]  # nur x,y
                    patches_list.append(patches.Polygon(xy, closed=True))
                
                poly = PatchCollection(patches_list, facecolor=lighten(color, 0.1), edgecolor=color, alpha=alpha, linewidths=linw, *args, **kwargs)
                bx.add_collection(poly)
                bx.autoscale()
        
            # Male Pfeile, TODO
            if self.elements.shape[1] == 2 and direction == True:
                mitte = np.sum(self.parent.points[self.elements, :], axis=1) /2
                vec = 0.2 * ( self.parent.points[self.elements[:, 1], :] - self.parent.points[self.elements[:, 0], :] )
                vec_normal = np.column_stack([vec[:, 1], -vec[:, 0], vec[:, 2]])
                p3 = mitte - vec + vec_normal
                p2 = mitte - vec - vec_normal
                p1 = mitte + vec
            
            
                if self.parent.dim == 3:
                    segments = np.concatenate([ np.stack([p2, p1], axis=1), np.stack([p1, p3], axis=1) ], axis=0)
                    lc = Line3DCollection(segments, colors="black", linewidths=1, *args, **kwargs)
                    bx.add_collection3d(lc)
                else:
                    segments_2d = np.concatenate([np.stack([p2[:,:2], p1[:,:2]], axis=1), np.stack([p1[:,:2], p3[:, :2]], axis=1)], axis=0)
                    lc = LineCollection(segments_2d, colors="black", linewidths=1, *args, **kwargs)
                    bx.add_collection(lc)
                
        # punkte malen        
        if self.elements.shape[1] == 1:
            verts = self.parent.points[self.elements.flatten()]
            if self.parent.dim == 2:
                bx.plot(verts[:, 0], verts[:, 1], 'o', color=color, *args, **kwargs)
            elif self.parent.dim == 3:
                bx.scatter(verts[:, 0], verts[:, 1], verts[:, 2], 'o', c=color, *args, **kwargs)            
                
                       
        
        if node_label:
            i_node = np.unique( self.elements.flatten() )
            for i in i_node:
                bx.text(*self.parent.points[i, 0:self.parent.dim], str(i), color="black", fontsize=9, fontweight='bold')
                
        return bx        
    
                    
    # Nachträglich noch objekte aus bekannten auswählen    
    def select(self, new_name, fkt):
        '''
        Aus den Elementen des aktuellen Attributes können Teilmengen selektiert werden
        hierzu wird eine Funktion definiert, z.B. lambda x: (x[:,0]<-1) | (x[:,0]>1)
        
        '''       
        num = self.elements.shape[1]
        mitte = np.sum(self.parent.points[self.elements, :], axis=1) / num
        new_elements = self.elements[fkt(mitte)]
        
        # Ursprungs-Attributnamen dynamisch im parent suchen ---
        org_attr = "Unbekannt"
        for attr_name, attr_wert in self.parent.__dict__.items():
            if attr_wert is self:
                org_attr = attr_name
                break        
        info_text = "Selektion aus %s" %(org_attr)
        
        if num == 2:
            setattr(self.parent, new_name, ElementMsh(self.parent, new_elements, info_text, node=True, connect=self.parent.edge_tab) )
        else:
            setattr(self.parent, new_name, ElementMsh(self.parent, new_elements, info_text) )
        
            
    def orientation(self, flip=None):
        
        '''
        Drehe Laufrichtung der Elemente um.
        Für Randelemente können auch diejenigen gedreht werden, die links/rechts orientiert sind.
        flip= 'all'  : flippe alle segemente
        flip='left'  : flippe nur die mit linkem Nachbar 
        flip='right' : flippe nur die mit rechtem Nachbar
        '''
        
        
        
        
        # Ursprungs-Attributnamen dynamisch im parent suchen ---
        org_attr = "Unbekannt"
        for attr_name, attr_wert in self.parent.__dict__.items():
            if attr_wert is self:
                org_attr = attr_name
                break        
        
        # finde Lage der connecting elements in Bezug auf den Umlaufsinn
        number_neighbors, l_idx, left_e, r_idx, right_e = FindLeftRight(org_attr, self.elements, self.connect)        
        
        if flip == 'all':
            new_elements = self.elements[::-1, ::-1]
            self.elements = new_elements
            if hasattr(self,'connect'):
                self.connect = [ self.parent.edge_tab[tuple(np.sort(t).astype(int))] for t in self.elements ]
            print("Richtungssinn von %s bei allen Segmenten umgedreht"%(org_attr) )    
        
        elif flip =='left':
            flip_idx = l_idx[number_neighbors[l_idx] == 1]
            self.elements[flip_idx] = self.elements[flip_idx,::-1]
            if hasattr(self,'connect'):
                self.connect = [ self.parent.edge_tab[tuple(np.sort(t).astype(int))] for t in self.elements ]
            print("Randkurve %s : Richtungssinn der Left-Elements umgedreht"%(org_attr) )    
        
        elif flip =='right':
            flip_idx = r_idx[number_neighbors[r_idx] == 1]
            self.elements[flip_idx] = self.elements[flip_idx,::-1]
            if hasattr(self,'connect'):
                self.connect = [ self.parent.edge_tab[tuple(np.sort(t).astype(int))] for t in self.elements ]
            print("Randkurve %s : Richtungssinn der Right-Elements umgedreht"%(org_attr) )
        elif flip is not None:
            print(" flip = all/left/right oder None")
            
        
                        
        
        
             
       
                
    
        
        

class MshHs:
    '''
        Erzeuge aus einem Model oder aus Listen verschiedene Objekte:
        
        -- Triangle, Quadritangle, ......
        -- points
        -- Vollständige Kantenliste (nur 2D)
        -- vordefinierte Objekte (über dict, oder physik. Gruppen)
        -- Achsenparameter für plot
        -- plot dimension
        
        Die geometrischen Objekte (Triangle,...) haben selbst wieder Attribute
             - elements
             - connect
             - info
             - node
        Sie enthalten auch Methoden wie
             - plot     --> .plot(mesh,ax=None, newaxis=False, color='k', alpha=0.7, node_label=False)
             - select   --> .select(newname,auswahlfunktion)    bsp.: lambda x: (x[:,0]<-1) | (x[:,0]>1)
             - orientation  --> .orientation(self, flip=None or 'all' or right' or 'left')   gibt auch info über die orientierung  
    '''
    def __init__(self, mod, points=[], triangles=[], segments=[]):
        
        self.ax = None
        self.dim = 3
        #gmsh file
        if mod is not None:
            # Finde Knoten
            nodeTags, coords, parametricCoord = mod.mesh.getNodes()
            self.points = np.array(coords).reshape(-1, 3)
            
            
            
            # finde Elemente der dimensionen 1,2,3 
            # tp info über knoten im element: z.B. 2 heisst drei Knoten bei einem Dreieck, 3 heisst 4 Knoten bei einem Viereck , 
            # nm Element id
            # el Knotennummern fortlaufend
            flag = ''
            for dim in [3, 2, 1]:
                # mit getElement werden nur Elementen zu Objekten erzeugt, die 
                # geometrisch definiert wurden
                tp,nm,el = mod.mesh.getElements(dim, -1)
                for i in range(len(tp)):
                    prop = mod.mesh.getElementProperties(tp[i])
                    eig = "%s"%(prop[0].split()[0])
                    elements = el[i].reshape(-1,prop[3])-1
                    #setattr(self,eig, elements )
                    if dim == 3:
                        flag ='3d'
                    if dim ==2 and flag == '':
                        flag = '2d'    
                    comment = "Element : " + prop[0] + " mit %i Knoten pro Element"%(prop[3])
                    if dim ==1 and flag == '3d':
                        comment += "\n da eindimensionale geom. Objekte in 3d Umgebung explizit oder implizit definiert wurden"
                    if dim ==1 and flag == '2d':
                        comment += "\n da eindimensionale geom. Objekte in 2d Umgebung explizit oder implizit definiert wurden"
                    if dim ==2 and flag == '3d':
                        comment += "\n da zweidimensionale geom. Objekte in 3d Umgebung explizit oder implizit definiert wurden"                    
                    setattr(self,eig, ElementMsh(self, elements, comment) )
        
        # meshtools oder handgemacht
        else:
            self.points = np.column_stack([points, len(points)*[0.0]])
            setattr(self,"Triangle", ElementMsh(self, triangles, "Triangle Elements 3 points") )
            
        
        #Laufe nun durch alle Elemente (2D,3D) und erstelle eine Kanten-Tabelle
        # (.,.) : [[],[],[]]
        # TODO trinagle_tab
        edge_tab = {}
        ElementTyp = ['Triangle#3', 'Quadriliteral#4', 'Tetrahedron#4', 'Hexaherdron#8', 'Prism#6', 'Pyramid#5']
        for elty in ElementTyp:
            # element
            attr = getattr(self, elty[:-2], None)
            if attr is not None:
                elements = attr.elements
                #Anzahl Knoten
                nn = int(elty[-1])
                # erstelle [[0,1],[1,2],[2,3],..[]]
                ii = np.arange(nn); jj = (ii + 1) % nn           
                idx = np.array( np.column_stack([ii, jj]))
                # erstelle alle kanten
                edges = np.sort(elements[:, idx], axis=2).reshape(-1, 2)
                # erstelle dict mit kanten und zugeordneten elementen
                [edge_tab.setdefault((int(ee[0]), int(ee[1])), []).append(elements[i//nn].astype(int).tolist()) for i, ee in enumerate(edges)]
                setattr(self,"edge_tab", edge_tab)
            
        
        ######### Zusätzliche Objekte ################
        if mod is not None:    
            # und die definierten phys. Gruppen. Die Gruppen sind typischerweise geo. Objekte (Linien, Flächen, Volumina)
            # nützlich für selbst definierte Kurven/Ränder, innere Linien
            groups = mod.getPhysicalGroups()
            for dim, tag in groups:
                name = mod.getPhysicalName(dim, tag)
                if name.startswith('_'):
                    continue          # interne Namen überspringen
                if name in ('__dict__', '__class__'):
                    continue          # Python-Interna nicht überschreiben
                rr = mod.getEntitiesForPhysicalName(name)
                rr_e = []
                for d,l in rr:
                    el_ty, el_tag, node_tag = mod.mesh.getElements(d,l)
                    rr_e.append( node_tag[0].reshape(-1,dim+1) -1 )
                rr_e = np.concatenate(rr_e, axis=0)
                # Elemente
                #setattr(self, name, rr_e)
                con = None
                if dim == 1:
                    con = self.edge_tab
                setattr(self,name, ElementMsh(self, rr_e, "Defined by a Physical Group", node=True, connect=con) )
                
        # Ränder         
        else:
            for key in segments:
                setattr(self,key, ElementMsh(self, np.array(segments[key]), "Defined by dict", node=True, connect=self.edge_tab) ) 


            

#
# Ermittle die relative Lage von angrenzenden Dreieckselementen
#
def FindLeftRight(name, segments, neighbor_elements):
    '''
    returns number_neighbors,left_idx, left_elements, right_idx, right_elements
    
    left_idx        contains the indices of the segments having left elements
    left_elements   elements lying on the left side   left_elements[k] refers to segments[left_idx[k]]
    
    same for right
    
    '''
    sh = segments.shape
    if sh[1] != 2:
        print("Lines only for the Moment")
        return
    
    if not np.array_equal(segments[1:, 0], segments[:-1, 1]):
        print("Warning: The single segments of the curve ", name, " are not in consecutive order")
    else:
        print("The single segments of the curve ", name, " are in consecutive order")    
    
    # Liste mit rechten und linken nachbarn
    right_elements = []
    left_elements = []
    left_idx = []
    right_idx = []
    number_neighbors = []
    # Laufe durch alle Dreieckselemente
    for i,xx in enumerate(neighbor_elements):
        # es können maximal 2 Dreiecke verknüpft sein, eins ist aber auch möglich
        number_neighbors += [ len(xx) ]
        for yy in xx:
            # beginne das Dreieckselement immer mit dem Knoten 0 aus seg
            tri = yy + yy[0:2]
            j = tri.index(segments[i, 0])
            tri = tri[j:j+3]
            # Zuordnung rechts links (zweite knotennummer im element ist zweiter knoten im seg) => Dreieck liegt in laufrichtung links
            if tri[1] == segments[i, 1]:
                left_elements += [tri]
                left_idx += [ i ]
            else:
                right_elements += [tri]
                right_idx += [ i ]
    
    
    if np.sum(number_neighbors) == sh[0]:
        if left_elements == []:
            print(name, ": Nur Randsegmente, Gebiet liegt rechts")
        elif right_elements ==[]:
            print(name, ": Nur Randsegmente, Gebiet liegt links")
        else:
            print(name, ": Nur Randsegmente, teilweise liegt das Gebiet aber links, teilweise rechts")
    elif np.sum(number_neighbors) == 2 *sh[0]:
        print(name, ": Segmente liegen innerhalb des Gebietes, Gebiet liegt links und rechts")
    else:
        print(name, ": Segmente sind teilweise am Rand, teilweise innen liegend")    
        
    return np.array(number_neighbors), np.array(left_idx), np.array(left_elements), np.array(right_idx), np.array(right_elements)
    
    

def NormalDerivative(name, mesh, sol, normal_direction='right'):
    '''
    Berechne die Normalenableitung, nur für Linien
    
       name  : Attribut der Line in mesh
       right : Normalenvektor zeigt nach rechts beim Durchlaufen der Kurve, also rechtsseitige Normalenableitung
       sol   : Lösung
       mesh  : Netz (Klasse MshHs)
       idx enthält die indizes der segmente, welche die normal_direction haben
       
       output:
       running length, normal derivative, idx, gradient, normal vector
       
    '''
    
    curve = getattr(mesh,name, None)
    if curve == None:
        print("Curve not found, check curve name")
        return
    seg = curve.elements
    
    if seg.shape[1] != 2:
        print("For the moment only line elements with 2 nodes are possible")
        return
    
    
    # Finde linke und rechte elemente der segmente
    number_neighbors, l_idx, left_e, r_idx, right_e = FindLeftRight(name, curve.elements, curve.connect)
    
    if normal_direction == 'left':
        idx = l_idx
        n = left_e
        dir_fac = -1
    else:
        idx = r_idx
        n = right_e
        dir_fac = 1
    
    if len(n) == 0:
        print("No %s elements found, check parameter normal_direction"%(normal_direction) )
        return None
    
    
        
    # Berechne Normalenableitung
    p = mesh.points
    # berechne alle bs und cs (shape: number of elements x 3)
    bs = p[n[:, [1, 2, 0]], 1] - p[n[:, [2, 0, 1]], 1]
    cs = p[n[:, [2, 0, 1]], 0] - p[n[:, [1, 2, 0]], 0]
    # alle Lösungen aus den elementen
    u = sol[n]
    twodelta = bs[:, 0]*cs[:, 1]-cs[:, 0]*bs[:, 1]
    # die ableitung in jedem element
    dudx = np.sum(bs * u, axis=1) / twodelta
    dudy = np.sum(cs * u, axis=1) / twodelta
    # der Vektor in Laufrichtung durch die Segmente
    vec = p[seg[idx, 1], :] - p[seg[idx, 0], :]
    vec_abs = np.sqrt(np.sum(vec**2, axis=1))
    # Normalenvektor vec x e_z, liegt also in Laufrichtung rechts
    normal = np.column_stack([vec[:, 1] / vec_abs * dir_fac, -vec[:, 0] / vec_abs * dir_fac])
    
    # Gradient
    grad_u = np.column_stack([dudx,dudy])
    normal_derivative = np.sum( grad_u*normal, axis=1)    
    
    # running length
    Ls = np.sqrt(np.sum(vec**2 , axis=1))
    # oder:
    # 0, L0/2+L1/2, L0/2+L1+L2/2, L0/2+L1+L2+L3/2, L0/2+L1+L2+L3+L4/2, ....
    #rl = np.cumsum([0, 0] + list(Ls[1:-1]))
    #rl[1:] += Ls[0] / 2
    #rl[1:] += Ls[1:] / 2
    rl = np.cumsum(Ls) - Ls[0]
    
    if len(idx) != len(seg):
        print("Warning: Not all segment elements were used due to orientation ( %i / %i )"%(len(idx), len(seg)) )
    
    return rl, normal_derivative, idx, normal, grad_u 
    



# Berechne den Gradienten für ein Netz
def ComputeGradient(p,t,u,location='middle',num=10,p_loc=[]):
    """
    Compute the Gradient of a triangular mesh

    Input:   p    array([[x1,y1],[x2,y2],...])          node points
             t    array([[n1,n2,n3],[n4,n5,n6],...])    elements
             u    array([u1,u2,u3,.....])               function at node values
             location                                   'middle', 'nodes','grid','set' 
                                                        middle: compute at middle points of triangles:
                                                        nodes: compute at node points, as mean value 
             p_loc  array([[X1,Y1],[X2,Y2],...])        points for gradient evaluation (location must be 'set')
             num  N                                     generate NxN points array for p_loc (location must be 'grid' )


    Output:  x     x-component of point
             y     y-component of point
             g_x   gradient, x-component at (x,y)
             g_y   gradient, y-component at (x,y)
    """

    if location=='grid':
        eps=1e-6
        h1=np.linspace(min(p[:,0])+eps,max(p[:,0])-eps,num)
        h2=np.linspace(min(p[:,1])+eps,max(p[:,1])-eps,num)
        h1,h2=np.meshgrid(h1,h2)
        h1.resize(num*num,1)
        h2.resize(num*num,1)
        points=np.append(h1,h2,axis=1)
    elif location=='set':
        if len(p_loc)==0:
            print("Error: p_loc is empty")
            return [],[],[],[]  
        points=p_loc
    else:
        points=np.sum(p[t],axis=1)/3. 




    # Compute all a,b,c
    a1=p[t[:,1],0]*p[t[:,2],1]-p[t[:,1],1]*p[t[:,2],0]
    a2=p[t[:,2],0]*p[t[:,0],1]-p[t[:,2],1]*p[t[:,0],0]
    b1=p[t[:,1],1]-p[t[:,2],1]
    b2=p[t[:,2],1]-p[t[:,0],1]
    c1=p[t[:,2],0]-p[t[:,1],0]
    c2=p[t[:,0],0]-p[t[:,2],0]    

    delta=0.5*(b1*c2-b2*c1);

    XYUV=np.array([])
    ii=0
    for x in points:
        x=np.array(x)
        ksi=0.5/delta*(a1+b1*x[0]+c1*x[1])
        eta=0.5/delta*(a2+b2*x[0]+c2*x[1])

        element=np.where( (ksi>=0) & (eta>=0) & (eta+ksi-1<=2e-13) )[0]

        if len(element)>0:
            element=element[0]

            bb1=b1[element]
            bb2=b2[element]
            bb3=p[t[element,0],1]-p[t[element,1],1]

            cc1=c1[element]
            cc2=c2[element]
            cc3=p[t[element,1],0]-p[t[element,0],0]

            dd=delta[element]

            u1=u[t[element,0]]
            u2=u[t[element,1]]
            u3=u[t[element,2]]

            gx=0.5/dd*(bb1*u1+bb2*u2+bb3*u3)
            gy=0.5/dd*(cc1*u1+cc2*u2+cc3*u3)

            help=np.append(x,np.array([gx,gy]))
            XYUV=np.append(XYUV,help)
        else:
            ii+=1
    #print(ii,"points not mapped to triangles")   
    XYUV.resize(len(XYUV)//4,4)  

    #
    if location=='nodes':
        xx=p[:,0]
        yy=p[:,1]
        nn=len(xx)
        gxx=np.zeros(nn)
        gyy=np.zeros(nn)
        counter=np.zeros(nn)
        for i,n in enumerate(t):
            gxx[n] += XYUV[i,2]
            gyy[n] += XYUV[i,3]
            counter[n] += 1
        gxx/=counter
        gyy/=counter
    else:    
        xx=XYUV[:,0]
        yy=XYUV[:,1]
        gxx=XYUV[:,2]
        gyy=XYUV[:,3]

    return xx,yy,gxx,gyy

    
# Aufhellung Flächenfarbe im Vergleich zu Kantenfarbe
def lighten(color, amount=0.5):
    r, g, b = to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = 1 - amount * (1 - l)   # Richtung Weiß
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2)




    

if __name__ == '__main__':
    

    
    
    
    # einfaches eigenes Netz und eine "Kurve" dazu
    p = np.array([[ 1, 0.7 ] , [ 0.5, 0.35 ] , [ 0.5, 0.18 ] , [ 0, 0 ] , [ 0, 0.7 ] , [ 1, 0 ]])
    t = np.array([[ 0, 4, 1 ] , [ 3, 1, 4 ] , [ 5, 0, 1 ] , [ 3, 5, 2 ] , [ 2, 1, 3 ] , [ 5, 1, 2 ]], dtype=np.int64)
    rand_seg = np.array([[ 3, 5 ] , [ 0, 4 ] , [ 4, 3 ],  [ 5, 0 ] ], dtype=np.int64)
    
    # punkte sind unter                            netz.nodes
    # Die Elemente unter                           netz.Triangle.elements
    # rand_seg unter dem key des dicts zu finden   netz.Rand.elements
    netz = MshHs(None, points=p, triangles=t, segments={'Rand': rand_seg})
    netz.Triangle.plot(color='blue', alpha=0.3, node_label=True)
    #netz.Rand.plot(color='red', alpha=0.8)
    netz.Rand.plot(color='red', alpha=0.8)
    plt.show()
    
    #############################################################
    # Erzeuge das Netz mit gmsh, sehr elementar nur mit Linien
    #
    #############################################################
    
    gmsh.initialize()
    name = 'TestGroups' 
    gmsh.model.add(name)
    
    #Äusseres Rechteck
    P1=(-1,-1,0); P2=(4,-1,0); P3=(4,4,0); P4=(-1,4,0)
    i1=gmsh.model.occ.addPoint(*P1)
    i2=gmsh.model.occ.addPoint(*P2)
    i3=gmsh.model.occ.addPoint(*P3)
    i4=gmsh.model.occ.addPoint(*P4)
    
    L1=gmsh.model.occ.addLine(i1,i2)
    L2=gmsh.model.occ.addLine(i2,i3)
    L3=gmsh.model.occ.addLine(i3,i4)
    L4=gmsh.model.occ.addLine(i4,i1)
    
    loop1 = gmsh.model.occ.addCurveLoop([L1,L2,L3,L4])
    
    # Dreiecksloch
    P5=(2.5,2,0); P6=(3.5,2,0); P7=(3.5,3,0);
    i5=gmsh.model.occ.addPoint(*P5)
    i6=gmsh.model.occ.addPoint(*P6)
    i7=gmsh.model.occ.addPoint(*P7)
      
    L5=gmsh.model.occ.addLine(i5,i6)
    L6=gmsh.model.occ.addLine(i6,i7)
    L7=gmsh.model.occ.addLine(i7,i5)
   
    loop2 = gmsh.model.occ.addCurveLoop([L5,L6,L7])
    
    # loop1 ist hauptrand, loop2 wird "abgezogen"
    surf1= gmsh.model.occ.addPlaneSurface([loop1, loop2])
    
    
    
    #Inneres Viereck
    P10=(0,0,0); P11=(2,0,0); P12=(2,1,0); P13=(0,1,0)
    i10=gmsh.model.occ.addPoint(*P10,tag=10)
    i11=gmsh.model.occ.addPoint(*P11)
    i12=gmsh.model.occ.addPoint(*P12)
    i13=gmsh.model.occ.addPoint(*P13)
    
    L10=gmsh.model.occ.addLine(i10,i11,tag=10)
    L11=gmsh.model.occ.addLine(i11,i12)
    L12=gmsh.model.occ.addLine(i12,i13)
    L13=gmsh.model.occ.addLine(i13,i10)
    
    gmsh.model.occ.synchronize()
    
    # innere Linie soll später ins Netz integriert werden, deshalb embed
    gmsh.model.mesh.embed(1,[L10,L11,L12,L13],2,surf1)
    
    gmsh.model.occ.synchronize()
    
    # Erzeuge Netz
    mesh = gmsh.model.mesh.generate(2)
    
    
    # Bilde nun Gruppen, z.B. Linien, die (bzw. zu zugehörigen knoten) 
    # später verwendet werden sollen, Fläche immer dazutun
    
    l0 = gmsh.model.addPhysicalGroup(1, [L5, L6, L7])
    gmsh.model.setPhysicalName(1, l0, "RandDreieck")
        
    l1 = gmsh.model.addPhysicalGroup(1, [L2,L4])
    gmsh.model.setPhysicalName(1, l1, "Rand_D")
    
    l2 = gmsh.model.addPhysicalGroup(1, [L10,L11])
    gmsh.model.setPhysicalName(1, l2, "Innen_1")
    
    f1 = gmsh.model.addPhysicalGroup(2, [surf1])
    gmsh.model.setPhysicalName(2, f1, "Flaeche")
    
    
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(name+".brep")
    gmsh.write(name+".msh")
    
    try:
        gmsh.fltk.run()
    except:
        print("Fehler bei gmsh.fltk.run()")  
    
    # Lege Daten in der Klasse ab
    netz = MshHs(gmsh.model)
    
    print("\nTeste Attribute von netz, mit netz.")
    
    gmsh.finalize()
    
    
    #Zuerst vorangegangenen File laden
    gmsh.initialize()
    name = 'TestGroups' 
    gmsh.open(name + ".msh")
    netz=MshHs(gmsh.model)
    
    netz.Triangle.plot(color='grey', alpha=0.3)
    netz.Innen_1.plot(color='black', node_label=True, direction=True)
    plt.show()
    print("Ende1")
    
    
    
    
    
    #####################################################################
    # Neues gmsh netz
    # mit dreiecken und Vierecken
    #####################################################################
    
    gmsh.initialize()
    
    name = 'bla' 
    gmsh.model.add(name)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.4)
    
    # Rechteck mit Rundungen kann auch damit gemacht werden (noch ein Parameter)
    #gmsh.model.occ.addDisk(1.5,0.5,0,0.3,0.3,1)
    gmsh.model.occ.addRectangle(0,0,0,1,1,10)
    gmsh.model.occ.addRectangle(1,0,0,1,1,11)
    gmsh.model.occ.fragment([(2, 10)], [(2, 11)])
    
    gmsh.model.occ.synchronize()
    #benutze rechtecke
    gmsh.model.mesh.setRecombine(2, 11)
    #vernetze
    mesh = gmsh.model.mesh.generate(2)
    gmsh.write(name+".msh")
    
    try:
        gmsh.fltk.run()
    except:
        print("Fehler bei gmsh.fltk.run()")  
    
    netz=MshHs(gmsh.model)
    
    netz.Triangle.plot(color='blue', node_label=True)
    netz.Quadrilateral.plot(color='red', node_label=True)
    plt.show()
    
    gmsh.finalize()
    
    
    
    
    
    ##############################################################################
    # nochmal ein Beispiel
    # Rechteck mit Loch und innerer Linie unter Verwendung von weiterführenden Objekten
    # + Verfeinerung
    ##############################################################################
    
    
    gmsh.initialize()
    
    name = 'bla' 
    gmsh.model.add(name)
    
    p0 = [0, 0, 0]; pr = [1.4, 0.5, 0]
    #erstellt tags zu den punkten, den linien und ein flächentag
    rechteck1 = gmsh.model.occ.addRectangle(*p0,2,1)
    ellipse1 = gmsh.model.occ.addDisk(*pr, 0.4, 0.3)
    gmsh.model.occ.synchronize()
    
    out = gmsh.model.getBoundary([(2, ellipse1)], oriented=False)
    rand_innen = [e[1] for e in out] 
      
    # Löcher bzw ausschneiden
    out, out2 = gmsh.model.occ.cut([(2, rechteck1)],[(2, ellipse1)])
    # (neues) Flächentag ermitteln 
    flaeche_id = out[0][1]
        
    # Innere Linie plazieren
    p1 = [0.5, 0.2, 0]; p2 =[0.5, 0.8, 0]
    pt1 = gmsh.model.occ.addPoint(*p1)
    pt2 = gmsh.model.occ.addPoint(*p2)
    l1 = gmsh.model.occ.addLine(pt1, pt2)
    
    gmsh.model.occ.synchronize()
         
    # Linie einbetten
    gmsh.model.mesh.embed(1,[l1],2,flaeche_id)
    
    # Wichtig scheint zu sein auch immer die flaeche mit in die physikalische Gruppe zu stecken
    sf = gmsh.model.addPhysicalGroup(2, [flaeche_id])
    gmsh.model.setPhysicalName(2, sf, "Flaeche")
    
    ri1 = gmsh.model.addPhysicalGroup(1, rand_innen )
    gmsh.model.setPhysicalName(1, ri1, "RandInnen")    
    
    ri2 = gmsh.model.addPhysicalGroup(1, [l1])
    gmsh.model.setPhysicalName(1, ri2, "LinieInnen")
    
    
    # noch eine Verfeinerung des netzes
    dist_tag = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_tag, "EdgesList", [l1])       # zu diesen Linien wird der (minimale) Abstand berechnet
    gmsh.model.mesh.field.setNumber(dist_tag, "Sampling", 100)          # auf den Linien werden dazu 100 pkt verwendet    
    
    # abstandsfunktion verwenden
    math_tag = gmsh.model.mesh.field.add("MathEval")
    # die Formel h=0.02 + 0.3*Fi gibt die ungefähre mittlere Dreieckslänge an
    gmsh.model.mesh.field.setString(math_tag, "F", "0.015 + 0.2*F%i"%(dist_tag))         # Fi = Distanzwert mit tag i
    gmsh.model.mesh.field.setAsBackgroundMesh(math_tag)     
    
    mesh = gmsh.model.mesh.generate(2)
    
    
    netz=MshHs(gmsh.model)
    # Man kann auch noch nachdem das Netz abgelegt wurde SubObjekte erzeugen
    # z.B. aus den Line-Elementen nur bestimmte durch eine Funktion auswählen 
    netz.Line.select("DirichletRand", lambda x: (x[:, 0] < 1e-10) | (x[:, 1]<1e-10) )
    
    # plotten
    netz.Triangle.plot(color='grey', alpha=0.2)
    netz.LinieInnen.plot(color='red')
    netz.RandInnen.plot(color='red', direction=True)
    netz.DirichletRand.plot(color='darkred', direction=True, node_label=True)
    plt.show()
    
    # Randrichtungen umkehren
    netz.Triangle.plot(color='grey', alpha=0.2)
    netz.RandInnen.flip(flip='right')
    netz.RandInnen.plot(color='darkred')
    plt.show()
    
    gmsh.write(name+".msh")
    gmsh.finalize()
       
        
    
    ##############################################################################
    # 3D - Körper
    #
    ##############################################################################
    
    gmsh.initialize()
    gmsh.model.add("3dBsp")
    
    # Großer Würfel (Startpunkt 0,0,0 – Größe 2×2×2)
    big = gmsh.model.occ.addBox(0, 0, 0, 2, 2, 2)
    
    # innere Punkte
    p1 = gmsh.model.occ.addPoint(0.5, 0.5, 0.5)
    p2 = gmsh.model.occ.addPoint(1.5, 0.5, 0.5)
    p3 = gmsh.model.occ.addPoint(1.5, 1.5, 0.5)
    p4 = gmsh.model.occ.addPoint(0.5, 1.5, 0.5)
    
    p5 = gmsh.model.occ.addPoint(1.5, 0.5, 1)
    p6 = gmsh.model.occ.addPoint(1.5, 1.5, 1)
    
    # innere Linien
    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)
    l5 = gmsh.model.occ.addLine(p2, p5)
    l6 = gmsh.model.occ.addLine(p5, p6)
    l7 = gmsh.model.occ.addLine(p6, p3)
    
    
    # innere Flächen
    loop1 = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    loop2 = gmsh.model.occ.addCurveLoop([l5, l6, l7, -l2])
    
    # Fläche erzeugen
    surf1 = gmsh.model.occ.addPlaneSurface([loop1])
    surf2 = gmsh.model.occ.addPlaneSurface([loop2])
    
    gmsh.model.occ.synchronize()
    
    # Flächen einbetten
    gmsh.model.mesh.embed(2,[surf1, surf2],3,big)
    
    # in 2D wars
    #gmsh.model.mesh.embed(1,[L10,L11,L12,L13],2,surf1)
    
    gmsh.model.mesh.generate(3)
    gmsh.write("3dBsp.msh")
    
    try:
        gmsh.fltk.run()
    except:
        print("Fehler bei gmsh.fltk.run()")  
    
    
    
    netz=MshHs(gmsh.model)
    
    netz.Triangle.plot(color='blue', alpha=0.3)
    netz.Line.plot(color='red', alpha=0.8, node_label=True)
    plt.show()
    
    netz.Triangle.plot(color='blue', alpha=0.3)
    netz.Tetrahedron.plot(color='grey', alpha=0.2)
    plt.show()
    
    gmsh.finalize()
    
    
    
    
    #######################################################
    #
    # Auswertungen von Lösungen auf dem Netz
    #
    #######################################################
    
    
    ########### Normalenableitung: ###################       
    #np.savetxt('GmshExample_points.dat', poi)
    #np.savetxt('GmshExample_phis.dat', phi)
    #np.savetxt('GmshExample_elements.dat', tri, fmt='%i')
    #np.savetxt('GmshExample_bound.dat', np.array(bseg[0]), fmt='%i')
    #np.savetxt('GmshExample_inner.dat', np.array(cseg[0]), fmt='%i')
    
    # Ein Netz welches über einfache Listen definiert ist 
    # (z.B. mit anderem programm definiert).
    # Wird verwendet um verschiedene Auswertedinge zu testen
    
    poi = np.loadtxt('GmshExample_points.dat')
    phi = np.loadtxt('GmshExample_phis.dat')
    tri = np.loadtxt('GmshExample_elements.dat', dtype=np.int64)
    # Randkurve
    bseg = np.loadtxt('GmshExample_bound.dat', dtype=np.int64)
    # Innere Kurve
    iseg = np.loadtxt('GmshExample_inner.dat', dtype=np.int64)    
    
    # Schreibe Netz in MshHs Klasse:
    netz = MshHs(None, points=poi, triangles=tri, segments={'KurveRand': bseg, 'KurveInnen': iseg})
    #netz.Triangle.plot(color='blue', alpha=0.3, node_label=True)
    #plt.show()
    
    
    #plots, alternativ auch mit net.Triangle.plot()
    plt.triplot(poi[:,0],poi[:,1],tri,color='gray')
    plt.title("Erzeuge dieses Netz mit gmsh, Parabel: y=7-(x-5)^2")
    plt.show()
    
    # Ergebnis plotten
    poi = netz.points
    tri = netz.Triangle.elements
    plt.tricontourf(poi[:,0],poi[:,1],tri,phi,levels=30,cmap='jet')
    plt.tricontour(poi[:,0],poi[:,1],tri,phi,levels=30,colors='black')
    plt.xlim((-0.2,10.2))
    plt.ylim((-0.2,8.2));
    plt.show()    
    
    
    
        
    # Normalenableitung mit MshHs, ergebnis
    netz.Triangle.plot()
    netz.KurveRand.plot(direction=True, node_label=True)
    plt.show()
    
    res_rand = NormalDerivative('KurveRand', netz, phi, normal_direction='left')
    res_innen_r = NormalDerivative('KurveInnen', netz, phi)
    res_innen_l = NormalDerivative('KurveInnen', netz, phi, normal_direction='left')
    
    
    
    
    #plot normal derivative
    plt.plot(res_rand[0],res_rand[1],'o',markersize=8,label='boundary')
    plt.plot(res_innen_r[0],res_innen_r[1],'o',markersize=8,label='inner, right')
    plt.plot(res_innen_l[0],res_innen_l[1],'o',markersize=8,label='inner_left')
    
    #indicate different parts of boundary
    s0 = res_rand[0]
    plt.plot([2,2],[0,10],'k--')
    plt.plot([s0[-1]-10,s0[-1]-10],[-2,5],'k--')
    plt.plot([s0[-1]-12,s0[-1]-12],[-5,5],'k--')
    plt.grid()
    
    plt.xlabel("curve length s")
    plt.ylabel(r"normal derivative $\;\;\frac{\partial \Phi}{\partial n}$")
    
    
    # theory, boundary
    R = 2
    #first part
    t=np.linspace(8,6,100)
    s=8-t
    dF=t
    plt.plot(s,dF,lw=2,color="cyan",label="Theory")
    
    # second part
    t=np.linspace(np.pi/2,-np.pi/2,100)
    s=2+(np.pi/2-t)*R
    dF=4*np.sin(t)*np.cos(t) + 4*np.cos(t) + 3*np.sin(t)
    plt.plot( s,dF,lw=2,color="cyan" )
    
    #third part (circle)
    t=np.linspace(2,0,100)
    s=2+np.pi*R+2-t
    dF=t
    plt.plot(s,dF,lw=2,color="cyan")
    
    #fourth part
    t=np.linspace(0,10,100)
    s=4+np.pi*R+t
    dF=(t+3)
    plt.plot(s,dF,lw=2,color="cyan")
    
    #theory inner curve
    t=np.linspace(2.5,7,500)
    # x(t)
    xx=t; yy=-(t-5)**2+7
    # ds
    ds=np.sqrt(np.diff(xx)**2+np.diff(yy)**2)
    # running length of curve
    s=np.append(np.array([0]),np.cumsum(ds))
    
    #left and right depends at inner curves on the orientation of the curve (starting point)
    # take phi on left side of curve, normal points to the right
    # grad(phi)*n
    dF=(yy*2*(xx-5) + (xx+3)*1)/(np.sqrt(4*(xx-5)**2+1))
    plt.plot(s,dF,lw=2,color="cyan")
    
    # take phi on the right side of curve, normal points to the left
    # grad(phi)*n
    dF= -( (yy+6*(xx-5))*2*(xx-5) + (xx+3+3)*1)/(np.sqrt(4*(xx-5)**2+1))
    plt.plot(s,dF,lw=2,color="cyan")
    
    
    plt.legend()
    plt.show()
    
    #Gradient plotten
    pp = netz.points[:,[0, 1]]
    xx,yy,ggx,ggy=ComputeGradient(pp,netz.Triangle.elements,phi,location='middle' )
    plt.triplot(pp[:, 0], pp[:, 1], netz.Triangle.elements)
    plt.quiver(xx,yy, ggx,ggy, color='green',scale=500)
    plt.show()
    
    print("Ende")