import os
import matplotlib.pyplot as plt
import numpy as np
import gmshtools as gm
import gmsh
basis = os.path.join(os.path.dirname(os.path.abspath(__file__)),"2D_Reference")
p = np.loadtxt(basis + r"\Netz2D_p.dat", dtype=float)




gmsh.initialize() 
name = 'TEST_2D'
gmsh.model.add(name)


#Coordinates
point1 = (1,1,0)
point2 = (2,1,0)
point3 = (3,1,0)
point4 = (3,4,0)
point5 = (2,4,0)
point6 = (1,4,0)
point7 = (1.25, 3,0)
point8 = (1.75, 3,0)
point9 = (1.75, 3.5,0)
point10 = (1.25,3.5,0)

Circle1mid = (1.5, 1.75)
radius1 = 0.35

Circle2mid= (2.5,2.5,0)
radius2 = 0.3

#koordinatenüberage an gmsh mit * rollout statt forschleife (25e-3, 80e-3/2, 0)
p1 = gmsh.model.occ.add_point(*point1) 
p2 = gmsh.model.occ.add_point(*point2)
p3 = gmsh.model.occ.add_point(*point3)
p4 = gmsh.model.occ.add_point(*point4)
p5 = gmsh.model.occ.add_point(*point5)
p6 = gmsh.model.occ.add_point(*point6)

c1 = gmsh.model.occ.add_circle(Circle1mid[0],Circle1mid[1],0,radius1)
c2 = gmsh.model.occ.add_circle(Circle2mid[0],Circle2mid[1],0,radius2)


#äußeres Viereck
L12 = gmsh.model.occ.add_line(p1, p2)
L23 = gmsh.model.occ.add_line(p2, p3)
L34 = gmsh.model.occ.add_line(p3, p4)
L45 = gmsh.model.occ.add_line(p4, p5)
L56 = gmsh.model.occ.add_line(p5, p6)
L61 = gmsh.model.occ.add_line(p6, p1)

#mittellinie
L25 = gmsh.model.occ.add_line(p2,p5)


#äußeres Rechteck als Loop definieren 
loop_außen = gmsh.model.occ.addCurveLoop([L12,L23,L34,L45,L56,L61])


#inneres Viereck
p7 = gmsh.model.occ.add_point(*point7) 
p8 = gmsh.model.occ.add_point(*point8)
p9 = gmsh.model.occ.add_point(*point9)
p10 = gmsh.model.occ.add_point(*point10)

L78 = gmsh.model.occ.add_line(p7,p8)
L89 = gmsh.model.occ.add_line(p8,p9)
L910 = gmsh.model.occ.add_line(p9,p10)
L107 = gmsh.model.occ.add_line(p10,p7)
loop_innen = gmsh.model.occ.addCurveLoop([L78, L89, L910,L107])
loop_c2 = gmsh.model.occ.addCurveLoop([c2])


#Fläche generieren erst jetzt kann man in Gmsh GUI ein Netz generieren
# passiert aber noch nicht selbstständig durch den Code...
surf= gmsh.model.occ.addPlaneSurface([loop_außen,loop_c2])
gmsh.model.occ.synchronize()


#Netz generieren !vorher synchronisieren!
f_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", [c1, c2])  # beide Kreise

f_thr = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", 0.0001)
gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", 0.2)
gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(f_thr, "DistMax", 0.1)

gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)
gmsh.model.occ.synchronize()
mesh = gmsh.model.mesh.generate(2)

#Gruppen bilden für Randbedingungen
f1 =gmsh.model.addPhysicalGroup(2,[surf])
gmsh.model.setPhysicalName(2,f1, "Fläche")

gmsh.option.setNumber("Mesh.SaveAll",1)
gmsh.write(name+".msh")
try: 
    gmsh.fltk.run()
except:
    print("Fehler bei gmsh.fltk.run()")

netz2=gm.MshHs(gmsh.model)
gmsh.finalize()
netz2.Triangle.plot()
plt.show()
