import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def prikazi_deformacijo_b_zlepki(center_point_index, premik, korak_mreze=50.0, velikost_slike=400):
    """
    Prikaže deformacijo točk z uporabo SimpleITK B-zlepkov.

    Args:
        center_point_index (int): Indeks parametra v seznamu parametrov transformacije, 
                                  ki ga želimo premakniti (npr. X komponenta pomika).
        premik (float): Velikost premika v pikslih (npr. 200.0).
        korak_mreze (float): Želeni razmik med kontrolnimi točkami (spacing).
        velikost_slike (int): Dimenzija fiktivne slike za inicializacijo mreže.
    """
    
    # --- 1. Inicializacija B-zlepkov ---
    slika_fake = sitk.Image(velikost_slike, velikost_slike, sitk.sitkUInt8)
    
    # Izračun števila mrežnih subdivizij (mesh size)
    num_subdiv = int(velikost_slike / korak_mreze)
    mesh_size_subdivisions = [num_subdiv, num_subdiv] 
    spline_order = 3 # Kubični B-zlepki (standard)
    
    # Ustvari B-zlepke (Transformacija je inicializirana na identiteto)
    transform_bspline = sitk.BSplineTransformInitializer(
        slika_fake, 
        mesh_size_subdivisions, 
        spline_order
    )

    # --- 2. Premik ene kontrolne točke (Simulacija deformacije) ---
    params = list(transform_bspline.GetParameters())
    
    # Premaknemo izbrano komponento (npr. X komponento) kontrolne točke
    # *POMEMBNO: Tukaj se predpostavlja, da je index pravilen glede na X/Y komponento.*
    if center_point_index < len(params):
        params[center_point_index] += premik 
    else:
        print(f"Opozorilo: Indeks {center_point_index} je izven obsega parametrov ({len(params)}).")
        return
        
    transform_bspline.SetParameters(params)

    # --- 3. Deformacija mreže točk ---
    # Ustvarimo mrežo točk za vizualizacijo (korak 20 pikslov)
    mesh = np.mgrid[0:velikost_slike:20, 0:velikost_slike:20].T.reshape(-1, 2) 

    # Uporabimo p.tolist() za pravilno pretvorbo NumPy vektorja v Python seznam za SimpleITK
    deformed_mesh = [transform_bspline.TransformPoint(p.tolist()) for p in mesh] 
    deformed_mesh = np.array(deformed_mesh)

    # --- 4. Prikaz rezultatov (Slike ena zraven druge) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'Deformacija mreže z B-zlepki (Premik indeksa {center_point_index} za {premik} px)', fontsize=14)

    # Levi podgraf: Originalna mreža
    axes[0].scatter(mesh[:, 0], mesh[:, 1], color='blue')
    axes[0].set_title('Originalna mreža točk')
    axes[0].set_xlim(0, velikost_slike); axes[0].set_ylim(velikost_slike, 0)

    # Desni podgraf: Deformirana mreža z vektorji premika
    axes[1].scatter(deformed_mesh[:, 0], deformed_mesh[:, 1], color='red')
    axes[1].quiver(mesh[:, 0], mesh[:, 1], 
                   deformed_mesh[:, 0] - mesh[:, 0], 
                   deformed_mesh[:, 1] - mesh[:, 1], 
                   angles='xy', scale_units='xy', scale=1, color='gray')
    axes[1].set_title('Deformirana mreža z vektorji')
    axes[1].set_xlim(0, velikost_slike); axes[1].set_ylim(velikost_slike, 0)

    plt.show()

######################
#               S1.6.2               #
######################	
# prikazi_deformacijo_b_zlepki(50, 200.0)

