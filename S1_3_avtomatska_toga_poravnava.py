import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from vaja_01_tranformacije_slik import nalozi_sliko_sitk, toga_trans_z_afino_matriko


# Pomožna funkcija za vizualizacijo prekrivanja (prej in potem)
def prikazi_prekrivanje(slika_fiksna, slika_gibljiva, naslov="Prekrivanje"):
    """Prikaže prekrivanje dveh slik kot barvno kompozicijo (rdeče-zeleno)."""
    slika_fiksna_np = sitk.GetArrayFromImage(slika_fiksna).astype(np.float32)
    slika_gibljiva_np = sitk.GetArrayFromImage(slika_gibljiva).astype(np.float32)
    
    # Normalizacija za boljši barvni prikaz
    min_val = min(slika_fiksna_np.min(), slika_gibljiva_np.min())
    max_val = max(slika_fiksna_np.max(), slika_gibljiva_np.max())
    slika_fiksna_norm = (slika_fiksna_np - min_val) / (max_val - min_val)
    slika_gibljiva_norm = (slika_gibljiva_np - min_val) / (max_val - min_val)

    kompozit = np.stack([slika_fiksna_norm, slika_gibljiva_norm, slika_fiksna_norm * 0], axis=2)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(kompozit)
    plt.title(naslov + " (Rdeča=Fiksna, Zelena=Gibljiva)")
    plt.axis('off')
    plt.show()

def osnovna_toga_registracija_2d(slika_fiksna, slika_gibljiva):
    """
    Izvede osnovno togo poravnavo 2D slik (Rotacija + Translacija)
    z uporabo prednastavljenih SITK parametrov.
    """

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,            
        minStep=1e-4, 
        numberOfIterations=500
    )

    # 1. Inicializacija in TIP TRANSFORMACIJE: Euler 2D (Toga)
    initial_transform = sitk.CenteredTransformInitializer(
        slika_fiksna, slika_gibljiva, 
        sitk.Euler2DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    R.SetInitialTransform(initial_transform)

    # Uporabimo CenteredTransformInitializer za določitev skal optimizatorja
    # Ta klic je najbolj stabilen za Euler transformacije.
    R.SetOptimizerScalesFromIndexShift()
    
    # 3. Zagon procesa registracije
    final_transform = R.Execute(slika_fiksna, slika_gibljiva)
    
    print("-" * 40)
    print(f"Končni strošek (Mean Squares): {R.GetMetricValue():.5f}")
    print(f"Končno število iteracij: {R.GetOptimizerIteration()}")
    print("-" * 40)

    # 5. Aplikacija dobljene transformacije (Resampling)
    slika_registered = sitk.Resample(
        slika_gibljiva,
        slika_fiksna, 
        final_transform,
        sitk.sitkLinear,
        0.0,
        slika_gibljiva.GetPixelID()
    )
    
    return slika_registered, final_transform

if __name__ == "__main__":
    ######################
    #               S1.3.4              #
    ######################
    
    pot_do_slike = 'mr-enhanced.png'
    
    # Naloži sliko in jo nastavi kot fiksno.
    slika_fixed = nalozi_sliko_sitk(pot_do_slike, ciljni_tip=sitk.sitkFloat32) 
    
    KOT_GEN = 10.0 # Stopinje
    TRANS_GEN = [-5.0, 15.0] # Piksli
    slika_moving = toga_trans_z_afino_matriko(slika_fixed, KOT_GEN, TRANS_GEN)
    # Prepričajte se, da je tudi slika_moving tipa Float, če ni avtomatsko.
    slika_moving = sitk.Cast(slika_moving, sitk.sitkFloat32)

    prikazi_prekrivanje(slika_fixed, slika_moving, "Neporavnane slike")

    # Klic funkcije (Primer izvajanja)
    slika_poravnana, transformacija = osnovna_toga_registracija_2d(slika_fixed, slika_moving)
    prikazi_prekrivanje(slika_fixed, slika_poravnana, "Poravnane slike (Toga registracija)")

