
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def nalozi_sliko_sitk(pot_do_datoteke: str, ciljni_tip=sitk.sitkUInt8) -> sitk.Image:
    """Naloži 2D sliko (preko PIL) in jo pretvori v SimpleITK sliko."""
    print(f"Nalaganje datoteke: {pot_do_datoteke}")
    try:
        # Naloži z PIL (podpira PNG) in pretvori v numpy polje
        img_np = np.array(Image.open(pot_do_datoteke).convert('L'))
        
        # Pretvori numpy v SimpleITK Image
        slika_sitk = sitk.GetImageFromArray(img_np)
        
        # Pretvori v ciljni tip
        slika_sitk = sitk.Cast(slika_sitk, ciljni_tip)
        
        print(f"Velikost slike: {slika_sitk.GetSize()}")
        return slika_sitk
        
    except FileNotFoundError:
        print(f"\n[NAPAKA]: Datoteka '{pot_do_datoteke}' ni najdena.")
        sys.exit(1)

def prikazi_sitk_sliko(slika: sitk.Image, naslov: str = "Slika"):
    """Prikaže 2D SimpleITK sliko z uporabo Matplotlib."""
    slika_np = sitk.GetArrayFromImage(slika)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(slika_np, cmap='gray')
    plt.title(naslov)
    plt.axis('off')
    plt.show()

##############################################

# Funkcija za izvedbo toge transformacije (Rotacija in Translacija)
def toga_trans_z_afino_matriko(slika_in, kot_stopinje, vektor_translacije):
    """Izvede togo (rigid) transformacijo slike z uporabo sitk.AffineTransform."""
    
    # 1. Pretvorba kota v radiane
    kot_radiani = kot_stopinje * np.pi / 180.0
    
    # 2. Določitev centra transformacije (pomembno za rotacijo)
    center = slika_in.TransformIndexToPhysicalPoint(
        [s // 2 for s in slika_in.GetSize()]
    )
    
    # 3. Inicializacija 2D afine transformacije
    transformacija_afina = sitk.AffineTransform(slika_in.GetDimension())
    transformacija_afina.SetCenter(center)

    # 4. Konstrukcija matrike rotacije [cos, -sin, sin, cos]
    matrika_rotacije = [
        np.cos(kot_radiani), -np.sin(kot_radiani),
        np.sin(kot_radiani), np.cos(kot_radiani)
    ]
    
    # Nastavitev matrike in translacije
    transformacija_afina.SetMatrix(matrika_rotacije)
    
    # Dodajanje translacije (premik se doda na že določeno rotacijo)
    transformacija_afina.SetTranslation(vektor_translacije)
    
    # 5. Aplikacija transformacije (Resampling)
    slika_out = sitk.Resample(
        slika_in,
        slika_in,             
        transformacija_afina,
        sitk.sitkLinear,      
        0.0,                  
        slika_in.GetPixelID()
    )
    return slika_out

def afina_transformacija_2d(slika_in, matrika_transformacije_2x2, vektor_translacije):
    """Izvede poljubno 2D afino transformacijo z uporabo sitk.AffineTransform."""
    
    # Določitev centra transformacije
    center = slika_in.TransformIndexToPhysicalPoint(
        [s // 2 for s in slika_in.GetSize()]
    )
    
    # Inicializacija afine transformacije
    transformacija_afina = sitk.AffineTransform(slika_in.GetDimension())
    transformacija_afina.SetCenter(center)
    
    # Nastavitev celotne transformacijske matrike (skaliranje, striženje, rotacija)
    transformacija_afina.SetMatrix(matrika_transformacije_2x2)
    
    # Nastavitev vektorja translacije
    transformacija_afina.SetTranslation(vektor_translacije)
    
    # Aplikacija transformacije
    slika_out = sitk.Resample(
        slika_in, slika_in, transformacija_afina,
        sitk.sitkLinear, 0.0, slika_in.GetPixelID()
    )
    return slika_out


if __name__ == "__main__":
    ######################
    #               S1.2.1               #
    ######################
    
    # Določitev poti do slike
    pot_do_slike = 'mr-enhanced.png'
    
    # Naloži sliko in jo nastavi kot fiksno
    slika_fixed = nalozi_sliko_sitk(pot_do_slike, ciljni_tip=sitk.sitkUInt8)
    
    # prikazi sliko:
    prikazi_sitk_sliko(slika_fixed)
    
    ######################
    #               S1.2.2               #
    ######################
    # TOGA TRANFORMACIJA
    vektor = [20.0, 10.0]
    slika_toga = toga_trans_z_afino_matriko(slika_fixed, -30.0, vektor)
    prikazi_sitk_sliko(slika_toga, f"Toga trans.: -30° rot. in [20, 10] transl.")

    ######################
    #               S1.2.3               #
    ######################
    # Afina poravnava
    matrika_test = [0.8, -0.3, 0.3, 1.2]
    vektor_test = [-15.0, 5.0]
    slika_poljubna = afina_transformacija_2d(slika_fixed, matrika_test, vektor_test)
    prikazi_sitk_sliko(slika_poljubna, f"Poljubna afina transformacija: Matrika={matrika_test}, Translacija={vektor_test}")
