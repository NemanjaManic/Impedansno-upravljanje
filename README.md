# Impedansno upravljanje robotom

Diplomski rad - Implementacija impedansnog upravljanja robotom.

## Preuzimanje i instalacija

### 1. Preuzimanje projekta

```bash
git clone https://github.com/NemanjaManic/Impedansno-upravljanje.git
cd Impedansno-upravljanje
```

### 2. Instalacija Python biblioteka
Koristim Python 3.10 verziju pythona, u virtuelnom okruženju (preporučeno, PyCharm automatski kreira venv):
```bash
# Ako koristiš PyCharm, samo aktiviraj venv koji on kreira
# Ili ručno kreiraj virtuelno okruženje:
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
pip install -r requirements.txt
```


### 3. Dodavanje objekta (kutije) u scenu

Ukoliko želite da testirate kontakt i interakciju robota sa okolinom, u `panda.xml` fajl je potrebno dodati sledeći objekat (geom):

```xml
<geom name="large_box"
      type="box"
      size="0.2 0.4 0.125"
      pos="0.6 0.0 0.125"
      rgba="0.8 0.3 0.3 0.7"/>
```

### 4. Pokretanje koda
```bash
python Panda_ImpedanceControl_TAU.py
```

## Struktura projekta

* **Panda_ImpedanceControl_TAU.py** - Glavna skripta za impedansno upravljanje
* **Funkcije.py** - Pomoćne funkcije
* **requirements.txt** - Lista potrebnih biblioteka

## Opis

Implementacija impedansnog upravljanja robota koji će prilagođavati svoje ponašanje u kontaktu sa okolinom.

