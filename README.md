# Impedansno upravljanje robotom

Diplomski rad - Implementacija impedansnog upravljanja robotom Franka Emika Panda.

## Preuzimanje i instalacija

### 1. Preuzimanje projekta

```bash
git clone https://github.com/NemanjaManic/Impedansno-upravljanje.git
cd Impedansno-upravljanje
```

### 2. Instalacija Python biblioteka
Koristim, Python 3.10 verziju pythona, u virtuelnom okruženju (preporučeno, PyCharm automatski kreira venv):
```bash
# Ako koristite PyCharm, samo aktivirajte venv koji on kreira
# Ili ručno kreirajte virtuelno okruženje:
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
pip install -r requirements.txt
```

### 3. Dodavanje objekta (kutije) u scenu

Ukoliko želite da testirate kontakt i interakciju robota sa okolinom, u `panda.xml` fajl je potrebno dodati sledeći objekat (geom) koji je rotiran za 10 stepeni:

```xml
   <body name="box_body" pos="0.6 0.0 0.125" euler="0.1745 0 0">
       <geom name="large_box" type="box" 
             size="0.2 0.4 0.125" 
             rgba="0.8 0.3 0.3 0.7"/>

   </body>
```

### 4. Pokretanje koda
Ukoliko želite da pokrenete scenario, gde je kutija pod nagibom i tu uočite primenu impedansnog upravljanja, pokrenite Panda_ImpedanceControl_Scenario1.
Ukoliko želite da pokrenete scenario, gde primenom neke sile posmatrate kako se robot ponaša podešavanjem parametara impedanse, pokrenite Panda_ImpedanceControl_Scenario2.
```bash
python Panda_ImpedanceControl_Scenario1.py
python Panda_ImpedanceControl_Scenario2.py
```

## Struktura projekta

* **Panda_ImpedanceControl_Scenario1.py** - Scenario1
* **Panda_ImpedanceControl_Scenario2.py** - Scenario2
* **Funkcije.py** - Pomoćne funkcije
* **requirements.txt** - Lista potrebnih biblioteka

## Opis

Implementacija impedansnog upravljanja robota Franka Emika Panda koji će prilagođavati svoje ponašanje u kontaktu sa okolinom. Ovaj rad obuhvata i proučavanja parametara impedanse.

