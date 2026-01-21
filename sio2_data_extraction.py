#!/usr/bin/env python3
"""
SiO2 Physical Properties Data Extraction Script

This script extracts SiO2 (silicon dioxide) physical property data from real databases
with proper literature citations. No synthetic data generation.

Data sources:
1. Materials Project API
2. AFLOW database (REST API)
3. Crystallography Open Database (COD)
4. Open Quantum Materials Database (OQMD)
5. Published literature and handbooks

All entries include proper reference citations (DOI, database ID, or publication info).
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import os
import hashlib
from typing import Dict, List, Optional, Set
import warnings
warnings.filterwarnings('ignore')

# CSV column names (matching the original file structure)
CSV_COLUMNS = [
    'crystal_structure', 'density_g/cm3', 'melting_point_degC', 'boiling_point_degC',
    'thermal_conductivity_W/(m*K)', 'specific_heat_J/(kg*K)', 'thermal_expansion_1e-6/K',
    'youngs_modulus_GPa', 'shear_modulus_GPa', 'bulk_modulus_GPa', 'poissons_ratio',
    'hardness_MPa', 'hardness_Mohs', 'tensile_strength_MPa', 'compressive_strength_MPa',
    'fracture_toughness_MPa*m^0.5', 'refractive_index_o', 'refractive_index_e',
    'dielectric_constant', 'resistivity_ohm*m', 'magnetic_susceptibility_cm3/mol',
    'band_gap_eV', 'std_enthalpy_kJ/mol', 'std_entropy_J/(mol*K)',
    'Sellmeier_B1', 'Sellmeier_B2', 'Sellmeier_B3', 'Sellmeier_C1_um2',
    'Sellmeier_C2_um2', 'Sellmeier_C3_um2', 'lattice_a_angstrom', 'lattice_b_angstrom',
    'lattice_c_angstrom', 'lattice_alpha_deg', 'lattice_beta_deg', 'lattice_gamma_deg',
    'space_group', 'volume_angstrom3', 'Z_formula_units', 'piezo_d11_pC/N',
    'piezo_d14_pC/N', 'piezo_e11_C/m2', 'piezo_e14_C/m2', 'elastic_C11_GPa',
    'elastic_C12_GPa', 'elastic_C13_GPa', 'elastic_C14_GPa', 'elastic_C33_GPa',
    'elastic_C44_GPa', 'elastic_C66_GPa', 'molar_volume_cm3/mol', 'melting_point_K',
    'boiling_point_K', 'thermal_expansion_parallel_1/K', 'thermal_expansion_perp_1/K',
    'dielectric_constant_parallel', 'dielectric_constant_perp', 'dielectric_loss_tan_delta',
    'thermal_conductivity_W/mK', 'poisson_ratio', 'bandgap_eV',
    'sound_velocity_longitudinal_m/s', 'sound_velocity_transverse_m/s', 'viscosity_Pa.s',
    'softening_point_degC', 'annealing_point_degC', 'strain_point_degC',
    'freq_temp_coeff_ppm/C', 'Q_factor', 'debye_temp_K', 'formation_gibbs_kJ/mol',
    'notes', 'reference'
]


class DataDeduplicator:
    """Track and prevent duplicate entries."""
    
    def __init__(self):
        self.seen_hashes: Set[str] = set()
    
    def _compute_hash(self, entry: Dict) -> str:
        """Compute hash of key properties to detect duplicates."""
        key_fields = [
            'crystal_structure', 'density_g/cm3', 'lattice_a_angstrom',
            'lattice_b_angstrom', 'lattice_c_angstrom', 'space_group',
            'band_gap_eV', 'bulk_modulus_GPa'
        ]
        hash_str = ""
        for field in key_fields:
            val = entry.get(field, '')
            if val != '' and val is not None:
                if isinstance(val, float):
                    hash_str += f"{field}:{val:.4f}|"
                else:
                    hash_str += f"{field}:{val}|"
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def is_duplicate(self, entry: Dict) -> bool:
        """Check if entry is a duplicate."""
        h = self._compute_hash(entry)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False


class MaterialsProjectExtractor:
    """Extract SiO2 data from Materials Project API."""
    
    BASE_URL = "https://api.materialsproject.org"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('MP_API_KEY', '')
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({'X-API-KEY': self.api_key})
    
    def search_sio2_entries(self, max_entries: int = 5000) -> List[Dict]:
        """Search for SiO2 entries in Materials Project."""
        entries = []
        
        if not self.api_key:
            print("    Warning: No Materials Project API key provided")
            return entries
        
        try:
            url = f"{self.BASE_URL}/materials/summary/"
            params = {
                'formula': 'SiO2',
                '_limit': min(max_entries, 1000),
                '_fields': 'material_id,formula_pretty,density,band_gap,formation_energy_per_atom,'
                          'energy_above_hull,volume,nsites,symmetry,structure'
            }
            
            response = self.session.get(url, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    entries = data['data']
                    print(f"    Retrieved {len(entries)} entries from Materials Project")
        except Exception as e:
            print(f"    Materials Project extraction error: {e}")
        
        return entries
    
    def parse_entry(self, entry: Dict) -> Dict:
        """Parse Materials Project entry to standard format."""
        result = {col: '' for col in CSV_COLUMNS}
        
        try:
            material_id = entry.get('material_id', '')
            
            symmetry = entry.get('symmetry', {})
            if isinstance(symmetry, dict):
                result['space_group'] = symmetry.get('symbol', '')
                crystal_system = symmetry.get('crystal_system', '')
                result['crystal_structure'] = self._classify_structure(result['space_group'], crystal_system)
            
            if 'density' in entry and entry['density']:
                result['density_g/cm3'] = entry['density']
            
            if 'band_gap' in entry and entry['band_gap']:
                result['band_gap_eV'] = entry['band_gap']
                result['bandgap_eV'] = entry['band_gap']
            
            if 'formation_energy_per_atom' in entry and entry['formation_energy_per_atom']:
                result['std_enthalpy_kJ/mol'] = entry['formation_energy_per_atom'] * 3 * 96.485
            
            if 'volume' in entry and entry['volume']:
                result['volume_angstrom3'] = entry['volume']
            
            structure = entry.get('structure', {})
            if isinstance(structure, dict):
                lattice = structure.get('lattice', {})
                if lattice:
                    result['lattice_a_angstrom'] = lattice.get('a', '')
                    result['lattice_b_angstrom'] = lattice.get('b', '')
                    result['lattice_c_angstrom'] = lattice.get('c', '')
                    result['lattice_alpha_deg'] = lattice.get('alpha', '')
                    result['lattice_beta_deg'] = lattice.get('beta', '')
                    result['lattice_gamma_deg'] = lattice.get('gamma', '')
            
            result['reference'] = f"Materials Project:{material_id}, DOI:10.17188/1190959"
            result['notes'] = f"DFT calculation, {entry.get('formula_pretty', 'SiO2')}"
            
        except Exception as e:
            print(f"    Error parsing MP entry: {e}")
        
        return result
    
    def _classify_structure(self, spacegroup: str, crystal_system: str) -> str:
        """Classify SiO2 structure based on space group."""
        sg_map = {
            'P3_121': 'alpha-quartz', 'P3_221': 'alpha-quartz',
            'P3121': 'alpha-quartz', 'P3221': 'alpha-quartz',
            'P6_222': 'beta-quartz', 'P6_422': 'beta-quartz',
            'P6222': 'beta-quartz', 'P6422': 'beta-quartz',
            'P4_12_12': 'alpha-cristobalite', 'P4_32_12': 'alpha-cristobalite',
            'P41212': 'alpha-cristobalite', 'P43212': 'alpha-cristobalite',
            'Fd-3m': 'beta-cristobalite', 'Fd3m': 'beta-cristobalite',
            'C222_1': 'alpha-tridymite', 'C2221': 'alpha-tridymite',
            'P6_3/mmc': 'beta-tridymite', 'P63/mmc': 'beta-tridymite',
            'C2/c': 'coesite',
            'P4_2/mnm': 'stishovite', 'P42/mnm': 'stishovite',
            'Pbcn': 'seifertite',
            'I-43d': 'melanophlogite'
        }
        
        for sg, struct in sg_map.items():
            if sg in str(spacegroup):
                return struct
        
        return f'SiO2_{crystal_system}' if crystal_system else 'SiO2_MP'


class AFLOWExtractor:
    """Extract SiO2 data from AFLOW database."""
    
    API_URL = "http://aflowlib.org/API/aflux/"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'SiO2-DataExtractor/1.0'})
    
    def search_sio2_entries(self, max_entries: int = 5000) -> List[Dict]:
        """Search for SiO2 entries in AFLOW database."""
        entries = []
        
        try:
            query = f"species(Si,O),nspecies(2),paging(1,{min(max_entries, 500)})"
            url = f"{self.API_URL}?{query}"
            
            response = self.session.get(url, timeout=120)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list):
                        entries = data
                    elif isinstance(data, dict) and 'entries' in data:
                        entries = data['entries']
                except json.JSONDecodeError:
                    lines = response.text.strip().split('\n')
                    for line in lines:
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except:
                            continue
        except Exception as e:
            print(f"    AFLOW extraction error: {e}")
        
        return entries[:max_entries]
    
    def parse_entry(self, entry: Dict) -> Dict:
        """Parse AFLOW entry to standard format."""
        result = {col: '' for col in CSV_COLUMNS}
        
        try:
            auid = entry.get('auid', entry.get('aurl', ''))
            
            spacegroup = entry.get('spacegroup_relax', entry.get('spacegroup', ''))
            result['space_group'] = spacegroup
            result['crystal_structure'] = self._classify_structure(spacegroup)
            
            if 'geometry' in entry:
                geom = entry['geometry']
                if isinstance(geom, list) and len(geom) >= 6:
                    result['lattice_a_angstrom'] = geom[0]
                    result['lattice_b_angstrom'] = geom[1]
                    result['lattice_c_angstrom'] = geom[2]
                    result['lattice_alpha_deg'] = geom[3]
                    result['lattice_beta_deg'] = geom[4]
                    result['lattice_gamma_deg'] = geom[5]
            
            if 'volume_cell' in entry:
                result['volume_angstrom3'] = entry['volume_cell']
            if 'density' in entry:
                result['density_g/cm3'] = entry['density']
            
            if 'Egap' in entry and entry['Egap']:
                result['band_gap_eV'] = entry['Egap']
                result['bandgap_eV'] = entry['Egap']
            
            if 'Bvoigt' in entry and entry['Bvoigt']:
                result['bulk_modulus_GPa'] = entry['Bvoigt']
            if 'Gvoigt' in entry and entry['Gvoigt']:
                result['shear_modulus_GPa'] = entry['Gvoigt']
            if 'poisson_ratio' in entry:
                result['poissons_ratio'] = entry['poisson_ratio']
            
            if 'enthalpy_formation_atom' in entry and entry['enthalpy_formation_atom']:
                result['std_enthalpy_kJ/mol'] = entry['enthalpy_formation_atom'] * 3 * 96.485
            
            result['reference'] = f"AFLOW:{auid}, Curtarolo et al. Comp. Mat. Sci. 58, 218 (2012), DOI:10.1016/j.commatsci.2012.02.005"
            result['notes'] = f"AFLOW DFT calculation"
            
        except Exception as e:
            print(f"    Error parsing AFLOW entry: {e}")
        
        return result
    
    def _classify_structure(self, spacegroup: str) -> str:
        """Classify SiO2 structure based on space group."""
        sg_map = {
            'P3121': 'alpha-quartz', 'P3221': 'alpha-quartz',
            'P6222': 'beta-quartz', 'P6422': 'beta-quartz',
            'P41212': 'alpha-cristobalite', 'P43212': 'alpha-cristobalite',
            'Fd3m': 'beta-cristobalite', 'Fd-3m': 'beta-cristobalite',
            'C2221': 'alpha-tridymite',
            'P63/mmc': 'beta-tridymite',
            'C2/c': 'coesite',
            'P42/mnm': 'stishovite',
            'Pbcn': 'seifertite'
        }
        
        for sg, struct in sg_map.items():
            if sg in str(spacegroup):
                return struct
        
        return 'SiO2_AFLOW'


class CODExtractor:
    """Extract SiO2 data from Crystallography Open Database."""
    
    SEARCH_URL = "https://www.crystallography.net/cod/result"
    CIF_URL = "https://www.crystallography.net/cod"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'SiO2-DataExtractor/1.0'})
    
    def search_sio2_entries(self, max_entries: int = 5000) -> List[Dict]:
        """Search for SiO2 entries in COD."""
        entries = []
        
        try:
            params = {
                'formula': 'O2 Si',
                'format': 'json'
            }
            
            response = self.session.get(self.SEARCH_URL, params=params, timeout=60)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list):
                        entries = data[:max_entries]
                except json.JSONDecodeError:
                    lines = response.text.strip().split('\n')
                    for line in lines[:max_entries]:
                        cod_id = line.strip()
                        if cod_id.isdigit():
                            entries.append({'file': cod_id})
        except Exception as e:
            print(f"    COD search error: {e}")
        
        detailed_entries = []
        for i, entry in enumerate(entries[:max_entries]):
            if i % 50 == 0 and i > 0:
                print(f"    Fetching COD entry {i}/{len(entries)}...")
            try:
                cod_id = entry.get('file', entry.get('cod_id', ''))
                if cod_id:
                    detail = self._fetch_entry_details(cod_id)
                    if detail:
                        detailed_entries.append(detail)
                time.sleep(0.1)
            except Exception:
                continue
        
        return detailed_entries
    
    def _fetch_entry_details(self, cod_id: str) -> Optional[Dict]:
        """Fetch detailed information for a COD entry."""
        try:
            url = f"{self.CIF_URL}/{cod_id}.json"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                data['cod_id'] = cod_id
                return data
        except Exception:
            pass
        return None
    
    def parse_entry(self, entry: Dict) -> Dict:
        """Parse COD entry to standard format."""
        result = {col: '' for col in CSV_COLUMNS}
        
        try:
            cod_id = entry.get('cod_id', entry.get('file', ''))
            
            result['lattice_a_angstrom'] = entry.get('a', '')
            result['lattice_b_angstrom'] = entry.get('b', '')
            result['lattice_c_angstrom'] = entry.get('c', '')
            result['lattice_alpha_deg'] = entry.get('alpha', 90)
            result['lattice_beta_deg'] = entry.get('beta', 90)
            result['lattice_gamma_deg'] = entry.get('gamma', 90)
            
            result['space_group'] = entry.get('sg', entry.get('spacegroup', ''))
            result['crystal_structure'] = self._classify_structure(result['space_group'])
            
            result['volume_angstrom3'] = entry.get('vol', '')
            result['Z_formula_units'] = entry.get('Z', '')
            
            if result['volume_angstrom3'] and result['Z_formula_units']:
                try:
                    vol = float(result['volume_angstrom3'])
                    z = int(result['Z_formula_units'])
                    molar_mass = 60.084
                    density = (z * molar_mass) / (vol * 0.6022)
                    result['density_g/cm3'] = round(density, 3)
                except:
                    pass
            
            authors = entry.get('authors', '')
            journal = entry.get('journal', '')
            year = entry.get('year', '')
            doi = entry.get('doi', '')
            
            ref_parts = [f"COD:{cod_id}"]
            if authors:
                ref_parts.append(authors)
            if journal:
                ref_parts.append(journal)
            if year:
                ref_parts.append(f"({year})")
            if doi:
                ref_parts.append(f"DOI:{doi}")
            
            result['reference'] = ', '.join(ref_parts)
            result['notes'] = 'X-ray crystallography'
            
        except Exception as e:
            print(f"    Error parsing COD entry: {e}")
        
        return result
    
    def _classify_structure(self, spacegroup: str) -> str:
        """Classify structure based on space group."""
        sg_map = {
            'P 3_1 2 1': 'alpha-quartz', 'P 3_2 2 1': 'alpha-quartz',
            'P3121': 'alpha-quartz', 'P3221': 'alpha-quartz',
            'P 6_2 2 2': 'beta-quartz', 'P6222': 'beta-quartz',
            'P 4_1 2_1 2': 'alpha-cristobalite', 'P41212': 'alpha-cristobalite',
            'F d -3 m': 'beta-cristobalite', 'Fd-3m': 'beta-cristobalite',
            'C 2 2 2_1': 'alpha-tridymite', 'C2221': 'alpha-tridymite',
            'P 6_3/m m c': 'beta-tridymite', 'P63/mmc': 'beta-tridymite',
            'C 2/c': 'coesite', 'C2/c': 'coesite',
            'P 4_2/m n m': 'stishovite', 'P42/mnm': 'stishovite',
            'P b c n': 'seifertite', 'Pbcn': 'seifertite'
        }
        
        for sg, struct in sg_map.items():
            if sg in str(spacegroup):
                return struct
        
        return 'SiO2_COD'


class OQMDExtractor:
    """Extract SiO2 data from Open Quantum Materials Database."""
    
    BASE_URL = "http://oqmd.org/oqmdapi"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'SiO2-DataExtractor/1.0'})
    
    def search_sio2_entries(self, max_entries: int = 5000) -> List[Dict]:
        """Search for SiO2 entries in OQMD."""
        entries = []
        
        try:
            url = f"{self.BASE_URL}/formationenergy"
            params = {
                'composition': 'SiO2',
                'limit': max_entries,
                'format': 'json'
            }
            
            response = self.session.get(url, params=params, timeout=120)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    entries = data['data']
                elif isinstance(data, list):
                    entries = data
        except Exception as e:
            print(f"    OQMD search error: {e}")
        
        return entries[:max_entries]
    
    def parse_entry(self, entry: Dict) -> Dict:
        """Parse OQMD entry to standard format."""
        result = {col: '' for col in CSV_COLUMNS}
        
        try:
            entry_id = entry.get('entry_id', entry.get('id', ''))
            
            if 'delta_e' in entry and entry['delta_e'] is not None:
                result['std_enthalpy_kJ/mol'] = entry['delta_e'] * 96.485
            
            if 'band_gap' in entry and entry['band_gap'] is not None:
                result['band_gap_eV'] = entry['band_gap']
                result['bandgap_eV'] = entry['band_gap']
            
            if 'volume' in entry and entry['volume']:
                result['volume_angstrom3'] = entry['volume']
            
            if 'spacegroup' in entry:
                result['space_group'] = entry['spacegroup']
                result['crystal_structure'] = self._classify_structure(entry['spacegroup'])
            else:
                result['crystal_structure'] = 'SiO2_OQMD'
            
            if 'unit_cell' in entry:
                uc = entry['unit_cell']
                if isinstance(uc, dict):
                    result['lattice_a_angstrom'] = uc.get('a', '')
                    result['lattice_b_angstrom'] = uc.get('b', '')
                    result['lattice_c_angstrom'] = uc.get('c', '')
                    result['lattice_alpha_deg'] = uc.get('alpha', '')
                    result['lattice_beta_deg'] = uc.get('beta', '')
                    result['lattice_gamma_deg'] = uc.get('gamma', '')
            
            result['reference'] = f"OQMD:{entry_id}, Saal et al. JOM 65, 1501 (2013), DOI:10.1007/s11837-013-0755-4"
            result['notes'] = 'OQMD DFT calculation'
            
        except Exception as e:
            print(f"    Error parsing OQMD entry: {e}")
        
        return result
    
    def _classify_structure(self, spacegroup: str) -> str:
        """Classify structure based on space group."""
        sg_map = {
            'P3121': 'alpha-quartz', 'P3221': 'alpha-quartz',
            'P6222': 'beta-quartz',
            'P41212': 'alpha-cristobalite',
            'Fd3m': 'beta-cristobalite', 'Fd-3m': 'beta-cristobalite',
            'C2/c': 'coesite',
            'P42/mnm': 'stishovite'
        }
        
        for sg, struct in sg_map.items():
            if sg in str(spacegroup):
                return struct
        
        return 'SiO2_OQMD'


class LiteratureDataExtractor:
    """Extract SiO2 data from published literature and handbooks."""
    
    def __init__(self):
        pass
    
    def get_handbook_data(self) -> List[Dict]:
        """Get SiO2 data from standard handbooks and review papers."""
        entries = []
        
        # CRC Handbook data
        crc_data = [
            {
                'crystal_structure': 'alpha-quartz',
                'density_g/cm3': 2.648,
                'melting_point_degC': 1713,
                'refractive_index_o': 1.5442,
                'refractive_index_e': 1.5533,
                'hardness_Mohs': 7.0,
                'reference': 'CRC Handbook of Chemistry and Physics, 97th ed., Haynes (2016-2017), DOI:10.1201/9781315380476',
                'notes': 'Room temperature values'
            },
            {
                'crystal_structure': 'alpha-quartz',
                'thermal_conductivity_W/(m*K)': 12.0,
                'specific_heat_J/(kg*K)': 740,
                'thermal_expansion_1e-6/K': 13.7,
                'reference': 'CRC Handbook of Chemistry and Physics, 97th ed., Haynes (2016-2017), DOI:10.1201/9781315380476',
                'notes': 'Thermal properties parallel to c-axis'
            },
            {
                'crystal_structure': 'alpha-quartz',
                'thermal_conductivity_W/(m*K)': 6.8,
                'thermal_expansion_1e-6/K': 7.5,
                'reference': 'CRC Handbook of Chemistry and Physics, 97th ed., Haynes (2016-2017), DOI:10.1201/9781315380476',
                'notes': 'Thermal properties perpendicular to c-axis'
            },
            {
                'crystal_structure': 'fused_silica',
                'density_g/cm3': 2.20,
                'melting_point_degC': 1713,
                'thermal_conductivity_W/(m*K)': 1.4,
                'specific_heat_J/(kg*K)': 730,
                'thermal_expansion_1e-6/K': 0.55,
                'refractive_index_o': 1.4585,
                'dielectric_constant': 3.8,
                'reference': 'CRC Handbook of Chemistry and Physics, 97th ed., Haynes (2016-2017), DOI:10.1201/9781315380476',
                'notes': 'Amorphous silica glass'
            },
            {
                'crystal_structure': 'stishovite',
                'density_g/cm3': 4.287,
                'hardness_Mohs': 9.5,
                'reference': 'CRC Handbook of Chemistry and Physics, 97th ed., Haynes (2016-2017), DOI:10.1201/9781315380476',
                'notes': 'High-pressure polymorph'
            },
            {
                'crystal_structure': 'coesite',
                'density_g/cm3': 2.911,
                'reference': 'CRC Handbook of Chemistry and Physics, 97th ed., Haynes (2016-2017), DOI:10.1201/9781315380476',
                'notes': 'High-pressure polymorph'
            },
            {
                'crystal_structure': 'alpha-cristobalite',
                'density_g/cm3': 2.32,
                'reference': 'CRC Handbook of Chemistry and Physics, 97th ed., Haynes (2016-2017), DOI:10.1201/9781315380476',
                'notes': 'Low-temperature cristobalite'
            },
            {
                'crystal_structure': 'beta-cristobalite',
                'density_g/cm3': 2.20,
                'reference': 'CRC Handbook of Chemistry and Physics, 97th ed., Haynes (2016-2017), DOI:10.1201/9781315380476',
                'notes': 'High-temperature cristobalite'
            },
            {
                'crystal_structure': 'alpha-tridymite',
                'density_g/cm3': 2.26,
                'reference': 'CRC Handbook of Chemistry and Physics, 97th ed., Haynes (2016-2017), DOI:10.1201/9781315380476',
                'notes': 'Low-temperature tridymite'
            }
        ]
        
        for data in crc_data:
            entry = {col: '' for col in CSV_COLUMNS}
            entry.update(data)
            entries.append(entry)
        
        # Landolt-Bornstein data
        lb_data = [
            {
                'crystal_structure': 'alpha-quartz',
                'lattice_a_angstrom': 4.9133,
                'lattice_c_angstrom': 5.4053,
                'lattice_alpha_deg': 90,
                'lattice_beta_deg': 90,
                'lattice_gamma_deg': 120,
                'space_group': 'P3121',
                'Z_formula_units': 3,
                'reference': 'Landolt-Bornstein III/29a, Every & McCurdy (1992), Springer',
                'notes': 'Room temperature X-ray diffraction'
            },
            {
                'crystal_structure': 'alpha-quartz',
                'elastic_C11_GPa': 87.26,
                'elastic_C12_GPa': 6.57,
                'elastic_C13_GPa': 11.95,
                'elastic_C14_GPa': -17.18,
                'elastic_C33_GPa': 105.8,
                'elastic_C44_GPa': 57.15,
                'elastic_C66_GPa': 40.35,
                'bulk_modulus_GPa': 37.67,
                'reference': 'Landolt-Bornstein III/29a, Every & McCurdy (1992), Springer',
                'notes': 'Mean experimental elastic constants'
            },
            {
                'crystal_structure': 'alpha-quartz',
                'dielectric_constant_parallel': 4.60,
                'dielectric_constant_perp': 4.52,
                'reference': 'Landolt-Bornstein, Mason AIP Handbook (1957)',
                'notes': 'Dielectric constants at 1 kHz'
            },
            {
                'crystal_structure': 'stishovite',
                'elastic_C11_GPa': 453,
                'elastic_C12_GPa': 211,
                'elastic_C13_GPa': 203,
                'elastic_C33_GPa': 776,
                'elastic_C44_GPa': 252,
                'elastic_C66_GPa': 302,
                'reference': 'Weidner et al. J Geophys Res 87:4740-4746 (1982), DOI:10.1029/JB087iB06p04740',
                'notes': 'Brillouin scattering measurement'
            }
        ]
        
        for data in lb_data:
            entry = {col: '' for col in CSV_COLUMNS}
            entry.update(data)
            entries.append(entry)
        
        # Research paper data
        paper_data = [
            # Wang et al. 2015
            {
                'crystal_structure': 'alpha-quartz',
                'density_g/cm3': 2.648,
                'bulk_modulus_GPa': 37.8,
                'elastic_C11_GPa': 86.6,
                'elastic_C12_GPa': 6.74,
                'elastic_C13_GPa': 12.4,
                'elastic_C14_GPa': 17.8,
                'elastic_C33_GPa': 106.4,
                'elastic_C44_GPa': 58.0,
                'elastic_C66_GPa': 40.3,
                'reference': 'Wang et al. Phys Chem Minerals 42:203-212 (2015), DOI:10.1007/s00269-014-0711-z',
                'notes': 'Elastic constants at ambient pressure'
            },
            {
                'crystal_structure': 'alpha-quartz',
                'density_g/cm3': 2.742,
                'bulk_modulus_GPa': 46.9,
                'elastic_C11_GPa': 90.3,
                'elastic_C12_GPa': 15.4,
                'elastic_C13_GPa': 23.7,
                'elastic_C14_GPa': 10.7,
                'elastic_C33_GPa': 122.3,
                'elastic_C44_GPa': 62.4,
                'elastic_C66_GPa': 37.5,
                'reference': 'Wang et al. Phys Chem Minerals 42:203-212 (2015), DOI:10.1007/s00269-014-0711-z',
                'notes': 'Elastic constants at 1.5 GPa'
            },
            {
                'crystal_structure': 'alpha-quartz',
                'density_g/cm3': 2.897,
                'elastic_C11_GPa': 103.4,
                'elastic_C12_GPa': 35.6,
                'elastic_C13_GPa': 38.9,
                'elastic_C14_GPa': 3.8,
                'elastic_C33_GPa': 160.1,
                'elastic_C44_GPa': 65.9,
                'elastic_C66_GPa': 33.9,
                'reference': 'Wang et al. Phys Chem Minerals 42:203-212 (2015), DOI:10.1007/s00269-014-0711-z',
                'notes': 'Elastic constants at 4.4 GPa'
            },
            # Ogi et al. 2006
            {
                'crystal_structure': 'alpha-quartz',
                'density_g/cm3': 2.6497,
                'bulk_modulus_GPa': 37.74,
                'elastic_C11_GPa': 87.17,
                'elastic_C12_GPa': 6.61,
                'elastic_C13_GPa': 12.02,
                'elastic_C14_GPa': -18.23,
                'elastic_C33_GPa': 105.80,
                'elastic_C44_GPa': 58.27,
                'elastic_C66_GPa': 40.28,
                'debye_temp_K': 563,
                'reference': 'Ogi et al. J Appl Phys 100:053511 (2006), DOI:10.1063/1.2335684',
                'notes': 'Resonant ultrasound spectroscopy measurement'
            },
            # Levien et al. 1980
            {
                'crystal_structure': 'alpha-quartz',
                'lattice_a_angstrom': 4.9133,
                'lattice_c_angstrom': 5.4053,
                'volume_angstrom3': 113.0,
                'Z_formula_units': 3,
                'space_group': 'P3121',
                'reference': 'Levien et al. Am Mineral 65:920-930 (1980)',
                'notes': 'Room temperature X-ray diffraction'
            },
            {
                'crystal_structure': 'alpha-quartz',
                'lattice_a_angstrom': 4.902,
                'lattice_c_angstrom': 5.400,
                'volume_angstrom3': 112.4,
                'Z_formula_units': 3,
                'space_group': 'P3121',
                'reference': 'Levien et al. Am Mineral 65:920-930 (1980)',
                'notes': 'At 2.07 GPa'
            },
            {
                'crystal_structure': 'alpha-quartz',
                'lattice_a_angstrom': 4.876,
                'lattice_c_angstrom': 5.364,
                'volume_angstrom3': 110.5,
                'Z_formula_units': 3,
                'space_group': 'P3121',
                'reference': 'Levien et al. Am Mineral 65:920-930 (1980)',
                'notes': 'At 5.58 GPa'
            },
            # Stishovite data
            {
                'crystal_structure': 'stishovite',
                'density_g/cm3': 4.287,
                'bulk_modulus_GPa': 313,
                'lattice_a_angstrom': 4.1773,
                'lattice_c_angstrom': 2.6654,
                'volume_angstrom3': 46.54,
                'Z_formula_units': 2,
                'space_group': 'P42/mnm',
                'reference': 'Yamanaka et al. Phys Chem Minerals 29:633-641 (2002), DOI:10.1007/s00269-002-0257-3',
                'notes': 'Rutile structure, 6-coordinate Si'
            },
            {
                'crystal_structure': 'stishovite',
                'density_g/cm3': 4.287,
                'elastic_C11_GPa': 455,
                'elastic_C12_GPa': 199,
                'elastic_C13_GPa': 192,
                'elastic_C33_GPa': 762,
                'elastic_C44_GPa': 258,
                'elastic_C66_GPa': 321,
                'bulk_modulus_GPa': 306,
                'reference': 'Jiang et al. Phys Earth Planet Inter 172:235-240 (2009), DOI:10.1016/j.pepi.2008.09.017',
                'notes': 'Brillouin scattering, experimental'
            },
            # Coesite data
            {
                'crystal_structure': 'coesite',
                'density_g/cm3': 2.911,
                'bulk_modulus_GPa': 97.5,
                'lattice_a_angstrom': 7.14,
                'lattice_b_angstrom': 12.38,
                'lattice_c_angstrom': 7.17,
                'lattice_beta_deg': 120.34,
                'volume_angstrom3': 548.0,
                'Z_formula_units': 16,
                'space_group': 'C2/c',
                'reference': 'Levien & Prewitt Am Mineral 66:324-333 (1981)',
                'notes': 'Monoclinic high-pressure phase'
            },
            # Seifertite data
            {
                'crystal_structure': 'seifertite',
                'density_g/cm3': 4.294,
                'bulk_modulus_GPa': 328,
                'lattice_a_angstrom': 4.097,
                'lattice_b_angstrom': 5.046,
                'lattice_c_angstrom': 4.495,
                'volume_angstrom3': 92.9,
                'Z_formula_units': 4,
                'space_group': 'Pbcn',
                'reference': 'Grocholski et al. Am Mineral 98:1420-1428 (2013), DOI:10.2138/am.2013.4409',
                'notes': 'Ultra-high pressure phase, >40 GPa formation'
            },
            # Malitson optical data
            {
                'crystal_structure': 'fused_silica',
                'density_g/cm3': 2.20,
                'refractive_index_o': 1.4585,
                'Sellmeier_B1': 0.6961663,
                'Sellmeier_B2': 0.4079426,
                'Sellmeier_B3': 0.8974794,
                'Sellmeier_C1_um2': 0.0684043,
                'Sellmeier_C2_um2': 0.1162414,
                'Sellmeier_C3_um2': 9.896161,
                'reference': 'Malitson J Opt Soc Am 55:1205-1209 (1965), DOI:10.1364/JOSA.55.001205',
                'notes': 'Sellmeier coefficients for fused silica'
            },
            # Fontanella dielectric data
            {
                'crystal_structure': 'alpha-quartz',
                'dielectric_constant_perp': 4.520,
                'dielectric_constant_parallel': 4.638,
                'reference': 'Fontanella et al. J Appl Phys 45:2852 (1974), DOI:10.1063/1.1663690',
                'notes': 'Dielectric constants at 1 kHz, 300K'
            },
            # Aerogel data
            {
                'crystal_structure': 'silica_aerogel',
                'density_g/cm3': 0.1,
                'thermal_conductivity_W/(m*K)': 0.015,
                'reference': 'Hrubesh J Non-Cryst Solids 225:335-342 (1998), DOI:10.1016/S0022-3093(98)00135-5',
                'notes': 'Thermal conductivity 13-20 mW/mK, porosity >90%'
            },
            {
                'crystal_structure': 'silica_aerogel',
                'density_g/cm3': 0.15,
                'thermal_conductivity_W/(m*K)': 0.02,
                'reference': 'Dorcheh & Abbasi J Mater Proc Tech 199:10-26 (2008), DOI:10.1016/j.jmatprotec.2007.10.060',
                'notes': 'BET 500-1000 m2/g, pore 5-100 nm'
            },
            # Mesoporous silica
            {
                'crystal_structure': 'mesoporous_silica_MCM41',
                'reference': 'Beck et al. J Am Chem Soc 114:10834 (1992), DOI:10.1021/ja00053a020',
                'notes': 'BET 1000-1200 m2/g, pore 2-3 nm, hexagonal'
            },
            {
                'crystal_structure': 'mesoporous_silica_SBA15',
                'reference': 'Zhao et al. Science 279:548-552 (1998), DOI:10.1126/science.279.5350.548',
                'notes': 'BET 600-1000 m2/g, pore 5-30 nm, hexagonal'
            },
            # Band gap data
            {
                'crystal_structure': 'alpha-quartz',
                'band_gap_eV': 8.9,
                'bandgap_eV': 8.9,
                'reference': 'Weinberg et al. J Appl Phys 50:5757 (1979), DOI:10.1063/1.326717',
                'notes': 'Optical band gap measurement'
            },
            {
                'crystal_structure': 'fused_silica',
                'band_gap_eV': 9.0,
                'bandgap_eV': 9.0,
                'reference': 'DiStefano & Eastman Solid State Commun 9:2259 (1971), DOI:10.1016/0038-1098(71)90643-0',
                'notes': 'Photoemission measurement'
            },
            # Thermodynamic data
            {
                'crystal_structure': 'alpha-quartz',
                'std_enthalpy_kJ/mol': -910.7,
                'std_entropy_J/(mol*K)': 41.46,
                'reference': 'NIST-JANAF Thermochemical Tables, Chase (1998), DOI:10.18434/T42S31',
                'notes': 'Standard thermodynamic data at 298.15 K'
            },
            {
                'crystal_structure': 'fused_silica',
                'std_enthalpy_kJ/mol': -903.5,
                'std_entropy_J/(mol*K)': 46.9,
                'reference': 'NIST-JANAF Thermochemical Tables, Chase (1998), DOI:10.18434/T42S31',
                'notes': 'Amorphous SiO2 at 298.15 K'
            }
        ]
        
        for data in paper_data:
            entry = {col: '' for col in CSV_COLUMNS}
            entry.update(data)
            entries.append(entry)
        
        return entries


class DataMerger:
    """Merge and validate collected data."""
    
    def __init__(self, original_csv_path: str):
        self.original_data = pd.read_csv(original_csv_path)
        self.all_data = []
        self.deduplicator = DataDeduplicator()
    
    def add_entries(self, entries: List[Dict], source: str = ""):
        """Add entries to the collection, checking for duplicates."""
        added = 0
        for entry in entries:
            if not self.deduplicator.is_duplicate(entry):
                self.all_data.append(entry)
                added += 1
        if source:
            print(f"    Added {added} unique entries from {source} (skipped {len(entries) - added} duplicates)")
    
    def merge_and_deduplicate(self) -> pd.DataFrame:
        """Merge all data and remove duplicates."""
        new_df = pd.DataFrame(self.all_data)
        
        for col in CSV_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = ''
        
        new_df = new_df[CSV_COLUMNS]
        combined = pd.concat([self.original_data, new_df], ignore_index=True)
        combined = combined.drop_duplicates()
        
        return combined
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data."""
        numeric_cols = [
            'density_g/cm3', 'melting_point_degC', 'thermal_conductivity_W/(m*K)',
            'youngs_modulus_GPa', 'bulk_modulus_GPa', 'band_gap_eV',
            'refractive_index_o', 'lattice_a_angstrom', 'lattice_c_angstrom'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        value_cols = [c for c in df.columns if c not in ['crystal_structure', 'notes', 'reference']]
        df = df.dropna(subset=value_cols, how='all')
        
        df = df[df['reference'].notna() & (df['reference'] != '')]
        
        return df


def main():
    """Main extraction workflow."""
    print("=" * 70)
    print("SiO2 Physical Properties Data Extraction")
    print("Real data only with proper literature citations")
    print("=" * 70)
    
    mp = MaterialsProjectExtractor()
    aflow = AFLOWExtractor()
    cod = CODExtractor()
    oqmd = OQMDExtractor()
    lit = LiteratureDataExtractor()
    
    original_csv = "/home/ubuntu/attachments/5c26f888-d2bd-45eb-9e2e-4ad49f84501a/sio2_properties.csv"
    
    merger = DataMerger(original_csv)
    
    print("\n[1/5] Extracting from Materials Project...")
    try:
        mp_entries = mp.search_sio2_entries(max_entries=5000)
        parsed_mp = [mp.parse_entry(e) for e in mp_entries if e]
        parsed_mp = [e for e in parsed_mp if e.get('reference')]
        merger.add_entries(parsed_mp, "Materials Project")
    except Exception as e:
        print(f"    Materials Project extraction failed: {e}")
    
    print("\n[2/5] Extracting from AFLOW database...")
    try:
        aflow_entries = aflow.search_sio2_entries(max_entries=5000)
        parsed_aflow = [aflow.parse_entry(e) for e in aflow_entries if e]
        parsed_aflow = [e for e in parsed_aflow if e.get('reference')]
        merger.add_entries(parsed_aflow, "AFLOW")
    except Exception as e:
        print(f"    AFLOW extraction failed: {e}")
    
    print("\n[3/5] Extracting from COD database...")
    try:
        cod_entries = cod.search_sio2_entries(max_entries=2000)
        parsed_cod = [cod.parse_entry(e) for e in cod_entries if e]
        parsed_cod = [e for e in parsed_cod if e.get('reference')]
        merger.add_entries(parsed_cod, "COD")
    except Exception as e:
        print(f"    COD extraction failed: {e}")
    
    print("\n[4/5] Extracting from OQMD database...")
    try:
        oqmd_entries = oqmd.search_sio2_entries(max_entries=5000)
        parsed_oqmd = [oqmd.parse_entry(e) for e in oqmd_entries if e]
        parsed_oqmd = [e for e in parsed_oqmd if e.get('reference')]
        merger.add_entries(parsed_oqmd, "OQMD")
    except Exception as e:
        print(f"    OQMD extraction failed: {e}")
    
    print("\n[5/5] Extracting from published literature...")
    try:
        lit_entries = lit.get_handbook_data()
        merger.add_entries(lit_entries, "Literature")
    except Exception as e:
        print(f"    Literature extraction failed: {e}")
    
    print("\n" + "=" * 70)
    print("Merging and validating data...")
    final_df = merger.merge_and_deduplicate()
    final_df = merger.validate_data(final_df)
    
    output_path = "/home/ubuntu/repos/machine-learning/sio2_properties_extended.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\nFinal dataset: {len(final_df)} entries (all with proper citations)")
    print(f"Saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("Summary by crystal structure:")
    print(final_df['crystal_structure'].value_counts().head(20))
    
    ref_count = final_df['reference'].notna().sum()
    print(f"\nEntries with references: {ref_count}/{len(final_df)} ({100*ref_count/len(final_df):.1f}%)")
    
    return final_df


if __name__ == "__main__":
    df = main()
