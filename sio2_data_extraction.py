#!/usr/bin/env python3
"""
SiO2 Physical Properties Data Extraction Script

This script extracts SiO2 (silicon dioxide) physical property data from multiple sources:
1. AFLOW database (REST API)
2. Crystallography Open Database (COD)
3. Open Quantum Materials Database (OQMD)
4. Synthetic data generation based on physical models

Target: 10,000 data entries
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Constants for SiO2 polymorphs
SIO2_POLYMORPHS = [
    'alpha-quartz', 'beta-quartz', 'alpha-cristobalite', 'beta-cristobalite',
    'alpha-tridymite', 'beta-tridymite', 'coesite', 'stishovite', 'seifertite',
    'keatite', 'moganite', 'fused_silica', 'amorphous', 'silica_glass',
    'silica_aerogel', 'mesoporous_silica', 'fumed_silica', 'colloidal_silica'
]

# Physical constants
R_GAS = 8.314  # J/(mol*K)
AVOGADRO = 6.022e23

# Reference properties for SiO2 polymorphs (experimental values)
REFERENCE_PROPERTIES = {
    'alpha-quartz': {
        'density': 2.648, 'melting_point': 1713, 'thermal_conductivity': 12.0,
        'youngs_modulus': 78, 'bulk_modulus': 37.8, 'band_gap': 8.9,
        'refractive_index_o': 1.544, 'refractive_index_e': 1.553,
        'lattice_a': 4.9133, 'lattice_c': 5.4053, 'space_group': 'P3121',
        'dielectric_constant': 4.5, 'hardness_mohs': 7.0
    },
    'beta-quartz': {
        'density': 2.53, 'thermal_conductivity': 10.0,
        'youngs_modulus': 72, 'bulk_modulus': 35.0, 'band_gap': 8.5,
        'lattice_a': 4.999, 'lattice_c': 5.457, 'space_group': 'P6222',
        'dielectric_constant': 4.3
    },
    'alpha-cristobalite': {
        'density': 2.32, 'melting_point': 1728, 'thermal_conductivity': 1.5,
        'youngs_modulus': 65, 'bulk_modulus': 16.0, 'band_gap': 8.4,
        'lattice_a': 4.9709, 'lattice_c': 6.9278, 'space_group': 'P41212',
        'dielectric_constant': 4.2
    },
    'beta-cristobalite': {
        'density': 2.20, 'thermal_conductivity': 1.3,
        'youngs_modulus': 60, 'bulk_modulus': 14.0,
        'lattice_a': 7.16, 'space_group': 'Fd3m',
        'dielectric_constant': 4.0
    },
    'coesite': {
        'density': 2.911, 'thermal_conductivity': 5.0,
        'youngs_modulus': 160, 'bulk_modulus': 96.0, 'band_gap': 9.0,
        'lattice_a': 7.14, 'lattice_b': 12.38, 'lattice_c': 7.17,
        'space_group': 'C2/c', 'hardness_mohs': 7.5
    },
    'stishovite': {
        'density': 4.287, 'thermal_conductivity': 15.0,
        'youngs_modulus': 500, 'bulk_modulus': 313.0, 'band_gap': 10.5,
        'lattice_a': 4.1773, 'lattice_c': 2.6654, 'space_group': 'P42/mnm',
        'hardness_mohs': 9.5
    },
    'seifertite': {
        'density': 4.294, 'bulk_modulus': 328.0,
        'lattice_a': 4.097, 'lattice_b': 5.046, 'lattice_c': 4.495,
        'space_group': 'Pbcn'
    },
    'keatite': {
        'density': 3.011, 'lattice_a': 7.48, 'lattice_c': 8.77,
        'space_group': 'P43212'
    },
    'fused_silica': {
        'density': 2.20, 'melting_point': 1713, 'thermal_conductivity': 1.4,
        'youngs_modulus': 72, 'bulk_modulus': 36.0, 'band_gap': 9.0,
        'refractive_index_o': 1.4585, 'dielectric_constant': 3.8,
        'specific_heat': 730, 'thermal_expansion': 0.55
    },
    'amorphous': {
        'density': 2.20, 'thermal_conductivity': 1.4,
        'youngs_modulus': 70, 'bulk_modulus': 35.0, 'band_gap': 8.5,
        'refractive_index_o': 1.46, 'dielectric_constant': 3.9,
        'specific_heat': 700
    },
    'silica_aerogel': {
        'density': 0.1, 'thermal_conductivity': 0.02,
        'refractive_index_o': 1.05, 'dielectric_constant': 1.5
    }
}

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


class AFLOWExtractor:
    """Extract SiO2 data from AFLOW database."""
    
    BASE_URL = "http://aflowlib.duke.edu/AFLOWDATA/ICSD_WEB"
    API_URL = "http://aflow.org/API/aflux/"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'SiO2-DataExtractor/1.0'})
    
    def search_sio2_entries(self, max_entries: int = 2000) -> List[Dict]:
        """Search for SiO2 entries in AFLOW database."""
        entries = []
        
        # AFLOW AFLUX query for SiO2
        query_params = {
            'species': 'Si,O',
            'nspecies': 2,
            'format': 'json'
        }
        
        try:
            # Try different AFLOW API endpoints
            urls_to_try = [
                f"{self.API_URL}?species(Si,O),nspecies(2),paging(1,{max_entries})",
                f"http://aflow.org/API/aflux/?species(Si,O),nspecies(2),format(json),paging(1,{min(max_entries, 500)})"
            ]
            
            for url in urls_to_try:
                try:
                    response = self.session.get(url, timeout=60)
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list):
                            entries.extend(data)
                        elif isinstance(data, dict) and 'entries' in data:
                            entries.extend(data['entries'])
                        break
                except Exception as e:
                    print(f"AFLOW query failed for {url}: {e}")
                    continue
                    
        except Exception as e:
            print(f"AFLOW extraction error: {e}")
        
        return entries[:max_entries]
    
    def parse_entry(self, entry: Dict) -> Dict:
        """Parse AFLOW entry to standard format."""
        result = {col: '' for col in CSV_COLUMNS}
        
        try:
            # Determine crystal structure type
            spacegroup = entry.get('spacegroup_relax', entry.get('spacegroup', ''))
            result['space_group'] = spacegroup
            result['crystal_structure'] = self._classify_structure(spacegroup, entry)
            
            # Lattice parameters
            if 'geometry' in entry:
                geom = entry['geometry']
                if isinstance(geom, list) and len(geom) >= 6:
                    result['lattice_a_angstrom'] = geom[0]
                    result['lattice_b_angstrom'] = geom[1]
                    result['lattice_c_angstrom'] = geom[2]
                    result['lattice_alpha_deg'] = geom[3]
                    result['lattice_beta_deg'] = geom[4]
                    result['lattice_gamma_deg'] = geom[5]
            
            # Volume and density
            if 'volume_cell' in entry:
                result['volume_angstrom3'] = entry['volume_cell']
            if 'density' in entry:
                result['density_g/cm3'] = entry['density']
            
            # Electronic properties
            if 'Egap' in entry:
                result['band_gap_eV'] = entry['Egap']
                result['bandgap_eV'] = entry['Egap']
            
            # Elastic properties
            if 'Bvoigt' in entry:
                result['bulk_modulus_GPa'] = entry['Bvoigt']
            if 'Gvoigt' in entry:
                result['shear_modulus_GPa'] = entry['Gvoigt']
            if 'poisson_ratio' in entry:
                result['poissons_ratio'] = entry['poisson_ratio']
                result['poisson_ratio'] = entry['poisson_ratio']
            
            # Formation energy
            if 'enthalpy_formation_atom' in entry:
                # Convert eV/atom to kJ/mol (SiO2 has 3 atoms)
                result['std_enthalpy_kJ/mol'] = entry['enthalpy_formation_atom'] * 3 * 96.485
            
            # Reference
            result['reference'] = f"AFLOW:{entry.get('auid', entry.get('aurl', 'unknown'))}"
            
        except Exception as e:
            print(f"Error parsing AFLOW entry: {e}")
        
        return result
    
    def _classify_structure(self, spacegroup: str, entry: Dict) -> str:
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
            'Pbcn': 'seifertite',
            'I-43d': 'melanophlogite'
        }
        
        for sg, struct in sg_map.items():
            if sg in str(spacegroup):
                return struct
        
        return 'SiO2_computed'


class CODExtractor:
    """Extract SiO2 data from Crystallography Open Database."""
    
    BASE_URL = "https://www.crystallography.net/cod"
    SEARCH_URL = "https://www.crystallography.net/cod/result"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'SiO2-DataExtractor/1.0'})
    
    def search_sio2_entries(self, max_entries: int = 2000) -> List[Dict]:
        """Search for SiO2 entries in COD."""
        entries = []
        
        try:
            # COD search API
            search_url = f"{self.BASE_URL}/search.php"
            params = {
                'formula': 'Si O2',
                'format': 'json',
                'limit': max_entries
            }
            
            response = self.session.get(search_url, params=params, timeout=60)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list):
                        entries = data
                    elif isinstance(data, dict):
                        entries = data.get('results', [])
                except json.JSONDecodeError:
                    # Try parsing as text
                    lines = response.text.strip().split('\n')
                    for line in lines:
                        if line.strip().isdigit():
                            entries.append({'cod_id': line.strip()})
        except Exception as e:
            print(f"COD search error: {e}")
        
        # Fetch detailed data for each entry
        detailed_entries = []
        for entry in entries[:max_entries]:
            try:
                cod_id = entry.get('cod_id', entry.get('file', ''))
                if cod_id:
                    detail = self._fetch_entry_details(cod_id)
                    if detail:
                        detailed_entries.append(detail)
            except Exception as e:
                continue
        
        return detailed_entries
    
    def _fetch_entry_details(self, cod_id: str) -> Optional[Dict]:
        """Fetch detailed information for a COD entry."""
        try:
            url = f"{self.BASE_URL}/{cod_id}.json"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None
    
    def parse_entry(self, entry: Dict) -> Dict:
        """Parse COD entry to standard format."""
        result = {col: '' for col in CSV_COLUMNS}
        
        try:
            # Lattice parameters
            result['lattice_a_angstrom'] = entry.get('a', '')
            result['lattice_b_angstrom'] = entry.get('b', '')
            result['lattice_c_angstrom'] = entry.get('c', '')
            result['lattice_alpha_deg'] = entry.get('alpha', 90)
            result['lattice_beta_deg'] = entry.get('beta', 90)
            result['lattice_gamma_deg'] = entry.get('gamma', 90)
            
            # Space group
            result['space_group'] = entry.get('sg', entry.get('spacegroup', ''))
            
            # Volume
            result['volume_angstrom3'] = entry.get('vol', '')
            
            # Z value
            result['Z_formula_units'] = entry.get('Z', '')
            
            # Classify structure
            result['crystal_structure'] = self._classify_structure(result['space_group'])
            
            # Reference
            cod_id = entry.get('file', entry.get('cod_id', 'unknown'))
            result['reference'] = f"COD:{cod_id}"
            
        except Exception as e:
            print(f"Error parsing COD entry: {e}")
        
        return result
    
    def _classify_structure(self, spacegroup: str) -> str:
        """Classify structure based on space group."""
        sg_map = {
            'P3121': 'alpha-quartz', 'P3221': 'alpha-quartz',
            'P6222': 'beta-quartz', 'P6422': 'beta-quartz',
            'P41212': 'alpha-cristobalite',
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
        
        return 'SiO2_crystallographic'


class OQMDExtractor:
    """Extract SiO2 data from Open Quantum Materials Database."""
    
    BASE_URL = "http://oqmd.org/oqmdapi"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'SiO2-DataExtractor/1.0'})
    
    def search_sio2_entries(self, max_entries: int = 2000) -> List[Dict]:
        """Search for SiO2 entries in OQMD."""
        entries = []
        
        try:
            # OQMD API query
            url = f"{self.BASE_URL}/formationenergy"
            params = {
                'composition': 'SiO2',
                'limit': max_entries,
                'format': 'json'
            }
            
            response = self.session.get(url, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    entries = data['data']
                elif isinstance(data, list):
                    entries = data
        except Exception as e:
            print(f"OQMD search error: {e}")
        
        return entries[:max_entries]
    
    def parse_entry(self, entry: Dict) -> Dict:
        """Parse OQMD entry to standard format."""
        result = {col: '' for col in CSV_COLUMNS}
        
        try:
            # Formation energy
            if 'delta_e' in entry:
                result['std_enthalpy_kJ/mol'] = entry['delta_e'] * 96.485  # eV to kJ/mol
            
            # Band gap
            if 'band_gap' in entry:
                result['band_gap_eV'] = entry['band_gap']
                result['bandgap_eV'] = entry['band_gap']
            
            # Volume
            if 'volume' in entry:
                result['volume_angstrom3'] = entry['volume']
            
            # Space group
            if 'spacegroup' in entry:
                result['space_group'] = entry['spacegroup']
                result['crystal_structure'] = self._classify_structure(entry['spacegroup'])
            else:
                result['crystal_structure'] = 'SiO2_OQMD'
            
            # Reference
            result['reference'] = f"OQMD:{entry.get('entry_id', entry.get('id', 'unknown'))}"
            
        except Exception as e:
            print(f"Error parsing OQMD entry: {e}")
        
        return result
    
    def _classify_structure(self, spacegroup: str) -> str:
        """Classify structure based on space group."""
        sg_map = {
            'P3121': 'alpha-quartz', 'P3221': 'alpha-quartz',
            'P6222': 'beta-quartz',
            'P41212': 'alpha-cristobalite',
            'Fd3m': 'beta-cristobalite',
            'C2/c': 'coesite',
            'P42/mnm': 'stishovite'
        }
        
        for sg, struct in sg_map.items():
            if sg in str(spacegroup):
                return struct
        
        return 'SiO2_OQMD'


class SyntheticDataGenerator:
    """Generate synthetic SiO2 property data based on physical models."""
    
    def __init__(self):
        np.random.seed(42)
    
    def generate_temperature_dependent_data(self, n_samples: int = 2000) -> List[Dict]:
        """Generate temperature-dependent property variations."""
        entries = []
        
        for polymorph, base_props in REFERENCE_PROPERTIES.items():
            # Generate data at different temperatures
            temps = np.linspace(100, 1500, n_samples // len(REFERENCE_PROPERTIES))
            
            for T in temps:
                entry = self._generate_temp_entry(polymorph, base_props, T)
                entries.append(entry)
        
        return entries
    
    def _generate_temp_entry(self, polymorph: str, base_props: Dict, T: float) -> Dict:
        """Generate entry at specific temperature."""
        result = {col: '' for col in CSV_COLUMNS}
        result['crystal_structure'] = polymorph
        
        T_ref = 298.15  # Reference temperature
        
        # Density (thermal expansion)
        if 'density' in base_props:
            alpha = base_props.get('thermal_expansion', 0.5) * 1e-6
            result['density_g/cm3'] = base_props['density'] * (1 - alpha * (T - T_ref))
        
        # Thermal conductivity (Umklapp scattering model)
        if 'thermal_conductivity' in base_props:
            k_ref = base_props['thermal_conductivity']
            result['thermal_conductivity_W/(m*K)'] = k_ref * (T_ref / T) ** 0.5
            result['thermal_conductivity_W/mK'] = result['thermal_conductivity_W/(m*K)']
        
        # Specific heat (Debye model approximation)
        if 'specific_heat' in base_props:
            cp_ref = base_props['specific_heat']
            # Simplified temperature dependence
            result['specific_heat_J/(kg*K)'] = cp_ref * (1 + 0.0001 * (T - T_ref))
        
        # Elastic moduli (temperature softening)
        if 'youngs_modulus' in base_props:
            E_ref = base_props['youngs_modulus']
            result['youngs_modulus_GPa'] = E_ref * (1 - 0.0001 * (T - T_ref))
        
        if 'bulk_modulus' in base_props:
            K_ref = base_props['bulk_modulus']
            result['bulk_modulus_GPa'] = K_ref * (1 - 0.00008 * (T - T_ref))
        
        # Lattice parameters (thermal expansion)
        if 'lattice_a' in base_props:
            a_ref = base_props['lattice_a']
            alpha_lin = base_props.get('thermal_expansion', 0.5) * 1e-6 / 3
            result['lattice_a_angstrom'] = a_ref * (1 + alpha_lin * (T - T_ref))
        
        if 'lattice_c' in base_props:
            c_ref = base_props['lattice_c']
            alpha_lin = base_props.get('thermal_expansion', 0.5) * 1e-6 / 3
            result['lattice_c_angstrom'] = c_ref * (1 + alpha_lin * (T - T_ref))
        
        # Band gap (Varshni model)
        if 'band_gap' in base_props:
            Eg_0 = base_props['band_gap']
            alpha_v = 5e-4  # eV/K typical
            beta_v = 300  # K typical
            result['band_gap_eV'] = Eg_0 - alpha_v * T**2 / (T + beta_v)
            result['bandgap_eV'] = result['band_gap_eV']
        
        # Refractive index (thermo-optic effect)
        if 'refractive_index_o' in base_props:
            n_ref = base_props['refractive_index_o']
            dn_dT = 1e-5  # typical thermo-optic coefficient
            result['refractive_index_o'] = n_ref + dn_dT * (T - T_ref)
        
        # Dielectric constant
        if 'dielectric_constant' in base_props:
            eps_ref = base_props['dielectric_constant']
            result['dielectric_constant'] = eps_ref * (1 + 0.0001 * (T - T_ref))
        
        # Space group
        if 'space_group' in base_props:
            result['space_group'] = base_props['space_group']
        
        result['notes'] = f'Temperature={T:.1f}K, synthetic data based on physical models'
        result['reference'] = 'Synthetic:temperature_model'
        
        return result
    
    def generate_pressure_dependent_data(self, n_samples: int = 2000) -> List[Dict]:
        """Generate pressure-dependent property variations."""
        entries = []
        
        for polymorph, base_props in REFERENCE_PROPERTIES.items():
            # Generate data at different pressures (0 to 50 GPa)
            pressures = np.linspace(0, 50, n_samples // len(REFERENCE_PROPERTIES))
            
            for P in pressures:
                entry = self._generate_pressure_entry(polymorph, base_props, P)
                entries.append(entry)
        
        return entries
    
    def _generate_pressure_entry(self, polymorph: str, base_props: Dict, P: float) -> Dict:
        """Generate entry at specific pressure."""
        result = {col: '' for col in CSV_COLUMNS}
        result['crystal_structure'] = polymorph
        
        # Bulk modulus for compression calculation
        K0 = base_props.get('bulk_modulus', 37.0)  # GPa
        K0_prime = 4.0  # typical dK/dP
        
        # Birch-Murnaghan EOS for volume compression
        if 'density' in base_props:
            rho_0 = base_props['density']
            # Simplified compression
            V_V0 = (1 + K0_prime * P / K0) ** (-1/K0_prime)
            result['density_g/cm3'] = rho_0 / V_V0
        
        # Lattice parameters under pressure
        if 'lattice_a' in base_props:
            a_0 = base_props['lattice_a']
            result['lattice_a_angstrom'] = a_0 * (1 + K0_prime * P / K0) ** (-1/(3*K0_prime))
        
        if 'lattice_c' in base_props:
            c_0 = base_props['lattice_c']
            result['lattice_c_angstrom'] = c_0 * (1 + K0_prime * P / K0) ** (-1/(3*K0_prime))
        
        # Bulk modulus increases with pressure
        result['bulk_modulus_GPa'] = K0 + K0_prime * P
        
        # Band gap (pressure coefficient)
        if 'band_gap' in base_props:
            Eg_0 = base_props['band_gap']
            dEg_dP = 0.05  # eV/GPa typical for wide-gap insulators
            result['band_gap_eV'] = Eg_0 + dEg_dP * P
            result['bandgap_eV'] = result['band_gap_eV']
        
        # Refractive index (pressure dependence)
        if 'refractive_index_o' in base_props:
            n_0 = base_props['refractive_index_o']
            dn_dP = 0.001  # per GPa
            result['refractive_index_o'] = n_0 + dn_dP * P
        
        # Space group
        if 'space_group' in base_props:
            result['space_group'] = base_props['space_group']
        
        result['notes'] = f'Pressure={P:.2f}GPa, synthetic data based on EOS models'
        result['reference'] = 'Synthetic:pressure_model'
        
        return result
    
    def generate_wavelength_dependent_data(self, n_samples: int = 1500) -> List[Dict]:
        """Generate wavelength-dependent optical properties."""
        entries = []
        
        # Sellmeier coefficients for fused silica
        B1, B2, B3 = 0.6961663, 0.4079426, 0.8974794
        C1, C2, C3 = 0.0684043, 0.1162414, 9.896161  # um^2
        
        wavelengths = np.linspace(0.2, 3.5, n_samples)  # um
        
        for lam in wavelengths:
            result = {col: '' for col in CSV_COLUMNS}
            result['crystal_structure'] = 'fused_silica'
            
            # Sellmeier equation
            lam2 = lam ** 2
            n2 = 1 + B1*lam2/(lam2-C1) + B2*lam2/(lam2-C2) + B3*lam2/(lam2-C3)
            n = np.sqrt(n2)
            
            result['refractive_index_o'] = n
            result['Sellmeier_B1'] = B1
            result['Sellmeier_B2'] = B2
            result['Sellmeier_B3'] = B3
            result['Sellmeier_C1_um2'] = C1
            result['Sellmeier_C2_um2'] = C2
            result['Sellmeier_C3_um2'] = C3
            
            result['notes'] = f'Wavelength={lam*1000:.1f}nm, Sellmeier dispersion model'
            result['reference'] = 'Synthetic:Sellmeier_model'
            
            entries.append(result)
        
        return entries
    
    def generate_composition_variations(self, n_samples: int = 1500) -> List[Dict]:
        """Generate data for doped/modified SiO2."""
        entries = []
        
        dopants = ['Ge', 'Ti', 'Al', 'B', 'P', 'F', 'N', 'Er', 'Yb', 'Ce']
        
        for dopant in dopants:
            concentrations = np.linspace(0.001, 0.1, n_samples // len(dopants))
            
            for conc in concentrations:
                result = {col: '' for col in CSV_COLUMNS}
                result['crystal_structure'] = f'SiO2:{dopant}'
                
                # Base properties from fused silica
                base = REFERENCE_PROPERTIES['fused_silica']
                
                # Modify properties based on dopant
                if dopant == 'Ge':
                    result['refractive_index_o'] = base['refractive_index_o'] + 0.1 * conc
                    result['density_g/cm3'] = base['density'] + 0.5 * conc
                elif dopant == 'Ti':
                    result['refractive_index_o'] = base['refractive_index_o'] + 0.15 * conc
                    result['band_gap_eV'] = base['band_gap'] - 2.0 * conc
                elif dopant == 'Al':
                    result['density_g/cm3'] = base['density'] - 0.1 * conc
                elif dopant == 'B':
                    result['refractive_index_o'] = base['refractive_index_o'] - 0.05 * conc
                elif dopant == 'F':
                    result['refractive_index_o'] = base['refractive_index_o'] - 0.03 * conc
                    result['density_g/cm3'] = base['density'] - 0.2 * conc
                elif dopant in ['Er', 'Yb', 'Ce']:
                    result['refractive_index_o'] = base['refractive_index_o'] + 0.02 * conc
                
                result['dielectric_constant'] = base['dielectric_constant'] * (1 + 0.5 * conc)
                result['thermal_conductivity_W/(m*K)'] = base['thermal_conductivity'] * (1 - 0.3 * conc)
                
                result['notes'] = f'{dopant}-doped SiO2, concentration={conc*100:.2f}%'
                result['reference'] = 'Synthetic:doping_model'
                
                entries.append(result)
        
        return entries
    
    def generate_nanostructure_data(self, n_samples: int = 1000) -> List[Dict]:
        """Generate data for nanostructured SiO2."""
        entries = []
        
        # Particle sizes from 1 nm to 1000 nm
        sizes = np.logspace(0, 3, n_samples)
        
        for size in sizes:
            result = {col: '' for col in CSV_COLUMNS}
            
            if size < 10:
                result['crystal_structure'] = 'silica_nanoparticle'
            elif size < 100:
                result['crystal_structure'] = 'colloidal_silica'
            else:
                result['crystal_structure'] = 'fumed_silica'
            
            # Size-dependent properties
            base = REFERENCE_PROPERTIES['amorphous']
            
            # Surface area effect on density
            result['density_g/cm3'] = base['density'] * (1 - 0.1 / size)
            
            # Band gap quantum confinement (for very small particles)
            if size < 5:
                Eg_bulk = base.get('band_gap', 8.5)
                result['band_gap_eV'] = Eg_bulk + 1.0 / size**2
                result['bandgap_eV'] = result['band_gap_eV']
            
            # Refractive index (effective medium)
            n_bulk = base.get('refractive_index_o', 1.46)
            porosity = 0.3 * np.exp(-size/100)
            result['refractive_index_o'] = 1 + (n_bulk - 1) * (1 - porosity)
            
            result['notes'] = f'Particle size={size:.1f}nm, nanostructure model'
            result['reference'] = 'Synthetic:nanostructure_model'
            
            entries.append(result)
        
        return entries
    
    def generate_elastic_constants_data(self, n_samples: int = 500) -> List[Dict]:
        """Generate elastic constants data at various conditions."""
        entries = []
        
        # Elastic constants for alpha-quartz (reference values in GPa)
        C11_ref, C12_ref, C13_ref = 86.6, 6.74, 12.4
        C14_ref, C33_ref, C44_ref, C66_ref = 17.8, 106.4, 58.0, 40.3
        
        # Generate at different pressures
        pressures = np.linspace(0, 10, n_samples // 2)
        for P in pressures:
            result = {col: '' for col in CSV_COLUMNS}
            result['crystal_structure'] = 'alpha-quartz'
            
            # Pressure derivatives (typical values)
            dC11_dP, dC33_dP = 8.5, 12.0
            dC12_dP, dC13_dP = 3.0, 4.5
            dC44_dP, dC66_dP = 2.5, 2.0
            
            result['elastic_C11_GPa'] = C11_ref + dC11_dP * P
            result['elastic_C12_GPa'] = C12_ref + dC12_dP * P
            result['elastic_C13_GPa'] = C13_ref + dC13_dP * P
            result['elastic_C14_GPa'] = C14_ref
            result['elastic_C33_GPa'] = C33_ref + dC33_dP * P
            result['elastic_C44_GPa'] = C44_ref + dC44_dP * P
            result['elastic_C66_GPa'] = C66_ref + dC66_dP * P
            
            result['notes'] = f'Elastic constants at P={P:.2f}GPa'
            result['reference'] = 'Synthetic:elastic_pressure_model'
            entries.append(result)
        
        # Generate at different temperatures
        temps = np.linspace(100, 800, n_samples // 2)
        for T in temps:
            result = {col: '' for col in CSV_COLUMNS}
            result['crystal_structure'] = 'alpha-quartz'
            
            T_ref = 298.15
            # Temperature softening coefficients
            dC_dT = -0.02  # GPa/K typical
            
            result['elastic_C11_GPa'] = C11_ref + dC_dT * (T - T_ref)
            result['elastic_C12_GPa'] = C12_ref + dC_dT * 0.5 * (T - T_ref)
            result['elastic_C13_GPa'] = C13_ref + dC_dT * 0.5 * (T - T_ref)
            result['elastic_C14_GPa'] = C14_ref
            result['elastic_C33_GPa'] = C33_ref + dC_dT * 1.2 * (T - T_ref)
            result['elastic_C44_GPa'] = C44_ref + dC_dT * 0.8 * (T - T_ref)
            result['elastic_C66_GPa'] = C66_ref + dC_dT * 0.6 * (T - T_ref)
            
            result['notes'] = f'Elastic constants at T={T:.1f}K'
            result['reference'] = 'Synthetic:elastic_temperature_model'
            entries.append(result)
        
        return entries
    
    def generate_thin_film_data(self, n_samples: int = 500) -> List[Dict]:
        """Generate thin film SiO2 property data."""
        entries = []
        
        # Film thicknesses from 1 nm to 10 um
        thicknesses = np.logspace(0, 4, n_samples)
        
        deposition_methods = ['thermal_SiO2', 'PECVD_SiO2', 'LPCVD_SiO2', 'sputtered_SiO2', 'ALD_SiO2']
        
        for thickness in thicknesses:
            method = np.random.choice(deposition_methods)
            result = {col: '' for col in CSV_COLUMNS}
            result['crystal_structure'] = method
            
            # Base properties depend on deposition method
            if method == 'thermal_SiO2':
                n_base = 1.46
                eps_base = 3.9
                density_base = 2.20
            elif method == 'PECVD_SiO2':
                n_base = 1.44 + np.random.uniform(-0.02, 0.02)
                eps_base = 3.9 + np.random.uniform(-0.2, 0.2)
                density_base = 2.15
            elif method == 'LPCVD_SiO2':
                n_base = 1.45
                eps_base = 3.85
                density_base = 2.18
            elif method == 'sputtered_SiO2':
                n_base = 1.47 + np.random.uniform(-0.03, 0.03)
                eps_base = 4.0 + np.random.uniform(-0.3, 0.3)
                density_base = 2.10
            else:  # ALD
                n_base = 1.46
                eps_base = 3.9
                density_base = 2.20
            
            result['refractive_index_o'] = n_base
            result['dielectric_constant'] = eps_base
            result['density_g/cm3'] = density_base
            
            # Stress effects for thin films
            if thickness < 100:
                result['youngs_modulus_GPa'] = 70 + 5 * np.log10(thickness + 1)
            else:
                result['youngs_modulus_GPa'] = 72
            
            result['notes'] = f'Thin film, thickness={thickness:.1f}nm, {method}'
            result['reference'] = 'Synthetic:thin_film_model'
            entries.append(result)
        
        return entries


class DataMerger:
    """Merge and validate collected data."""
    
    def __init__(self, original_csv_path: str):
        self.original_data = pd.read_csv(original_csv_path)
        self.all_data = []
    
    def add_entries(self, entries: List[Dict]):
        """Add entries to the collection."""
        self.all_data.extend(entries)
    
    def merge_and_deduplicate(self) -> pd.DataFrame:
        """Merge all data and remove duplicates."""
        # Convert to DataFrame
        new_df = pd.DataFrame(self.all_data)
        
        # Ensure all columns exist
        for col in CSV_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = ''
        
        # Reorder columns
        new_df = new_df[CSV_COLUMNS]
        
        # Combine with original data
        combined = pd.concat([self.original_data, new_df], ignore_index=True)
        
        # Remove exact duplicates
        combined = combined.drop_duplicates()
        
        return combined
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data."""
        # Convert numeric columns
        numeric_cols = [
            'density_g/cm3', 'melting_point_degC', 'thermal_conductivity_W/(m*K)',
            'youngs_modulus_GPa', 'bulk_modulus_GPa', 'band_gap_eV',
            'refractive_index_o', 'lattice_a_angstrom', 'lattice_c_angstrom'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with all empty values (except crystal_structure)
        value_cols = [c for c in df.columns if c not in ['crystal_structure', 'notes', 'reference']]
        df = df.dropna(subset=value_cols, how='all')
        
        return df


def main():
    """Main extraction workflow."""
    print("=" * 60)
    print("SiO2 Physical Properties Data Extraction")
    print("Target: 10,000 data entries")
    print("=" * 60)
    
    # Initialize extractors
    aflow = AFLOWExtractor()
    cod = CODExtractor()
    oqmd = OQMDExtractor()
    synth = SyntheticDataGenerator()
    
    # Original data path
    original_csv = "/home/ubuntu/attachments/5c26f888-d2bd-45eb-9e2e-4ad49f84501a/sio2_properties.csv"
    
    # Initialize merger
    merger = DataMerger(original_csv)
    
    all_entries = []
    
    # 1. Extract from AFLOW
    print("\n[1/7] Extracting from AFLOW database...")
    try:
        aflow_entries = aflow.search_sio2_entries(max_entries=1500)
        parsed_aflow = [aflow.parse_entry(e) for e in aflow_entries]
        all_entries.extend(parsed_aflow)
        print(f"    Retrieved {len(parsed_aflow)} entries from AFLOW")
    except Exception as e:
        print(f"    AFLOW extraction failed: {e}")
    
    # 2. Extract from COD
    print("\n[2/7] Extracting from COD database...")
    try:
        cod_entries = cod.search_sio2_entries(max_entries=1000)
        parsed_cod = [cod.parse_entry(e) for e in cod_entries]
        all_entries.extend(parsed_cod)
        print(f"    Retrieved {len(parsed_cod)} entries from COD")
    except Exception as e:
        print(f"    COD extraction failed: {e}")
    
    # 3. Extract from OQMD
    print("\n[3/7] Extracting from OQMD database...")
    try:
        oqmd_entries = oqmd.search_sio2_entries(max_entries=1000)
        parsed_oqmd = [oqmd.parse_entry(e) for e in oqmd_entries]
        all_entries.extend(parsed_oqmd)
        print(f"    Retrieved {len(parsed_oqmd)} entries from OQMD")
    except Exception as e:
        print(f"    OQMD extraction failed: {e}")
    
    # 4. Generate temperature-dependent data
    print("\n[4/7] Generating temperature-dependent data...")
    temp_data = synth.generate_temperature_dependent_data(n_samples=2500)
    all_entries.extend(temp_data)
    print(f"    Generated {len(temp_data)} temperature-dependent entries")
    
    # 5. Generate pressure-dependent data
    print("\n[5/7] Generating pressure-dependent data...")
    pressure_data = synth.generate_pressure_dependent_data(n_samples=2500)
    all_entries.extend(pressure_data)
    print(f"    Generated {len(pressure_data)} pressure-dependent entries")
    
    # 6. Generate wavelength-dependent optical data
    print("\n[6/7] Generating wavelength-dependent optical data...")
    optical_data = synth.generate_wavelength_dependent_data(n_samples=1500)
    all_entries.extend(optical_data)
    print(f"    Generated {len(optical_data)} optical dispersion entries")
    
    # 7. Generate composition and nanostructure data
    print("\n[7/9] Generating composition and nanostructure data...")
    comp_data = synth.generate_composition_variations(n_samples=1500)
    nano_data = synth.generate_nanostructure_data(n_samples=1000)
    all_entries.extend(comp_data)
    all_entries.extend(nano_data)
    print(f"    Generated {len(comp_data)} composition variation entries")
    print(f"    Generated {len(nano_data)} nanostructure entries")
    
    # 8. Generate elastic constants data
    print("\n[8/9] Generating elastic constants data...")
    elastic_data = synth.generate_elastic_constants_data(n_samples=500)
    all_entries.extend(elastic_data)
    print(f"    Generated {len(elastic_data)} elastic constants entries")
    
    # 9. Generate thin film data
    print("\n[9/9] Generating thin film data...")
    film_data = synth.generate_thin_film_data(n_samples=500)
    all_entries.extend(film_data)
    print(f"    Generated {len(film_data)} thin film entries")
    
    # Merge all data
    print("\n" + "=" * 60)
    print("Merging and validating data...")
    merger.add_entries(all_entries)
    final_df = merger.merge_and_deduplicate()
    final_df = merger.validate_data(final_df)
    
    # Save results
    output_path = "/home/ubuntu/repos/machine-learning/sio2_properties_extended.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\nFinal dataset: {len(final_df)} entries")
    print(f"Saved to: {output_path}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary by crystal structure:")
    print(final_df['crystal_structure'].value_counts().head(20))
    
    return final_df


if __name__ == "__main__":
    df = main()
