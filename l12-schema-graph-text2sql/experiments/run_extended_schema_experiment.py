#!/usr/bin/env python3
"""
Extended Schema Experiment:
Tests Schema Graph Text-to-SQL performance on 20-table complex schema.
Compares success rates with original 7-table schema experiments.
"""
import json
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from openai import OpenAI

DB_CONFIG = {
    'dbname': 'l12_materials',
    'user': 'l12_user',
    'password': 'l12_password',
    'host': 'localhost',
    'port': 5432
}

# Extended schema YAML for prompt injection (30 tables - matching OQMD real scale)
EXTENDED_SCHEMA_YAML = """
tables:
  material_entry:
    columns: [entry_id (PK, text), source_db, source_material_id, formula, reduced_formula, chemical_system, number_of_elements]
    
  composition:
    columns: [composition_id (PK), entry_id (FK→material_entry), element, atomic_fraction, site_label]
    
  structure:
    columns: [structure_id (PK), entry_id (FK→material_entry), prototype, strukturbericht, formula_type, space_group_number, crystal_system, lattice_a, lattice_b, lattice_c, volume_per_atom, space_group]
    
  phase_stability:
    columns: [stability_id (PK), entry_id (FK→material_entry), formation_energy_per_atom, energy_above_hull, is_stable, band_gap]
    
  calculation:
    columns: [calculation_id (PK, text), entry_id (FK→material_entry), method, functional, calculation_type]
    
  calculated_property:
    columns: [property_id (PK), calculation_id (FK→calculation), property_name, value, unit, tensor_component]
    
  prototype_definition:
    columns: [prototype_id (PK), prototype_name, strukturbericht, formula_type, description]
    
  element:
    columns: [element_id (PK), symbol (UNIQUE), name, atomic_number, atomic_mass, electronegativity, atomic_radius, group_number, period_number, block, category]
    
  element_property:
    columns: [element_property_id (PK), element_id (FK→element), property_name, value, unit, temperature_k, source]
    
  space_group:
    columns: [space_group_id (PK), space_group_number (UNIQUE), hermann_mauguin, crystal_system, point_group, laue_class, is_centrosymmetric]
    
  application_domain:
    columns: [domain_id (PK), domain_name, description, parent_domain_id (FK→application_domain, self-ref)]
    note: hierarchical structure, parent_domain_id references own table
    
  material_application:
    columns: [material_application_id (PK), entry_id (FK→material_entry), domain_id (FK→application_domain), relevance_score, notes]
    
  literature_reference:
    columns: [reference_id (PK), doi, title, authors, journal, year, volume, pages]
    
  material_reference:
    columns: [material_reference_id (PK), entry_id (FK→material_entry), reference_id (FK→literature_reference), context]
    
  experimental_measurement:
    columns: [measurement_id (PK), entry_id (FK→material_entry), reference_id (FK→literature_reference), method, temperature_k, pressure_gpa]
    
  measured_property:
    columns: [measured_property_id (PK), measurement_id (FK→experimental_measurement), property_name, value, uncertainty, unit]
    
  synthesis_method:
    columns: [synthesis_id (PK), method_name, category, description]
    
  material_synthesis:
    columns: [material_synthesis_id (PK), entry_id (FK→material_entry), synthesis_id (FK→synthesis_method), reference_id (FK→literature_reference), temperature_k, duration_hours, atmosphere, success]
    
  defect_type:
    columns: [defect_type_id (PK), defect_name, category, description]
    
  material_defect:
    columns: [material_defect_id (PK), entry_id (FK→material_entry), defect_type_id (FK→defect_type), formation_energy, concentration, site, dopant_element_id (FK→element)]

  band_structure:
    columns: [band_structure_id (PK), calculation_id (FK→calculation), entry_id (FK→material_entry), is_direct_gap, cbm_energy, vbm_energy, band_gap_type, num_bands, num_kpoints]

  density_of_states:
    columns: [dos_id (PK), calculation_id (FK→calculation), entry_id (FK→material_entry), total_dos_at_fermi, efermi, is_metallic, spin_polarized]

  elastic_tensor:
    columns: [elastic_id (PK), entry_id (FK→material_entry), calculation_id (FK→calculation), bulk_modulus_vrh, shear_modulus_vrh, youngs_modulus, poisson_ratio, is_stable]

  magnetic_property:
    columns: [magnetic_id (PK), entry_id (FK→material_entry), total_magnetization, magnetic_ordering, curie_temperature_k, magnetic_anisotropy_energy]

  thermal_property:
    columns: [thermal_id (PK), entry_id (FK→material_entry), calculation_id (FK→calculation), debye_temperature_k, thermal_conductivity, specific_heat_cv, gruneisen_parameter, temperature_k]

  surface_energy:
    columns: [surface_id (PK), entry_id (FK→material_entry), miller_index, surface_energy_j_m2, work_function, is_reconstructed]

  grain_boundary:
    columns: [grain_boundary_id (PK), entry_id (FK→material_entry), sigma_value, rotation_axis, tilt_angle, gb_energy_j_m2, excess_volume]

  phase_diagram_entry:
    columns: [phase_entry_id (PK), entry_id (FK→material_entry), chemical_system, is_on_hull, decomposition_products, hull_distance]

  alloy_system:
    columns: [alloy_system_id (PK), system_name, num_components, category, description]

  material_alloy_system:
    columns: [material_alloy_id (PK), entry_id (FK→material_entry), alloy_system_id (FK→alloy_system), phase, composition_type]

foreign_keys:
  - composition.entry_id → material_entry.entry_id
  - structure.entry_id → material_entry.entry_id
  - phase_stability.entry_id → material_entry.entry_id
  - calculation.entry_id → material_entry.entry_id
  - calculated_property.calculation_id → calculation.calculation_id
  - element_property.element_id → element.element_id
  - application_domain.parent_domain_id → application_domain.domain_id (self-ref)
  - material_application.entry_id → material_entry.entry_id
  - material_application.domain_id → application_domain.domain_id
  - material_reference.entry_id → material_entry.entry_id
  - material_reference.reference_id → literature_reference.reference_id
  - experimental_measurement.entry_id → material_entry.entry_id
  - experimental_measurement.reference_id → literature_reference.reference_id
  - measured_property.measurement_id → experimental_measurement.measurement_id
  - material_synthesis.entry_id → material_entry.entry_id
  - material_synthesis.synthesis_id → synthesis_method.synthesis_id
  - material_synthesis.reference_id → literature_reference.reference_id
  - material_defect.entry_id → material_entry.entry_id
  - material_defect.defect_type_id → defect_type.defect_type_id
  - material_defect.dopant_element_id → element.element_id
  - band_structure.calculation_id → calculation.calculation_id
  - band_structure.entry_id → material_entry.entry_id
  - density_of_states.calculation_id → calculation.calculation_id
  - density_of_states.entry_id → material_entry.entry_id
  - elastic_tensor.entry_id → material_entry.entry_id
  - elastic_tensor.calculation_id → calculation.calculation_id
  - magnetic_property.entry_id → material_entry.entry_id
  - thermal_property.entry_id → material_entry.entry_id
  - thermal_property.calculation_id → calculation.calculation_id
  - surface_energy.entry_id → material_entry.entry_id
  - grain_boundary.entry_id → material_entry.entry_id
  - phase_diagram_entry.entry_id → material_entry.entry_id
  - material_alloy_system.entry_id → material_entry.entry_id
  - material_alloy_system.alloy_system_id → alloy_system.alloy_system_id
"""

# Test queries categorized by JOIN complexity (150 queries: 25 per category)
EXTENDED_QUERIES = [
    # === Category 1: Simple (1-2 tables) — 25 queries ===
    {"id": "E001", "query": "B2プロトタイプの化合物を全て出して", "category": "simple", "min_tables": 1, "expected_tables": ["structure"]},
    {"id": "E002", "query": "Feを含む安定な化合物は？", "category": "simple", "min_tables": 3, "expected_tables": ["material_entry", "composition", "phase_stability"]},
    {"id": "E003", "query": "band_gapが2以上の化合物を出して", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "phase_stability"]},
    {"id": "E004", "query": "NaCl型でband_gapが0の金属的な化合物は？", "category": "simple", "min_tables": 3, "expected_tables": ["material_entry", "structure", "phase_stability"]},
    {"id": "E005", "query": "L12型でenergy_above_hullが0.01未満のものを出して", "category": "simple", "min_tables": 3, "expected_tables": ["material_entry", "structure", "phase_stability"]},
    {"id": "E006", "query": "Niを含む化合物の一覧を出して", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "composition"]},
    {"id": "E007", "query": "formation_energyが-1.0以下の化合物は？", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "phase_stability"]},
    {"id": "E008", "query": "fcc結晶系の化合物を全て出して", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "structure"]},
    {"id": "E009", "query": "AlとNiの両方を含む化合物は？", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "composition"]},
    {"id": "E010", "query": "磁気的に秩序化した化合物を出して", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "magnetic_property"]},
    {"id": "E011", "query": "direct band gapを持つ化合物は？", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "band_structure"]},
    {"id": "E012", "query": "bulk_modulusが200GPa以上の化合物を出して", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "elastic_tensor"]},
    {"id": "E013", "query": "デバイ温度が500K以上の化合物は？", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "thermal_property"]},
    {"id": "E014", "query": "表面エネルギーが2.0 J/m2以上の(111)面を持つ化合物は？", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "surface_energy"]},
    {"id": "E015", "query": "Sigma5粒界を持つ化合物は？", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "grain_boundary"]},
    {"id": "E016", "query": "convex hull上にある化合物を全て出して", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "phase_diagram_entry"]},
    {"id": "E017", "query": "二元系合金に分類される化合物は？", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "material_alloy_system"]},
    {"id": "E018", "query": "金属的な化合物のDOSをフェルミ面で出して", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "density_of_states"]},
    {"id": "E019", "query": "poisson_ratioが0.3以上の化合物は？", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "elastic_tensor"]},
    {"id": "E020", "query": "thermal_conductivityが100以上の高熱伝導化合物は？", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "thermal_property"]},
    {"id": "E021", "query": "NiAsプロトタイプの化合物のentry_idと化学式を出して", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "structure"]},
    {"id": "E022", "query": "BiF3型で安定な化合物は？", "category": "simple", "min_tables": 3, "expected_tables": ["material_entry", "structure", "phase_stability"]},
    {"id": "E023", "query": "Cuを含む化合物でband_gapが0のもの", "category": "simple", "min_tables": 3, "expected_tables": ["material_entry", "composition", "phase_stability"]},
    {"id": "E024", "query": "磁気異方性エネルギーが正の化合物を出して", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "magnetic_property"]},
    {"id": "E025", "query": "hull_distanceが0.05未満の化合物を出して", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "phase_diagram_entry"]},

    # === Category 2: Medium (3-4 tables, multi-hop) — 25 queries ===
    {"id": "E026", "query": "Aerospace Alloys用途に適した安定なB2化合物を探して", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "structure", "phase_stability", "material_application", "application_domain"]},
    {"id": "E027", "query": "Arc Meltingで合成されたNiを含む化合物の安定性は？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "composition", "phase_stability", "material_synthesis", "synthesis_method"]},
    {"id": "E028", "query": "XRDで測定された安定なL12化合物のlattice_parameterを出して", "category": "medium", "min_tables": 5, "expected_tables": ["material_entry", "structure", "phase_stability", "experimental_measurement", "measured_property"]},
    {"id": "E029", "query": "遷移金属(dブロック)を含む化合物で電池用途のものは？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "material_application", "application_domain"]},
    {"id": "E030", "query": "2020年以降に出版された論文で報告されたB2化合物は？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "structure", "material_reference", "literature_reference"]},
    {"id": "E031", "query": "direct band gapを持つ安定なNaCl型化合物は？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "band_structure", "phase_stability", "structure"]},
    {"id": "E032", "query": "弾性的に安定でbulk_modulusが高いB2化合物は？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "elastic_tensor", "structure"]},
    {"id": "E033", "query": "ferromagnetic秩序で安定な化合物の一覧", "category": "medium", "min_tables": 3, "expected_tables": ["material_entry", "magnetic_property", "phase_stability"]},
    {"id": "E034", "query": "高温(1000K以上)で合成されたL12化合物のデバイ温度は？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "material_synthesis", "structure", "thermal_property"]},
    {"id": "E035", "query": "表面エネルギーが低い安定な化合物は？", "category": "medium", "min_tables": 3, "expected_tables": ["material_entry", "surface_energy", "phase_stability"]},
    {"id": "E036", "query": "粒界エネルギーが高いB2化合物は？", "category": "medium", "min_tables": 3, "expected_tables": ["material_entry", "grain_boundary", "structure"]},
    {"id": "E037", "query": "convex hull上にあるL12合金系の化合物は？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "phase_diagram_entry", "material_alloy_system", "structure"]},
    {"id": "E038", "query": "Ni-Al合金系で安定な化合物の弾性定数は？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "material_alloy_system", "alloy_system", "elastic_tensor"]},
    {"id": "E039", "query": "DOSがメタリックでband_gapが0の化合物のformation_energyは？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "density_of_states", "phase_stability"]},
    {"id": "E040", "query": "電気陰性度が高い元素(2.0以上)を含むNaCl型化合物は？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "structure"]},
    {"id": "E041", "query": "Sputteringで合成された化合物のband構造でdirect gapのものは？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "material_synthesis", "synthesis_method", "band_structure"]},
    {"id": "E042", "query": "curie温度が500K以上の安定な化合物は？", "category": "medium", "min_tables": 3, "expected_tables": ["material_entry", "magnetic_property", "phase_stability"]},
    {"id": "E043", "query": "Thermoelectrics用途でデバイ温度が高い化合物は？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "material_application", "application_domain", "thermal_property"]},
    {"id": "E044", "query": "reconstructed表面を持つ安定な化合物は？", "category": "medium", "min_tables": 3, "expected_tables": ["material_entry", "surface_energy", "phase_stability"]},
    {"id": "E045", "query": "三元系合金で磁気的に秩序化したものは？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "material_alloy_system", "alloy_system", "magnetic_property"]},
    {"id": "E046", "query": "分解生成物がある不安定化合物のband_gapは？", "category": "medium", "min_tables": 3, "expected_tables": ["material_entry", "phase_diagram_entry", "phase_stability"]},
    {"id": "E047", "query": "spin_polarizedなDOS計算がある安定化合物は？", "category": "medium", "min_tables": 3, "expected_tables": ["material_entry", "density_of_states", "phase_stability"]},
    {"id": "E048", "query": "youngs_modulusが300GPa以上のB2化合物は？", "category": "medium", "min_tables": 3, "expected_tables": ["material_entry", "elastic_tensor", "structure"]},
    {"id": "E049", "query": "gruneisen_parameterが2以上の化合物で安定なものは？", "category": "medium", "min_tables": 3, "expected_tables": ["material_entry", "thermal_property", "phase_stability"]},
    {"id": "E050", "query": "work_functionが5eV以上の表面を持つNaCl型化合物は？", "category": "medium", "min_tables": 3, "expected_tables": ["material_entry", "surface_energy", "structure"]},

    # === Category 3: Complex (5+ tables, multi-hop chains) — 25 queries ===
    {"id": "E051", "query": "Vacancy欠陥を持つ安定なNaCl型化合物で、その構成元素の電気陰性度が2.0以上のものは？", "category": "complex", "min_tables": 6, "expected_tables": ["material_entry", "structure", "phase_stability", "material_defect", "defect_type", "composition", "element"]},
    {"id": "E052", "query": "XRDで測定されたhardnessが10GPa以上の化合物のうち、Arc Meltingで合成されたものは？", "category": "complex", "min_tables": 6, "expected_tables": ["material_entry", "experimental_measurement", "measured_property", "material_synthesis", "synthesis_method"]},
    {"id": "E053", "query": "Aerospace Alloysに適したB2化合物で、実験データがあるものの格子定数は？", "category": "complex", "min_tables": 6, "expected_tables": ["material_entry", "structure", "material_application", "application_domain", "experimental_measurement", "measured_property"]},
    {"id": "E054", "query": "Nature Materialsに掲載された化合物のうち、Vacancy形成エネルギーが1eV未満のものは？", "category": "complex", "min_tables": 5, "expected_tables": ["material_entry", "material_reference", "literature_reference", "material_defect", "defect_type"]},
    {"id": "E055", "query": "4族元素(Ti,Zr,Hf)を含むB2化合物で、触媒用途があり安定なものを出して", "category": "complex", "min_tables": 5, "expected_tables": ["material_entry", "composition", "element", "structure", "phase_stability", "material_application", "application_domain"]},
    {"id": "E056", "query": "direct band gapで弾性的に安定な化合物のうち、表面エネルギーが低いものは？", "category": "complex", "min_tables": 4, "expected_tables": ["material_entry", "band_structure", "elastic_tensor", "surface_energy"]},
    {"id": "E057", "query": "ferromagneticで安定な化合物のうち、デバイ温度が300K以上でbulk_modulusも高いものは？", "category": "complex", "min_tables": 5, "expected_tables": ["material_entry", "magnetic_property", "phase_stability", "thermal_property", "elastic_tensor"]},
    {"id": "E058", "query": "Ni-Al合金系でconvex hull上にあり、粒界エネルギーのデータもある化合物は？", "category": "complex", "min_tables": 5, "expected_tables": ["material_entry", "material_alloy_system", "alloy_system", "phase_diagram_entry", "grain_boundary"]},
    {"id": "E059", "query": "Ball Millingで合成されたferromagnetic化合物でcurie温度が高いものは？", "category": "complex", "min_tables": 5, "expected_tables": ["material_entry", "material_synthesis", "synthesis_method", "magnetic_property"]},
    {"id": "E060", "query": "pブロック元素を含むNaCl型化合物で、DOSがmetallic、かつ表面reconstructionありのものは？", "category": "complex", "min_tables": 6, "expected_tables": ["material_entry", "composition", "element", "structure", "density_of_states", "surface_energy"]},
    {"id": "E061", "query": "Thermoelectrics用途の化合物でband_gapが0.5以上、thermal_conductivityが低いものは？", "category": "complex", "min_tables": 5, "expected_tables": ["material_entry", "material_application", "application_domain", "band_structure", "thermal_property"]},
    {"id": "E062", "query": "2019年以降の論文で報告されたVacancy欠陥を持つ安定化合物は？", "category": "complex", "min_tables": 6, "expected_tables": ["material_entry", "material_reference", "literature_reference", "material_defect", "defect_type", "phase_stability"]},
    {"id": "E063", "query": "elastic_tensorがis_stable=trueでpoisson_ratioが低い化合物のうち、DFTで計算されたものは？", "category": "complex", "min_tables": 4, "expected_tables": ["material_entry", "elastic_tensor", "calculation"]},
    {"id": "E064", "query": "高温合成(1200K以上)された化合物でdirect band gap、かつconvex hull上のものは？", "category": "complex", "min_tables": 5, "expected_tables": ["material_entry", "material_synthesis", "band_structure", "phase_diagram_entry"]},
    {"id": "E065", "query": "Sigma3粒界を持つL12化合物で弾性的に安定なものは？", "category": "complex", "min_tables": 4, "expected_tables": ["material_entry", "grain_boundary", "structure", "elastic_tensor"]},
    {"id": "E066", "query": "alkali_metal元素を含む化合物でthermal_conductivityが高く、表面エネルギーが低いものは？", "category": "complex", "min_tables": 5, "expected_tables": ["material_entry", "composition", "element", "thermal_property", "surface_energy"]},
    {"id": "E067", "query": "DSCで測定された化合物のうち、磁気的に秩序化しており、かつconvex hull上のものは？", "category": "complex", "min_tables": 5, "expected_tables": ["material_entry", "experimental_measurement", "magnetic_property", "phase_diagram_entry"]},
    {"id": "E068", "query": "三元系合金で安定かつdirect band gapの化合物のうち、弾性テンソルデータがあるものは？", "category": "complex", "min_tables": 6, "expected_tables": ["material_entry", "material_alloy_system", "alloy_system", "phase_stability", "band_structure", "elastic_tensor"]},
    {"id": "E069", "query": "Substitutional欠陥を持ち、DOSがmetallicで、かつArc Meltingで合成された化合物は？", "category": "complex", "min_tables": 6, "expected_tables": ["material_entry", "material_defect", "defect_type", "density_of_states", "material_synthesis", "synthesis_method"]},
    {"id": "E070", "query": "curie温度が1000K以上でbulk_modulusが200GPa以上のferromagnetic化合物は？", "category": "complex", "min_tables": 4, "expected_tables": ["material_entry", "magnetic_property", "elastic_tensor"]},
    {"id": "E071", "query": "Energy Materials用途でconvex hull上の化合物のうち、粒界データがあるものは？", "category": "complex", "min_tables": 5, "expected_tables": ["material_entry", "material_application", "application_domain", "phase_diagram_entry", "grain_boundary"]},
    {"id": "E072", "query": "VASPで計算されたband構造のうち、band gapが間接的で弾性的に安定な化合物は？", "category": "complex", "min_tables": 4, "expected_tables": ["material_entry", "band_structure", "calculation", "elastic_tensor"]},
    {"id": "E073", "query": "2元素化合物でbulk_modulus/shear_modulus比(Pugh ratio)が高く安定なものは？", "category": "complex", "min_tables": 4, "expected_tables": ["material_entry", "elastic_tensor", "phase_stability"]},
    {"id": "E074", "query": "spin_polarized DOSでferromagneticかつNi含有化合物の表面エネルギーは？", "category": "complex", "min_tables": 5, "expected_tables": ["material_entry", "density_of_states", "magnetic_property", "composition", "surface_energy"]},
    {"id": "E075", "query": "Sputtering合成で粒界データと弾性データの両方がある化合物は？", "category": "complex", "min_tables": 5, "expected_tables": ["material_entry", "material_synthesis", "synthesis_method", "grain_boundary", "elastic_tensor"]},

    # === Category 4: Very Complex (self-ref, subqueries, aggregation) — 25 queries ===
    {"id": "E076", "query": "親カテゴリがEnergy Materialsであるすべてのサブカテゴリに属する化合物数を数えて", "category": "very_complex", "min_tables": 3, "expected_tables": ["application_domain", "material_application"]},
    {"id": "E077", "query": "ドーパント元素として使われている元素で電気陰性度が1.5以上のものを列挙して", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_defect", "element"]},
    {"id": "E078", "query": "3つ以上の異なるドメインに紐付けられた化合物のうち安定なものだけ出して", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_entry", "material_application", "phase_stability"]},
    {"id": "E079", "query": "同一化合物に対して実験値と計算値の両方が存在するものを出して", "category": "very_complex", "min_tables": 4, "expected_tables": ["material_entry", "calculated_property", "calculation", "experimental_measurement", "measured_property"]},
    {"id": "E080", "query": "vacuum雰囲気で1500K以上で合成された化合物のうちband_gapが正で論文引用があるものは？", "category": "very_complex", "min_tables": 6, "expected_tables": ["material_entry", "material_synthesis", "phase_stability", "material_reference", "literature_reference"]},
    {"id": "E081", "query": "band_structureとDOSの両方の計算データがある化合物は？", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_entry", "band_structure", "density_of_states"]},
    {"id": "E082", "query": "弾性テンソルと熱物性の両方のデータがある化合物でデバイ温度が最も高いものは？", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_entry", "elastic_tensor", "thermal_property"]},
    {"id": "E083", "query": "同一化合物に3種類以上の欠陥タイプが報告されているものは？", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_entry", "material_defect", "defect_type"]},
    {"id": "E084", "query": "磁気特性と弾性特性の両方がある化合物でbulk_modulusが最高のものTop5", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_entry", "magnetic_property", "elastic_tensor"]},
    {"id": "E085", "query": "表面エネルギーと粒界エネルギーの両方のデータがある化合物は？", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_entry", "surface_energy", "grain_boundary"]},
    {"id": "E086", "query": "2つ以上の合金系に属する化合物のうちconvex hull上のものは？", "category": "very_complex", "min_tables": 4, "expected_tables": ["material_entry", "material_alloy_system", "phase_diagram_entry"]},
    {"id": "E087", "query": "band_gapが0(metallic)でありながらthermal_conductivityが低い異常な化合物は？", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_entry", "band_structure", "thermal_property"]},
    {"id": "E088", "query": "同一計算IDでband_structureとDOSとelastic_tensorの3つが揃っている化合物は？", "category": "very_complex", "min_tables": 4, "expected_tables": ["material_entry", "band_structure", "density_of_states", "elastic_tensor", "calculation"]},
    {"id": "E089", "query": "formation_energyが最も低い化合物Top5とそのプロトタイプ", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_entry", "phase_stability", "structure"]},
    {"id": "E090", "query": "各合金系カテゴリごとの平均hull_distanceを計算して", "category": "very_complex", "min_tables": 4, "expected_tables": ["material_alloy_system", "alloy_system", "phase_diagram_entry"]},
    {"id": "E091", "query": "実験測定と計算の両方でlattice_parameterが報告されている化合物は？", "category": "very_complex", "min_tables": 5, "expected_tables": ["material_entry", "experimental_measurement", "measured_property", "calculated_property", "calculation"]},
    {"id": "E092", "query": "同一化合物で複数のmiller_indexの表面エネルギーが報告されているものは？", "category": "very_complex", "min_tables": 2, "expected_tables": ["material_entry", "surface_energy"]},
    {"id": "E093", "query": "phase_diagram上で分解生成物が3つ以上ある化合物は？", "category": "very_complex", "min_tables": 2, "expected_tables": ["material_entry", "phase_diagram_entry"]},
    {"id": "E094", "query": "磁気秩序がferromagneticかつantiferromagneticの両方の報告がある化合物は？", "category": "very_complex", "min_tables": 2, "expected_tables": ["material_entry", "magnetic_property"]},
    {"id": "E095", "query": "band_gapの計算値が0なのにDOSではnon-metallicとされている矛盾する化合物は？", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_entry", "band_structure", "density_of_states"]},
    {"id": "E096", "query": "全ての物性データ(elastic, magnetic, thermal, surface, grain_boundary)が揃っている化合物は？", "category": "very_complex", "min_tables": 6, "expected_tables": ["material_entry", "elastic_tensor", "magnetic_property", "thermal_property", "surface_energy", "grain_boundary"]},
    {"id": "E097", "query": "親ドメインと子ドメインの両方に紐付く化合物は？", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_entry", "material_application", "application_domain"]},
    {"id": "E098", "query": "同一元素をドーパントとして使っている化合物が3つ以上あるドーパント元素は？", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_defect", "element"]},
    {"id": "E099", "query": "最も多くの文献に引用されている化合物Top10は？", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_entry", "material_reference"]},
    {"id": "E100", "query": "thermal_conductivityとbulk_modulusの相関が見られる化合物のペアを出して", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_entry", "thermal_property", "elastic_tensor"]},

    # === Category 5: Cross-domain (element properties + material properties) — 25 queries ===
    {"id": "E101", "query": "原子番号が26以上30以下の元素を含む化合物でformation_energyが負のものは？", "category": "cross_domain", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "phase_stability"]},
    {"id": "E102", "query": "alkali_metalカテゴリの元素を含む化合物でThermoelectrics用途のものは？", "category": "cross_domain", "min_tables": 5, "expected_tables": ["material_entry", "composition", "element", "material_application", "application_domain"]},
    {"id": "E103", "query": "cubic結晶系の空間群に属する化合物のうちDSCで測定されたものを出して", "category": "cross_domain", "min_tables": 5, "expected_tables": ["material_entry", "structure", "space_group", "experimental_measurement"]},
    {"id": "E104", "query": "Substitutional欠陥のドーパント元素がpブロックの化合物で安定なものは？", "category": "cross_domain", "min_tables": 5, "expected_tables": ["material_entry", "material_defect", "defect_type", "element", "phase_stability"]},
    {"id": "E105", "query": "Ball Millingで合成されnanoindentationで測定されたhardnessデータがある化合物は？", "category": "cross_domain", "min_tables": 6, "expected_tables": ["material_entry", "material_synthesis", "synthesis_method", "experimental_measurement", "measured_property"]},
    {"id": "E106", "query": "dブロック元素を含み磁気的に秩序化した安定な化合物は？", "category": "cross_domain", "min_tables": 5, "expected_tables": ["material_entry", "composition", "element", "magnetic_property", "phase_stability"]},
    {"id": "E107", "query": "原子量が100以上の元素を含む化合物のbulk_modulusは？", "category": "cross_domain", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "elastic_tensor"]},
    {"id": "E108", "query": "電気陰性度差が大きい(max-min > 1.5)元素を含む化合物のband_gapは？", "category": "cross_domain", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "band_structure"]},
    {"id": "E109", "query": "fブロック元素(ランタノイド)を含む化合物でcurie温度が高いものは？", "category": "cross_domain", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "magnetic_property"]},
    {"id": "E110", "query": "atomic_radiusが大きい元素(200pm以上)を含むB2化合物で安定なものは？", "category": "cross_domain", "min_tables": 5, "expected_tables": ["material_entry", "composition", "element", "structure", "phase_stability"]},
    {"id": "E111", "query": "transition_metalを含む化合物でthermal_conductivityが高いものは？", "category": "cross_domain", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "thermal_property"]},
    {"id": "E112", "query": "Noble gas以外の元素のみで構成される安定化合物でband_gapが2以上のものは？", "category": "cross_domain", "min_tables": 5, "expected_tables": ["material_entry", "composition", "element", "phase_stability", "band_structure"]},
    {"id": "E113", "query": "dブロック元素のみで構成される化合物のpoisson_ratioの分布は？", "category": "cross_domain", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "elastic_tensor"]},
    {"id": "E114", "query": "5族元素(V,Nb,Ta)を含む化合物で表面エネルギーのデータがあるものは？", "category": "cross_domain", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "surface_energy"]},
    {"id": "E115", "query": "電気陰性度が1.0未満の元素を含む合金系化合物は？", "category": "cross_domain", "min_tables": 5, "expected_tables": ["material_entry", "composition", "element", "material_alloy_system"]},
    {"id": "E116", "query": "原子番号40以上の元素を含む化合物で粒界データがあるものは？", "category": "cross_domain", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "grain_boundary"]},
    {"id": "E117", "query": "dブロック元素を含む化合物のphase diagram上での安定性は？", "category": "cross_domain", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "phase_diagram_entry"]},
    {"id": "E118", "query": "alkali_metal含有化合物でdebye温度が400K以上かつ安定なものは？", "category": "cross_domain", "min_tables": 5, "expected_tables": ["material_entry", "composition", "element", "thermal_property", "phase_stability"]},
    {"id": "E119", "query": "原子番号が偶数の元素のみで構成される化合物のうちDOSがmetallicなものは？", "category": "cross_domain", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "density_of_states"]},
    {"id": "E120", "query": "3d遷移金属を含む化合物で磁気異方性エネルギーが高いものは？", "category": "cross_domain", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "magnetic_property"]},
    {"id": "E121", "query": "pブロック元素を含む化合物でthermal_conductivityが低く表面エネルギーも低いものは？", "category": "cross_domain", "min_tables": 5, "expected_tables": ["material_entry", "composition", "element", "thermal_property", "surface_energy"]},
    {"id": "E122", "query": "lanthanide元素を含む化合物のband構造でdirect gapのものは？", "category": "cross_domain", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "band_structure"]},
    {"id": "E123", "query": "電気陰性度2.5以上の元素を含む化合物で弾性的に安定なものは？", "category": "cross_domain", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "elastic_tensor"]},
    {"id": "E124", "query": "原子番号20以下の軽元素のみで構成される化合物のformation_energyは？", "category": "cross_domain", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "phase_stability"]},
    {"id": "E125", "query": "sブロック元素を含む化合物でconvex hull上かつ磁気秩序があるものは？", "category": "cross_domain", "min_tables": 5, "expected_tables": ["material_entry", "composition", "element", "phase_diagram_entry", "magnetic_property"]},

    # === Category 6: Aggregation & Comparison — 25 queries ===
    {"id": "E126", "query": "プロトタイプ別の平均formation_energyを出して", "category": "aggregation", "min_tables": 3, "expected_tables": ["material_entry", "structure", "phase_stability"]},
    {"id": "E127", "query": "合成方法ごとの化合物数を多い順に並べて", "category": "aggregation", "min_tables": 2, "expected_tables": ["material_synthesis", "synthesis_method"]},
    {"id": "E128", "query": "アプリケーションドメインごとの安定化合物数を多い順に並べて", "category": "aggregation", "min_tables": 3, "expected_tables": ["material_application", "application_domain", "phase_stability"]},
    {"id": "E129", "query": "各元素が含まれる化合物数のランキングTop10を出して", "category": "aggregation", "min_tables": 1, "expected_tables": ["composition"]},
    {"id": "E130", "query": "出版年ごとの論文数と紐付く化合物数を出して", "category": "aggregation", "min_tables": 3, "expected_tables": ["literature_reference", "material_reference"]},
    {"id": "E131", "query": "プロトタイプ別の平均bulk_modulusを計算して", "category": "aggregation", "min_tables": 3, "expected_tables": ["material_entry", "structure", "elastic_tensor"]},
    {"id": "E132", "query": "合金系カテゴリ別の化合物数は？", "category": "aggregation", "min_tables": 3, "expected_tables": ["material_alloy_system", "alloy_system"]},
    {"id": "E133", "query": "磁気秩序タイプ別の平均curie温度を出して", "category": "aggregation", "min_tables": 1, "expected_tables": ["magnetic_property"]},
    {"id": "E134", "query": "欠陥タイプ別の平均形成エネルギーを計算して", "category": "aggregation", "min_tables": 2, "expected_tables": ["material_defect", "defect_type"]},
    {"id": "E135", "query": "miller_index別の平均表面エネルギーを出して", "category": "aggregation", "min_tables": 1, "expected_tables": ["surface_energy"]},
    {"id": "E136", "query": "プロトタイプ別の平均デバイ温度の比較", "category": "aggregation", "min_tables": 3, "expected_tables": ["material_entry", "structure", "thermal_property"]},
    {"id": "E137", "query": "空間群ごとの化合物数を出して", "category": "aggregation", "min_tables": 2, "expected_tables": ["structure", "space_group"]},
    {"id": "E138", "query": "各化学系(binary, ternary等)に含まれる化合物数は？", "category": "aggregation", "min_tables": 1, "expected_tables": ["material_entry"]},
    {"id": "E139", "query": "band_gap_type(direct/indirect)ごとの化合物数は？", "category": "aggregation", "min_tables": 1, "expected_tables": ["band_structure"]},
    {"id": "E140", "query": "is_on_hull=trueとfalseの化合物数の比較", "category": "aggregation", "min_tables": 1, "expected_tables": ["phase_diagram_entry"]},
    {"id": "E141", "query": "合成温度帯別(500K未満, 500-1000K, 1000K以上)の化合物数は？", "category": "aggregation", "min_tables": 1, "expected_tables": ["material_synthesis"]},
    {"id": "E142", "query": "プロトタイプ別の安定化合物の割合を計算して", "category": "aggregation", "min_tables": 3, "expected_tables": ["material_entry", "structure", "phase_stability"]},
    {"id": "E143", "query": "元素カテゴリ別の平均電気陰性度は？", "category": "aggregation", "min_tables": 1, "expected_tables": ["element"]},
    {"id": "E144", "query": "各ドメインに紐付く化合物の平均band_gapは？", "category": "aggregation", "min_tables": 4, "expected_tables": ["material_application", "application_domain", "material_entry", "phase_stability"]},
    {"id": "E145", "query": "粒界のsigma値ごとの平均gb_energyを出して", "category": "aggregation", "min_tables": 1, "expected_tables": ["grain_boundary"]},
    {"id": "E146", "query": "合金系のnum_components別の平均formation_energyは？", "category": "aggregation", "min_tables": 4, "expected_tables": ["material_alloy_system", "alloy_system", "material_entry", "phase_stability"]},
    {"id": "E147", "query": "測定手法ごとの測定データ件数を出して", "category": "aggregation", "min_tables": 2, "expected_tables": ["experimental_measurement"]},
    {"id": "E148", "query": "安定/不安定化合物別の平均band_gap比較", "category": "aggregation", "min_tables": 3, "expected_tables": ["material_entry", "phase_stability", "band_structure"]},
    {"id": "E149", "query": "各プロトタイプのyoungs_modulus平均値をランキングで出して", "category": "aggregation", "min_tables": 3, "expected_tables": ["material_entry", "structure", "elastic_tensor"]},
    {"id": "E150", "query": "thermal_propertyのtemperature_k別のデータ件数の分布を出して", "category": "aggregation", "min_tables": 1, "expected_tables": ["thermal_property"]},

    # === Verification Gap Queries (①〜⑤: 検証不足パターンの補完) ===

    # --- ① 同名カラム多義性 (band_gap: phase_stability vs band_structure) ---
    {"id": "V001", "query": "band_gapが小さい化合物のバンド構造データを出して", "category": "ambiguity", "min_tables": 3, "expected_tables": ["material_entry", "band_structure", "phase_stability"]},
    {"id": "V002", "query": "band_gapが1eV以上の安定な化合物を出して", "category": "ambiguity", "min_tables": 2, "expected_tables": ["material_entry", "phase_stability"]},
    {"id": "V003", "query": "band_gapのDFT計算値と実験値を比較して", "category": "ambiguity", "min_tables": 4, "expected_tables": ["material_entry", "band_structure", "experimental_measurement", "measured_property"]},
    {"id": "V004", "query": "direct band gapを持つ化合物のphase_stabilityでのband_gapは？", "category": "ambiguity", "min_tables": 3, "expected_tables": ["material_entry", "band_structure", "phase_stability"]},
    {"id": "V005", "query": "band_gapが0の化合物のDOS（density_of_states）を出して", "category": "ambiguity", "min_tables": 3, "expected_tables": ["material_entry", "phase_stability", "density_of_states"]},
    {"id": "V006", "query": "band_structureのband_gap_typeがindirectで、phase_stabilityのband_gapが2以上の化合物", "category": "ambiguity", "min_tables": 3, "expected_tables": ["material_entry", "band_structure", "phase_stability"]},
    {"id": "V007", "query": "formation_energy_per_atomが低い化合物の弾性テンソルデータを出して", "category": "ambiguity", "min_tables": 3, "expected_tables": ["material_entry", "phase_stability", "elastic_tensor"]},
    {"id": "V008", "query": "space_group_numberが225の化合物のstructureとspace_group情報を両方出して", "category": "ambiguity", "min_tables": 3, "expected_tables": ["material_entry", "structure", "space_group"]},
    {"id": "V009", "query": "is_stableがtrueの化合物のバンド構造でdirect gapのものだけ出して", "category": "ambiguity", "min_tables": 3, "expected_tables": ["material_entry", "phase_stability", "band_structure"]},
    {"id": "V010", "query": "thermal_conductivityが高い化合物のband_gapをband_structureから出して", "category": "ambiguity", "min_tables": 3, "expected_tables": ["material_entry", "thermal_property", "band_structure"]},
    {"id": "V011", "query": "磁気秩序がferromagneticな化合物のband_gap（phase_stability）は？", "category": "ambiguity", "min_tables": 3, "expected_tables": ["material_entry", "magnetic_property", "phase_stability"]},
    {"id": "V012", "query": "band_structureとphase_stabilityの両方にデータがある化合物は何件？", "category": "ambiguity", "min_tables": 3, "expected_tables": ["material_entry", "band_structure", "phase_stability"]},
    {"id": "V013", "query": "元素のelectronegativityが高い化合物のband_gap分布を出して", "category": "ambiguity", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "phase_stability"]},
    {"id": "V014", "query": "surface_energyが低い化合物のband構造でdirect gapのものは？", "category": "ambiguity", "min_tables": 3, "expected_tables": ["material_entry", "surface_energy", "band_structure"]},
    {"id": "V015", "query": "bulk_modulusが高くband_gapが大きい化合物（両方のテーブルから）", "category": "ambiguity", "min_tables": 3, "expected_tables": ["material_entry", "elastic_tensor", "phase_stability"]},

    # --- ② 自己参照FK (application_domain.parent_domain_id → domain_id) ---
    {"id": "V016", "query": "Energy Materialsとそのサブカテゴリに属する化合物を全て出して", "category": "self_ref", "min_tables": 3, "expected_tables": ["material_entry", "material_application", "application_domain"]},
    {"id": "V017", "query": "親カテゴリがNullの最上位ドメイン一覧を出して", "category": "self_ref", "min_tables": 1, "expected_tables": ["application_domain"]},
    {"id": "V018", "query": "サブカテゴリを持つドメインとその子カテゴリ数を出して", "category": "self_ref", "min_tables": 1, "expected_tables": ["application_domain"]},
    {"id": "V019", "query": "Aerospace Alloysの親ドメインに属する化合物も含めて全て出して", "category": "self_ref", "min_tables": 3, "expected_tables": ["material_entry", "material_application", "application_domain"]},
    {"id": "V020", "query": "ドメイン階層の深さ（親→子→孫）を持つカテゴリを出して", "category": "self_ref", "min_tables": 1, "expected_tables": ["application_domain"]},

    # --- ③ isolated table（FK未接続: prototype_definition, space_group） ---
    {"id": "V021", "query": "strukturberichtがB2の原型定義の詳細を出して", "category": "isolated", "min_tables": 1, "expected_tables": ["prototype_definition"]},
    {"id": "V022", "query": "空間群221の詳細情報を出して", "category": "isolated", "min_tables": 1, "expected_tables": ["space_group"]},
    {"id": "V023", "query": "prototype_definitionに登録されているプロトタイプ名の一覧を出して", "category": "isolated", "min_tables": 1, "expected_tables": ["prototype_definition"]},
    {"id": "V024", "query": "空間群番号が200以上の空間群名を全て出して", "category": "isolated", "min_tables": 1, "expected_tables": ["space_group"]},
    {"id": "V025", "query": "cubic結晶系に属する空間群の一覧を出して", "category": "isolated", "min_tables": 1, "expected_tables": ["space_group"]},

    # --- ④ Traversal追加すぎ（余分なテーブル追加で型不一致等を誘発） ---
    {"id": "V026", "query": "structureテーブルのspace_group_number別に化合物数を集計して", "category": "over_traversal", "min_tables": 1, "expected_tables": ["structure"]},
    {"id": "V027", "query": "structureのcrystal_system別のlattice_a平均値を出して", "category": "over_traversal", "min_tables": 1, "expected_tables": ["structure"]},
    {"id": "V028", "query": "composition.elementのみでFeを含む化合物数を数えて", "category": "over_traversal", "min_tables": 1, "expected_tables": ["composition"]},
    {"id": "V029", "query": "phase_stabilityテーブルのis_stableがtrueの件数を出して", "category": "over_traversal", "min_tables": 1, "expected_tables": ["phase_stability"]},
    {"id": "V030", "query": "calculationテーブルのsoftware_name別のデータ件数を出して", "category": "over_traversal", "min_tables": 1, "expected_tables": ["calculation"]},

    # --- ⑤ aggregation + Traversal効果の検証 ---
    {"id": "V031", "query": "全テーブルのデータ件数をテーブル別に出して", "category": "agg_traversal", "min_tables": 1, "expected_tables": ["material_entry"]},
    {"id": "V032", "query": "合金系ごとに、その系に属する化合物のformation_energy平均と弾性定数平均を出して", "category": "agg_traversal", "min_tables": 5, "expected_tables": ["material_entry", "material_alloy_system", "alloy_system", "phase_stability", "elastic_tensor"]},
    {"id": "V033", "query": "各合成手法ごとの安定化合物の平均band_gapと化合物数を出して", "category": "agg_traversal", "min_tables": 4, "expected_tables": ["material_entry", "material_synthesis", "synthesis_method", "phase_stability"]},
    {"id": "V034", "query": "元素カテゴリ別に、含有化合物のbulk_modulus平均とformation_energy平均を出して", "category": "agg_traversal", "min_tables": 5, "expected_tables": ["material_entry", "composition", "element", "elastic_tensor", "phase_stability"]},
    {"id": "V035", "query": "用途ドメイン別の化合物数と平均thermal_conductivityを集計して", "category": "agg_traversal", "min_tables": 4, "expected_tables": ["material_entry", "material_application", "application_domain", "thermal_property"]},

    # --- ⑥ 意味的誤り耐性クエリ（テーブル削除で条件消失→rows爆増を検出） ---
    {"id": "V036", "query": "磁気的に秩序化した三元系合金を出して", "category": "semantic_trap", "min_tables": 3, "expected_tables": ["material_entry", "composition", "magnetic_property"]},
    {"id": "V037", "query": "band_gapが2eV以上でかつ弾性的に安定な化合物を出して", "category": "semantic_trap", "min_tables": 3, "expected_tables": ["material_entry", "phase_stability", "elastic_tensor"]},
    {"id": "V038", "query": "表面エネルギーが低くかつ熱伝導率が高い化合物を出して", "category": "semantic_trap", "min_tables": 3, "expected_tables": ["material_entry", "surface_energy", "thermal_property"]},
    {"id": "V039", "query": "粒界エネルギーが高いfcc結晶系の化合物を出して", "category": "semantic_trap", "min_tables": 3, "expected_tables": ["material_entry", "grain_boundary", "structure"]},
    {"id": "V040", "query": "DOSデータがあり、かつis_stableがtrueの化合物を出して", "category": "semantic_trap", "min_tables": 3, "expected_tables": ["material_entry", "density_of_states", "phase_stability"]},
    {"id": "V041", "query": "欠陥タイプがvacancyでかつbulk_modulusが高い化合物を出して", "category": "semantic_trap", "min_tables": 4, "expected_tables": ["material_entry", "material_defect", "defect_type", "elastic_tensor"]},
    {"id": "V042", "query": "文献引用のあるエネルギー材料（Energy Materials）を出して", "category": "semantic_trap", "min_tables": 5, "expected_tables": ["material_entry", "material_reference", "literature_reference", "material_application", "application_domain"]},
    {"id": "V043", "query": "実験的に測定されたband_gapとDFT計算のband_gapを持つ化合物を出して", "category": "semantic_trap", "min_tables": 4, "expected_tables": ["material_entry", "experimental_measurement", "measured_property", "phase_stability"]},
    {"id": "V044", "query": "Aerospace Alloys用途かつ高温でthermal_conductivityデータのある化合物を出して", "category": "semantic_trap", "min_tables": 4, "expected_tables": ["material_entry", "material_application", "application_domain", "thermal_property"]},
    {"id": "V045", "query": "相図データのある合金系でformation_energyが最も低い化合物を出して", "category": "semantic_trap", "min_tables": 5, "expected_tables": ["material_entry", "material_alloy_system", "alloy_system", "phase_diagram_entry", "phase_stability"]},
]


def build_schema_subset(tables: list) -> str:
    """Build a minimal schema YAML containing only the specified tables and their FK relations."""
    import re
    lines = EXTENDED_SCHEMA_YAML.strip().split('\n')
    
    # Parse table definitions
    table_defs = {}
    current_table = None
    for line in lines:
        m = re.match(r'^  (\w+):', line)
        if m and 'columns:' not in line and 'note:' not in line and 'foreign_keys:' not in line:
            current_table = m.group(1)
            table_defs[current_table] = []
        if current_table:
            table_defs[current_table].append(line)
        if line.strip() == '' and current_table:
            current_table = None
    
    # Build subset
    subset = "tables:\n"
    for t in tables:
        if t in table_defs:
            for l in table_defs[t]:
                subset += l + '\n'
            subset += '\n'
    
    # Add relevant FK lines
    subset += "foreign_keys:\n"
    fk_section = False
    for line in lines:
        if 'foreign_keys:' in line:
            fk_section = True
            continue
        if fk_section and line.strip().startswith('-'):
            # Check if both sides of FK reference tables in our subset
            parts = line.strip('- ').split('→')
            if len(parts) == 2:
                left_table = parts[0].strip().split('.')[0].strip('- ')
                right_table = parts[1].strip().split('.')[0].strip()
                # Remove trailing annotations like " (self-ref)"
                right_table = right_table.split(' ')[0].split('(')[0].strip()
                if left_table in tables and right_table in tables:
                    subset += line + '\n'
    
    return subset


def build_prompt(query: str, schema_mode: str = "full", relevant_tables: list = None) -> str:
    """Build the LLM prompt for SQL generation.
    
    schema_mode:
      - "full": Include all 30 tables (simulates: schema in prompt, no traversal)
      - "traversed": Include only relevant tables (simulates: Schema Graph traversal)
      - "none": No schema information (baseline)
    """
    base = f"""あなたは材料科学データベースのSQL生成エンジンです。
ユーザーの自然言語クエリを、PostgreSQL SQLに変換してください。

重要なルール:
- SELECT文のみ生成すること（INSERT/UPDATE/DELETE禁止）
- テーブル名・カラム名は正確に使用すること
- JOINは必要最小限のテーブルのみ使用すること
- 結果は最大100件に制限すること（LIMIT 100）
"""
    if schema_mode == "full":
        base += f"\n=== データベーススキーマ（全30テーブル） ===\n{EXTENDED_SCHEMA_YAML}\n"
    elif schema_mode == "traversed" and relevant_tables:
        subset = build_schema_subset(relevant_tables)
        base += f"\n=== データベーススキーマ（関連テーブルのみ） ===\n{subset}\n"
    # "none" mode: no schema
    
    base += f"\n=== ユーザークエリ ===\n{query}\n\n生成SQL:"
    return base


def execute_sql(sql: str) -> dict:
    """Execute SQL and return results or error."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description] if cur.description else []
        cur.close()
        return {"success": True, "rows": len(rows), "columns": cols}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if conn is not None:
            conn.close()


def count_joins_in_sql(sql: str) -> int:
    """Count the number of JOIN clauses in generated SQL."""
    import re
    return len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))


def count_tables_in_sql(sql: str) -> list:
    """Extract table names referenced in SQL (FROM and JOIN clauses)."""
    import re
    tables = set()
    # FROM clause
    for m in re.finditer(r'\bFROM\s+(\w+)', sql, re.IGNORECASE):
        tables.add(m.group(1).lower())
    # JOIN clause
    for m in re.finditer(r'\bJOIN\s+(\w+)', sql, re.IGNORECASE):
        tables.add(m.group(1).lower())
    return sorted(tables)


def run_llm_query(query: str, model: str = "gpt-5.5", schema_mode: str = "full",
                  relevant_tables: list = None) -> dict:
    """Generate SQL via LLM and execute."""
    client = OpenAI()
    prompt = build_prompt(query, schema_mode, relevant_tables)
    
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000,
        )
        sql = response.choices[0].message.content.strip()
        # Clean SQL
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[1] if "\n" in sql else sql[3:]
            sql = sql.rsplit("```", 1)[0]
        sql = sql.strip()
    except Exception as e:
        return {"success": False, "error": f"LLM error: {e}", "latency_ms": int((time.time()-start)*1000)}
    
    latency = int((time.time() - start) * 1000)
    
    # Execute the generated SQL
    result = execute_sql(sql)
    result["sql"] = sql
    result["latency_ms"] = latency
    result["model"] = model
    result["schema_mode"] = schema_mode
    result["join_count"] = count_joins_in_sql(sql)
    result["tables_used"] = count_tables_in_sql(sql)
    
    return result


def run_experiment(quick=False):
    """Run the full extended schema experiment with 3 conditions:
    1. LLM + Full Schema (30 tables in prompt, no traversal)
    2. LLM + Traversed Schema (only relevant tables, simulating SG traversal)
    3. LLM without Schema (baseline)

    Args:
        quick: If True, run only medium+complex (50 queries) for fast validation.
    """
    queries = EXTENDED_QUERIES
    if quick:
        queries = [q for q in EXTENDED_QUERIES if q["category"] in ("medium", "complex")]
        print("=" * 60)
        print(f"QUICK MODE: medium+complex only ({len(queries)} queries)")
        print("=" * 60)
    else:
        print("=" * 60)
        print("EXTENDED SCHEMA EXPERIMENT (30 tables)")
        print("=" * 60)
    print("Conditions: Full Schema (30t) | Traversed (subset) | No Schema")
    print("=" * 60)
    
    results = []
    
    for q in queries:
        print(f"\n[{q['id']}] {q['query'][:60]}...")
        
        # Condition 1: LLM + Full Schema (all 30 tables, no traversal)
        r1 = run_llm_query(q["query"], model="gpt-5.5", schema_mode="full")
        r1["condition"] = "llm_full_schema"
        r1["query_id"] = q["id"]
        r1["query_text"] = q["query"]
        r1["category"] = q["category"]
        r1["min_tables"] = q["min_tables"]
        r1["expected_tables"] = q["expected_tables"]
        print(f"  Full(30t):  {'✓' if r1['success'] else '✗'} JOINs={r1.get('join_count',0)} ({r1['latency_ms']}ms)")
        if not r1['success']:
            print(f"    Error: {r1.get('error', '')[:80]}")
        
        # Condition 2: LLM + Traversed Schema (only relevant tables)
        r2 = run_llm_query(q["query"], model="gpt-5.5", schema_mode="traversed",
                          relevant_tables=q["expected_tables"])
        r2["condition"] = "llm_traversed"
        r2["query_id"] = q["id"]
        r2["query_text"] = q["query"]
        r2["category"] = q["category"]
        r2["min_tables"] = q["min_tables"]
        r2["expected_tables"] = q["expected_tables"]
        print(f"  Traversed:  {'✓' if r2['success'] else '✗'} JOINs={r2.get('join_count',0)} ({r2['latency_ms']}ms)")
        if not r2['success']:
            print(f"    Error: {r2.get('error', '')[:80]}")
        
        # Condition 3: LLM without schema (baseline)
        r3 = run_llm_query(q["query"], model="gpt-5.5", schema_mode="none")
        r3["condition"] = "llm_no_schema"
        r3["query_id"] = q["id"]
        r3["query_text"] = q["query"]
        r3["category"] = q["category"]
        r3["min_tables"] = q["min_tables"]
        r3["expected_tables"] = q["expected_tables"]
        print(f"  No-Schema:  {'✓' if r3['success'] else '✗'} ({r3['latency_ms']}ms)")
        if not r3['success']:
            print(f"    Error: {r3.get('error', '')[:80]}")
        
        # Compute unnecessary JOINs for Full schema condition
        expected_join_count = max(0, len(q["expected_tables"]) - 1)
        r1["unnecessary_joins"] = max(0, r1.get("join_count", 0) - expected_join_count)
        r2["unnecessary_joins"] = max(0, r2.get("join_count", 0) - expected_join_count)
        
        results.append({
            "query": q,
            "llm_full_schema": r1,
            "llm_traversed": r2,
            "llm_no_schema": r3
        })
    
    # Save results
    output_path = Path(__file__).parent / "results" / "extended_schema_experiment.json"
    output_path.parent.mkdir(exist_ok=True)
    
    # Make JSON serializable
    for r in results:
        for key in ['llm_full_schema', 'llm_traversed', 'llm_no_schema']:
            if 'rows' not in r[key]:
                r[key]['rows'] = 0
            if 'columns' not in r[key]:
                r[key]['columns'] = []
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (30 tables)")
    print("=" * 60)
    
    categories = ['simple', 'medium', 'complex', 'very_complex', 'cross_domain', 'aggregation']
    for cat in categories:
        cat_results = [r for r in results if r['query']['category'] == cat]
        full_ok = sum(1 for r in cat_results if r['llm_full_schema']['success'])
        trav_ok = sum(1 for r in cat_results if r['llm_traversed']['success'])
        no_ok = sum(1 for r in cat_results if r['llm_no_schema']['success'])
        total = len(cat_results)
        print(f"  {cat:15s}: Full={full_ok}/{total} ({100*full_ok/total:.0f}%)  Traversed={trav_ok}/{total} ({100*trav_ok/total:.0f}%)  No-Schema={no_ok}/{total} ({100*no_ok/total:.0f}%)")
    
    total = len(results)
    full_total = sum(1 for r in results if r['llm_full_schema']['success'])
    trav_total = sum(1 for r in results if r['llm_traversed']['success'])
    no_total = sum(1 for r in results if r['llm_no_schema']['success'])
    print(f"\n  {'TOTAL':15s}: Full={full_total}/{total} ({100*full_total/total:.0f}%)  Traversed={trav_total}/{total} ({100*trav_total/total:.0f}%)  No-Schema={no_total}/{total} ({100*no_total/total:.0f}%)")
    
    # JOIN analysis
    print("\n" + "=" * 60)
    print("JOIN ANALYSIS (Traversal Effect)")
    print("=" * 60)
    full_joins = [r['llm_full_schema'].get('join_count', 0) for r in results if r['llm_full_schema']['success']]
    trav_joins = [r['llm_traversed'].get('join_count', 0) for r in results if r['llm_traversed']['success']]
    full_unnecessary = [r['llm_full_schema'].get('unnecessary_joins', 0) for r in results if r['llm_full_schema']['success']]
    trav_unnecessary = [r['llm_traversed'].get('unnecessary_joins', 0) for r in results if r['llm_traversed']['success']]
    
    if full_joins:
        print(f"  Full Schema avg JOINs:     {sum(full_joins)/len(full_joins):.1f}")
        print(f"  Traversed avg JOINs:       {sum(trav_joins)/len(trav_joins):.1f}" if trav_joins else "  Traversed: N/A")
        full_with_unnecessary = sum(1 for u in full_unnecessary if u > 0)
        trav_with_unnecessary = sum(1 for u in trav_unnecessary if u > 0)
        print(f"  Full: queries with unnecessary JOINs: {full_with_unnecessary}/{len(full_unnecessary)} ({100*full_with_unnecessary/len(full_unnecessary):.0f}%)")
        if trav_unnecessary:
            print(f"  Traversed: queries with unnecessary JOINs: {trav_with_unnecessary}/{len(trav_unnecessary)} ({100*trav_with_unnecessary/len(trav_unnecessary):.0f}%)")
        print(f"  Full avg unnecessary JOINs:    {sum(full_unnecessary)/len(full_unnecessary):.2f}")
        if trav_unnecessary:
            print(f"  Traversed avg unnecessary JOINs: {sum(trav_unnecessary)/len(trav_unnecessary):.2f}")
    
    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == '__main__':
    quick_mode = '--quick' in sys.argv
    run_experiment(quick=quick_mode)
