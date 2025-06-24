# Comprehensive Materials Engineering Scenarios
# 30+ scenarios across 8 material categories

# Scenario data structure and functions
get_scenarios_by_category <- function(category) {
  scenarios <- list(
    alloy = list(
      "Steel Alloy Optimization" = "steel_alloy",
      "Aluminum Alloy Design" = "aluminum_alloy", 
      "Titanium Alloy Processing" = "titanium_alloy",
      "Superalloy Development" = "superalloy",
      "Steel Transfer Learning" = "steel_transfer_learning",
      "Aluminum Transfer Learning" = "aluminum_transfer_learning",
      "Titanium Transfer Learning" = "titanium_transfer_learning",
      "Superalloy Transfer Learning" = "superalloy_transfer_learning"
    ),
    ceramic = list(
      "Alumina Ceramic Sintering" = "alumina_ceramic",
      "Silicon Carbide Processing" = "silicon_carbide",
      "Zirconia Toughening" = "zirconia_toughening",
      "Piezoelectric Ceramic Design" = "piezoelectric_ceramic"
    ),
    polymer = list(
      "Thermoplastic Processing" = "thermoplastic",
      "Thermoset Curing" = "thermoset",
      "Polymer Blend Optimization" = "polymer_blend",
      "Biodegradable Polymer Design" = "biodegradable_polymer",
      "SMILES-based Properties" = "polymer_smiles_properties",
      "SMILES Degradation Analysis" = "polymer_degradation_smiles",
      "SMILES Mechanical Properties" = "polymer_mechanical_smiles",
      "SMILES Solubility Prediction" = "polymer_solubility_smiles"
    ),
    alloy = list(
      "Steel Alloy Optimization" = "steel_alloy",
      "Aluminum Alloy Design" = "aluminum_alloy",
      "Titanium Alloy Processing" = "titanium_alloy",
      "Superalloy Development" = "superalloy",
      "Steel Transfer Learning" = "steel_transfer_learning",
      "Aluminum Transfer Learning" = "aluminum_transfer_learning", 
      "Titanium Transfer Learning" = "titanium_transfer_learning",
      "Superalloy Transfer Learning" = "superalloy_transfer_learning",
    ),
    composite = list(
      "Carbon Fiber Composites" = "carbon_fiber",
      "Glass Fiber Reinforcement" = "glass_fiber",
      "Natural Fiber Composites" = "natural_fiber",
      "Metal Matrix Composites" = "metal_matrix"
    ),
    nano = list(
      "Nanoparticle Synthesis" = "nanoparticle",
      "Carbon Nanotube Processing" = "carbon_nanotube",
      "Graphene Applications" = "graphene",
      "Quantum Dot Design" = "quantum_dot"
    ),
    bio = list(
      "Biocompatible Implants" = "biocompatible_implants",
      "Drug Delivery Systems" = "drug_delivery",
      "Tissue Engineering Scaffolds" = "tissue_scaffolds",
      "Dental Materials" = "dental_materials"
    ),
    electronic = list(
      "Semiconductor Processing" = "semiconductor",
      "Conductive Polymers" = "conductive_polymers",
      "Dielectric Materials" = "dielectric",
      "Magnetic Materials" = "magnetic"
    ),
    energy = list(
      "Battery Electrode Materials" = "battery_electrode",
      "Solar Cell Optimization" = "solar_cell",
      "Fuel Cell Catalysts" = "fuel_cell",
      "Thermoelectric Materials" = "thermoelectric"
    )
  )
  
  return(scenarios[[category]])
}

load_scenario_data <- function(category, scenario) {
  scenario_info <- get_scenario_info(category, scenario)
  scenario_data <- generate_scenario_data(category, scenario)
  
  list(
    description = scenario_info$description,
    variables = scenario_info$variables,
    relationships = scenario_info$relationships,
    data = scenario_data
  )
}

get_scenario_info <- function(category, scenario) {
  scenarios_info <- list(
    # Alloy Development Scenarios
    steel_alloy = list(
      description = "Optimization of steel alloy composition and processing parameters to achieve desired mechanical properties. This scenario focuses on the relationship between carbon content, alloying elements, heat treatment, and resulting strength and toughness.",
      variables = c(
        "Carbon Content (%): Primary strengthening element in steel",
        "Manganese Content (%): Improves hardenability and strength", 
        "Heat Treatment Temperature (°C): Affects microstructure formation",
        "Cooling Rate (°C/min): Controls phase transformation kinetics",
        "Tensile Strength (MPa): Target mechanical property"
      ),
      relationships = c(
        "Higher carbon content increases strength but reduces ductility",
        "Manganese addition improves hardenability and impact toughness",
        "Optimal heat treatment temperature depends on alloy composition",
        "Faster cooling rates promote martensitic transformation"
      )
    ),
    
    aluminum_alloy = list(
      description = "Design of aluminum alloys for aerospace applications, focusing on the balance between strength, weight, and corrosion resistance. Precipitation hardening and solution treatment effects are key considerations.",
      variables = c(
        "Copper Content (%): Primary strengthening element",
        "Magnesium Content (%): Forms strengthening precipitates",
        "Solution Treatment Temperature (°C): Dissolves alloying elements",
        "Aging Time (hours): Controls precipitate formation",
        "Yield Strength (MPa): Target mechanical property"
      ),
      relationships = c(
        "Cu-Mg precipitates provide primary strengthening mechanism",
        "Solution treatment temperature must be below solidus",
        "Aging time controls precipitate size and distribution",
        "Over-aging reduces strength due to precipitate coarsening"
      )
    ),
    
    titanium_alloy = list(
      description = "Processing optimization for titanium alloys used in biomedical implants, focusing on biocompatibility, mechanical properties, and corrosion resistance in physiological environments.",
      variables = c(
        "Vanadium Content (%): Beta stabilizer element",
        "Aluminum Content (%): Alpha stabilizer element", 
        "Processing Temperature (°C): Affects microstructure",
        "Oxygen Content (ppm): Interstitial strengthening element",
        "Elastic Modulus (GPa): Target property for bone matching"
      ),
      relationships = c(
        "Alpha-beta balance controls mechanical properties",
        "Lower elastic modulus reduces stress shielding",
        "Oxygen content significantly affects ductility",
        "Processing temperature controls grain size and texture"
      )
    ),
    
    superalloy = list(
      description = "Development of nickel-based superalloys for high-temperature turbine applications, focusing on creep resistance, oxidation resistance, and thermal stability at elevated temperatures.",
      variables = c(
        "Chromium Content (%): Oxidation resistance",
        "Aluminum Content (%): Forms protective oxide scale",
        "Service Temperature (°C): Operating condition",
        "Stress Level (MPa): Applied mechanical load",
        "Creep Rate (1/s): Target performance metric"
      ),
      relationships = c(
        "Cr content must exceed 15% for oxidation resistance",
        "Al forms protective alumina scale at high temperatures",
        "Creep rate increases exponentially with temperature",
        "Gamma-prime precipitates provide high-temperature strength"
      )
    ),
    
    # Ceramic Processing Scenarios
    alumina_ceramic = list(
      description = "Sintering optimization of alumina ceramics for structural applications, focusing on density, grain size control, and mechanical properties through processing parameter optimization.",
      variables = c(
        "Sintering Temperature (°C): Controls densification rate",
        "Sintering Time (hours): Duration of heat treatment",
        "Particle Size (μm): Starting powder characteristics",
        "Additive Content (%): Sintering aids",
        "Final Density (%): Target property"
      ),
      relationships = c(
        "Higher sintering temperature increases densification rate",
        "Longer sintering time promotes grain growth",
        "Smaller particle size enhances sintering kinetics",
        "Sintering aids lower required temperature"
      )
    ),
    
    silicon_carbide = list(
      description = "Processing of silicon carbide ceramics for high-temperature applications, focusing on the relationship between processing conditions and thermal conductivity, strength, and oxidation resistance.",
      variables = c(
        "Sintering Pressure (MPa): Applied during hot pressing",
        "Sintering Temperature (°C): Processing temperature",
        "Carbon Content (%): Stoichiometry control",
        "Grain Size (μm): Microstructural parameter",
        "Thermal Conductivity (W/m·K): Target property"
      ),
      relationships = c(
        "Higher pressure promotes densification",
        "Stoichiometric composition maximizes properties",
        "Smaller grain size improves strength",
        "Dense microstructure enhances thermal conductivity"
      )
    ),
    
    zirconia_toughening = list(
      description = "Toughening mechanisms in zirconia ceramics through transformation toughening, focusing on the relationship between stabilizer content, microstructure, and fracture toughness.",
      variables = c(
        "Yttria Content (%): Stabilizer concentration",
        "Grain Size (μm): Microstructural parameter",
        "Processing Temperature (°C): Sintering condition",
        "Cooling Rate (°C/min): Thermal history",
        "Fracture Toughness (MPa·m^0.5): Target property"
      ),
      relationships = c(
        "Optimal yttria content for tetragonal phase stability",
        "Grain size affects transformation zone size",
        "Controlled cooling preserves metastable phases",
        "Transformation toughening provides crack resistance"
      )
    ),
    
    piezoelectric_ceramic = list(
      description = "Design of piezoelectric ceramics for sensor and actuator applications, focusing on the relationship between composition, processing, and electromechanical properties.",
      variables = c(
        "PZT Composition Ratio: Pb(Zr,Ti)O3 ratio",
        "Poling Field (kV/mm): Electric field for domain alignment",
        "Poling Temperature (°C): Temperature during poling",
        "Grain Size (μm): Microstructural parameter",
        "Piezoelectric Coefficient (pC/N): Target property"
      ),
      relationships = c(
        "Morphotropic phase boundary maximizes properties",
        "Higher poling field improves domain alignment",
        "Optimal poling temperature near Curie point",
        "Grain size affects domain wall mobility"
      )
    ),
    
    # Additional scenarios for all 30+ scenarios...
    # (Continuing with remaining scenarios to reach 30+)
    
    # Polymer Engineering Scenarios
    thermoplastic = list(
      description = "Processing optimization for thermoplastic polymers, focusing on the relationship between processing conditions, molecular weight, and mechanical properties for injection molding applications.",
      variables = c(
        "Processing Temperature (°C): Melt processing temperature",
        "Cooling Rate (°C/min): Solidification rate",
        "Molecular Weight (g/mol): Polymer chain length",
        "Crystallinity (%): Degree of crystalline structure",
        "Tensile Strength (MPa): Target mechanical property"
      ),
      relationships = c(
        "Higher molecular weight improves mechanical properties",
        "Faster cooling reduces crystallinity",
        "Processing temperature affects molecular degradation",
        "Crystallinity enhances strength but reduces ductility"
      )
    ),
    
    # SMILES-based Polymer Scenarios
    polymer_smiles_properties = list(
      description = "Prediction of polymer properties from SMILES notation and molecular fingerprints, focusing on structure-property relationships for thermoplastic design and optimization.",
      variables = c(
        "SMILES: Simplified molecular input line entry system notation",
        "Morgan Fingerprint: Circular molecular fingerprint (radius=2)",
        "Molecular Weight (g/mol): Calculated from SMILES structure",
        "LogP: Octanol-water partition coefficient",
        "TPSA: Topological polar surface area (Ų)",
        "Glass Transition Temperature (°C): Target thermal property"
      ),
      relationships = c(
        "Aromatic rings increase glass transition temperature",
        "Flexible chains reduce Tg and increase ductility",
        "Polar groups increase intermolecular interactions",
        "Molecular weight affects processing viscosity"
      )
    ),
    
    polymer_degradation_smiles = list(
      description = "Prediction of polymer thermal degradation from molecular structure using SMILES notation and chemical descriptors for stability assessment.",
      variables = c(
        "SMILES: Polymer repeat unit structure",
        "Molecular Weight (g/mol): Average molecular weight",
        "Bond Dissociation Energy (kJ/mol): Weakest bond strength",
        "Aromatic Content (%): Fraction of aromatic rings",
        "Heteroatom Count: Number of N, O, S atoms",
        "Degradation Temperature (°C): 5% weight loss temperature"
      ),
      relationships = c(
        "Aromatic structures increase thermal stability",
        "Weak bonds (C-O, C-N) reduce degradation temperature",
        "Higher molecular weight improves thermal stability",
        "Heteroatoms can catalyze degradation reactions"
      )
    ),
    
    polymer_mechanical_smiles = list(
      description = "Prediction of mechanical properties from polymer molecular structure using SMILES-based descriptors and fingerprinting for materials design.",
      variables = c(
        "SMILES: Polymer backbone structure",
        "Crystallinity Index: Calculated from structure regularity",
        "Chain Flexibility: Rotational bond count",
        "Cross-link Density: Estimated from functional groups",
        "Molecular Weight (g/mol): Number average MW",
        "Tensile Strength (MPa): Ultimate tensile strength"
      ),
      relationships = c(
        "Regular structures increase crystallinity and strength",
        "Flexible chains reduce modulus but increase toughness",
        "Cross-linking increases strength and brittleness",
        "Higher MW generally improves mechanical properties"
      )
    ),
    
    polymer_solubility_smiles = list(
      description = "Prediction of polymer solubility parameters from SMILES notation using group contribution methods and molecular descriptors.",
      variables = c(
        "SMILES: Polymer repeat unit structure",
        "Hansen Solubility Parameters: δd, δp, δh components",
        "Polar Surface Area (Ų): Topological polar surface area",
        "Hydrogen Bond Donors: Count of H-bond donor groups",
        "Hydrogen Bond Acceptors: Count of H-bond acceptor groups",
        "Solubility Parameter (MPa^0.5): Hildebrand parameter"
      ),
      relationships = c(
        "Polar groups increase hydrogen bonding component",
        "Aromatic rings contribute to dispersion forces",
        "Hydrogen bonding groups dominate solubility behavior",
        "Molecular symmetry affects crystallization tendency"
      )
    ),
    
    # Transfer Learning Metal Scenarios
    steel_transfer_learning = list(
      description = "Transfer learning for steel alloy property prediction using pre-trained models from general metal databases, demonstrating domain adaptation from broad materials data to specific steel compositions.",
      variables = c(
        "Base Model Features: General metal properties (density, melting point, crystal structure)",
        "Steel-Specific Features: Carbon content (%), alloying elements",
        "Processing Parameters: Heat treatment temperature, cooling rate",
        "Microstructure: Grain size, phase fractions",
        "Target Properties: Yield strength, ultimate tensile strength, hardness"
      ),
      relationships = c(
        "Pre-trained features capture fundamental metal behavior",
        "Fine-tuning adapts to steel-specific composition effects",
        "Processing parameters modify microstructure-property relationships",
        "Transfer learning reduces data requirements for new steel grades"
      )
    ),
    
    aluminum_transfer_learning = list(
      description = "Transfer learning for aluminum alloy design using knowledge from steel alloy models, focusing on precipitation hardening and age-hardening mechanisms across different metal systems.",
      variables = c(
        "Source Domain: Steel alloy composition and properties",
        "Target Domain: Aluminum alloy composition (Cu, Mg, Si content)",
        "Shared Features: Elastic modulus, thermal properties, grain size",
        "Domain-Specific Features: Precipitation kinetics, solutionizing temperature",
        "Target Properties: Age-hardening response, corrosion resistance"
      ),
      relationships = c(
        "Fundamental strengthening mechanisms transfer between metals",
        "Precipitation hardening principles apply across alloy systems",
        "Domain adaptation accounts for different crystal structures",
        "Shared processing-property relationships enable knowledge transfer"
      )
    ),
    
    titanium_transfer_learning = list(
      description = "Transfer learning for titanium alloy biocompatibility prediction using models pre-trained on general biomedical materials, demonstrating cross-material knowledge transfer.",
      variables = c(
        "Base Model: Biocompatibility data from various materials",
        "Titanium Features: Alpha/beta phase balance, oxygen content",
        "Surface Properties: Roughness, oxide layer thickness",
        "Mechanical Match: Elastic modulus similarity to bone",
        "Target Properties: Cell adhesion, corrosion resistance, biocompatibility"
      ),
      relationships = c(
        "General biocompatibility principles transfer across materials",
        "Surface properties dominate biological response",
        "Mechanical property matching reduces stress shielding",
        "Transfer learning leverages broader biomedical materials database"
      )
    ),
    
    superalloy_transfer_learning = list(
      description = "Transfer learning for superalloy high-temperature performance using models trained on refractory metals, focusing on creep resistance and oxidation behavior knowledge transfer.",
      variables = c(
        "Source Domain: Refractory metal high-temperature data",
        "Target Domain: Nickel-based superalloy composition",
        "Shared Features: Melting point, thermal expansion, diffusion rates",
        "Superalloy-Specific: Gamma prime precipitation, carbide formation",
        "Target Properties: Creep life, oxidation resistance, thermal fatigue"
      ),
      relationships = c(
        "High-temperature deformation mechanisms are transferable",
        "Oxidation kinetics follow similar principles across metals",
        "Diffusion-controlled processes share common physics",
        "Transfer learning captures complex multi-component interactions"
      )
    ),
    
    polymer_degradation_smiles = list(
      description = "Prediction of polymer thermal degradation behavior from SMILES notation using molecular descriptors and structural features.",
      variables = c(
        "SMILES: Polymer repeat unit structure",
        "MACCS Keys: 166-bit structural fingerprint",
        "Aromatic Fraction: Percentage of aromatic carbons",
        "Heteroatom Count: Number of non-carbon atoms",
        "Rotatable Bonds: Molecular flexibility indicator",
        "Degradation Temperature (°C): Thermal stability measure"
      ),
      relationships = c(
        "Aromatic structures improve thermal stability",
        "Heteroatoms can create weak points",
        "Flexible chains degrade at lower temperatures",
        "Crosslinking improves thermal resistance"
      )
    ),
    
    polymer_solubility_prediction = list(
      description = "Prediction of polymer solubility parameters from SMILES notation using Hansen solubility parameters and molecular descriptors.",
      variables = c(
        "SMILES: Polymer chemical structure",
        "RDKit Descriptors: Molecular property descriptors",
        "Hansen δD: Dispersion parameter (MPa^0.5)",
        "Hansen δP: Polar parameter (MPa^0.5)", 
        "Hansen δH: Hydrogen bonding parameter (MPa^0.5)",
        "Solubility Parameter (MPa^0.5): Total Hansen parameter"
      ),
      relationships = c(
        "Polar groups increase δP and δH components",
        "Aromatic rings contribute to δD component",
        "Hydrogen bonding groups affect δH significantly",
        "Total solubility parameter affects compatibility"
      )
    ),
    
    polymer_mechanical_smiles = list(
      description = "Prediction of mechanical properties from polymer SMILES notation using molecular fingerprints and structural descriptors for materials design.",
      variables = c(
        "SMILES: Polymer backbone structure",
        "Atom Pair Fingerprint: Structural connectivity fingerprint",
        "Chain Flexibility: Calculated rotatable bond ratio",
        "Crystallinity Index: Structural regularity measure",
        "Crosslink Density: Estimated from structure",
        "Young's Modulus (GPa): Target mechanical property"
      ),
      relationships = c(
        "Rigid backbones increase modulus",
        "Crystalline regions enhance stiffness",
        "Crosslinking improves mechanical properties",
        "Side chain bulkiness affects packing"
      )
    ),
    
    # Composite Materials Scenarios
    carbon_fiber = list(
      description = "Optimization of carbon fiber reinforced composites for aerospace applications, focusing on fiber-matrix interface, processing conditions, and mechanical performance.",
      variables = c(
        "Fiber Volume Fraction (%): Reinforcement content",
        "Fiber Surface Treatment: Interface modification",
        "Cure Pressure (MPa): Consolidation pressure",
        "Cure Temperature (°C): Processing temperature",
        "Tensile Modulus (GPa): Target stiffness property"
      ),
      relationships = c(
        "Higher fiber content increases stiffness",
        "Surface treatment improves fiber-matrix bonding",
        "Adequate cure pressure eliminates voids",
        "Optimal cure temperature ensures complete crosslinking"
      )
    ),
    
    # Nanomaterials Scenarios
    nanoparticle = list(
      description = "Synthesis of nanoparticles for catalytic applications, focusing on size control, surface area, and the relationship between synthesis parameters and catalytic activity.",
      variables = c(
        "Synthesis Temperature (°C): Reaction temperature",
        "Reaction Time (minutes): Duration of synthesis",
        "Precursor Concentration (M): Starting material concentration",
        "pH Level: Solution acidity/basicity",
        "Particle Size (nm): Target nanoparticle dimension"
      ),
      relationships = c(
        "Higher temperature increases particle growth rate",
        "Longer reaction time promotes particle coarsening",
        "Higher concentration leads to more nucleation sites",
        "pH affects precursor solubility and nucleation"
      )
    )
  )
}

generate_scenario_data <- function(category, scenario, n_samples = 200) {
  set.seed(42)
  
  if (scenario == "polymer_smiles_properties") {
    smiles_strings <- sample(c(
      "CC(C)(C)c1ccc(O)cc1", "CCc1ccc(O)cc1", "c1ccc2c(c1)ccc3c2ccc4c3cccc4",
      "CC(C)c1ccc(C(C)(C)C)cc1", "c1ccc(cc1)c2ccccc2", "CCCCc1ccc(O)cc1",
      "CC(C)(C)c1ccc(C(=O)O)cc1", "c1ccc(cc1)C(c2ccccc2)c3ccccc3"
    ), n_samples, replace = TRUE)
    
    molecular_weight <- ifelse(grepl("CC\\(C\\)\\(C\\)", smiles_strings), 
                              runif(n_samples, 200, 400),
                              runif(n_samples, 100, 250))
    logp <- ifelse(grepl("O", smiles_strings), 
                   runif(n_samples, 1, 4),
                   runif(n_samples, 3, 6))
    tpsa <- ifelse(grepl("O", smiles_strings), 
                   runif(n_samples, 20, 80),
                   runif(n_samples, 0, 20))
    
    glass_transition_temp <- 50 + 0.3 * molecular_weight + 
                           ifelse(grepl("c1ccc", smiles_strings), 40, 0) +
                           ifelse(grepl("O", smiles_strings), 20, 0) +
                           rnorm(n_samples, 0, 10)
    
    data.frame(smiles = smiles_strings, molecular_weight, logp, tpsa, 
               glass_transition_temp = pmax(glass_transition_temp, -50))
  
  } else if (scenario == "polymer_degradation_smiles") {
    smiles_strings <- sample(c(
      "CC(C)OC(=O)C", "CCOC(=O)C=C", "c1ccc(cc1)C(=O)O",
      "CC(C)(C)OC(=O)C", "CCN(CC)C(=O)C", "c1ccc2c(c1)ccc3c2cccc3"
    ), n_samples, replace = TRUE)
    
    molecular_weight <- runif(n_samples, 1000, 50000)
    aromatic_content <- ifelse(grepl("c1ccc", smiles_strings), 
                              runif(n_samples, 30, 80),
                              runif(n_samples, 0, 20))
    heteroatom_count <- nchar(gsub("[^NOSPFClBrI]", "", smiles_strings))
    
    degradation_temp <- 200 + 0.002 * molecular_weight + 
                       2 * aromatic_content - 10 * heteroatom_count +
                       rnorm(n_samples, 0, 15)
    
    data.frame(smiles = smiles_strings, molecular_weight, aromatic_content,
               heteroatom_count, degradation_temp = pmax(degradation_temp, 150))
  
  } else if (scenario == "polymer_mechanical_smiles") {
    smiles_strings <- sample(c(
      "CC(C)(C)c1ccc(cc1)C(C)(C)C", "c1ccc(cc1)c2ccccc2", "CCN(CC)C(=O)C",
      "CC(C)c1ccc(C(C)(C)C)cc1", "c1ccc2c(c1)ccc3c2cccc3", "CCOC(=O)C=C"
    ), n_samples, replace = TRUE)
    
    molecular_weight <- runif(n_samples, 5000, 100000)
    crystallinity_index <- ifelse(grepl("CC\\(C\\)\\(C\\)", smiles_strings), 
                                 runif(n_samples, 60, 90),
                                 runif(n_samples, 20, 60))
    chain_flexibility <- nchar(gsub("[^C]", "", smiles_strings)) / nchar(smiles_strings)
    
    tensile_strength <- 20 + 0.001 * molecular_weight + 
                       2 * crystallinity_index - 50 * chain_flexibility +
                       rnorm(n_samples, 0, 10)
    
    data.frame(smiles = smiles_strings, molecular_weight, crystallinity_index,
               chain_flexibility, tensile_strength = pmax(tensile_strength, 10))
  
  } else if (scenario == "polymer_solubility_smiles") {
    smiles_strings <- sample(c(
      "CC(C)OC(=O)C", "c1ccc(cc1)O", "CCN(CC)C(=O)C",
      "CC(C)(C)c1ccc(O)cc1", "CCOC(=O)C=C", "c1ccc2c(c1)ccc3c2cccc3"
    ), n_samples, replace = TRUE)
    
    polar_surface_area <- ifelse(grepl("O", smiles_strings), 
                                runif(n_samples, 40, 120),
                                runif(n_samples, 0, 40))
    h_bond_donors <- nchar(gsub("[^O]H", "", smiles_strings))
    h_bond_acceptors <- nchar(gsub("[^ON]", "", smiles_strings))
    
    solubility_parameter <- 15 + 0.5 * polar_surface_area + 
                           2 * h_bond_donors + 1.5 * h_bond_acceptors +
                           rnorm(n_samples, 0, 2)
    
    data.frame(smiles = smiles_strings, polar_surface_area, h_bond_donors,
               h_bond_acceptors, solubility_parameter = pmax(solubility_parameter, 10))
  
  } else if (scenario == "steel_transfer_learning") {
    base_features <- data.frame(
      density = runif(n_samples, 7.6, 8.1),
      melting_point = runif(n_samples, 1450, 1550),
      crystal_structure = sample(c("BCC", "FCC", "Mixed"), n_samples, replace = TRUE)
    )
    
    steel_features <- data.frame(
      carbon_content = runif(n_samples, 0.1, 1.5),
      manganese_content = runif(n_samples, 0.3, 2.0),
      heat_treatment_temp = runif(n_samples, 800, 1200)
    )
    
    base_strength <- 200 + 100 * (base_features$density - 7.8) + 
                    0.1 * (base_features$melting_point - 1500)
    
    steel_adjustment <- 300 * steel_features$carbon_content + 
                       50 * steel_features$manganese_content +
                       0.2 * (steel_features$heat_treatment_temp - 1000)
    
    tensile_strength <- base_strength + steel_adjustment + rnorm(n_samples, 0, 20)
    
    cbind(base_features, steel_features, 
          tensile_strength = pmax(tensile_strength, 200))
  
  } else if (scenario == "aluminum_transfer_learning") {
    source_features <- data.frame(
      elastic_modulus = runif(n_samples, 60, 80),
      thermal_conductivity = runif(n_samples, 150, 250),
      grain_size = runif(n_samples, 10, 100)
    )
    
    aluminum_features <- data.frame(
      copper_content = runif(n_samples, 1, 6),
      magnesium_content = runif(n_samples, 0.5, 3),
      aging_temp = runif(n_samples, 150, 200)
    )
    
    base_strength <- 100 + 2 * source_features$elastic_modulus - 
                    0.5 * source_features$grain_size
    
    aluminum_adjustment <- 40 * aluminum_features$copper_content + 
                          25 * aluminum_features$magnesium_content +
                          0.5 * aluminum_features$aging_temp
    
    yield_strength <- base_strength + aluminum_adjustment + rnorm(n_samples, 0, 15)
    
    cbind(source_features, aluminum_features,
          yield_strength = pmax(yield_strength, 100))
  
  } else if (scenario == "titanium_transfer_learning") {
    base_features <- data.frame(
      biocompatibility_score = runif(n_samples, 0.6, 1.0),
      surface_roughness = runif(n_samples, 0.1, 2.0),
      elastic_modulus = runif(n_samples, 100, 120)
    )
    
    titanium_features <- data.frame(
      vanadium_content = runif(n_samples, 0, 6),
      aluminum_content = runif(n_samples, 4, 8),
      oxygen_content = runif(n_samples, 0.1, 0.4)
    )
    
    base_compatibility <- 50 + 30 * base_features$biocompatibility_score - 
                         10 * base_features$surface_roughness +
                         0.2 * (base_features$elastic_modulus - 110)
    
    titanium_adjustment <- -2 * titanium_features$vanadium_content + 
                          3 * titanium_features$aluminum_content -
                          20 * titanium_features$oxygen_content
    
    biocompatibility_index <- base_compatibility + titanium_adjustment + rnorm(n_samples, 0, 5)
    
    cbind(base_features, titanium_features,
          biocompatibility_index = pmax(biocompatibility_index, 20))
  
  } else if (scenario == "superalloy_transfer_learning") {
    source_features <- data.frame(
      melting_point = runif(n_samples, 1400, 1700),
      thermal_expansion = runif(n_samples, 10, 20),
      diffusion_rate = runif(n_samples, 0.1, 1.0)
    )
    
    superalloy_features <- data.frame(
      chromium_content = runif(n_samples, 15, 25),
      aluminum_content = runif(n_samples, 3, 8),
      service_temp = runif(n_samples, 800, 1200)
    )
    
    base_performance <- 100 + 0.05 * (source_features$melting_point - 1550) - 
                       2 * source_features$thermal_expansion +
                       20 * source_features$diffusion_rate
    
    superalloy_adjustment <- 3 * superalloy_features$chromium_content + 
                            5 * superalloy_features$aluminum_content +
                            0.1 * superalloy_features$service_temp
    
    creep_life <- base_performance + superalloy_adjustment + rnorm(n_samples, 0, 10)
    
    cbind(source_features, superalloy_features,
          creep_life = pmax(creep_life, 50))
  
  } else if (scenario == "steel_alloy") {
    carbon_content <- runif(n_samples, 0.1, 1.5)
    manganese_content <- runif(n_samples, 0.3, 2.0)
    heat_treatment_temp <- runif(n_samples, 800, 1200)
    cooling_rate <- runif(n_samples, 1, 100)
    
    tensile_strength <- 300 + 200 * carbon_content + 50 * manganese_content + 
                       0.1 * heat_treatment_temp - 0.5 * cooling_rate +
                       rnorm(n_samples, 0, 20)
    
    data.frame(carbon_content, manganese_content, heat_treatment_temp, 
               cooling_rate, tensile_strength = pmax(tensile_strength, 200))
               
  } else {
    # Default scenario
    x1 <- runif(n_samples, 0, 10)
    x2 <- runif(n_samples, 0, 10)
    x3 <- runif(n_samples, 0, 10)
    x4 <- runif(n_samples, 0, 10)
    y <- 2 * x1 + 3 * x2 - x3 + 0.5 * x4 + rnorm(n_samples, 0, 5)
    
    data.frame(x1, x2, x3, x4, y = pmax(y, 10))
  }
}
