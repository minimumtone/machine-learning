#!/bin/bash
# Extract optimized lattice constants from L12 calculations
# Usage: bash extract_results.sh > l12_results.csv

echo "formula,element_A,element_B,count_A,count_B,lattice_constant,converged"

# --- Fe3Mn ---
DIR="Fe3Mn"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Fe3Mn,Fe,Mn,3,1,$A,$CONV"
else
    echo "Fe3Mn,Fe,Mn,3,1,NA,not_run"
fi

# --- Mn3Fe ---
DIR="Mn3Fe"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Mn3Fe,Mn,Fe,3,1,$A,$CONV"
else
    echo "Mn3Fe,Mn,Fe,3,1,NA,not_run"
fi

# --- Cr3Mn ---
DIR="Cr3Mn"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Cr3Mn,Cr,Mn,3,1,$A,$CONV"
else
    echo "Cr3Mn,Cr,Mn,3,1,NA,not_run"
fi

# --- Mn3Cr ---
DIR="Mn3Cr"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Mn3Cr,Mn,Cr,3,1,$A,$CONV"
else
    echo "Mn3Cr,Mn,Cr,3,1,NA,not_run"
fi

# --- Al3Mn ---
DIR="Al3Mn"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Al3Mn,Al,Mn,3,1,$A,$CONV"
else
    echo "Al3Mn,Al,Mn,3,1,NA,not_run"
fi

# --- Mn3Al ---
DIR="Mn3Al"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Mn3Al,Mn,Al,3,1,$A,$CONV"
else
    echo "Mn3Al,Mn,Al,3,1,NA,not_run"
fi

# --- Cr3Mo ---
DIR="Cr3Mo"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Cr3Mo,Cr,Mo,3,1,$A,$CONV"
else
    echo "Cr3Mo,Cr,Mo,3,1,NA,not_run"
fi

# --- Mo3Cr ---
DIR="Mo3Cr"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Mo3Cr,Mo,Cr,3,1,$A,$CONV"
else
    echo "Mo3Cr,Mo,Cr,3,1,NA,not_run"
fi

# --- Fe3Mo ---
DIR="Fe3Mo"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Fe3Mo,Fe,Mo,3,1,$A,$CONV"
else
    echo "Fe3Mo,Fe,Mo,3,1,NA,not_run"
fi

# --- Mo3Fe ---
DIR="Mo3Fe"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Mo3Fe,Mo,Fe,3,1,$A,$CONV"
else
    echo "Mo3Fe,Mo,Fe,3,1,NA,not_run"
fi

# --- Mo3Ni ---
DIR="Mo3Ni"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Mo3Ni,Mo,Ni,3,1,$A,$CONV"
else
    echo "Mo3Ni,Mo,Ni,3,1,NA,not_run"
fi

# --- Ni3Mo ---
DIR="Ni3Mo"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Ni3Mo,Ni,Mo,3,1,$A,$CONV"
else
    echo "Ni3Mo,Ni,Mo,3,1,NA,not_run"
fi

# --- Ir3Pd ---
DIR="Ir3Pd"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Ir3Pd,Ir,Pd,3,1,$A,$CONV"
else
    echo "Ir3Pd,Ir,Pd,3,1,NA,not_run"
fi

# --- Pd3Ir ---
DIR="Pd3Ir"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Pd3Ir,Pd,Ir,3,1,$A,$CONV"
else
    echo "Pd3Ir,Pd,Ir,3,1,NA,not_run"
fi

# --- Ir3Ru ---
DIR="Ir3Ru"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Ir3Ru,Ir,Ru,3,1,$A,$CONV"
else
    echo "Ir3Ru,Ir,Ru,3,1,NA,not_run"
fi

# --- Ru3Ir ---
DIR="Ru3Ir"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Ru3Ir,Ru,Ir,3,1,$A,$CONV"
else
    echo "Ru3Ir,Ru,Ir,3,1,NA,not_run"
fi

# --- Pd3Pt ---
DIR="Pd3Pt"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Pd3Pt,Pd,Pt,3,1,$A,$CONV"
else
    echo "Pd3Pt,Pd,Pt,3,1,NA,not_run"
fi

# --- Pt3Pd ---
DIR="Pt3Pd"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Pt3Pd,Pt,Pd,3,1,$A,$CONV"
else
    echo "Pt3Pd,Pt,Pd,3,1,NA,not_run"
fi

# --- Pd3Rh ---
DIR="Pd3Rh"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Pd3Rh,Pd,Rh,3,1,$A,$CONV"
else
    echo "Pd3Rh,Pd,Rh,3,1,NA,not_run"
fi

# --- Rh3Pd ---
DIR="Rh3Pd"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Rh3Pd,Rh,Pd,3,1,$A,$CONV"
else
    echo "Rh3Pd,Rh,Pd,3,1,NA,not_run"
fi

# --- Pd3Ru ---
DIR="Pd3Ru"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Pd3Ru,Pd,Ru,3,1,$A,$CONV"
else
    echo "Pd3Ru,Pd,Ru,3,1,NA,not_run"
fi

# --- Ru3Pd ---
DIR="Ru3Pd"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Ru3Pd,Ru,Pd,3,1,$A,$CONV"
else
    echo "Ru3Pd,Ru,Pd,3,1,NA,not_run"
fi

# --- Pt3Ru ---
DIR="Pt3Ru"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Pt3Ru,Pt,Ru,3,1,$A,$CONV"
else
    echo "Pt3Ru,Pt,Ru,3,1,NA,not_run"
fi

# --- Ru3Pt ---
DIR="Ru3Pt"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Ru3Pt,Ru,Pt,3,1,$A,$CONV"
else
    echo "Ru3Pt,Ru,Pt,3,1,NA,not_run"
fi

# --- Ni3Pd ---
DIR="Ni3Pd"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Ni3Pd,Ni,Pd,3,1,$A,$CONV"
else
    echo "Ni3Pd,Ni,Pd,3,1,NA,not_run"
fi

# --- Pd3Ni ---
DIR="Pd3Ni"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Pd3Ni,Pd,Ni,3,1,$A,$CONV"
else
    echo "Pd3Ni,Pd,Ni,3,1,NA,not_run"
fi

# --- Os3Pd ---
DIR="Os3Pd"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Os3Pd,Os,Pd,3,1,$A,$CONV"
else
    echo "Os3Pd,Os,Pd,3,1,NA,not_run"
fi

# --- Pd3Os ---
DIR="Pd3Os"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Pd3Os,Pd,Os,3,1,$A,$CONV"
else
    echo "Pd3Os,Pd,Os,3,1,NA,not_run"
fi

# --- Os3Pt ---
DIR="Os3Pt"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Os3Pt,Os,Pt,3,1,$A,$CONV"
else
    echo "Os3Pt,Os,Pt,3,1,NA,not_run"
fi

# --- Pt3Os ---
DIR="Pt3Os"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Pt3Os,Pt,Os,3,1,$A,$CONV"
else
    echo "Pt3Os,Pt,Os,3,1,NA,not_run"
fi

# --- Os3Rh ---
DIR="Os3Rh"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Os3Rh,Os,Rh,3,1,$A,$CONV"
else
    echo "Os3Rh,Os,Rh,3,1,NA,not_run"
fi

# --- Rh3Os ---
DIR="Rh3Os"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Rh3Os,Rh,Os,3,1,$A,$CONV"
else
    echo "Rh3Os,Rh,Os,3,1,NA,not_run"
fi

# --- Os3Ru ---
DIR="Os3Ru"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Os3Ru,Os,Ru,3,1,$A,$CONV"
else
    echo "Os3Ru,Os,Ru,3,1,NA,not_run"
fi

# --- Ru3Os ---
DIR="Ru3Os"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Ru3Os,Ru,Os,3,1,$A,$CONV"
else
    echo "Ru3Os,Ru,Os,3,1,NA,not_run"
fi

# --- Cr3V ---
DIR="Cr3V"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Cr3V,Cr,V,3,1,$A,$CONV"
else
    echo "Cr3V,Cr,V,3,1,NA,not_run"
fi

# --- V3Cr ---
DIR="V3Cr"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "V3Cr,V,Cr,3,1,$A,$CONV"
else
    echo "V3Cr,V,Cr,3,1,NA,not_run"
fi

# --- Fe3V ---
DIR="Fe3V"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Fe3V,Fe,V,3,1,$A,$CONV"
else
    echo "Fe3V,Fe,V,3,1,NA,not_run"
fi

# --- V3Fe ---
DIR="V3Fe"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "V3Fe,V,Fe,3,1,$A,$CONV"
else
    echo "V3Fe,V,Fe,3,1,NA,not_run"
fi

# --- Ni3V ---
DIR="Ni3V"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Ni3V,Ni,V,3,1,$A,$CONV"
else
    echo "Ni3V,Ni,V,3,1,NA,not_run"
fi

# --- V3Ni ---
DIR="V3Ni"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "V3Ni,V,Ni,3,1,$A,$CONV"
else
    echo "V3Ni,V,Ni,3,1,NA,not_run"
fi

# --- Al3Cr ---
DIR="Al3Cr"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Al3Cr,Al,Cr,3,1,$A,$CONV"
else
    echo "Al3Cr,Al,Cr,3,1,NA,not_run"
fi

# --- Cr3Al ---
DIR="Cr3Al"
if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then
    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk '{print $1}')
    CONV="no"
    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then
        CONV="yes"
    fi
    echo "Cr3Al,Cr,Al,3,1,$A,$CONV"
else
    echo "Cr3Al,Cr,Al,3,1,NA,not_run"
fi
