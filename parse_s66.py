#!/usr/bin/env python3
"""
Parse a two-row-header basis-set energy file and perform CBS extrapolation.

Expected file format
---------------------
Line 1 : SYS  <basis label for each data column>   (basis label repeated
         once per quantity it has, e.g. "DZ DZ DZ TZ TZ TZ QZ QZ 5Z 5Z CBS")
Line 2 : SYS  <quantity label for each data column> (SCF / MP2 / CCSD(T) / REF)
Line 3+: <system name>  <value> <value> ...          (one row per system)

Assumptions (edit the constants below if your convention differs)
-------------------------------------------------------------------
* Cardinal numbers used in the extrapolation formulas:
      DZ -> 2, TZ -> 3, QZ -> 4, 5Z -> 5, 6Z -> 6
* SCF/HF energies: two-point exponential extrapolation (Halkier et al.),
  with alpha = 1.63:
      E_CBS = (E_Y*exp(-alpha*X) - E_X*exp(-alpha*Y))
              / (exp(-alpha*X) - exp(-alpha*Y))
* Correlation energies (delta-MP2, delta-CCSD(T)): standard Helgaker
  two-point 1/X^3 extrapolation:
      E_CBS = (X^3*E_X - Y^3*E_Y) / (X^3 - Y^3)
* delta-MP2      = E(MP2) - E(SCF)
* delta-CCSD(T)  = E(CCSD(T)) - E(MP2)
* A pair is only extrapolated for delta-CCSD(T) if BOTH basis sets in the
  pair actually report a CCSD(T) value; otherwise the result is nan
  (this is why TZ-QZ and QZ-5Z show "nan" in the example: QZ and 5Z have
  no CCSD(T) column).
"""

import sys
import numpy as np
from collections import OrderedDict

ALPHA = 1.63
CARDINAL = {'DZ': 2, 'TZ': 3, 'QZ': 4, '5Z': 5, '6Z': 6}
PAIRS = [('TZ','TZ'),('DZ', 'TZ'), ('TZ', 'QZ'), ('QZ', '5Z')]


def parse_file(path):
    """Read the file and return a list of (system_name, {(basis, qty): value})."""
    with open(path) as fh:
        lines = [ln.rstrip('\n') for ln in fh if ln.strip()]

    header_basis = lines[0].split()
    header_qty = lines[1].split()

    if header_basis[0] != 'SYS' or header_qty[0] != 'SYS':
        raise ValueError("Expected both header lines to start with 'SYS'")
    if len(header_basis) != len(header_qty):
        raise ValueError("Header lines have a different number of columns")

    columns = list(zip(header_basis[1:], header_qty[1:]))  # [(DZ,SCF), (DZ,MP2), ...]

    systems = []
    for line in lines[2:]:
        tokens = line.split()
        name, raw_values = tokens[0], tokens[1:]
        if len(raw_values) != len(columns):
            raise ValueError(
                f"Row '{name}' has {len(raw_values)} values, "
                f"expected {len(columns)} based on the header"
            )
        values = list(map(float, raw_values))
        col_data = dict(zip(columns, values))
        systems.append((name, col_data))
    return systems


def basis_energies(col_data):
    """Reshape {(basis, qty): val} -> {basis: {qty: val}}, dropping REF."""
    out = OrderedDict()
    for (basis, qty), val in col_data.items():
        if qty == 'REF':
            continue
        out.setdefault(basis, {})[qty] = val
    return out


def get_ref(col_data):
    for (basis, qty), val in col_data.items():
        if qty == 'REF':
            return val
    return None


def extrapolate_scf(X, Y, EX, EY):
    """Two-point exponential extrapolation for SCF/HF energies."""
    if EX is None or EY is None:
        return float('nan')
    dX, dY = np.exp(-ALPHA * X), np.exp(-ALPHA * Y)
    return (EY * dX - EX * dY) / (dX - dY)


def extrapolate_corr(X, Y, EX, EY):
    """Two-point 1/X^3 (Helgaker) extrapolation for correlation energies."""
    if EX is None or EY is None or np.isnan(EX) or np.isnan(EY):
        return float('nan')
    return (X ** 3 * EX - Y ** 3 * EY) / (X ** 3 - Y ** 3)


def build_table(col_data):
    """Return list of (ext_label, SCF_cbs, deltaMP2_cbs, deltaCCSDT_cbs)."""
    energies = basis_energies(col_data)
    rows = []
    for b1, b2 in PAIRS:
        if b1 not in energies or b2 not in energies:
            continue

        e1, e2 = energies[b1], energies[b2]
        scf1, scf2 = e1.get('SCF'), e2.get('SCF')
        mp2_1, mp2_2 = e1.get('MP2'), e2.get('MP2')
        ccsdt1, ccsdt2 = e1.get('CCSD(T)'), e2.get('CCSD(T)')

        dmp2_1 = (mp2_1 - scf1) if (scf1 is not None and mp2_1 is not None) else float('nan')
        dmp2_2 = (mp2_2 - scf2) if (scf2 is not None and mp2_2 is not None) else float('nan')

        dccsdt_1 = (ccsdt1 - mp2_1) if (ccsdt1 is not None and mp2_1 is not None) else float('nan')
        dccsdt_2 = (ccsdt2 - mp2_2) if (ccsdt2 is not None and mp2_2 is not None) else float('nan')

        X, Y = CARDINAL[b1], CARDINAL[b2]

        # if "5Z" == b2:
            # scf_cbs = energies['CBS'].get('SCF')#extrapolate_scf(X, Y, scf1, scf2)
            # dmp2_cbs =  extrapolate_corr(X, Y, dmp2_1, dmp2_2) 
            # dccsdt_cbs = extrapolate_corr(X, Y, dccsdt_1, dccsdt_2)
        # elif b1 == 'DZ' and b2 == 'TZ':
            # scf_cbs = extrapolate_scf(X, Y, scf1, scf2)
            # dmp2_cbs = energies['QZ'].get('MP2') - energies['QZ'].get('SCF')
            # b1 = 'DZ/QZ(MP2)'
            # dccsdt_cbs = extrapolate_corr(X, Y, dccsdt_1, dccsdt_2)
        if b1 != b2:
            scf_cbs = extrapolate_scf(X, Y, scf1, scf2)
            dmp2_cbs = extrapolate_corr(X, Y, dmp2_1, dmp2_2)
            dccsdt_cbs = extrapolate_corr(X, Y, dccsdt_1, dccsdt_2)
        else:
            assert np.isclose(ccsdt1,ccsdt2)
            scf_cbs = 0 
            dmp2_cbs = 0
            dccsdt_cbs = dccsdt_1
        rows.append([f'{b1}-{b2}', scf_cbs, dmp2_cbs, dccsdt_cbs])
    return rows


def build_raw_table(col_data):
    """
    Return the *unextrapolated* per-basis data in the same row shape as
    build_table(): (label, SCF, delta MP2, delta CCSD(T)).
 
    Here 'label' is the basis set itself (DZ, TZ, QZ, 5Z, ...) rather than
    a pair, and the values are exactly what was in the file (no CBS limit
    involved) - delta MP2 / delta CCSD(T) are still computed as
    MP2-SCF / CCSD(T)-MP2, but for a single basis, so nan only appears
    where a quantity is genuinely missing for that basis (e.g. no CCSD(T)
    at QZ/5Z).
    """
    energies = basis_energies(col_data)  # preserves header column order
    rows = []
    for basis, qty_vals in energies.items():
        scf = qty_vals.get('SCF', float('nan'))
        mp2 = qty_vals.get('MP2')
        ccsdt = qty_vals.get('CCSD(T)')

        dmp2 = (mp2 - scf) if (mp2 is not None and not np.isnan(scf)) else float('nan')
        # if basis == '5Z':
            # dmp2 = np.nan 
        dccsdt = (ccsdt - mp2) if (ccsdt is not None and mp2 is not None) else float('nan')
 
        rows.append([basis, scf, dmp2, dccsdt])
    return rows


def format_table(name, rows):
    lines = [f"\nSystem: {name}",
             f"{'ext':<8}{'SCF':>14}{'delta MP2':>14}{'delta CCSD(T)':>16}"]
    for ext, scf, dmp2, dccsdt in rows:
        scf_s = 'nan' if np.isnan(scf) else f'{scf:.5f}'
        dmp2_s = 'nan' if np.isnan(dmp2) else f'{dmp2:.5f}'
        dccsdt_s = 'nan' if np.isnan(dccsdt) else f'{dccsdt:.5f}'
        lines.append(f"{ext:<8}{scf_s:>14}{dmp2_s:>14}{dccsdt_s:>16}")
    return '\n'.join(lines)


def fill_table(path):
    systems = parse_file(path)

    all_tables = {}   # system_name -> rows
    orig_tables = {}
    ref_values = {}   # REF column, in file order

    for name, col_data in systems:
        rows = build_table(col_data)
        all_tables[name] = rows

        ref = get_ref(col_data)
        ref_values[name] = ref

        raw_rows = build_raw_table(col_data)
        orig_tables[name] = raw_rows
    
    return all_tables, orig_tables, ref_values


if __name__ == '__main__':
    infile = sys.argv[1] if len(sys.argv) > 1 else 's66.dat'
    ext_tables, original_tables, refs = fill_table(infile)
    for k in ext_tables.keys():
        if len(sys.argv) > 2 and sys.argv[2] not in k:
            continue
        print(format_table(k,ext_tables[k]))
        print(format_table(k,original_tables[k]))
        print("Ref",refs[k])

