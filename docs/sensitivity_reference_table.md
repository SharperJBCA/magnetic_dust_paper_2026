# Reference sensitivity table (μK·arcmin)

This table is used by `src/mde_pipeline/qc/check_map_sensitivities.py` to compare simulated-map sensitivities derived from variance maps:

\[
\sigma_{\mu K\cdot arcmin} = \sqrt{\mathrm{median}(\mathrm{variance}) \times \Omega_{pix,\,arcmin^2}}
\]

- Machine-readable source: `configs/preprocessing/reference_sensitivities_uK_arcmin.csv`
- Notes:
  - LiteBIRD rows are channel-level values used in the PTEP-era instrument model.
  - Planck rows are common polarization white-noise-equivalent values (μK·arcmin).
  - WMAP rows are DA-level band values converted to μK·arcmin.
  - If you want strict reproducibility to a specific paper/table version, update the CSV `source` text and value columns accordingly.

| family | map_id | freq [GHz] | reference [μK·arcmin] |
|---|---|---:|---:|
| litebird | litebird_LFT_L1-040 | 40.0 | 37.42 |
| litebird | litebird_LFT_L2-050 | 50.0 | 33.46 |
| litebird | litebird_LFT_L1-060 | 60.0 | 21.31 |
| litebird | litebird_LFT_L3-068 | 68.0 | 19.91 |
| litebird | litebird_LFT_L2-068 | 68.0 | 31.77 |
| litebird | litebird_LFT_L4-078 | 78.0 | 15.55 |
| litebird | litebird_LFT_L1-078 | 78.0 | 18.65 |
| litebird | litebird_LFT_L3-089 | 89.0 | 12.67 |
| litebird | litebird_LFT_L2-089 | 89.0 | 28.77 |
| litebird | litebird_LFT_L4-100 | 100.0 | 10.34 |
| litebird | litebird_LFT_L3-119 | 119.0 | 8.80 |
| litebird | litebird_LFT_L4-140 | 140.0 | 7.42 |
| litebird | litebird_MFT_M1-100 | 100.0 | 9.83 |
| litebird | litebird_MFT_M2-119 | 119.0 | 7.84 |
| litebird | litebird_MFT_M1-140 | 140.0 | 7.31 |
| litebird | litebird_MFT_M2-166 | 166.0 | 5.74 |
| litebird | litebird_MFT_M1-195 | 195.0 | 7.26 |
| litebird | litebird_HFT_H1-195 | 195.0 | 10.50 |
| litebird | litebird_HFT_H2-235 | 235.0 | 10.79 |
| litebird | litebird_HFT_H1-280 | 280.0 | 13.80 |
| litebird | litebird_HFT_H2-337 | 337.0 | 21.95 |
| litebird | litebird_HFT_H3-402 | 402.0 | 47.45 |
| planck | planck030_cosmo | 28.4 | 145.0 |
| planck | planck044_cosmo | 44.1 | 149.0 |
| planck | planck070_cosmo | 70.4 | 137.0 |
| planck | planck100_pr3 | 100.0 | 64.6 |
| planck | planck143_pr3 | 143.0 | 42.6 |
| planck | planck217_pr3 | 217.0 | 65.5 |
| planck | planck353_pr3 | 353.0 | 406.0 |
| wmap | wmapK_cosmo | 22.5 | 887.0 |
| wmap | wmapKa_cosmo | 33.0 | 675.0 |
| wmap | wmapQ1_cosmo | 40.7 | 430.0 |
| wmap | wmapQ2_cosmo | 40.7 | 430.0 |
| wmap | wmapV1_cosmo | 60.8 | 316.0 |
| wmap | wmapV2_cosmo | 60.8 | 316.0 |
| wmap | wmapW1_cosmo | 93.5 | 490.0 |
| wmap | wmapW2_cosmo | 93.5 | 490.0 |
| wmap | wmapW3_cosmo | 93.5 | 490.0 |
| wmap | wmapW4_cosmo | 93.5 | 490.0 |

## Example usage

```bash
python src/mde_pipeline/qc/check_map_sensitivities.py --family litebird --field QQ
python src/mde_pipeline/qc/check_map_sensitivities.py --family planck --field QQ --rtol 0.30
python src/mde_pipeline/qc/check_map_sensitivities.py --map-id planck143_pr3 --map-id wmapV1_cosmo
```
