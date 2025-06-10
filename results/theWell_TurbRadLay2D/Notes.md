## The Well: Turbulent radiative layer 2D


- Project: Custom_CondAttUnet_lowSkip_turbulent_radiative_layer_2D
- Sort by: train loss

| SDE name | Energy loss | Batch sz | Sigma min | Sigma max | Beta min | Beta max | tot epochs | Train loss | Val loss | eLambda | Train eLoss | ID                        |
|----------|-------------|----------|-----------|-----------|----------|----------|------------|------------|----------|---------|-------------|---------------------------|
| vpsde    | False       | 128      | -         | -         | 0.1      | 25       | 525        | 6.8e-03    | 6.7e-03  | -       | -           | y2025_m04_d27_23h_30m_37s |
| subvpsde | False       | 128      | -         | -         | 0.4      | 30       | 960        | 1.2e-02    | 1.2e-02  | -       | -           | y2025_m04_d02_09h_25m_13s |
| vesde    | False       | 128      | 0.15      | 4         | -        | -        | 689        | 1.2e-02    | 1.3e-02  | -       | -           | y2025_m04_d28_03h_18m_54s |
| vpsde    | True        | 128      | -         | -         | 0.1      | 27       | 699        | 9.3e-03    | 1.1e-02  | 0.4     | 1.3e-02     | y2025_m04_d24_10h_54m_08s |
| subvpsde | True        | 128      | -         | -         | 0.25     | 30       | 646        | 2.2e-02    | 2.2e-02  | 0.4     | 2.0e-02     | y2025_m04_d24_16h_28m_53s |
| vesde    | True        | 128      | 0.05      | 5         | -        | -        | 699        | 2.0e-02    | 2.1e-02  | 0.4     | 3.0e-03     | y2025_m04_d24_19h_19m_14s |

['y2025_m04_d27_23h_30m_37s', 'y2025_m04_d02_09h_25m_13s', 'y2025_m04_d28_03h_18m_54s', 'y2025_m04_d24_10h_54m_08s', 'y2025_m04_d24_16h_28m_53s', 'y2025_m04_d24_19h_19m_14s']


- Project: Custom_CondAttUnet_lowSkip_turbulent_radiative_layer_2D
- Sort by: val loss

| SDE name | Energy loss | Batch sz | Sigma min | Sigma max | Beta min | Beta max | tot epochs | Train loss | Val loss | eLambda | Train eLoss | ID                        |
|----------|-------------|----------|-----------|-----------|----------|----------|------------|------------|----------|---------|-------------|---------------------------|
| vpsde    | False       | 128      | -         | -         | 0.1      | 25       | 525        | 6.8e-03    | 6.7e-03  | -       | -           | y2025_m04_d27_23h_30m_37s |
| subvpsde | False       | 128      | -         | -         | 0.4      | 30       | 960        | 1.2e-02    | 1.2e-02  | -       | -           | y2025_m04_d02_09h_25m_13s |
| vesde    | False       | 128      | 0.15      | 9.5       | -        | -        | 630        | 1.8e-02    | 1.1e-02  | -       | -           | y2025_m04_d11_09h_51m_09s |
| vpsde    | True        | 144      | -         | -         | 0.1      | 25       | 1064       | 1.3e-02    | 5.6e-03  | 0.15    | 2.5e-02     | y2025_m03_d12_14h_04m_25s |
| subvpsde | True        | 128      | -         | -         | 0.4      | 8.5      | 420        | 3.4e-02    | 2.0e-02  | 0.25    | 4.6e-02     | y2025_m04_d11_14h_43m_59s |
| vesde    | True        | 128      | 0.05      | 5         | -        | -        | 699        | 2.0e-02    | 2.1e-02  | 0.4     | 3.0e-03     | y2025_m04_d24_19h_19m_14s |

['y2025_m04_d27_23h_30m_37s', 'y2025_m04_d02_09h_25m_13s', 'y2025_m04_d11_09h_51m_09s', 'y2025_m03_d12_14h_04m_25s', 'y2025_m04_d11_14h_43m_59s', 'y2025_m04_d24_19h_19m_14s']



<u>old sampling</u>:

['y2025_m04_d14_10h_52m_35s', 'y2025_m04_d02_09h_25m_13s', 'y2025_m04_d11_09h_51m_09s', 'y2025_m04_d24_10h_54m_08s', 'y2025_m04_d24_16h_28m_53s', 'y2025_m04_d24_19h_19m_14s']

<u>sampling</u>:

['y2025_m04_d27_23h_30m_37s', 'y2025_m04_d02_09h_25m_13s', 'y2025_m04_d28_03h_18m_54s', 'y2025_m04_d24_10h_54m_08s', 'y2025_m04_d24_16h_28m_53s', 'y2025_m04_d24_19h_19m_14s']

