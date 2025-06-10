## JHTDB: Transonic flow 128 x 64


- Project: Custom_CondAttUnet_lowSkip_JHTDB
- Sort by: train loss

| SDE name | Energy loss | Batch sz | Sigma min | Sigma max | Beta min | Beta max | tot epochs | Train loss | Val loss | eLambda | Train eLoss | ID                        |
|----------|-------------|----------|-----------|-----------|----------|----------|------------|------------|----------|---------|-------------|---------------------------|
| vpsde    | False       | 164      | -         | -         | 0.01     | 5        | 599        | 2.3e-03    | 2.7e-03  | -       | -           | y2025_m04_d24_11h_11m_40s |
| subvpsde | False       | 210      | -         | -         | 0.35     | 30       | 499        | 1.2e-02    | 5.7e-03  | -       | -           | y2025_m04_d09_23h_16m_29s |
| vesde    | False       | 210      | 0.15      | 1         | -        | -        | 463        | 1.5e-03    | 2.9e-03  | -       | -           | y2025_m04_d10_06h_23m_07s |
| vpsde    | True        | 220      | -         | -         | 0.39     | 5.6      | 437        | 1.8e-03    | 1.9e-03  | 0.25    | 1.3e-05     | y2025_m04_d24_00h_29m_31s |
| subvpsde | True        | 220      | -         | -         | 0.4      | 28       | 380        | 1.0e-02    | 9.6e-03  | 0.25    | 9.1e-04     | y2025_m04_d24_03h_35m_40s |
| vesde    | True        | 220      | 0.05      | 8         | -        | -        | 599        | 5.7e-03    | 5.9e-03  | 0.25    | 2.4e-04     | y2025_m04_d24_06h_14m_58s |

['y2025_m04_d24_11h_11m_40s', 'y2025_m04_d09_23h_16m_29s', 'y2025_m04_d10_06h_23m_07s', 'y2025_m04_d24_00h_29m_31s', 'y2025_m04_d24_03h_35m_40s', 'y2025_m04_d24_06h_14m_58s']


- Project: Custom_CondAttUnet_lowSkip_JHTDB
- Sort by: val loss

| SDE name | Energy loss | Batch sz | Sigma min | Sigma max | Beta min | Beta max | tot epochs | Train loss | Val loss | eLambda | Train eLoss | ID                        |
|----------|-------------|----------|-----------|-----------|----------|----------|------------|------------|----------|---------|-------------|---------------------------|
| vpsde    | False       | 164      | -         | -         | 0.01     | 5        | 599        | 2.3e-03    | 2.7e-03  | -       | -           | y2025_m04_d24_11h_11m_40s |
| subvpsde | False       | 210      | -         | -         | 0.35     | 30       | 499        | 1.2e-02    | 5.7e-03  | -       | -           | y2025_m04_d09_23h_16m_29s |
| vesde    | False       | 210      | 0.15      | 1         | -        | -        | 463        | 1.5e-03    | 2.9e-03  | -       | -           | y2025_m04_d10_06h_23m_07s |
| vpsde    | True        | 220      | -         | -         | 0.39     | 5.6      | 437        | 1.8e-03    | 1.9e-03  | 0.25    | 1.3e-05     | y2025_m04_d24_00h_29m_31s |
| subvpsde | True        | 220      | -         | -         | 0.4      | 28       | 380        | 1.0e-02    | 9.6e-03  | 0.25    | 9.1e-04     | y2025_m04_d24_03h_35m_40s |
| vesde    | True        | 200      | 0.05      | 8.5       | -        | -        | 599        | 1.0e-02    | 5.5e-03  | 0.35    | 1.1e-01     | y2025_m04_d11_02h_18m_48s |

['y2025_m04_d24_11h_11m_40s', 'y2025_m04_d09_23h_16m_29s', 'y2025_m04_d10_06h_23m_07s', 'y2025_m04_d24_00h_29m_31s', 'y2025_m04_d24_03h_35m_40s', 'y2025_m04_d11_02h_18m_48s']

<u>sampling</u>:

['y2025_m04_d24_11h_11m_40s', 'y2025_m04_d09_23h_16m_29s', 'y2025_m04_d28_08h_53m_22s', 'y2025_m04_d24_00h_29m_31s', 'y2025_m04_d24_03h_35m_40s', 'y2025_m04_d24_06h_14m_58s']

ve false: "y2025_m04_d28_08h_53m_22s"

Batch size train:210
Batch size valid:100

Dataset name:"JHTDB"
Energy loss:false

Max epochs:600

SDE name:"vesde"

Sigma max:8
Sigma min:0.04
T_max (T):1

train loss:0.0099
val_loss:0.01
