## The Well: MHD 64


- Project: Custom_CondAttUnet_lowSkip_3D_MHD_64
- Sort by: train loss

| SDE name | Energy loss | Batch sz | Sigma min | Sigma max | Beta min | Beta max | tot epochs | Train loss | Val loss | eLambda | Train eLoss | ID                        |
|----------|-------------|----------|-----------|-----------|----------|----------|------------|------------|----------|---------|-------------|---------------------------|
| vpsde    | False       | 12       | -         | -         | 0.1      | 25       | 5          | 1.1e-01    | 1.2e-01  | -       | -           | y2025_m03_d13_15h_27m_31s |
| subvpsde | False       | 10       | -         | -         | 0.25     | 30       | 87         | 2.3e-01    | 1.2e-01  | -       | -           | y2025_m04_d17_22h_15m_44s |
| vesde    | False       | 10       | 0.1       | 6         | -        | -        | 145        | 3.7e-01    | 1.8e-01  | -       | -           | y2025_m04_d18_16h_00m_24s |
| vpsde    | True        | 10       | -         | -         | 0.15     | 7        | 83         | 1.2e-01    | 1.3e-01  | 0.15    | 9.7e-03     | y2025_m04_d26_03h_37m_16s |
| subvpsde | True        | 10       | -         | -         | 0.1      | 15       | 14         | 1.3e-01    | 2.1e-01  | 0.1     | 2.4e-02     | y2025_m04_d27_11h_19m_09s |
| vesde    | True        | 10       | 0.6       | 4         | -        | -        | 146        | 4.3e-02    | 4.8e-02  | 0.28    | 1.5e-02     | y2025_m04_d25_15h_48m_56s |

['y2025_m03_d13_15h_27m_31s', 'y2025_m04_d17_22h_15m_44s', 'y2025_m04_d18_16h_00m_24s', 'y2025_m04_d26_03h_37m_16s', 'y2025_m04_d27_11h_19m_09s', 'y2025_m04_d25_15h_48m_56s']


- Project: Custom_CondAttUnet_lowSkip_3D_MHD_64
- Sort by: val loss

| SDE name | Energy loss | Batch sz | Sigma min | Sigma max | Beta min | Beta max | tot epochs | Train loss | Val loss | eLambda | Train eLoss | ID                        |
|----------|-------------|----------|-----------|-----------|----------|----------|------------|------------|----------|---------|-------------|---------------------------|
| vpsde    | False       | 10       | -         | -         | 0.1      | 30       | 73         | 1.4e-01    | 7.1e-02  | -       | -           | y2025_m04_d17_19h_24m_01s |
| subvpsde | False       | 10       | -         | -         | 0.25     | 30       | 87         | 2.3e-01    | 1.2e-01  | -       | -           | y2025_m04_d17_22h_15m_44s |
| vesde    | False       | 10       | 0.1       | 6         | -        | -        | 145        | 3.7e-01    | 1.8e-01  | -       | -           | y2025_m04_d18_16h_00m_24s |
| vpsde    | True        | 10       | -         | -         | 0.05     | 20       | 76         | 1.8e-01    | 8.7e-02  | 0.25    | 3.0e-02     | y2025_m04_d19_12h_20m_01s |
| subvpsde | True        | 9        | -         | -         | 0.1      | 12       | 55         | 1.9e-01    | 2.0e-01  | 0.1     | 1.1e-02     | y2025_m04_d27_15h_04m_53s |
| vesde    | True        | 10       | 0.6       | 4         | -        | -        | 146        | 4.3e-02    | 4.8e-02  | 0.28    | 1.5e-02     | y2025_m04_d25_15h_48m_56s |

['y2025_m04_d17_19h_24m_01s', 'y2025_m04_d17_22h_15m_44s', 'y2025_m04_d18_16h_00m_24s', 'y2025_m04_d19_12h_20m_01s', 'y2025_m04_d27_15h_04m_53s', 'y2025_m04_d25_15h_48m_56s']


<u>sampling</u>:

['y2025_m04_d17_19h_24m_01s', 'y2025_m04_d17_22h_15m_44s', 'y2025_m04_d18_16h_00m_24s', 'y2025_m04_d26_03h_37m_16s', 'y2025_m04_d27_15h_04m_53s', 'y2025_m04_d27_17h_17m_00s']

ve true: y2025_m04_d27_17h_17m_00s
Energy loss:true
ID:"y2025_m04_d27_17h_17m_00s"
Lambda energy loss:0.1
Learning rate:0.0001
Likelihood weighting:false
Max epochs:350
Model name:"Custom_CondAttUnet_lowSkip_3D"
Num scales (N):1,000
SDE name:"vesde"
Seed:123,456,789
Sigma max:4
Sigma min:0.01
T_max (T):1
train loss:0.4717815220355987
val_loss:0.4587558507919311