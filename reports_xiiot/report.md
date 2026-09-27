# X-IIoTID — prima baseline con split bloccato

Base: 5bb5b62 (merge PR #4). Questa consegna è il primo commit del branch `feat/xiiotid-baseline`;
l'hash è riportato sulla scheda Trello.

## Dataset registrato

| Voce | Valore |
|---|---|
| File | `X-IIoTID dataset.csv` |
| sha256 | `7b9290057ee42e784da3c0d84b781815502c9205c74175c96374e71a5ffd98a0` |
| Righe × colonne | 820.834 × 68 |
| Target | `class2`, 10 classi |
| Feature usate | 54 |
| Seed | 42 |

Registrazione completa in [`manifest.json`](manifest.json), prodotto dall'audit.

## Feature escluse e perché

| Gruppo | Colonne | Motivo |
|---|---|---|
| Etichette | `class1`, `class2`, `class3` | sono il target e le sue varianti |
| Alert di altri IDS | `anomaly_alert`, `OSSEC_alert`, `OSSEC_alert_level` | derivano dal target: sono già una decisione di rilevamento |
| Identificativi del testbed | `Date`, `Timestamp`, `Scr_IP`, `Des_IP`, `Scr_port`, `Des_port` | legano il modello alla topologia dell'esperimento |
| Costanti | `Bad_checksum`, `is_SYN_with_RST` | un solo valore in tutto il dataset |

## Split

**Finestra canonica.** Le feature `Avg_*`/`Std_*` sono aggregati che il testbed ha calcolato su finestre
di 10 secondi, quindi la finestra è esattamente il tratto contiguo di flussi che condividono lo stesso
vettore di aggregati. Il bin `Timestamp // 10` ne è solo una proxy, e i dati la smentiscono: flussi
dentro lo stesso bin portano vettori di aggregati diversi, quindi il bin non è la finestra usata dal
testbed. Adottare la definizione canonica riduce la duplicazione residua dal 51,0% al 6,1% delle righe
di test e porta `crypto-ransomware` da 5 righe di training a 274.

**Tre parti.** `StratifiedGroupKFold` a 5 fold per isolare il test, poi a 4 fold per la validazione;
a ogni taglio si tiene il fold che massimizza la quota per classe più piccola sul lato più debole, così
nessuna classe manca da una parte. Regola deterministica, seed 42. Nessuna finestra è divisa fra due parti.

| Voce | Valore |
|---|---|
| Finestre canoniche | 59.493 |
| Righe train / validation / test | 492.604 / 164.104 / 164.126 (60/20/20) |
| sha256 indici train | `19e58a59e0523bd2…` |
| sha256 indici validation | `0da40240b4c2ad25…` |
| sha256 indici test | `31acb2ea12578f45…` |

Lo split è ricostruito da `iot_audit.xiiotid.split()` e i tre hash sono verificati dall'audit, quindi
tutti i confronti girano sullo stesso split. La validazione serve al confronto fra modelli, il test
produce i numeri riportati.

| class2 | train | validation | test |
|---|---|---|---|
| Normal | 252.850 | 84.284 | 84.283 |
| RDOS | 84.863 | 28.187 | 28.211 |
| Reconnaissance | 76.554 | 25.518 | 25.518 |
| Weaponization | 40.356 | 13.452 | 13.452 |
| Lateral _movement | 18.958 | 6.319 | 6.319 |
| Exfiltration | 13.280 | 4.427 | 4.427 |
| Tampering | 3.073 | 1.025 | 1.024 |
| C&C | 1.717 | 573 | 573 |
| Exploitation | 679 | 227 | 227 |
| crypto-ransomware | 274 | 92 | 92 |

**Perché non uno split cronologico.** Ogni classe di attacco è confinata in una finestra temporale:
Exfiltration e crypto-ransomware stanno interamente nell'ultimo quarto del periodo, Reconnaissance
nei primi due, Lateral movement nel terzo. Un taglio passato/futuro lascerebbe intere classi fuori
dal training.

## Audit target leakage: PASS

```
python scripts/xiiotid_audit.py --csv "data/X-IIoTID dataset.csv"

PASS [xiiotid] single target fixed (class2, 10 classes)
PASS [xiiotid] no label column among predictors
PASS [xiiotid] no IDS alert among predictors
PASS [xiiotid] no testbed identifier among predictors
PASS [xiiotid] train, validation and test are disjoint
PASS [xiiotid] no canonical window shared by any two parts
PASS [xiiotid] preprocessing fitted on train rows only
PASS [xiiotid] every class present in every part
PASS [xiiotid] split reproducible from the manifest
WARN [xiiotid] 6.1% of test rows share a host-resource vector with train
exit 0
```

Suite completa: `python -m pytest -q` → 19 passed (14 della base più 5 nuovi in
`tests/test_xiiotid_leakage.py`, che girano su dati sintetici e non richiedono il dataset).

## Baseline

```
python scripts/train_xiiotid_baseline.py --csv "data/X-IIoTID dataset.csv"
python scripts/train_xiiotid_baseline.py --csv "data/X-IIoTID dataset.csv" --models rf logreg --drop-host-resources
```

Quattro modelli sullo stesso split: LightGBM, XGBoost, Random Forest e Logistic Regression.
Gli iperparametri degli alberi seguono quelli già usati negli script `train_mc_*.py` del repository.

| Modello | Macro-F1 test | Macro-F1 validation | Balanced Accuracy | Costo attacchi mancati | Costo falsi allarmi | Tempo |
|---|---|---|---|---|---|---|
| Random Forest | 0,9837 | 0,9309 | 0,9752 | 4.417 | 94 | 45 s |
| XGBoost | 0,9814 | 0,9346 | 0,9864 | 2.863 | 114 | 149 s |
| LightGBM | 0,9773 | 0,9332 | 0,9876 | 2.606 | 130 | 109 s |
| Logistic Regression | 0,8157 | 0,7982 | 0,9648 | 4.723 | 5.364 | 81 s |

Recall per classe sul test, stesso split per tutti:

| class2 | Righe test | Random Forest | XGBoost | LightGBM | LogReg |
|---|---|---|---|---|---|
| Normal | 84.283 | 0,999 | 0,999 | 0,999 | 0,936 |
| RDOS | 28.211 | 1,000 | 1,000 | 1,000 | 1,000 |
| Reconnaissance | 25.518 | 0,978 | 0,983 | 0,980 | 0,867 |
| Weaponization | 13.452 | 0,991 | 0,999 | 0,995 | 0,969 |
| Lateral _movement | 6.319 | 0,979 | 0,984 | 0,986 | 0,968 |
| Exfiltration | 4.427 | 0,999 | 0,999 | 0,999 | 0,986 |
| Tampering | 1.024 | 0,993 | 0,993 | 0,991 | 0,997 |
| C&C | 573 | 0,914 | 0,958 | 0,974 | 0,969 |
| Exploitation | 227 | 0,899 | 0,960 | 0,974 | 0,956 |
| crypto-ransomware | 92 | 1,000 | 0,989 | 0,978 | 1,000 |

I tre modelli ad albero si equivalgono: Random Forest ha la Macro-F1 più alta, LightGBM la Balanced
Accuracy migliore e il costo di attacchi mancati più basso (2.606 contro 4.417), a parità di falsi
allarmi. La regressione logistica resta indietro di 17 punti di Macro-F1 e produce 40 volte più falsi
allarmi. La recall per classe sulla validazione è in `recall_per_class_validation` dentro ogni file
di metriche.

**Divario fra validazione e test.** La Macro-F1 di validazione è circa 5 punti sotto quella di test per
tutti e tre gli alberi. La differenza viene da una sola classe: `Exploitation`, 227 righe per parte, su
cui Random Forest ha precisione 0,287 in validazione (719 predizioni per 227 casi veri) contro 0,990 nel
test. La recall è invece stabile, 0,907 contro 0,899. È varianza di fold su una classe rara: i numeri di
test vanno letti come il lato favorevole di quella oscillazione.

Dettaglio e matrici di confusione in `metrics_rf.json`, `metrics_xgb.json`, `metrics_lgbm.json`,
`metrics_logreg.json`. Variante senza feature host, sullo stesso split, in `metrics_rf_no_host.json`
e `metrics_logreg_no_host.json`:

| Modello senza feature host | Macro-F1 test | Balanced Accuracy |
|---|---|---|
| Random Forest | 0,9596 | 0,9694 |
| Logistic Regression | 0,8143 | 0,9476 |

## Matrice attacco → gravità → costo

Prima versione in [`configs/xiiotid_severity_cost.json`](../configs/xiiotid_severity_cost.json):
gravità da 1 a 5 secondo la fase della kill chain e l'impatto sul processo industriale, costo del
falso negativo separato dal costo del falso allarme, in unità relative. I valori sono la proposta
della responsabile della linea, sottoposta al relatore per validazione.

| class2 | Gravità | Costo FN | Costo FP |
|---|---|---|---|
| Normal | 0 | — | 1 |
| Reconnaissance | 1 | 2 | 1 |
| Weaponization | 2 | 4 | 1 |
| RDOS | 3 | 8 | 2 |
| Exploitation | 4 | 12 | 2 |
| Lateral _movement | 4 | 15 | 2 |
| C&C | 4 | 15 | 2 |
| Tampering | 5 | 20 | 2 |
| Exfiltration | 5 | 25 | 2 |
| crypto-ransomware | 5 | 30 | 2 |

Il costo riportato nella tabella della baseline somma il costo FN degli attacchi classificati come
`Normal` e il costo FP del traffico normale classificato come attacco.

## Limiti dichiarati

1. **crypto-ransomware**: 458 righe in tutto, divise in 274 / 92 / 92. Restano poche, ma la classe è
   ora allenabile e misurabile in tutte e tre le parti.
2. **Misure host ripetute**: il 6,1% delle righe di test (10.057) ha un vettore `Avg_*`/`Std_*` identico
   a una riga di training, perché lo stesso vettore può ripresentarsi in finestre distinte e non
   contigue. Nessuna finestra è divisa, e la definizione canonica ha ridotto il fenomeno da oltre metà
   del test a un ventesimo. Le righe identiche su tutte e 54 le feature sono lo 0,6% del test (1.018,
   in maggioranza RDOS), quindi il livello delle metriche non è spiegato da duplicazione.
3. **Exploitation instabile**: 227 righe per parte, precisione fra 0,29 e 0,99 secondo la parte. Le
   conclusioni su quella classe vanno prese con cautela.
4. **Definizione di finestra**: la finestra canonica è dedotta dai dati, non documentata dal dataset.
   Se il relatore disponesse di un identificativo ufficiale di finestra, va sostituito a questa deduzione.
5. **Costi**: la matrice gravità-costo è la proposta della responsabile, sottoposta al relatore.

## Ambiente

Python 3.14.7, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3, scipy 1.18.1,
Linux 7.2.4 x86_64. Tempi: audit ~50 s, Random Forest 45 s, Logistic Regression 81 s, LightGBM 109 s, XGBoost 149 s.
