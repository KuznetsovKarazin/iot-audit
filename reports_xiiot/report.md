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

## Schema dei predittori e feature escluse

Lo schema è deciso sulle sole righe di training: le colonne costanti e la distinzione fra numeriche e
categoriche si ricavano dal training e si applicano invariate a validation e test.

| Gruppo | Colonne | Motivo |
|---|---|---|
| Etichette | `class1`, `class2`, `class3` | sono il target e le sue varianti |
| Alert di altri IDS | `anomaly_alert`, `OSSEC_alert`, `OSSEC_alert_level` | sono output di altri sistemi di rilevamento, cioè una decisione già presa, non etichette derivate dal target |
| Identificativi del testbed | `Date`, `Timestamp`, `Scr_IP`, `Des_IP`, `Scr_port`, `Des_port` | legano il modello alla topologia dell'esperimento |
| Costanti | `Bad_checksum`, `is_SYN_with_RST` | un solo valore in tutto il dataset |

## Split

**Finestra inferita, non ufficiale.** Il dataset documenta che le feature `Avg_*`/`Std_*` sono aggregate
su 10 secondi (DOI 10.1109/JIOT.2021.3102056) ma non contiene un identificativo di finestra. I gruppi
usati qui sono quindi una ricostruzione: tratti contigui di righe che condividono lo stesso vettore di
aggregati. Il bin `Timestamp // 10` è una proxy più debole, perché flussi dentro lo stesso bin portano
vettori diversi. Raggruppare per questi tratti tiene aggregati identici dallo stesso lato dello split;
non dimostra indipendenza temporale né assenza di leakage in generale. Rispetto alla proxy, la
duplicazione residua delle misure host scende dal 51,0% al 6,1% delle righe di test e
`crypto-ransomware` passa da 5 righe di training a 274.

**Diagnostiche dei gruppi inferiti**, in `manifest.json` sotto `group_diagnostics`:

| Voce | Valore |
|---|---|
| Gruppi costruiti | 59.493 |
| Righe per gruppo (mediana / p95 / max) | 5 / 34 / 5.707 |
| Durata del gruppo in secondi (mediana / p95) | 6 / 9 |
| Gruppi più lunghi della finestra documentata di 10 s | 7 (144 righe), di cui 3 oltre un'ora |
| Righe con timestamp non numerico | 451 |
| Timestamp che portano più di un gruppo | 1.364 |
| Gruppi che attraversano più di un bin da 10 s | 28.470 |

La durata mediana di 6 secondi e il p95 di 9 sono compatibili con l'aggregazione documentata, e solo
7 gruppi su 59.493 la superano; il più lungo copre 96 giorni, segno che la ricostruzione non è esatta
in coda. I 1.364 timestamp con più di un gruppo sono la prova che il bin da 10 secondi non coincide
con la finestra del testbed.

**Tre parti.** `StratifiedGroupKFold` a 5 fold per isolare il test, poi a 4 fold per la validazione;
a ogni taglio si tiene il fold che massimizza la quota per classe più piccola sul lato più debole, così
nessuna classe manca da una parte. Regola deterministica, seed 42. Nessuno dei gruppi costruiti è diviso fra due parti; la
disgiunzione è verificata dall'audit sulle tre coppie.

| Voce | Valore |
|---|---|
| Gruppi (finestre inferite) | 59.493 |
| Righe train / validation / test | 492.604 / 164.104 / 164.126 (60/20/20) |
| sha256 indici train | `19e58a59e0523bd2…` |
| sha256 indici validation | `0da40240b4c2ad25…` |
| sha256 indici test | `31acb2ea12578f45…` |

Lo split è ricostruito da `iot_audit.xiiotid.split()`. Il manifest è scritto una volta sola, con
`--write`; da allora audit, training e controllo della Definition of Done confrontano sha256 del
dataset e i tre hash degli indici con quanto registrato, e si fermano sulle differenze senza
sovrascrivere il file. Ogni file di metriche riporta commit, hash e configurazione del modello, quindi
l'appartenenza allo stesso split è verificabile a posteriori. La validazione serve al confronto fra modelli, il test
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
python scripts/xiiotid_audit.py --csv "data/X-IIoTID dataset.csv"   # --write per registrarlo

PASS [xiiotid] single target fixed (class2, 10 classes)
PASS [xiiotid] no label column among predictors
PASS [xiiotid] no third-party IDS alert among predictors
PASS [xiiotid] no testbed identifier among predictors
PASS [xiiotid] train, validation and test are disjoint
PASS [xiiotid] feature schema decided on training rows only
PASS [xiiotid] no inferred window shared by any two parts
PASS [xiiotid] preprocessing fitted on train rows only
PASS [xiiotid] every class present in every part
PASS [xiiotid] dataset and split match the registered manifest
WARN [xiiotid] 6.1% of test rows share a host-resource vector with train
exit 0
```

Suite completa: `python -m pytest -q` → 22 passed (14 della base più 8 in
`tests/test_xiiotid_leakage.py`, che girano su dati sintetici e non richiedono il dataset). Fra questi
due test negativi: un manifest in disaccordo con lo split fa fallire l'esecuzione senza essere
sovrascritto, e modificare le righe di validation e test non cambia né le feature né i tipi scelti.

Controprova eseguita a mano sul dataset vero, alterando l'hash dell'indice di test nel manifest:
l'audit stampa `ERROR ... dataset and split match the registered manifest` ed esce con 1, il training
si ferma con `ValueError: manifest does not match this run`, e il manifest resta alterato, cioè non
viene riscritto.

## Baseline

```
python scripts/train_xiiotid_baseline.py --csv "data/X-IIoTID dataset.csv"
python scripts/train_xiiotid_baseline.py --csv "data/X-IIoTID dataset.csv" --models rf logreg --drop-host-resources
```

Quattro modelli sullo stesso split: LightGBM, XGBoost, Random Forest e Logistic Regression.
Gli iperparametri degli alberi seguono quelli già usati negli script `train_mc_*.py` del repository.

| Modello | Macro-F1 test | Macro-F1 validation | Balanced Accuracy | Costo attacchi mancati | Costo falsi allarmi | Attacchi mal classificati (non pesati) | Tempo |
|---|---|---|---|---|---|---|---|
| Random Forest | 0,9837 | 0,9309 | 0,9752 | 4.417 | 117 | 95 | 55 s |
| XGBoost | 0,9814 | 0,9346 | 0,9864 | 2.863 | 151 | 89 | 145 s |
| LightGBM | 0,9773 | 0,9332 | 0,9876 | 2.606 | 188 | 180 | 562 s |
| Logistic Regression | 0,8157 | 0,7982 | 0,9648 | 4.723 | 9.538 | 1.985 | 93 s |

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

Lettura, senza sintesi forzate. I tre modelli ad albero sono vicini ma non equivalenti, e si ordinano
in modo diverso secondo la metrica: Random Forest ha la Macro-F1 di test più alta (0,9837 contro 0,9773
di LightGBM) e il costo di falsi allarmi più basso (117 contro 188), mentre LightGBM ha la Balanced
Accuracy migliore (0,9876) e il costo di attacchi mancati più basso (2.606 contro 4.417, il 41% in meno). Quale sia preferibile dipende da come si pesano i due costi, e la matrice della sezione
successiva è lo strumento per deciderlo. XGBoost sta in mezzo su tutte le voci. La regressione
logistica resta indietro di 17 punti di Macro-F1 e produce da 50 a 80 volte più costo di falsi allarmi.
Sulla validazione l'ordine cambia: XGBoost è primo (0,9346) e Random Forest ultimo dei tre alberi
(0,9309), un'altra ragione per non dichiarare un vincitore su una singola cifra. La recall per classe
sulla validazione e la matrice di confusione di validazione sono in ogni file di metriche.

**Divario fra validazione e test.** La Macro-F1 di validazione è circa 5 punti sotto quella di test per
tutti e tre gli alberi. La differenza viene da una sola classe: `Exploitation`, 227 righe per parte, su
cui Random Forest ha precisione 0,287 in validazione (719 predizioni per 227 casi veri) contro 0,990 nel
test. La recall è invece stabile, 0,907 contro 0,899. È varianza di fold su una classe rara: i numeri di
test vanno letti come il lato favorevole di quella oscillazione.

Dettaglio e matrici di confusione in `metrics_rf.json`, `metrics_xgb.json`, `metrics_lgbm.json`,
`metrics_logreg.json`. Variante senza feature host, sullo stesso split, in `metrics_rf_no_host.json`
e `metrics_logreg_no_host.json`:

| Modello senza feature host | Macro-F1 test | Macro-F1 validation | Balanced Accuracy | Feature |
|---|---|---|---|---|
| Random Forest | 0,9596 | 0,9399 | 0,9694 | 30 |
| Logistic Regression | 0,8143 | 0,7950 | 0,9476 | 30 |

I due file no-host hanno gli stessi campi degli altri, compresi provenance, recall per classe di
validazione e matrice di confusione di validazione.

## Matrice attacco → gravità → costo

Prima versione in [`configs/xiiotid_severity_cost.json`](../configs/xiiotid_severity_cost.json):
gravità da 0 a 5 secondo la fase della kill chain e l'impatto sul processo industriale, costo del
falso negativo separato dal costo del falso allarme, in unità relative. È un'ipotesi di scenario con
pesi relativi motivati uno per uno, non una misura empirica del danno, e resta sottoposta al relatore
per validazione.

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

Come la applica `expected_cost()`: un attacco classificato come `Normal` pesa il `cost_fn` della sua
classe vera; traffico normale classificato come attacco pesa il `cost_fp` della classe che il modello
ha alzato, quindi i valori per classe della tabella sono quelli effettivamente usati. Le confusioni fra
attacchi diversi sono **contate ma non pesate**, nel campo `mistriaged_attacks_not_weighted`: un
attacco riconosciuto come attacco di altra fase resta un allarme alzato, e attribuirgli un costo
richiederebbe un'ipotesi sulla risposta operativa che qui non facciamo.

## Limiti dichiarati

1. **crypto-ransomware**: 458 righe in tutto, divise in 274 / 92 / 92. Restano poche, ma la classe è
   ora allenabile e misurabile in tutte e tre le parti.
2. **Misure host ripetute**: il 6,1% delle righe di test (10.057) ha un vettore `Avg_*`/`Std_*` identico
   a una riga di training, perché lo stesso vettore può ripresentarsi in gruppi distinti e non contigui.
   Nessun gruppo è diviso, e il passaggio dalla proxy temporale ai gruppi inferiti ha ridotto il
   fenomeno da oltre metà del test a un ventesimo. Le righe identiche su tutte le feature sono lo 0,6%
   del test (1.018, in maggioranza RDOS): una duplicazione esatta così limitata non spiega da sola il
   livello delle metriche, ma non esclude altre forme di somiglianza fra righe vicine.
3. **Exploitation instabile**: 227 righe per parte, precisione fra 0,29 e 0,99 secondo la parte. Le
   conclusioni su quella classe vanno prese con cautela; la matrice di confusione di validazione è
   esportata in ogni file di metriche per permettere la verifica.
4. **Definizione dei gruppi**: sono gruppi inferiti dai dati, non identificativi ufficiali di finestra.
   Le diagnostiche sopra ne mostrano l'aderenza ai 10 secondi documentati e i casi fuori scala. Se
   esiste un identificativo ufficiale, va sostituito a questa ricostruzione.
5. **Costi**: ipotesi di scenario, non misura empirica; le confusioni fra attacchi sono contate e non
   pesate.
6. **Portata della verifica**: 22 test PASS, audit PASS e coerenza fra metriche e matrici di confusione
   non equivalgono alla validazione completa del protocollo.

## Ambiente

Python 3.14.7, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3, scipy 1.18.1,
Linux 7.2.4 x86_64. Tempi dell'ultima esecuzione: audit circa 50 s, Random Forest 55 s, Logistic
Regression 93 s, XGBoost 145 s, LightGBM 562 s, varianti no-host 30 e 113 s, controllo della
Definition of Done meno di un secondo.

Il campo `provenance.commit` di ogni file di metriche è il commit da cui l'esecuzione è partita, quindi
precede il commit che aggiunge gli artefatti; `provenance.source_sha256` riporta gli sha256 dei file
eseguiti — `xiiotid.py`, lo script di training e la config dei costi — così la versione esatta resta
verificabile anche quando `tree_clean` è falso.
