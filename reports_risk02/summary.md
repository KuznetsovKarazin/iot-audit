# X-IIoTID — analisi dei costi e regioni di scelta (primo checkpoint)

Base: `f35be31` (upstream/main, PR #5 integrata). Nessun nuovo addestramento: tutto è ricostruito
dalle matrici di confusione di validation e test dei sei JSON già accettati, che restano immutati.
Gli sha256 degli otto input sono in [`inputs_sha256.csv`](inputs_sha256.csv).

Comando: `python scripts/xiiotid_cost_regions.py`.

## Domanda

Quanto cambia il classificatore preferito quando cambia il peso relativo fra attacchi mancati e
falsi allarmi? Analisi esplorativa: non si dichiara un vincitore universale.

## 1. Conteggi e costi ricostruiti

Per ogni variante e parte, da ogni matrice: attacchi classificati `Normal` (colonna `Normal`),
traffico `Normal` classificato come attacco (riga `Normal`), i costi corrispondenti con i pesi per
classe della configurazione accettata, e le confusioni fra attacchi diversi, che restano **contate e
non pesate**. Tabella completa in [`counts_costs.csv`](counts_costs.csv); FPR calcolato sul solo
traffico `Normal`.

| Variante | Parte | Mancati | C_miss | Falsi all. | C_fa | Confusioni | C_miss/1k | C_fa/1k | FPR |
|---|---|---|---|---|---|---|---|---|---|
| lgbm | validation | 444 | 2.530 | 149 | 228 | 640 | 15,42 | 1,39 | 0,00177 |
| lgbm | test | 509 | 2.606 | 130 | 188 | 180 | 15,88 | 1,15 | 0,00154 |
| xgb | validation | 499 | 3.047 | 123 | 175 | 586 | 18,57 | 1,07 | 0,00146 |
| xgb | test | 504 | 2.863 | 114 | 151 | 89 | 17,44 | 0,92 | 0,00135 |
| rf | validation | 773 | 5.281 | 108 | 131 | 585 | 32,18 | 0,80 | 0,00128 |
| rf | test | 808 | 4.417 | 94 | 117 | 95 | 26,91 | 0,71 | 0,00112 |
| logreg | validation | 2.166 | 4.972 | 5.281 | 9.436 | 1.782 | 30,30 | 57,50 | 0,06266 |
| logreg | test | 2.121 | 4.723 | 5.364 | 9.538 | 1.985 | 28,78 | 58,11 | 0,06364 |
| rf_no_host | validation | 424 | 2.946 | 483 | 764 | 443 | 17,95 | 4,66 | 0,00573 |
| rf_no_host | test | 485 | 2.754 | 481 | 758 | 400 | 16,78 | 4,62 | 0,00571 |
| logreg_no_host | validation | 2.352 | 6.028 | 2.558 | 4.480 | 2.822 | 36,73 | 27,30 | 0,03035 |
| logreg_no_host | test | 2.288 | 5.478 | 2.656 | 4.639 | 3.997 | 33,38 | 28,27 | 0,03151 |

## 2. C(λ) e intervalli di preferenza

C(λ) = C_miss + λ·C_falsi_allarmi, λ ≥ 0. λ pesa il lato degli allarmi rispetto a quello degli
attacchi mancati, **a parità di pesi per classe** della configurazione accettata: λ = 1 riproduce
esattamente quella configurazione. I pesi restano ipotesi relative, non danni monetari misurati.
Ogni variante è una retta, il preferito è l'inviluppo inferiore. Nei punti di incrocio le varianti
costano uguale: il confine appartiene all'intervallo che inizia lì, cioè alla variante con il costo
di falsi allarmi più basso, quindi gli intervalli sono chiusi a sinistra e aperti a destra. Intervalli in
[`choice_regions.csv`](choice_regions.csv), grafico in [`choice_regions.png`](choice_regions.png).

| Parte | λ < soglia₁ | fra le soglie | λ > soglia₂ |
|---|---|---|---|
| validation | lgbm per λ < 9,75 | xgb fra 9,75 e 50,77 | rf oltre 50,77 |
| test | lgbm per λ < 6,95 | xgb fra 6,95 e 45,71 | rf oltre 45,71 |

`logreg` non entra mai nell'inviluppo, su nessuna delle due parti.

## 3. Selettore fissato sulla validation

Regola dichiarata in [`configs/xiiotid_lambda.json`](../configs/xiiotid_lambda.json) prima di
leggere qualunque cifra di test: minimo di C(λ) sul pool dei quattro modelli host, pareggi risolti
per C_miss crescente e poi per nome. Sul test la regola si applica invariata. Tabella completa in
[`selector.csv`](selector.csv).

| λ | Scelta su validation | Costo su test della scelta | Scelta per Macro-F1 di validation | Costo su test | Minimo sul test (solo diagnostico) |
|---|---|---|---|---|---|
| 0 | lgbm | 2.606 | xgb | 2.863 | lgbm, 2.606 |
| 1 | lgbm | 2.794 | xgb | 3.014 | lgbm, 2.794 |
| 5 | lgbm | 3.546 | xgb | 3.618 | lgbm, 3.546 |
| 10 | xgb | 4.373 | xgb | 4.373 | xgb, 4.373 |
| 50 | xgb | 10.413 | xgb | 10.413 | rf, 10.267 |
| 100 | rf | 16.117 | xgb | 17.963 | rf, 16.117 |

Il selettore per Macro-F1 di validation sceglie `xgb` (0,9346 contro 0,9332 di `lgbm` e 0,9309 di
`rf`) per ogni λ, perché non dipende da λ. Il selettore a costo coincide con quello solo nella fascia
intermedia: a λ = 1 differiscono, e la scelta per Macro-F1 costa sul test 3.014 contro 2.794, cioè
il 7,9% in più. Il minimo ottenibile scegliendo direttamente sul test coincide con la scelta fatta
su validation in otto casi su nove della griglia; l'unica differenza è a λ = 50, dove il minimo sul
test sarebbe `rf` con 10.267 contro 10.413, l'1,4% in meno. È un riferimento diagnostico, non una
regola di selezione.

## 4. Controlli appaiati senza feature host

Confronto fra configurazioni dello stesso algoritmo, non esperimento fattoriale: le due varianti
no-host non entrano nel pool del selettore. Dati in [`paired_no_host.csv`](paired_no_host.csv).

| Coppia | Parte | C_miss con host → senza | C_falsi allarmi con host → senza |
|---|---|---|---|
| rf | validation | 5.281 → 2.946 | 131 → 764 |
| rf | test | 4.417 → 2.754 | 117 → 758 |
| logreg | validation | 4.972 → 6.028 | 9.436 → 4.480 |
| logreg | test | 4.723 → 5.478 | 9.538 → 4.639 |

Le due coppie si muovono in direzioni opposte sul lato mancati: per `rf` togliere le feature host
abbassa C_miss di circa il 38% ma moltiplica per sei il costo degli allarmi; per `logreg` alza
C_miss e dimezza gli allarmi. Con due sole coppie non si può separare l'effetto delle feature da
quello dell'algoritmo.

## 5. Verifiche eseguite

Lo script termina con codice 1 se una verifica fallisce. Esito attuale, quattro PASS:
costi ricalcolati uguali al blocco `cost` dei JSON accettati; mancati, falsi allarmi e confusioni fra
attacchi che partizionano esattamente gli errori fuori diagonale; normalizzazione per 1.000 righe
che usa le righe della propria parte; e una prova che altera i soli dati di test, verificando che le
rette di validation e tutte le scelte restino identiche mentre quelle di test cambiano. I casi limite
dell'inviluppo — intercette uguali, più rette che si incrociano nello stesso punto, rette dominate — e
un caso sintetico con conteggi noti per mancati, falsi allarmi, confusioni fra attacchi e
normalizzazione sono in `tests/test_cost_regions.py`: `python -m pytest -q` dà 28 passed.

## 6. Cosa resiste al confronto, e cosa no

Resiste l'ordine dell'inviluppo: `lgbm` per λ piccoli, `xgb` nella fascia intermedia, `rf` per λ
grandi, identico su validation e test, con `logreg` sempre fuori. Resiste il segno dei controlli
appaiati: per `rf` senza feature host meno attacchi mancati e molti più allarmi, su entrambe le
parti. Non resiste la posizione esatta delle soglie, che si spostano da 9,75 a 6,95 (−29%) e da
50,77 a 45,71 (−10%): con due sole partizioni non sappiamo se sia rumore o struttura. Non è
dimostrata una scelta unica robusta a tutti i λ, e non viene dichiarata.

Limite generale: il test era già stato esaminato nella baseline, quindi questo confronto non è una
conferma cieca; il selettore non è stato modificato dopo averne letto i risultati.

## 7. Esperimento minimo successivo

Ripetere lo split con semi diversi, a parità di regola, pesi e algoritmi, e osservare la
distribuzione delle due soglie di λ: è il minimo che distingue rumore da struttura. Richiede nuovi
addestramenti, quindi va concordato dopo il checkpoint.
