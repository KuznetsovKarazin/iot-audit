# X-IIoTID — robustezza della scelta del modello a costi incerti

Base: `a0eb070`. Nessun nuovo fit e nessuna nuova inferenza: aritmetica razionale esatta sulle matrici
di `reports_xiiot/`. Protocollo fissato prima dei calcoli in
[`configs/xiiotid_cost_robustness.json`](../configs/xiiotid_cost_robustness.json); hash in
[`manifest.json`](manifest.json). Comando: `python scripts/xiiotid_cost_robustness.py`.

## Domanda

λ = 1. I 18 pesi (`cost_fn` e `cost_fp` delle 9 classi di attacco) variano ciascuno in modo indipendente
in [(1−δ)·w⁰, (1+δ)·w⁰], con δ in [0, 1]. Fino a quale δ regge la scelta su validation? È un insieme di
scenari, non una probabilità, una stima del danno o un intervallo di confidenza.

## 1. Margini e raggi su validation

Il raggio è il δ oltre il quale esiste un vettore di pesi che rende m più costoso di k. Con δ minore del
raggio m è strettamente migliore in ogni scenario; al raggio c'è parità nel caso estremo. I 18 pesi di
quel caso, che attestano la parità, sono in [`pairwise.csv`](pairwise.csv), con i conteggi in
[`counts.csv`](counts.csv).

| m | k | Costo m | Costo k | Margine | Denominatore | Raggio esatto | Raggio |
|---|---|---|---|---|---|---|---|
| lgbm | xgb | 2.758 | 3.222 | 464 | 652 | 116/163 | 0,7117 |
| lgbm | logreg | 2.758 | 14.408 | 11.650 | 14.042 | 5825/7021 | 0,8297 |
| lgbm | rf | 2.758 | 5.412 | 2.654 | 2.942 | 1327/1471 | 0,9021 |
| xgb | logreg | 3.222 | 14.408 | 11.186 | 14.520 | 5593/7260 | 0,7704 |
| xgb | rf | 3.222 | 5.412 | 2.190 | 2.426 | 1095/1213 | 0,9027 |
| rf | logreg | 5.412 | 14.408 | 8.996 | 16.262 | 26/47 | 0,5532 |

`lgbm` è quindi il meno costoso in ogni scenario per δ < 116/163 ≈ 0,7117. Al confine pareggia con `xgb`;
oltre il confine esistono scenari in cui `xgb` costa meno.

## 2. Minimax regret e confronto fra le scelte

Spareggio dichiarato prima del calcolo: costo nominale di validation, poi Macro-F1 di validation, poi
ordine fisso lgbm, xgb, rf, logreg. Non è mai servito. Intervalli esatti in
[`minimax_regret.csv`](minimax_regret.csv), grafico in [`regret_frontier.png`](regret_frontier.png).

| Scelta su validation | Modello | Regret massimo a δ = 0 / 116/163 / 1 |
|---|---|---|
| Minimax regret, per ogni δ in [0, 1] | lgbm | 0 / 0 / 2.392 |
| Nominale, λ = 1 | lgbm | 0 / 0 / 2.392 |
| Macro-F1 (0,9346 contro 0,9332 di lgbm) | xgb | 464 / 928 / 3.334 |

Le scelte minimax e nominale coincidono; quella per Macro-F1 no. Su [0, 1] il regret massimo di `xgb`
supera quello di `lgbm` di una quantità fra 464 e 942. Il minimax regret è un criterio standard: qui non
c'è novità algoritmica, e non minimizza il costo in ogni scenario, ma limita il rimpianto peggiore.

## 3. Controllo retrospettivo sul test

Il test era già stato osservato, quindi questo è un controllo descrittivo e non una conferma
indipendente; non ha influito su selezione, ipotesi o spareggio. `lgbm` resta il modello a regret minimo
per ogni δ, ma il primo confine scende da 0,7117 a 55/214 ≈ 0,2570 (costi 2.794 e 3.014, 220/856). Il
margine su `xgb` si riduce soprattutto perché sul test `lgbm` manca più `Weaponization` (57 contro 6, cioè
+204 di costo). Il vantaggio è quindi molto meno ampio di quanto dica la sola validation.

## 4. Errori attacco→altro attacco

Sono descritti a parte e fuori dal costo: validation lgbm 640, xgb 586, rf 585, logreg 1.782; test lgbm
180, xgb 89, rf 95, logreg 1.985. `lgbm` ne fa di più degli altri due alberi, e la scelta per costo non lo
vede. Negli alberi il divario fra le due parti è la varianza su `Exploitation` già documentata nella
baseline.

## 5. Verifiche

Quattro controlli dello script danno PASS: etichette, costi uguali a RISK-02, parità ai vettori estremi,
copertura esatta di [0, 1]. In `tests/test_cost_robustness.py`: δ = 0, esempio sintetico con cambio di
vincitore e con parità, formula contro enumerazione dei vertici in 2 e 3 dimensioni, invarianza della
selezione su validation quando cambia il test, riproduzione dei valori della scheda.
`python -m pytest -q` dà 54 passed. Non sono emersi errori negli ingressi.

## 6. Limiti

- Una sola partizione: i confini non hanno un'incertezza stimata, e il test mostra che possono spostarsi
  molto.
- Le variazioni indipendenti possono cambiare l'ordine dei costi: `Lateral _movement` e `C&C` si separano
  per ogni δ > 0, `Exfiltration` può superare `crypto-ransomware` da δ > 1/11. A δ = 1 un peso può essere
  nullo: è una prova di stress, non uno scenario plausibile.
- I pesi w⁰ restano ipotesi relative, e i 18 pesi coprono solo gli errori verso e da `Normal`.
- Il risultato vale per questi 4 modelli: non certifica una generalizzazione.

## 7. Proposta minima di conferma successiva

Non avviata. Il test attuale non può più fare da conferma, e un nuovo split dello stesso dataset
misurerebbe la variabilità della partizione, non l'indipendenza. Il minimo è applicare il protocollo
congelato a dati etichettati non ancora osservati (una nuova cattura o un dataset con classi mappabili),
dichiarando prima di guardarli la scelta (`lgbm`) e il confronto da riportare: segno del margine e raggio
rispetto a ogni altro modello. Richiede nuove inferenze, quindi va concordata.
