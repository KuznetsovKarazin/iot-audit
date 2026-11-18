# Scelta dell'IDS per Macro-F1, costo nominale e minimax regret

Nota di lavoro per le sezioni Methods/Results dell'articolo comune. Stato: X-IIoTID completo,
CICIoT2023 in attesa degli artefatti Validation/Test (scheda NUNZIO CIC-RISK-01). Protocollo comune
v1 fissato nel commit `a613981`, prima di ogni calcolo; codice in `b5e9e9a`; tabelle in
[`xiiotid/`](xiiotid/) con [`manifest.json`](xiiotid/manifest.json) (hash di input, protocollo,
codice e output; zero nuovi fit, zero nuove inferenze); figure in [`figures/`](figures/).

## 1. Domanda

Se invece del Macro-F1 si sceglie il modello con un costo che separa attacchi mancati e falsi
allarmi, o con il minimax regret su pesi incerti, la scelta cambia? Quando cambia, quanto resta
stabile al variare dei costi e della frequenza degli attacchi, e quali errori spiegano la
differenza? Non è richiesto che il minimax vinca o cambi modello: coincidenze e risultati
sfavorevoli sono riportati come tali.

## 2. Metodo

L'unità di analisi è la matrice di confusione completa di ciascun modello, split e seed, con le
etichette esplicite e l'hash del file. Il pool principale è rf, xgb, lgbm; logreg è un confronto
supplementare su X-IIoTID perché i suoi artefatti c'erano già. Su X-IIoTID esiste un solo seed (42),
sullo split a finestre canoniche già accettato (492.604 / 164.104 / 164.126 righe).

Un attacco classificato come benigno è un attacco mancato, il traffico benigno classificato come
attacco è un falso allarme; le confusioni fra tipi di attacco restano fuori dal costo e sono
riportate a parte. Il costo atteso per osservazione è

C = Σ_i [ λ·π·q_i·FN_i / n_i + (1 − π)·FP_i / n_benigno ],

dove FN_i e FP_i sono i conteggi della matrice, n_i i supporti della stessa matrice e q_i la quota
della classe i fra gli attacchi di Validation, tenuta fissa anche sul Test. Usare tassi condizionati
alla classe evita di confrontare costi grezzi fra dataset di dimensioni diverse. Gli scenari sono
π ∈ {0,001; 0,01; 0,1} per la quota di attacchi e λ ∈ {1; 10; 100} per il rapporto di costo FN/FP:
ipotesi di sensibilità, non frequenze o danni misurati. Con pesi uniformi il costo si scrive
C = λπ·A + (1 − π)·B, con A = Σ q_i·FNR_i e B = Σ FPR_i, quindi la scelta dipende solo dal rapporto
r = λπ / (1 − π); i nove scenari coprono r da 0,001 a 11,1.

Per la robustezza ciascuno dei pesi varia in modo indipendente in [(1 − δ)w, (1 + δ)w], δ ∈ [0, 1].
Margini, raggi e minimax regret usano la formula esatta già verificata in RISK-03, in aritmetica
razionale; le frontiere sono esatte su tutto [0, 1] e i riepiloghi sono a δ = 0; 0,25; 0,5; 1.
δ = 1 è una prova di stress che ammette pesi nulli.

Tre politiche scelgono su Validation: massimo Macro-F1 (ricalcolato in modo esatto dalla matrice,
uguale al valore riportato alla quarta cifra), minimo costo nominale dello scenario, minimo regret
massimo. Pareggi: costo nominale, Macro-F1, ordine fisso lgbm, xgb, rf. Le scelte sono congelate e
applicate al Test con gli stessi π, λ, δ e q. Il Test era già stato osservato nella baseline: è un
controllo retrospettivo e il miglior modello sul Test serve solo a misurare il regret.

## 3. Risultati su X-IIoTID

| π | λ | r | Macro-F1 | Costo nominale = minimax | Raggio | Regret sul Test, scelta F1 / scelta di costo | Migliore sul Test |
|---|---|---|---|---|---|---|---|
| 0,001 | 1 | 0,001 | xgb | rf | 0,34 | 23,3 / 0 | rf |
| 0,001 | 10 | 0,010 | xgb | rf | 0,26 | 19,9 / 0 | rf |
| 0,01 | 1 | 0,010 | xgb | rf | 0,26 | 19,7 / 0 | rf |
| 0,001 | 100 | 0,100 | xgb | xgb | 0,19 | 0 / 0 | xgb |
| 0,01 | 10 | 0,101 | xgb | xgb | 0,20 | 0 / 0 | xgb |
| 0,1 | 1 | 0,111 | xgb | xgb | 0,23 | 0 / 0 | xgb |
| 0,01 | 100 | 1,01 | xgb | lgbm | 0,32 | 0 / 25,1 | xgb |
| 0,1 | 10 | 1,11 | xgb | lgbm | 0,36 | 0 / 23,4 | xgb |
| 0,1 | 100 | 11,1 | xgb | lgbm | 0,80 | 0 / 79,8 | xgb |

Regret a δ = 0 per 100.000 osservazioni. Il raggio è il δ fino al quale la scelta nominale non costa
più di nessun altro modello, per qualunque combinazione di pesi; i valori esatti sono in
[`radii.csv`](xiiotid/radii.csv). Figure: [mappa delle scelte](figures/choice_map.png),
[stabilità](figures/stability.png), [errori per classe](figures/errors_by_class.png).

**Quando divergono le scelte.** Il Macro-F1 sceglie sempre xgb (0,9346 contro 0,9332 di lgbm e
0,9309 di rf). Le due politiche di costo se ne allontanano in sei scenari su nove: scelgono rf
quando pesano di più i falsi allarmi (r ≤ 0,01), lgbm quando pesano di più gli attacchi mancati
(r ≥ 1), e tornano a xgb solo nella fascia intermedia (r ≈ 0,1). Su Validation le soglie sono
r = 0,052 fra rf e xgb e r = 0,447 fra xgb e lgbm. Il minimax regret coincide con la scelta nominale
in tutti e nove gli scenari e per ogni δ in [0, 1]: non cambia mai modello, aggiunge soltanto il
limite al rimpianto peggiore.

**Quanto sono stabili.** Il raggio della scelta nominale va da 0,19 a 0,80. È più piccolo quando la
scelta è xgb (0,19–0,23), perché lì rf le resta vicino, ed è massimo a r = 11,1. Oltre il raggio
esistono combinazioni di pesi che rendono più economico un altro modello, ma il modello scelto
mantiene il regret massimo più basso fino a δ = 1. Anche la frontiera retrospettiva sul Test è
costante su tutto [0, 1] in ogni scenario: rf per r ≤ 0,01, xgb negli altri sei.

**Controllo retrospettivo sul Test.** Dove la scelta di costo è rf, rf resta il migliore anche sul
Test e xgb costerebbe il 16–21% in più (per esempio 135,8 contro 112,4 per 100.000 osservazioni a
π = 0,001, λ = 1). Nei tre scenari intermedi tutte le politiche e il Test indicano xgb. Dove la
scelta di costo è lgbm il risultato è sfavorevole: sul Test lgbm costa il 3,3%, il 3,1% e l'1,2% in
più di xgb, cioè la scelta per Macro-F1 sarebbe stata migliore. Il bilancio sul Test è quindi tre
scenari a favore della scelta di costo, tre a favore del Macro-F1 e tre di parità.

**Quali errori spiegano le differenze.** A separare rf da xgb sono due errori opposti. rf genera meno
falsi allarmi (B = 128 contro 146 per 100.000 su Validation), soprattutto verso Lateral _movement
(3 contro 19) e C&C (16 contro 23). In compenso manca molti più attacchi (A = 968 contro 625):
Reconnaissance (447 contro 351), RDOS (83 contro 1 su Validation, nessuno sul Test), Lateral
_movement (141 contro 98) e C&C (52 contro 20). Fra lgbm e xgb, su Validation lgbm manca meno Lateral
_movement (77 contro 98) e Reconnaissance (322 contro 351), ma dà più falsi allarmi verso C&C (36
contro 23), Lateral _movement (30 contro 19) e Tampering (12 contro 8). Sul Test lgbm manca 57
Weaponization contro 6: con q = 0,17 sono circa 64 punti per 100.000 in A, che annullano il suo
vantaggio (A = 638 contro 631, mentre B resta 154 contro 135). Sul Test xgb costa quindi meno di
lgbm in ogni scenario. È la stessa varianza fra le due parti già osservata in RISK-03. Le confusioni
fra attacchi, fuori dal costo, sono più numerose per lgbm: 640 contro 586 (xgb) e 585 (rf) su
Validation, 180 contro 89 e 95 sul Test, quasi tutte in Reconnaissance.

**Pool supplementare e seed.** Con logreg nel pool le scelte non cambiano: logreg non è mai scelta e
aumenta solo il regret massimo a δ = 1 negli scenari con r ≥ 1. X-IIoTID ha un solo seed, quindi non
c'è dispersione da riportare: le righe per seed di [`choices.csv`](xiiotid/choices.csv) coincidono
con quelle della media.

## 4. CICIoT2023

Non ancora analizzato. Il contratto degli input richiede 40 matrici: rf, xgb e lgbm, più kan se
disponibile, per i seed 101, 202, 303, 404 e 505, su Validation e Test, con le 34 etichette nel loro
ordine e la conferma del nome della classe benigna. All'arrivo si dichiarano i file con il loro hash
in `configs/common_inputs_v1.json` e si rieseguono gli stessi comandi, senza cambiare il protocollo.
Se i seed cambiano anche lo split, i supporti Validation differiscono e q non è unico: va
documentato prima di qualsiasi modifica.

## 5. Limiti

- Su X-IIoTID c'è una sola partizione e un solo seed: soglie e raggi non hanno un'incertezza
  stimata, e il Test mostra che il vantaggio di lgbm può sparire da una parte all'altra.
- Entrambi i Test erano già stati esaminati: è un confronto retrospettivo fra politiche di scelta,
  non una conferma prospettica né una valutazione su traffico reale.
- La riponderazione assume invariati i tassi di errore per classe; π e λ sono ipotesi, e i pesi
  sono uniformi fra le classi. I pesi per gravità di RISK-03 restano un'analisi supplementare.
- Le variazioni indipendenti dei pesi possono invertire l'ordine fra classi; a δ = 1 un peso può
  essere nullo.
- Le confusioni fra attacchi non entrano nel costo, e lgbm ne fa di più.
- I risultati valgono per questi modelli; il confronto fra i due dataset manca finché non arriva
  CICIoT2023.

## 6. Riproduzione

    python scripts/cost_selection.py                    # controllo degli input
    python scripts/cost_selection.py --risk03           # RISK-03 invariato attraverso il percorso generico
    python scripts/cost_selection.py --dataset xiiotid  # tabelle e manifest
    python scripts/cost_selection_figures.py            # le tre figure
    python -m pytest -q
