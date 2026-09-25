# Patch breve del solo §4.4 (nessuna nuova sezione)

Riferimento: `Rapporto1_1.pdf`, pag. 6, §4.4 «Soluzione D».

Testo attuale (4 righe, ~330 caratteri):

> **Statistiche rater fold-safe.** Nell'Excel, rater-accuracy e rater-confidence sono medie
> globali su tutto il dataset: usarle in cross-validation leakerebbe il test. Per ogni outer fold
> si ricalcolano solo sui casi di training; sulle righe di training si usa leave-one-out
> (la decisione corrente non entra nella propria feature).

Cosa manca: *su quali casi* è calcolato il profilo, e il collegamento con l'inferenza (dove i
profili vengono «dai casi storici»). Le due versioni sotto chiudono l'ambiguità restando nello
stesso paragrafo.

---

## Versione A — breve, consigliata (~5 righe, ~640 caratteri)

**Statistiche rater fold-safe.** Nell'Excel, rater-accuracy e rater-confidence sono medie globali su
tutto il dataset: usarle in cross-validation leakerebbe il test. Per ogni outer fold si ricalcolano
solo sui casi di training, **escludendo l'intero caso della decisione**: poiché ogni rater valuta ogni
caso una volta sola, il leave-one-out sulla riga equivale a un leave-one-case-out (differenza misurata
**0** su tutti e cinque i fold). Il profilo di una decisione non contiene quindi mai decisioni dello
stesso caso, esattamente come in inferenza, dove i profili vengono dai soli casi storici. `rater-accuracy`
è inoltre calcolata su `error-rating`, che su un caso nuovo non è disponibile per definizione: le
etichette di errore delle altre decisioni sullo stesso caso non entrano mai nel profilo.

Chiude: l'ambiguità «decisione vs caso», la coerenza con l'inferenza, e il ramo "alternativo" del
revisore (che non è applicabile).

---

## Versione B — ultra-breve (~4 righe, ~470 caratteri, stessa lunghezza dell'attuale)

**Statistiche rater fold-safe.** Nell'Excel, rater-accuracy e rater-confidence sono medie globali su
tutto il dataset: usarle in cross-validation leakerebbe il test. Per ogni outer fold si ricalcolano
solo sui casi di training, **escludendo l'intero caso della decisione**: ogni rater valuta ogni caso
una volta sola, quindi escludere la riga equivale a escludere il caso, e il profilo non contiene mai
decisioni dello stesso caso — come in inferenza, dove i profili vengono dai soli casi storici.

Lascia fuori: il numero della verifica e il punto su `error-rating` (li citi solo se il revisore
insiste). L'ambiguità è comunque chiusa.

---

## Aggiunta facoltativa (una clausola, zero sezioni nuove)

Nel §6 (pag. 14), l'elenco dei punti verificati:

> …che la baseline sia definita come dichiarato, che D impari dal bersaglio giusto,
> … e che il protocollo nested non sia contaminato.

si può integrare con una sola clausola:

> …, **che i profili rater non contengano decisioni dello stesso caso**, …

Così la frase di §4.4 («differenza misurata 0 su tutti e cinque i fold») resta agganciata all'elenco
delle verifiche senza creare un nuovo §6.7. Se preferisci zero numeri nel rapporto, usa la versione B
e tieni lo script di controllo (`scratch/verify_q_ir_and_rater_stats.py`) come evidenza a parte,
da mostrare solo su richiesta.
