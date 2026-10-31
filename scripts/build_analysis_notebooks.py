"""Classification, exploratory clustering, and calibrated anomaly scoring."""

from build_application_notebooks import DATA
from build_foundation_notebooks import code, md, notebook


def main():
    notebook(
        "18a",
        "Nearest neighbor classification",
        "B",
        ["04"],
        [
            md("""
        ## The engineering question
        Can labeled neighbors classify a new support request, and when should the classifier abstain?

        **Learning outcomes:** compare a majority baseline with k-neighbor voting; isolate held-out labels; inspect the confusion matrix and neighbors; separate classification confidence from geometric proximity.

        The fictional training set contains billing, account-access, and network requests. Test sentences are separately authored paraphrases plus an out-of-scope song request. The encoder is frozen, k=3 and similarity threshold 0.4 are fixed before test scoring. No model is fine-tuned. This is a small supervised decision layer on top of a learned representation, not evidence of general support-ticket accuracy.
        """),
            code(DATA),
            code("""
        from collections import Counter
        from sklearn.metrics import confusion_matrix
        from vector_masterclass.models import embed
        from vector_masterclass.applications import neighbor_predict
        from vector_masterclass.retrieval import exact_search
        cfg = fixtures['classification']
        texts,labels = zip(*cfg['train'])
        vectors = embed(list(texts))
        majority = sorted(Counter(labels),key=lambda label:(-Counter(labels)[label],label))[0]
        rows = []
        for (text,expected),query in zip(cfg['test'],embed([t for t,l in cfg['test']])):
            predicted = neighbor_predict(query,vectors,labels,k=3,threshold=.4)
            neighbors = exact_search(query,vectors,[str(i) for i in range(len(texts))],k=3)
            rows.append({'text':text,'expected':expected,'predicted':predicted,'majority_baseline':majority,'correct':predicted==expected,'neighbors':[{'text':texts[int(i)],'label':labels[int(i)],'score':score} for i,score in neighbors]})
        results = lesson.table(pd.DataFrame(rows))
        display(results[['text','expected','predicted','correct']])
        for row in rows:
            print(row['text'], '->',row['predicted'])
            print(row['neighbors'])
        categories = ['billing','access','network','abstain']
        cm = confusion_matrix(results.expected,results.predicted,labels=categories)
        fig,ax = plt.subplots(figsize=(6,5))
        picture = ax.imshow(cm,cmap='Blues',vmin=0)
        ax.set(xticks=range(4),xticklabels=categories,yticks=range(4),yticklabels=categories,xlabel='Predicted',ylabel='Expected',title='Held-out classification; inspect abstention separately')
        for i in range(4):
            for j in range(4): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
        fig.colorbar(picture,ax=ax,label='Examples')
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Majority voting can be wrong when nearby labels are mixed, while a similarity threshold can reject a correct low-similarity example. The held-out out-of-scope sentence probes abstention; it is not enough to calibrate all unknown requests. A point close to several billing examples is not automatically a calibrated billing probability.

        **Exercise:** inspect the three neighbors of each error. Compare k=1 and k=5 on a newly defined development set before selecting another k. Explain why duplicating one training example can bias unweighted voting, and why balancing counts is different from representing the real deployment distribution.

        Production checks should include per-class precision/recall, rare classes, label ambiguity, annotator agreement, source/customer separation, and drift. Similarity-based classification is useful when labels are scarce and explanations via examples help; a trained classifier may be cheaper or better once enough labels exist.
        """),
            code("""
        manifest = lesson.finish({'no_exact_train_test_overlap':not set(texts)&{t for t,l in cfg['test']},'all_test_examples_scored':len(results)==7,'negative_case_present':bool(results.expected.eq('abstain').any()),'all_neighbors_from_training':all(len(r['neighbors'])==3 for r in rows)},observations={'k':3,'threshold':.4,'test_accuracy':float(results.correct.mean()),'majority_accuracy':float(results.expected.eq(majority).mean()),'label_provenance':fixtures['provenance']})
        """),
        ],
        docker=False,
    )

    notebook(
        "18b",
        "Clusters are hypotheses to inspect",
        "B",
        ["02", "04"],
        [
            md("""
        ## The engineering question
        Are clusters stable semantic groupings, or artifacts of k, initialization, and a two-dimensional projection?

        **Learning outcomes:** cluster real embeddings; compare initialization stability and silhouette diagnostics; inspect representative source examples; distinguish a projection from the original distance space.

        We use the twelve fictional support training sentences as an unlabeled exploratory corpus. K-means minimizes squared Euclidean distance to centroids. On normalized points, pairwise squared distance relates to cosine, but unconstrained centroid updates do not make ordinary k-means identical to spherical k-means. We compare k=2,3,4 and five seeds with one initialization each so instability can remain visible.

        Silhouette is computed in the original embedding space. PCA is only a display projection; apparent two-dimensional gaps are not a clustering validity test. The known support labels are not used to fit or select the clusters.
        """),
            code(DATA),
            code("""
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import adjusted_rand_score,silhouette_score
        from vector_masterclass.models import embed
        texts = [t for t,l in fixtures['classification']['train']]
        vectors = embed(texts)
        rows,assignments = [],{}
        for k in [2,3,4]:
            for seed in range(5):
                labels = KMeans(n_clusters=k,random_state=seed,n_init=1).fit_predict(vectors)
                assignments[k,seed]=labels
                rows.append({'k':k,'seed':seed,'silhouette':silhouette_score(vectors,labels,metric='cosine'),'ARI_vs_seed0':adjusted_rand_score(assignments[k,0],labels)})
        results = lesson.table(pd.DataFrame(rows))
        display(results)
        # Display k=3 for the declared teaching example; do not select by known labels.
        labels = assignments[3,0]
        projection = PCA(n_components=2,random_state=17).fit_transform(vectors)
        members = lesson.table(pd.DataFrame({'example':range(len(texts)),'text':texts,'cluster':labels}),'members')
        display(members)
        fig,axes = plt.subplots(1,2,figsize=(11,4))
        axes[0].scatter(projection[:,0],projection[:,1],c=labels,cmap='tab10',s=60)
        for i,(x,y) in enumerate(projection): axes[0].annotate(str(i),(x,y),xytext=(4,4),textcoords='offset points')
        axes[0].set(xlabel='PCA component 1',ylabel='PCA component 2',title='Illustrative projection; labels identify source rows')
        results.boxplot(column='ARI_vs_seed0',by='k',ax=axes[1])
        axes[1].set(title='Initialization stability',ylabel='Adjusted Rand index vs seed 0',xlabel='Cluster count k',ylim=(-.1,1.1))
        fig.suptitle('')
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Stable clusters can still be irrelevant to a business task; unstable clusters may reflect legitimate overlapping topics. Silhouette rewards geometric separation and can prefer a coarse partition. Read the member texts before naming a cluster, and do not equate a topic cluster with a verified category.

        **Exercise:** add one sentence combining billing and login problems. Inspect how membership changes across seeds. Then compare the original-space nearest neighbors with distances in the PCA plot. Explain a case where the projection suggests a relationship that the original vectors do not support.

        Production exploration needs deduplication, sampling, language/source slices, drift monitoring, and a plan for human naming and review. Density-based methods or hierarchical clustering may suit another shape; no method removes the need to inspect source examples.
        """),
            code("""
        manifest = lesson.finish({'all_initializations_measured':len(results)==15,'finite_diagnostics':bool(np.isfinite(results[['silhouette','ARI_vs_seed0']]).all().all()),'projection_labeled_illustrative':projection.shape==(12,2),'source_examples_available':len(members)==12},observations={'k_values':[2,3,4],'seeds':list(range(5)),'label_use':'none during fit or cluster-count selection','projection':'PCA display only'})
        """),
        ],
        docker=False,
    )

    notebook(
        "18c",
        "Anomaly scores need a reference population",
        "B",
        ["04", "18b"],
        [
            md("""
        ## The engineering question
        Is a request unusual relative to the reference corpus, and what errors does a chosen threshold create?

        **Learning outcomes:** construct a normal-only reference set; score distance to nearby examples; choose a threshold on development data; report held-out false positives/negatives; avoid interpreting unusualness as wrongdoing.

        The reference population is the twelve support sentences. Anomaly score is one minus mean cosine similarity to the three closest reference points. Development and test fixtures contain normal paraphrases and deliberately unrelated requests. A threshold is selected on development F1 with a conservative tie-break, then frozen. This detects topic novelty in an artificial workload; it does not detect maliciousness, fraud, or a system outage.
        """),
            code(DATA),
            code("""
        from vector_masterclass.models import embed
        from vector_masterclass.applications import binary_metrics
        reference_texts = [t for t,l in fixtures['classification']['train']]
        reference = embed(reference_texts)
        def scores(texts):
            similarities = embed(texts)@reference.T
            return 1-np.sort(similarities,axis=1)[:,-3:].mean(axis=1)
        cfg = fixtures['anomaly']
        dev_scores = scores([t for t,l in cfg['development']])
        dev_labels = [l for t,l in cfg['development']]
        sweep = pd.DataFrame([{'threshold':float(th),**binary_metrics(dev_labels,dev_scores>th)} for th in np.linspace(0,1,51)])
        chosen = float(sweep.sort_values(['f1','precision','threshold'],ascending=[False,False,False]).iloc[0].threshold)
        rows = []
        for split in ['development','test']:
            for (text,expected),score in zip(cfg[split],scores([t for t,l in cfg[split]])):
                rows.append({'split':split,'text':text,'expected_anomaly':expected,'score':float(score),'predicted_anomaly':bool(score>chosen)})
        results = lesson.table(pd.DataFrame(rows))
        lesson.table(sweep,'development-thresholds')
        test = results[results.split.eq('test')]
        metrics = binary_metrics(test.expected_anomaly,test.predicted_anomaly)
        display(results)
        print('Frozen threshold:',chosen,'Test metrics:',metrics)
        fig,ax = plt.subplots(figsize=(10,5))
        ax.barh(range(len(test)),test.score,color=['#f59e0b' if value else '#2563eb' for value in test.expected_anomaly])
        ax.axvline(chosen,color='black',linestyle='--',label='Development-selected threshold')
        ax.set(yticks=range(len(test)),yticklabels=[t[:46] for t in test.text],xlabel='1 − mean top-3 cosine similarity',title='Topic novelty relative to support examples')
        ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        An unusual but legitimate request is a false positive if the goal is support triage. A novel failure can sound similar to routine network messages and become a false negative. The reference distribution and labeling objective define what the score means. Thresholds calibrated on a few synthetic outliers do not transfer automatically to real traffic.

        **Exercise:** add a legitimate new product line to the test set without adding it to the reference set. Measure the false-positive change, then propose a reviewed reference update. Explain why adding every flagged request to the reference can hide persistent anomalies and why a high anomaly score is not evidence of wrongdoing.

        Production monitoring needs base rates, false-alert cost, time/source splits, drift, human review, and retraining/invalidation policy. Keep anomaly detection separate from authorization or punitive actions.
        """),
            code("""
        manifest = lesson.finish({'normal_reference_only':len(reference)==12,'both_test_classes':set(test.expected_anomaly)=={False,True},'held_out_count':len(test)==6,'finite_scores':bool(np.isfinite(results.score).all()),'no_reference_test_overlap':not set(reference_texts)&set(test.text)},observations={'frozen_threshold':chosen,'test_metrics':metrics,'score_meaning':'topic novelty, not maliciousness','threshold_split':'development'})
        """),
        ],
        docker=False,
    )
    print("Authored analysis lessons 18a, 18b, 18c")


if __name__ == "__main__":
    main()
