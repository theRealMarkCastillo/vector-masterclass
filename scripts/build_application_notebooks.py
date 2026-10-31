"""M4 task-specific applications with labels, baselines, policy and failure analysis."""

from build_foundation_notebooks import code, md, notebook

DATA = """import json
fixtures = json.loads((root()/'datasets/applications.json').read_text())
"""


def main():
    notebook(
        "14",
        "Recommendations without future leakage",
        "B",
        ["04", "08", "11"],
        [
            md("""
        ## The engineering question
        Does a content-similarity recommender find useful eligible items without learning from the future?

        **Learning outcomes:** construct profiles from pre-cutoff interactions; exclude owned, unavailable, and incompatible items; compare popularity, content retrieval, and diversity; measure recommendation hit rate and catalog coverage.

        The fictional catalog has travel and office accessories. Each user's next interaction occurs after a fixed cutoff. We build both the content profile and popularity baseline only from earlier interactions. Descriptions are item features, while future interactions are labels. A future product is present as a deliberate leakage trap and must never become an eligible recommendation at the earlier timestamp.

        The user profile is the normalized mean of owned-item embeddings. Qdrant retrieves eligible vectors; MMR diversifies the returned candidates. This is a tiny content-based demonstration, not collaborative filtering or a customer-preference study. Two synthetic users provide diagnostic examples, not statistically reliable product quality.
        """),
            code(DATA),
            code("""
        from uuid import uuid4
        from collections import Counter
        from vector_masterclass.schema import Document
        from vector_masterclass.chunking import chunk_document
        from vector_masterclass.models import embed
        from vector_masterclass.retrieval import normalized,mmr
        from vector_masterclass.store import VectorIndex,tenant_filter
        cfg = fixtures['recommendation']
        catalog = {item['id']:item for item in cfg['catalog']}
        docs = [Document(id=i['id'],revision='1',title=i['id'],text=i['text'],published=i['published'],family=i['family']) for i in catalog.values()]
        matrix = embed([d.text for d in docs])
        by_id = {d.id:matrix[i] for i,d in enumerate(docs)}
        historical = [e for e in cfg['interactions'] if e['date'] < cfg['cutoff']]
        popularity = Counter(e['item'] for e in historical)
        index = VectorIndex('vmc_14_'+uuid4().hex[:8])
        index.create()
        rows = []
        try:
            for doc in docs:
                index.upsert(chunk_document(doc,'document'),document_id=doc.id,include_sparse=False)
            for user in cfg['users']:
                owned = {e['item'] for e in historical if e['user']==user['id']}
                future = {e['item'] for e in cfg['interactions'] if e['user']==user['id'] and e['date']>=cfg['cutoff']}
                eligible = {id_ for id_,item in catalog.items() if id_ not in owned and item['published']<cfg['cutoff'] and user['context'] in item['compatible']}
                profile = normalized(np.mean([by_id[id_] for id_ in sorted(owned)],axis=0))
                response = index.client.query_points(index.name,query=profile.tolist(),using='dense',query_filter=tenant_filter('public',[id_+'@1' for id_ in eligible]),limit=20,with_payload=True)
                candidates = [p.payload['document_id'] for p in response.points]
                diverse = [candidates[i] for i in mmr(profile,np.array([by_id[id_] for id_ in candidates]),k=3,weight=.6)]
                variants = {'popularity':sorted(eligible,key=lambda id_:(-popularity[id_],id_))[:3],'content':candidates[:3],'diverse content':diverse}
                for method,ids in variants.items():
                    rows.append({'user':user['id'],'method':method,'recommended':ids,'future_positive_items':sorted(future),'hit_at_3':bool(set(ids)&future),'catalog_coverage':len(set(ids))/len(eligible),'family_diversity':len({catalog[id_]['family'] for id_ in ids}),'exclusions_hold':set(ids).issubset(eligible) and not set(ids)&owned})
        finally:
            index.drop()
            index.close()
        results = lesson.table(pd.DataFrame(rows))
        display(results)
        fig,axes = plt.subplots(1,2,figsize=(11,4))
        results.groupby('method').hit_at_3.mean().plot.bar(ax=axes[0])
        results.groupby('method').family_diversity.mean().plot.bar(ax=axes[1],color='#f59e0b')
        axes[0].set(ylabel='Hit@3',ylim=(0,1.05),title='Temporal holdout, two fictional users')
        axes[1].set(ylabel='Mean distinct product families',title='Diversity has a separate objective')
        for ax in axes: ax.tick_params(axis='x',rotation=20)
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        A relevant recommendation can lose to an ineligible near neighbor unless exclusions are enforced within retrieval. Popularity provides a strong cheap baseline but cannot personalize an unusual preference. Content methods address some cold-start items but still need a useful user profile. Diversity may improve catalog coverage while lowering short-term clicks; choose the objective explicitly.

        **Exercise:** move the future case's publication date before the cutoff and observe eligibility. Then accidentally include the post-cutoff click in the profile and explain why the apparent quality gain is leakage. Propose a cold-start fallback for a user with no history and keep eligibility checks on that path.

        Production recommendation evaluation also needs exposure bias, negative sampling, repeat purchases, inventory changes, exploration, and online outcome measurement. An unclicked item is not automatically a true negative. These labels are author-created instructional fixtures.
        """),
            code("""
        manifest = lesson.finish({'all_exclusions_hold':bool(results.exclusions_hold.all()),'future_item_excluded':all('future-case' not in ids for ids in results.recommended),'training_precedes_cutoff':all(e['date']<cfg['cutoff'] for e in historical),'three_baselines':results.method.nunique()==3},observations={'cutoff':cfg['cutoff'],'evaluation_users':2,'label_provenance':fixtures['provenance'],'recommendation_k':3})
        """),
        ],
    )

    notebook(
        "15",
        "Duplicate candidates are not merge decisions",
        "B",
        ["04", "08"],
        [
            md("""
        ## The engineering question
        When does high semantic similarity hide an entity mismatch or a contradictory statement?

        **Learning outcomes:** calibrate a candidate threshold on development pairs; freeze it before a separate entity-family test; compare exact text, embedding candidates, and typed-field verification; count false merges explicitly.

        We generate fictional record pairs from disjoint device names in development and test. A positive pair describes the same serial-numbered device. Negative pairs differ in serial identity or explicitly disagree about whether power is supplied. This lab evaluates candidate generation and a conservative verifier, not a universal entity-resolution system. Candidate similarity is evidence to inspect, not permission to merge records destructively.
        """),
            code("""
        from vector_masterclass.models import embed
        from vector_masterclass.applications import binary_metrics
        pairs = []
        for split,names in [('development',['Atlas','Boreal']),('test',['Cobalt','Delta'])]:
            for name in names:
                base = f'{name} device serial 100 provides power over Ethernet.'
                pairs.extend([
                    {'split':split,'family':name,'left':base,'right':f'Power over Ethernet is provided by {name}, serial 100.','same_entity':True,'left_serial':'100','right_serial':'100','compatible_claim':True},
                    {'split':split,'family':name,'left':base,'right':base,'same_entity':True,'left_serial':'100','right_serial':'100','compatible_claim':True},
                    {'split':split,'family':name,'left':base,'right':f'{name} device serial 101 provides power over Ethernet.','same_entity':False,'left_serial':'100','right_serial':'101','compatible_claim':True},
                    {'split':split,'family':name,'left':base,'right':f'{name} device serial 100 does not provide power over Ethernet.','same_entity':False,'left_serial':'100','right_serial':'100','compatible_claim':False},
                ])
        vectors = embed([text for p in pairs for text in [p['left'],p['right']]])
        for i,pair in enumerate(pairs):
            pair['similarity'] = float(vectors[2*i]@vectors[2*i+1])
        frame = pd.DataFrame(pairs)
        dev = frame[frame.split.eq('development')]
        sweep = pd.DataFrame([{'threshold':float(t),**binary_metrics(dev.same_entity,dev.similarity.ge(t))} for t in np.linspace(.5,.99,30)])
        chosen = sweep.sort_values(['f1','precision','threshold'],ascending=[False,False,False]).iloc[0].threshold
        rows = []
        for split,group in frame.groupby('split'):
            candidates = group.similarity.ge(chosen)
            # The typed compatibility flag is supplied by the source schema, not inferred by cosine.
            verified = candidates & group.left_serial.eq(group.right_serial) & group.compatible_claim
            for method,predicted in [('exact text',group.left.eq(group.right)),('semantic candidates',candidates),('typed verification',verified)]:
                rows.append({'split':split,'method':method,**binary_metrics(group.same_entity,predicted)})
        results = lesson.table(pd.DataFrame(rows))
        lesson.table(frame,'pairs')
        lesson.table(sweep,'development-thresholds')
        display(frame[['split','left','right','same_entity','similarity']])
        display(results)
        print('Frozen development-selected threshold:',chosen)
        fig,axes = plt.subplots(1,2,figsize=(11,4))
        sweep.plot(x='threshold',y=['precision','recall'],ax=axes[0])
        axes[0].axvline(chosen,color='black',linestyle='--')
        axes[0].set(title='Development threshold selection',ylim=(0,1.05))
        results[results.split.eq('test')].set_index('method')[['fp','fn']].rename(columns={'fp':'False merges','fn':'Missed matches'}).plot.bar(ax=axes[1])
        axes[1].set(title='Held-out entity families',ylabel='Pair count')
        axes[1].tick_params(axis='x',rotation=20)
        axes[1].legend(loc='upper left',bbox_to_anchor=(1,1))
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        The `same_entity` column is the task's **merge-safe match label**: even a record sharing the serial number is not safe to merge when its typed capability claim conflicts. The conservative verifier relies on trusted structured identity/capability fields. In arbitrary free text those fields may be missing or wrong, requiring extraction verification and human adjudication. This controlled fixture does not establish a general contradiction detector.

        **Exercise:** inspect the highest-similarity negative pair. Explain whether a threshold can separate it from the least-similar positive pair. Remove the typed claim field and document which merge decisions become uncertain. Report the cost of false merges separately from missed duplicates; undoing a false merge can be far more expensive.

        At scale, retrieve candidate neighbors instead of scoring every pair, then evaluate candidate recall before the verifier. Blocking by tenant/entity type and preserving merge provenance are essential. Never silently deduplicate by vector similarity alone.
        """),
            code("""
        manifest = lesson.finish({'source_families_disjoint':not set(frame[frame.split.eq('development')].family)&set(frame[frame.split.eq('test')].family),'held_out_scored':len(results[results.split.eq('test')])==3,'typed_false_merges_zero':bool(results.loc[results.method.eq('typed verification'),'fp'].eq(0).all()),'negative_cases_present':bool((~frame.same_entity).any())},observations={'selected_threshold':float(chosen),'threshold_selection_split':'development','label_semantics':'merge-safe match, not physical entity identity alone','typed_verifier_is_not_general_NLI':True})
        """),
        ],
        docker=False,
    )

    notebook(
        "16a",
        "Semantic caches need exact context",
        "B",
        ["04", "08"],
        [
            md("""
        ## The engineering question
        When is a similar question safe to answer from a cache, and when must the cache miss?

        **Learning outcomes:** separate similarity from cache eligibility; include tenant, permission/source revisions, conversation state, and answer-model revision; enforce expiration and invalidation in Qdrant; measure false hits on negation and product changes.

        The cached answer is an explicit fixture, not a generated response. Qdrant stores the query embedding and exact scope hash. Scope and expiry filters run before similarity search. A hit still requires semantic equivalence for the application; a cosine threshold cannot prove that. The threshold 0.9 is fixed before this tiny diagnostic set, not optimized on it.
        """),
            code("""
        from uuid import uuid4
        from vector_masterclass.semantic_cache import SemanticCache,CacheScope
        cache = SemanticCache('vmc_16a_'+uuid4().hex[:8])
        scope = CacheScope(tenant='red',authorization_revision='1',source_id='atlas-warranty',source_revision='1',conversation_hash='product=Atlas',answer_model_revision='fixture-v1')
        tests = [('exact','How long is the Atlas warranty?',True),('paraphrase','What is the warranty duration for Atlas?',True),('wrong product','How long is the Boreal warranty?',False),('negated request','Does the Atlas warranty not last two years?',False),('different fact','What is the Atlas waterproof rating?',False)]
        rows = []
        try:
            cache.put(tests[0][1],'Fixture answer: two years.',scope,now=100,ttl=30)
            for label,query,safe in tests:
                hit = cache.get(query,scope,now=110,threshold=.9)
                rows.append({'case':label,'expected_safe_reuse':safe,'cache_hit':hit is not None,'false_hit':hit is not None and not safe,'score':None if hit is None else hit['score'],'scope_constraint':False})
            for field in CacheScope.model_fields:
                changed = scope.model_copy(update={field:'changed'})
                hit = cache.get(tests[0][1],changed,now=110,threshold=.9)
                rows.append({'case':'changed '+field,'expected_safe_reuse':False,'cache_hit':hit is not None,'false_hit':hit is not None,'score':None,'scope_constraint':True})
            expired = cache.get(tests[0][1],scope,now=130,threshold=.9)
            cache.invalidate_source('atlas-warranty')
            deleted = cache.get(tests[0][1],scope,now=110,threshold=.9)
        finally:
            cache.close(drop=True)
        results = lesson.table(pd.DataFrame(rows))
        display(results)
        fig,ax = plt.subplots(figsize=(10,5))
        values = results.cache_hit.astype(int)
        ax.barh(results['case'],values,color=['#dc2626' if false else '#2563eb' for false in results.false_hit])
        ax.set(xlabel='Cache hit (1) or miss (0)',xticks=[0,1],title='Context changes must miss, regardless of query similarity')
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Exact scope protects eligibility, not meaning. A negated question may remain close enough to the original for a false hit; if no false hit appears at 0.9, lower the threshold on a separate development set and inspect the first failure. A missed paraphrase costs latency; a false answer can change the user's decision. Choose asymmetric error costs and consider exact intent/entity guards or a verifier.

        **Exercise:** warm a cache, revoke access, and show why the authoritative authorization revision must change before the next lookup. Then change only the conversation's product identity. Explain why TTL alone cannot implement immediate revocation or source deletion. The notebook uses an explicit synthetic clock for deterministic expiry tests; a deployed service needs one trusted clock and a skew policy.

        Invalidation removes all entries for the source across scopes. Request callers must not manufacture their own scope; the service constructs it from authoritative state. A distributed cache also needs deletion propagation, observability, and memory/eviction limits. Cached values are prior outputs or fixtures, never evidence of a new model call.
        """),
            code("""
        manifest = lesson.finish({'exact_query_hits':bool(results.iloc[0].cache_hit),'all_context_changes_miss':bool((~results.loc[results.scope_constraint,'cache_hit']).all()),'expiry_boundary_excluded':expired is None,'source_invalidation_excluded':deleted is None,'negative_semantic_cases_measured':int((~results.expected_safe_reuse & ~results.scope_constraint).sum())==3},observations={'clock':'synthetic seconds 100..130','threshold':.9,'answers':'explicit fixtures; no generation claim','semantic_false_hits':int(results.false_hit.sum())})
        """),
        ],
    )

    notebook(
        "16b",
        "Routing with abstention and separate authority",
        "B",
        ["04", "08"],
        [
            md("""
        ## The engineering question
        Can similarity choose a useful intent while declining ambiguous or out-of-scope requests?

        **Learning outcomes:** build intent prototypes; calibrate threshold/margin on development data; freeze them for test evaluation; measure abstention and errors; keep tool authorization outside routing.

        Routing compares each query with labeled prototype embeddings and keeps the best score per intent. It accepts only when both the best score and separation from the runner-up exceed thresholds. Development examples include an ambiguous reset/export request and an unrelated poem. A route is a classification suggestion. It never grants permission to reset a device or export private data.
        """),
            code(DATA),
            code("""
        from sklearn.metrics import confusion_matrix
        from vector_masterclass.models import embed
        from vector_masterclass.applications import route
        cfg = fixtures['routing']
        train_text,labels = zip(*cfg['train'])
        prototypes = embed(list(train_text))
        dev_vectors = embed([t for t,l in cfg['development']],role='query')
        sweeps = []
        for threshold in [.3,.4,.5,.6,.7]:
            for margin in [0,.05,.1,.2]:
                predictions = [route(v,prototypes,labels,threshold=threshold,margin=margin)['intent'] for v in dev_vectors]
                sweeps.append({'threshold':threshold,'margin':margin,'development_accuracy':np.mean([p==label for p,(_,label) in zip(predictions,cfg['development'])])})
        chosen = sorted(sweeps,key=lambda r:(-r['development_accuracy'],-r['threshold'],-r['margin']))[0]
        rows = []
        for split in ['development','test']:
            for (text,label),vector in zip(cfg[split],embed([t for t,l in cfg[split]],role='query')):
                prediction = route(vector,prototypes,labels,threshold=chosen['threshold'],margin=chosen['margin'])
                rows.append({'split':split,'text':text,'expected':label,**prediction,'correct':prediction['intent']==label})
        results = lesson.table(pd.DataFrame(rows))
        lesson.table(pd.DataFrame(sweeps),'development-calibration')
        display(results)
        print('Frozen development configuration:',chosen)
        test = results[results.split.eq('test')]
        categories = ['status','reset','export','abstain']
        cm = confusion_matrix(test.expected,test.intent,labels=categories)
        fig,ax = plt.subplots(figsize=(6,5))
        picture = ax.imshow(cm,cmap='Blues',vmin=0)
        ax.set(xticks=range(4),xticklabels=categories,yticks=range(4),yticklabels=categories,xlabel='Predicted',ylabel='Expected',title='Held-out routing including out-of-scope requests')
        for i in range(4):
            for j in range(4): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
        fig.colorbar(picture,ax=ax,label='Examples')
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Raising thresholds can improve accepted-request precision while reducing coverage. Overall accuracy hides that tradeoff, so inspect accepted and abstained examples separately. A high similarity to a destructive intent is not authorization. The caller still needs a verified identity, policy permission, validated arguments, and the applicable action approval boundary.

        **Exercise:** add “restart without erasing settings” to development data and inspect confusion with factory reset. Add a new prototype only after defining a distinct intent and corresponding policy. Explain why a prototype copied from the test sentence would invalidate the holdout.

        Five test examples are a diagnostic fixture. Production routing requires representative languages, ambiguity, distribution shift, adversarial wording, and safe fallback behavior. Similarity margins are not calibrated confidence probabilities.
        """),
            code("""
        manifest = lesson.finish({'no_route_grants_authority':bool((~results.authorized).all()),'both_splits':set(results.split)=={'development','test'},'out_of_scope_tested':bool(test.expected.eq('abstain').any()),'finite_scores':bool(np.isfinite(results.score).all())},observations={'frozen_parameters':chosen,'test_accuracy':float(test.correct.mean()),'test_coverage':float(test.intent.ne('abstain').mean()),'label_provenance':fixtures['provenance']})
        """),
        ],
        docker=False,
    )

    notebook(
        "17",
        "Memory needs correction and forgetting",
        "B",
        ["06", "08"],
        [
            md("""
        ## The engineering question
        Can retrieval honor a correction, expiration, and deletion even when obsolete text is semantically close?

        **Learning outcomes:** distinguish episodic records from current facts; retain correction provenance; filter by validity time and tenant; propagate deletion to the index; reject replay of a tombstoned revision.

        The fictional travel preference changes from window to aisle. We treat the correction as a new revision of the same source, keep the event history in the ingestion manifest, and remove obsolete points. A temporary appointment record expires at a synthetic timestamp. Semantic similarity provides ranking only; currentness and permission are explicit constraints. This is a memory lifecycle experiment, not a claim that a model has human memory.
        """),
            code("""
        from uuid import uuid4
        from vector_masterclass.schema import Document
        from vector_masterclass.store import VectorIndex
        from vector_masterclass.ingestion import Ingestor
        index = VectorIndex('vmc_17_'+uuid4().hex[:8])
        index.create()
        ingest = Ingestor(index,lesson.path/'memory-events.sqlite')
        old = Document(id='seat-preference',revision='1',title='Seat preference',text='The red traveler prefers a window seat.',tenant='red',published='2026-01-01')
        corrected = old.model_copy(update={'revision':'2','text':'The red traveler corrected the preference: choose an aisle seat. The earlier window preference is superseded.','published':'2026-02-01'})
        appointment = Document(id='appointment',revision='1',title='Temporary appointment',text='The red traveler has a temporary appointment at 09:00.',tenant='red')
        other = Document(id='blue-preference',revision='1',title='Blue traveler',text='The blue traveler prefers a window seat.',tenant='blue')
        rows = []
        current = {old.id:old,appointment.id:appointment}
        expires = {old.id:float('inf'),appointment.id:120}
        def retrieve(now):
            allowed = [doc.key for id_,doc in current.items() if expires[id_]>now]
            return index.search('What seat does the traveler prefer and what appointment is planned?',tenant='red',document_keys=allowed,k=10)
        try:
            ingest.ingest([old,appointment,other])
            before = retrieve(100)
            rows.append({'stage':'initial','retrieved_sources':[h.chunk.document_key for h in before],'visible_points':len(index.all_points())})
            ingest.ingest([corrected])
            current[old.id]=corrected
            after = retrieve(100)
            rows.append({'stage':'after correction','retrieved_sources':[h.chunk.document_key for h in after],'visible_points':len(index.all_points())})
            expired = retrieve(120)
            rows.append({'stage':'after expiry filter','retrieved_sources':[h.chunk.document_key for h in expired],'visible_points':len(index.all_points())})
            ingest.delete(corrected.id,'red')
            del current[corrected.id]
            deleted = retrieve(120)
            rows.append({'stage':'after explicit deletion','retrieved_sources':[h.chunk.document_key for h in deleted],'visible_points':len(index.all_points())})
            rejected = False
            try: ingest.ingest([corrected])
            except ValueError: rejected=True
            events = lesson.table(pd.DataFrame(ingest.events()).drop(columns=['source_json']),'events')
            consistent = ingest.reconcile()['consistent']
        finally:
            ingest.close()
            index.drop()
            index.close()
        results = lesson.table(pd.DataFrame(rows))
        display(results)
        display(events[['sequence','source_id','operation','state']])
        fig,ax = plt.subplots(figsize=(10,4))
        ax.bar(results.stage,[len(ids) for ids in results.retrieved_sources])
        ax.set(ylabel='Eligible retrieved records',title='Validity and deletion constrain memory before ranking')
        ax.tick_params(axis='x',rotation=15)
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Expiration hides a record from retrieval while leaving stored points until cleanup; deletion removes indexed points and leaves a tombstone/event for reconciliation. The correction retains history for audit while preferring the current record. Audit retention and erasure requirements must be defined separately, including source logs, backups, caches, and derived indexes.

        **Exercise:** move the clock just before and exactly to expiration. Then attempt to reingest the deleted revision. Explain why an ordered event log or source-version authority is needed to reject older events in a multi-writer system. The single-writer teaching pipeline does not infer ordering from arbitrary revision strings.

        Memory ranking can combine similarity with recency and importance, but those heuristics must operate only after tenant, validity, correction, and deletion constraints. Do not preserve an obsolete fact merely because it has the highest score.
        """),
            code("""
        manifest = lesson.finish({'old_revision_removed':all(h.chunk.document_key!='seat-preference@1' for h in after),'correction_present':any(h.chunk.document_key=='seat-preference@2' for h in after),'expiry_enforced':all(h.chunk.document_key!='appointment@1' for h in expired),'deleted_memory_absent':len(deleted)==0,'tenant_boundary':all(h.chunk.tenant=='red' for h in before+after+expired),'tombstone_replay_rejected':rejected,'manifest_converged':consistent},observations={'clock':'synthetic seconds 100..120','expiry_is_filtering_not_physical_erasure':True,'history_retention':'SQLite teaching event log; explicit production retention decision required'})
        """),
        ],
    )
    print("Authored application lessons 14, 15, 16a, 16b, 17")


if __name__ == "__main__":
    main()
