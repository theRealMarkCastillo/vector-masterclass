"""M4 multilingual, code, and aligned image/text retrieval."""

from build_foundation_notebooks import code, md, notebook


def main():
    notebook(
        "19a",
        "Multilingual retrieval and representation contracts",
        "B",
        ["04", "10"],
        [
            md("""
        ## The engineering question
        Can a question in one language retrieve an English source, and what happens when equal-dimensional models are mixed?

        **Learning outcomes:** use a suitable multilingual encoder; report per-language slices; compare the English baseline; enforce model compatibility beyond dimension counts; inspect translation and tokenization limits.

        Bootstrap the separate model once with `uv run vmc bootstrap --model multilingual`. The pinned multilingual MiniLM maps text into 384 dimensions with a 128-token limit. The core English encoder also has 384 dimensions, but its axes have a different learned meaning. Equal shapes do not make their query/document vectors compatible.

        Ten author-written questions cover English, Spanish, French, German, and Japanese against the same English source corpus. Each language asks about Atlas warranty and Boreal PoE capacity. These tiny diagnostic translations are not a multilingual benchmark or a human-rated translation study.
        """),
            code("""
        from uuid import uuid4
        from vector_masterclass.models import embed
        from vector_masterclass.data import documents
        from vector_masterclass.chunking import chunk_document
        from vector_masterclass.store import VectorIndex
        from vector_masterclass.retrieval import exact_search
        docs = documents(tenant='public')
        ids = [d.key for d in docs]
        questions = [
            ('en','How long is the Atlas warranty?','atlas-warranty@1'),('en','What is the Boreal PoE power budget?','boreal-ports@1'),
            ('es','¿Cuánto dura la garantía del router Atlas?','atlas-warranty@1'),('es','¿Cuál es la potencia total de alimentación por Ethernet de Boreal?','boreal-ports@1'),
            ('fr','Quelle est la durée de la garantie du routeur Atlas ?','atlas-warranty@1'),('fr','Quelle est la puissance totale PoE du commutateur Boreal ?','boreal-ports@1'),
            ('de','Wie lange gilt die Garantie für den Atlas-Router?','atlas-warranty@1'),('de','Wie hoch ist das gesamte PoE-Leistungsbudget von Boreal?','boreal-ports@1'),
            ('ja','Atlasルーターの保証期間はどのくらいですか？','atlas-warranty@1'),('ja','BorealのPoE電力予算は合計何ワットですか？','boreal-ports@1'),
        ]
        rows = []
        for key in ['dense','multilingual']:
            vectors = embed([d.text for d in docs],key=key)
            query_vectors = embed([q for lang,q,gold in questions],key=key,role='query')
            for (language,text,gold),vector in zip(questions,query_vectors):
                ranked = [id_ for id_,score in exact_search(vector,vectors,ids,k=3)]
                rows.append({'encoder':key,'language':language,'question':text,'top_source':ranked[0],'hit_at_1':ranked[0]==gold,'recall_at_3':gold in ranked})
        index = VectorIndex('vmc_19a_'+uuid4().hex[:8],key='multilingual')
        index.create()
        rejected = False
        try:
            for doc in docs:
                index.upsert(chunk_document(doc,'document'),document_id=doc.id,include_sparse=False)
            server = index.search(questions[-2][1],tenant='public',k=3,exact=True)
            wrong = VectorIndex(index.name,key='dense')
            try: wrong.verify_contract()
            except ValueError: rejected=True
            finally: wrong.close()
            print('Japanese query Qdrant results:',[h.chunk.document_key for h in server])
        finally:
            index.drop()
            index.close()
        results = lesson.table(pd.DataFrame(rows))
        display(results)
        fig,ax = plt.subplots(figsize=(9,4))
        results.groupby(['language','encoder']).hit_at_1.mean().unstack().plot.bar(ax=ax)
        ax.set(ylabel='Hit@1 (two questions per language)',ylim=(0,1.05),title='Language slices are diagnostics, not population estimates')
        ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        ax.tick_params(axis='x',rotation=0)
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        A multilingual model can lose to an English model on some queries, and performance can vary sharply by language and domain. Brand names and PoE abbreviations offer shared lexical anchors; easy examples should not be generalized to long legal or technical passages. Translation-based retrieval is another baseline, with its own latency, privacy, and fidelity costs.

        **Exercise:** add a query containing a language-specific negation and inspect the retrieved source. Compare token counts against the two model limits. Then attempt to query the multilingual collection using the English model configuration and explain why the representation contract rejects it before relying on dimension validation.

        [Multilingual MiniLM model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). Production evaluation needs native-language judgments, scripts, locale-specific terminology, mixed-language queries, and source-language coverage.
        """),
            code("""
        manifest = lesson.finish({'five_languages':results.language.nunique()==5,'both_encoders_measured':results.encoder.nunique()==2,'equal_dimension_mismatch_rejected':rejected,'real_multilingual_qdrant_query':len(server)>0},observations={'queries_per_language':2,'judgments':'author-written diagnostic translations; not a representative benchmark','multilingual_model':'pinned local CPU inference'})
        """),
        ],
    )

    notebook(
        "19b",
        "Code search needs identifiers and intent",
        "B",
        ["04", "10"],
        [
            md("""
        ## The engineering question
        Which code-search queries need exact symbols, and which need a semantic description of behavior?

        **Learning outcomes:** index function boundaries with file/line provenance; compare lexical, dense, and fused retrieval; keep development/test function families distinct; inspect generic-encoder limitations and source-code trust boundaries.

        The corpus consists of eight author-created Python functions. We parse their AST boundaries without importing or executing them. Development queries target collection utilities; held-out queries target I/O and date/retry functions. All functions remain searchable, but test questions do not tune retrieval parameters. Function comments/docstrings are indexed as untrusted source text.
        """),
            code("""
        import ast
        from vector_masterclass.models import embed
        from vector_masterclass.retrieval import LexicalIndex,exact_search,reciprocal_rank_fusion
        from vector_masterclass.metrics import ranking_metrics
        records = []
        for path in sorted((root()/'datasets/code').glob('*.py')):
            source = path.read_text()
            for node in ast.parse(source).body:
                if isinstance(node,ast.FunctionDef):
                    records.append({'id':node.name,'family':path.stem,'file':str(path.relative_to(root())),'line':node.lineno,'code':ast.get_source_segment(source,node)})
        ids = [r['id'] for r in records]
        texts = [r['code'] for r in records]
        lexical = LexicalIndex(texts,ids)
        vectors = embed(texts)
        queries = [
            ('development','unique_in_order','unique_in_order'),('development','remove duplicates without changing order','unique_in_order'),
            ('development','find_device_by_serial','find_device_by_serial'),('development','look up a device by its serial number','find_device_by_serial'),
            ('test','read_json','read_json'),('test','load an object from a JSON file','read_json'),
            ('test','parse_iso_date','parse_iso_date'),('test','turn a year month day string into a date','parse_iso_date'),
            ('test','calculate a retry delay that doubles but has an upper limit','exponential_delay'),
        ]
        rows = []
        for split,text,gold in queries:
            dense = [id_ for id_,score in exact_search(embed([text],role='query')[0],vectors,ids,k=8)]
            words = [id_ for id_,score in lexical.search(text,k=8)]
            fused = [id_ for id_,score in reciprocal_rank_fusion([dense,words])]
            for method,ranked in [('BM25',words),('generic dense',dense),('RRF60',fused)]:
                rows.append({'split':split,'query':text,'method':method,'top_function':ranked[0],**ranking_metrics(ranked,{gold:1},k=3)})
        results = lesson.table(pd.DataFrame(rows))
        display(results[['split','query','method','top_function','recall','mrr']])
        display(lesson.table(pd.DataFrame(records)[['id','family','file','line']],'source-map'))
        fig,ax = plt.subplots(figsize=(9,4))
        results.groupby(['split','method']).mrr.mean().unstack().plot.bar(ax=ax)
        ax.set(ylabel='Mean reciprocal rank@3',ylim=(0,1.05),title='Exact symbols and natural-language behavior share one corpus')
        ax.tick_params(axis='x',rotation=0)
        ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Exact symbol queries give a cheap lexical baseline. Natural-language queries can benefit from embeddings, but the generic encoder used here was not selected through a code-retrieval benchmark. Function-level chunks retain signatures/docstrings; a call graph or surrounding module may be needed for behavior spread across functions.

        **Exercise:** rename a function while preserving its body, then compare symbol and intent queries on development data. Remove a docstring and inspect the change. Explain why retrieved code must not be automatically executed and why comments can contain prompt injection just like other documents.

        Production code search also requires branch/commit identity, generated/vendor-code filtering, repository permissions, secret handling, and evaluation across repositories rather than nearby duplicated functions. This lab's source-family split is a small controlled analogue, not proof of cross-repository generalization.
        """),
            code("""
        by_id = {r['id']:r for r in records}
        dev_families = {by_id[gold]['family'] for split,text,gold in queries if split=='development'}
        test_families = {by_id[gold]['family'] for split,text,gold in queries if split=='test'}
        manifest = lesson.finish({'eight_parsed_functions':len(records)==8,'source_family_split':not dev_families&test_families,'lexical_baseline_present':'BM25' in set(results.method),'source_lines_recorded':all(r['line']>0 for r in records)},observations={'encoder':'generic pinned MiniLM; no code-specific performance claim','code_execution':'AST parsing and text embedding only','k':3})
        """),
        ],
        docker=False,
    )

    notebook(
        "19c",
        "Aligned image and text search",
        "B",
        ["04", "05a"],
        [
            md("""
        ## The engineering question
        Can text retrieve images through a jointly trained representation, and which compositional details does it miss?

        **Learning outcomes:** encode actual images and text locally with the same CLIP projection space; store image vectors in Qdrant; inspect curated matches and missing-image failures; record preprocessing, attribution, and domain limitations.

        Bootstrap once with `uv run vmc bootstrap --model clip`. The CPU CLIP ViT-B/32 model produces normalized 512-dimensional image and text features. Its pinned processor handles resizing/cropping and a 77-token text limit. An unrelated text encoder's vectors cannot be substituted into this space.

        The corpus combines two licensed scikit-learn sample photographs with four programmatically drawn geometric fixtures. The shapes intentionally test color, count, and composition outside ordinary photo-caption examples. These are real model encodings, not manually assigned or random vectors. The photographs' original attribution/license record is copied into the run artifacts.
        """),
            code("""
        from uuid import uuid4
        from pathlib import Path
        import shutil
        import sklearn
        from sklearn.datasets import load_sample_image
        from PIL import Image,ImageDraw
        from qdrant_client import QdrantClient,models as qm
        from vector_masterclass.config import qdrant_url,model_config,fingerprint
        from vector_masterclass.multimodal import clip_encode
        from vector_masterclass.hardware import resolve_device
        ids = ['china','flower','red-square','blue-circle','two-red-squares','green-triangle']
        images = [Image.fromarray(load_sample_image('china.jpg')),Image.fromarray(load_sample_image('flower.jpg'))]
        for kind in ids[2:]:
            canvas = Image.new('RGB',(224,224),'white')
            draw = ImageDraw.Draw(canvas)
            if kind=='red-square': draw.rectangle((55,55,169,169),fill='red')
            elif kind=='blue-circle': draw.ellipse((45,45,179,179),fill='blue')
            elif kind=='two-red-squares':
                draw.rectangle((20,75,85,140),fill='red')
                draw.rectangle((135,75,200,140),fill='red')
            else: draw.polygon([(112,30),(30,190),(194,190)],fill='green')
            images.append(canvas)
        for id_,picture in zip(ids,images): picture.save(lesson.path/(id_+'.png'))
        shutil.copyfile(Path(sklearn.__file__).parent/'datasets/images/README.txt',lesson.path/'photo-attribution.txt')
        vectors = clip_encode(images=images)
        questions = [('a photograph of a Chinese temple building','china'),('a photograph of a flower','flower'),('a red square on a white background','red-square'),('a blue circle on a white background','blue-circle'),('two red squares on a white background','two-red-squares'),('a green triangle on a white background','green-triangle'),('a photograph of a dog',None)]
        query_vectors = clip_encode(texts=[q for q,gold in questions])
        client = QdrantClient(url=qdrant_url(),timeout=20)
        name = 'vmc_19c_'+uuid4().hex[:8]
        client.create_collection(name,vectors_config=qm.VectorParams(size=512,distance=qm.Distance.COSINE),metadata={'representation':fingerprint(model_config('clip'))})
        rows = []
        try:
            client.upsert(name,points=[qm.PointStruct(id=i,vector=v.tolist(),payload={'image_id':ids[i],'provenance':'sklearn CC-BY-2.0 sample' if i<2 else 'author-created geometric fixture'}) for i,v in enumerate(vectors)],wait=True)
            for (text,gold),query in zip(questions,query_vectors):
                hit = client.query_points(name,query=query.tolist(),limit=1,with_payload=True,search_params=qm.SearchParams(exact=True)).points[0]
                rows.append({'query':text,'expected_image':gold,'retrieved_image':hit.payload['image_id'],'cosine_score':hit.score,'hit_at_1':hit.payload['image_id']==gold if gold else None,'unsupported_forced_return':gold is None})
        finally:
            client.delete_collection(name)
            client.close()
        results = lesson.table(pd.DataFrame(rows))
        display(results)
        fig,axes = plt.subplots(2,3,figsize=(10,7))
        for ax,id_,picture in zip(axes.flat,ids,images):
            ax.imshow(picture)
            ax.set_title(id_)
            ax.axis('off')
        fig.suptitle('Actual model inputs: licensed photographs and authored geometric fixtures')
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        The absent dog image still produces a nearest neighbor. Retrieval is not an existence test. Compare the red-square and two-red-squares queries for compositional confusion, even if both happen to pass this run. CLIP cosine scores are not probabilities; a softmax over a chosen label list changes with that list and is not calibrated confidence.

        **Exercise:** add “a red circle” without adding such an image, then inspect the returned shape and score. Change image crop/background and compare ranking stability. Explain why layout-heavy document retrieval may need patch/token representations rather than a single global vector; lesson 23a examines that boundary.

        The photographs are distributed by scikit-learn under CC BY 2.0; see `photo-attribution.txt` for original Flickr links and attribution. The upstream flower record contains inconsistent account naming, which is retained verbatim rather than silently corrected. Shapes are author-created CC0 fixtures. [CLIP model card](https://huggingface.co/openai/clip-vit-base-patch32), [original CLIP research](https://arxiv.org/abs/2103.00020). This diagnostic set does not qualify a general image-search product.
        """),
            code("""
        manifest = lesson.finish({'real_aligned_dimensions':vectors.shape==(6,512) and query_vectors.shape==(7,512),'normalized_features':bool(np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)),'all_curated_queries_measured':len(results)==7,'missing_image_failure_recorded':int(results.unsupported_forced_return.sum())==1,'attribution_preserved':(lesson.path/'photo-attribution.txt').exists()},observations={'model':model_config('clip'),'device':resolve_device('clip'),'photo_source':'scikit-learn packaged samples; original license/attribution copied','shape_source':'author-created geometric fixtures','known_match_hit_rate':float(results.hit_at_1.dropna().mean())})
        """),
        ],
    )
    print("Authored modality lessons 19a, 19b, 19c")


if __name__ == "__main__":
    main()
